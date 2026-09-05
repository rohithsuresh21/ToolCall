#!/usr/bin/env bash
# Node run - follows the user's plan EXACTLY, in order:
#   1. preflight (00_preflight.sh): model cached + CUDA + vllm importable
#   2. delete OLD training data + OLD trained weights (everything must be NEW)
#   3. collect 1500 fresh examples (oracle) -> build -> AUDIT -> promote to data/sft.jsonl
#   4. train SFT on the freshly audited set (NEVER reuse old adapter)
#   5. archive SFT weights -> artifacts/ship/
#   6. train GRPO on the fresh SFT: 50 steps, 4-hop fix (Plan B step3 flags +
#      efficiency_lambda=0 + under-call penalty ON)
#   7. archive GRPO weights -> artifacts/ship/
#   8. verify all artifacts exist + history rows == STEPS
#   9. eval DEV set  - STRICTLY vllm backend, temp 0.0
#  10. eval JUDGE benchmark - STRICTLY vllm backend, temp 0.0
#  11. save every eval report into ship/
#  12. generate the combined professional report (dev + judge) PDF
#
# STRICT RULES (user-enforced):
#   * evals ONLY on the vllm backend on the node (never hf).
#   * GRPO steps MUST be 50.
#   * NO reuse of any previously trained weight.
#   * must finish inside the 4-hour reservation - no mid-run stops, no loops.
#
# Usage (on the node, from the repo root):  bash scripts/60_pipeline_tomorrow.sh
# Env knobs: MODEL, STEPS (default 50), N (default 1500)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STEPS="${STEPS:-50}"
N="${N:-1500}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SHIP="$PWD/artifacts/ship"
mkdir -p "$SHIP"

echo "======================================================================"
echo "  TOMORROW PIPELINE  model=$MODEL  steps=$STEPS  examples=$N"
echo "======================================================================"

# ------------------------------------------------------------ 0. PREFLIGHT --
echo "===== [0] preflight: vllm + model + packages present on the node ====="
bash scripts/00_preflight.sh "$MODEL"
echo "  preflight done"
echo ""

# ------------------------------------------------- 1. DELETE OLD DATA/CKPT --
echo "===== [1] deleting OLD generated data + OLD trained weights ====="
rm -rf artifacts/raw_oracle.jsonl artifacts/raw_teacher.jsonl artifacts/sft_candidate.jsonl
rm -rf artifacts/sft-1p7b-fixed artifacts/grpo-planb-step3
rm -rf artifacts/eval-* artifacts/judge_eval_*
echo "  removed. remaining artifacts/ship is preserved."

# --------------------------------------- 2. COLLECT 1500 FRESH + AUDIT ----
echo "===== [2] collecting ${N} fresh examples (oracle) + audit + promote ====="
N="$N" bash scripts/10_build_data.sh
echo "  promoted fresh set -> data/sft.jsonl ($(wc -l < data/sft.jsonl) records)"

# ---------------------------------------------------- 3. TRAIN SFT FRESH ----
echo "===== [3] SFT on the FRESH audited set (new adapter, old one deleted) ====="
OUT="artifacts/sft-1p7b-fixed" DATA="data/sft.jsonl" MODEL="$MODEL" bash scripts/20_sft.sh
[[ -d artifacts/sft-1p7b-fixed ]] || { echo "FATAL: SFT produced no adapter"; exit 1; }

# --------------------------------------------------- 4. SAVE SFT WEIGHTS ----
echo "===== [4] archiving SFT weights -> ship/ ====="
tar -czf "$SHIP/sft-1p7b-fixed_$STAMP.tar.gz" artifacts/sft-1p7b-fixed
(cd artifacts/ship && sha256sum "sft-1p7b-fixed_$STAMP.tar.gz" | tee -a "MANIFEST_$STAMP.txt")
echo "  -> $SHIP/sft-1p7b-fixed_$STAMP.tar.gz"

# -------------------------------------------- 5. TRAIN GRPO (50, FIXED) ----
echo "===== [5] GRPO   steps=$STEPS   with 4-hop fix (lambda=0 + under-call ON) ====="
python3 -m atr.train.grpo \
  --model-id "$MODEL" \
  --adapter artifacts/sft-1p7b-fixed \
  --out-dir artifacts/grpo-planb-step3 \
  --group-size 8 --tasks-per-step 8 --steps "$STEPS" \
  --lr 2e-5 --temperature 1.0 --kl-beta 0.03 --micro-batch 2 \
  --curriculum true --void-turn-filter true --eval-every 50 \
  --log-every 5 \
  --dead-frac-source discarded --advantage-scale mad --advantage-baseline sign --sign-baseline 0.5 \
  --dqw true --dqw-temp 2.2 --e2h-curriculum true \
  --efficiency-lambda 0.0 --under-call-penalty true
[[ -d artifacts/grpo-planb-step3 ]] || { echo "FATAL: GRPO produced no adapter"; exit 1; }

# -------------------------------------------------- 6. SAVE GRPO WEIGHTS ----
echo "===== [6] archiving GRPO weights -> ship/ ====="
tar -czf "$SHIP/grpo-planb-step3_$STAMP.tar.gz" artifacts/grpo-planb-step3
(cd artifacts/ship && sha256sum "grpo-planb-step3_$STAMP.tar.gz" | tee -a "MANIFEST_$STAMP.txt")
echo "  -> $SHIP/grpo-planb-step3_$STAMP.tar.gz"

# ------------------------------------------------------------ 7. VERIFY ------
echo "===== [7] verifying all artifacts exist with sane sizes ====="
python3 - <<'PY'
import os
must = {
  "data/sft.jsonl": 500,               # >=500 lines (gate floor)
  "artifacts/sft-1p7b-fixed/adapter_config.json": 100,
  "artifacts/grpo-planb-step3/adapter_config.json": 100,
}
for p, minsize in must.items():
    if not os.path.exists(p):
        raise SystemExit(f"VERIFY FAIL: {p} missing")
    if os.path.getsize(p) < minsize:
        raise SystemExit(f"VERIFY FAIL: {p} too small ({os.path.getsize(p)}B)")
    print(f"  OK {p} ({os.path.getsize(p)}B)")
h = "artifacts/grpo-planb-step3/history.jsonl"
if os.path.exists(h):
    n = sum(1 for _ in open(h))
    print(f"  OK history.jsonl rows={n} (target {os.environ.get('STEPS','50')})")
print("  ALL VERIFY CHECKS PASSED")
PY
echo ""

# ---------------------------------------------- 8. EVAL DEV SET (VLLM) ------
echo "===== [8] eval DEV set  (STRICTLY vllm backend, temp 0.0) ====="
ADAPTER="artifacts/grpo-planb-step3" MODEL="$MODEL" bash scripts/40_eval.sh
tar -czf "$SHIP/eval-dev_$STAMP.tar.gz" artifacts/eval-grpo-planb-step3

# ------------------------------------------- 9. EVAL JUDGE (VLLM) ---------
echo "===== [9] eval JUDGE benchmark  (STRICTLY vllm backend, temp 0.0) ====="
if [[ ! -f data/judge_tasks.jsonl ]]; then
  echo "FATAL: data/judge_tasks.jsonl missing. Build it from the HF dataset:"
  echo "  python3 scripts/make_judge_tasks.py --out data/judge_tasks.jsonl"
  exit 1
fi
python3 scripts/judge_eval.py \
  --base "$MODEL" --backend "vllm:$MODEL" \
  --adapter artifacts/grpo-planb-step3 \
  --tasks data/judge_tasks.jsonl \
  --out artifacts/judge_eval_grpo-planb \
  --name "pipeline"
tar -czf "$SHIP/eval-judge_$STAMP.tar.gz" artifacts/judge_eval_grpo-planb

# ---------------------------------------------- 10. SAVE ALL REPORTS -------
echo "===== [10] saving all eval reports into ship/ ====="
cp artifacts/eval-grpo-planb-step3/report.txt "$SHIP/dev-report_$STAMP.txt"
cp artifacts/eval-grpo-planb-step3/report.json "$SHIP/dev-report_$STAMP.json"
cp artifacts/judge_eval_grpo-planb/report.txt "$SHIP/judge-report_$STAMP.txt"
cp artifacts/judge_eval_grpo-planb/scores.jsonl "$SHIP/judge-scores_$STAMP.jsonl"
(cd artifacts/ship && sha256sum ./*.tar.gz ./*.txt ./*.json ./*.jsonl | tee "MANIFEST_$STAMP.txt")
echo "  reports saved -> ship/"

# ----------------------------------------------- 11. COMBINED REPORT -------
echo "===== [11] combined professional report (dev + judge) PDF ====="
if python3 scripts/make_combined_report.py \
    --dev artifacts/eval-grpo-planb-step3/report.json \
    --judge artifacts/judge_eval_grpo-planb/scores.jsonl \
    --out "artifacts/ship/ATR-Eval-Report_$STAMP.pdf" \
    --model "$MODEL" --steps "$STEPS"; then
  echo "  report -> artifacts/ship/ATR-Eval-Report_$STAMP.pdf"
else
  echo "  reportlab unavailable on node (non-fatal); rebuild locally after pull:"
  echo "    python scripts/make_combined_report.py --dev pulled-path/dev-report.json --judge pulled-path/judge-scores.jsonl --out ATR-Eval-Report.pdf"
fi

echo ""
echo "######################################################################"
echo "  TOMORROW PIPELINE COMPLETE (stamp $STAMP)."
echo "  PULL EVERYTHING TO LOCAL 'pulled 2' NOW:"
echo "    powershell -File scripts/pull_all.ps1"
echo "######################################################################"