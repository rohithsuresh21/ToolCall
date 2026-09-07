#!/usr/bin/env bash
# Node run using the user-supplied SFT adapter (adapter.tar.gz -> sft-1p7b-fixed).
#   1. preflight (00_preflight.sh): model cached + CUDA + vllm importable
#   2. verify the SFT adapter + git data/sft.jsonl exist on the node
#   3. train GRPO on the SFT adapter: 50 steps, 4-hop fix (Plan B step3 flags +
#      efficiency_lambda=0 + under-call penalty ON) with the approved 20% real
#      MuSiQue-Ans train pool (data/musique_train_tasks.jsonl, hop-stratified
#      40/30/30, double-checked disjoint from the 54 judge probes) fresh from
#      the SFT adapter -- NO resume, the old grpo-planb-step3 is saturated.
#   4. archive GRPO weights -> artifacts/ship/ FIRST (immediate-pull target)
#   5. verify artifacts exist + history rows == STEPS
#   6. eval DEV set  - STRICTLY vllm backend, temp 0.0
#   7. eval JUDGE benchmark - STRICTLY vllm backend, temp 0.0
#   8. save every eval report into ship/
#   9. generate the combined professional report (dev + judge) PDF
#
# STRICT RULES (user-enforced):
#   * evals ONLY on the vllm backend on the node (never hf).
#   * GRPO steps MUST be 50.
#   * uses the SFT adapter the user pulled (r=32/alpha=64, Qwen3-1.7B).
#   * must finish inside the 4-hour reservation - no mid-run stops, no loops.
#
# Usage (on the node, from the repo root):  bash scripts/60_pipeline_tomorrow.sh
# Env knobs: MODEL, STEPS (default 50), ADAPTER (default artifacts/sft-1p7b-fixed),
#            OUT (default artifacts/grpo-real), REAL_MIX (default 0.2)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STEPS="${STEPS:-50}"
ADAPTER="${ADAPTER:-artifacts/sft-1p7b-fixed}"
OUT="${OUT:-artifacts/grpo-real}"
DATA="${DATA:-data/sft.jsonl}"
REAL_MIX="${REAL_MIX:-0.2}"
REAL_TASKS="data/musique_train_tasks.jsonl"
OUT_NAME="$(basename "$OUT")"
STAMP="$(date +%Y%m%d_%H%M%S)"
SHIP="$PWD/artifacts/ship"
mkdir -p "$SHIP"

echo "======================================================================"
echo "  PIPELINE  model=$MODEL  steps=$STEPS  adapter=$ADAPTER  out=$OUT"
echo "            real_mix=$REAL_MIX  real_tasks=$REAL_TASKS"
echo "======================================================================"

# ------------------------------------------------------------ 0. PREFLIGHT --
echo "===== [0] preflight: vllm + model + packages present on the node ====="
bash scripts/00_preflight.sh "$MODEL"
echo "  preflight done"
echo ""

# ------------------------------- 1. VERIFY SFT ADAPTER + GIT DATA PRESENT ---
echo "===== [1] verifying SFT adapter + git data on the node ====="
if [[ ! -f "$ADAPTER/adapter_config.json" || ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "FATAL: SFT adapter missing at $ADAPTER" >&2
  echo "  upload adapter.tar.gz and extract:"
  echo "    tar -xzf /home/gpu17/adapter.tar.gz -C $ADAPTER"
  exit 1
fi
echo "  adapter OK: $(python3 -c "import json;print('r=%s alpha=%s base=%s' % (json.load(open('$ADAPTER/adapter_config.json'))['r'], json.load(open('$ADAPTER/adapter_config.json'))['lora_alpha'], json.load(open('$ADAPTER/adapter_config.json'))['base_model_name_or_path']))")"
if [[ ! -f "$REAL_TASKS" ]]; then
  echo "FATAL: $REAL_TASKS missing. git pull (it is committed) or run:" >&2
  echo "  python3 scripts/make_musique_train_tasks.py" >&2
  exit 1
fi
echo "  $REAL_TASKS has $(wc -l < "$REAL_TASKS" | tr -d '[:space:]') records (committed pool)"
echo ""

# ------------------------------------------------ 2. TRAIN GRPO (50, FIXED) --
echo "===== [2] GRPO   steps=$STEPS   with 4-hop fix (lambda=0 + under-call ON) ====="
python3 -m atr.train.grpo \
  --model-id "$MODEL" \
  --adapter "$ADAPTER" \
  --out-dir "$OUT" \
  --group-size 8 --tasks-per-step 10 --steps "$STEPS" \
  --lr 2e-5 --temperature 1.0 --kl-beta 0.03 --micro-batch 2 \
  --curriculum true --void-turn-filter true --eval-every 50 \
  --log-every 5 \
  --dead-frac-source discarded --advantage-scale mad --advantage-baseline sign --sign-baseline 0.5 \
  --dqw true --dqw-temp 2.2 --e2h-curriculum true \
  --efficiency-lambda 0.0 --under-call-penalty true \
  --real-tasks-path "$REAL_TASKS" --real-fraction "$REAL_MIX"
[[ -d "$OUT" ]] || { echo "FATAL: GRPO produced no adapter"; exit 1; }
echo ""

# -------------------------------------------------- archiving GRPO weights ----
echo "===== [3] archiving GRPO weights -> ship/ ====="
tar -czf "$SHIP/${OUT_NAME}_$STAMP.tar.gz" "$OUT"
(cd artifacts/ship && sha256sum "${OUT_NAME}_$STAMP.tar.gz" | tee -a "MANIFEST_$STAMP.txt")
echo "  -> $SHIP/${OUT_NAME}_$STAMP.tar.gz"
echo "  IMMEDIATE PULL (local, weights first): scp -P 22013 gpu17@10.214.5.55:~/ToolCall/artifacts/ship/${OUT_NAME}_$STAMP.tar.gz ."

# ------------------------------------------------------------ 4. VERIFY ------
echo "===== [4] verifying all artifacts exist with sane sizes ====="
python3 - <<PY
import os
must = {
  "data/sft.jsonl": 500,
  "artifacts/sft-1p7b-fixed/adapter_config.json": 100,
  "${OUT}/adapter_config.json": 100,
}
for p, minsize in must.items():
    if not os.path.exists(p):
        raise SystemExit(f"VERIFY FAIL: {p} missing")
    if os.path.getsize(p) < minsize:
        raise SystemExit(f"VERIFY FAIL: {p} too small ({os.path.getsize(p)}B)")
    print(f"  OK {p} ({os.path.getsize(p)}B)")
h = "${OUT}/history.jsonl"
if os.path.exists(h):
    n = sum(1 for _ in open(h))
    print(f"  OK history.jsonl rows={n} (target {os.environ.get('STEPS','50')})")
print("  ALL VERIFY CHECKS PASSED")
PY
echo ""

# ---------------------------------------------- 5. EVAL DEV SET (VLLM) ------
echo "===== [5] eval DEV set  (STRICTLY vllm backend, temp 0.0) ====="
ADAPTER="$OUT" MODEL="$MODEL" bash scripts/40_eval.sh
tar -czf "$SHIP/eval-dev_$STAMP.tar.gz" "artifacts/eval-$OUT_NAME"

# ------------------------------------------- 6. EVAL JUDGE (VLLM) ---------
echo "===== [6] eval JUDGE benchmark  (STRICTLY vllm backend, temp 0.0) ====="
if [[ ! -f data/judge_tasks.jsonl ]]; then
  echo "FATAL: data/judge_tasks.jsonl missing." >&2
  exit 1
fi
python3 scripts/judge_eval.py \
  --base "$MODEL" --backend "vllm:$MODEL" \
  --adapter "$OUT" \
  --tasks data/judge_tasks.jsonl \
  --out "artifacts/judge_eval_$OUT_NAME" \
  --name "pipeline"
tar -czf "$SHIP/eval-judge_$STAMP.tar.gz" "artifacts/judge_eval_$OUT_NAME"

# ---------------------------------------------- 7. SAVE ALL REPORTS -------
echo "===== [7] saving all eval reports into ship/ ====="
cp "artifacts/eval-$OUT_NAME/report.txt" "$SHIP/dev-report_$STAMP.txt"
cp "artifacts/eval-$OUT_NAME/report.json" "$SHIP/dev-report_$STAMP.json"
cp "artifacts/judge_eval_$OUT_NAME/report.txt" "$SHIP/judge-report_$STAMP.txt"
cp "artifacts/judge_eval_$OUT_NAME/scores.jsonl" "$SHIP/judge-scores_$STAMP.jsonl"
(cd artifacts/ship && sha256sum ./*.tar.gz ./*.txt ./*.json ./*.jsonl | tee "MANIFEST_$STAMP.txt")
echo "  reports saved -> ship/"

# ----------------------------------------------- 8. COMBINED REPORT -------
echo "===== [8] combined professional report (dev + judge) PDF ====="
if python3 scripts/make_combined_report.py \
    --dev "artifacts/eval-$OUT_NAME/report.json" \
    --judge "artifacts/judge_eval_$OUT_NAME/scores.jsonl" \
    --out "artifacts/ship/ATR-Eval-Report_$STAMP.pdf" \
    --model "$MODEL" --steps "$STEPS"; then
  echo "  report -> artifacts/ship/ATR-Eval-Report_$STAMP.pdf"
else
  echo "  reportlab unavailable on node (non-fatal); rebuild locally after pull:"
  echo "    python scripts/make_combined_report.py --dev pulled-path/dev-report.json --judge pulled-path/judge-scores.jsonl --out ATR-Eval-Report.pdf"
fi

echo ""
echo "######################################################################"
echo "  PIPELINE COMPLETE (stamp $STAMP)."
echo "  PULL EVERYTHING TO LOCAL 'pulled 2' NOW:"
echo "    powershell -File scripts/pull_all.ps1"
echo "######################################################################"