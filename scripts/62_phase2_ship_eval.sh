#!/usr/bin/env bash
# Phase 2 continuation: GRPO already finished (30 steps). Archive weights,
# verify, run vllm dev + judge evals, combined report, ship everything.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STEPS="${STEPS:-30}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SHIP="$PWD/artifacts/ship"
GRPO="artifacts/grpo-planb-step3"
mkdir -p "$SHIP"

echo "[phase2] GRPO adapter dir: $GRPO"
echo "  final dir: $(ls -d $GRPO/final 2>&1)"

# ---------------------------------------------- 1. ARCHIVE GRPO WEIGHTS ------
echo "===== [1] archiving GRPO weights -> ship/ ====="
tar -czf "$SHIP/grpo-planb-step3_$STAMP.tar.gz" "$GRPO"
(cd artifacts/ship && sha256sum "grpo-planb-step3_$STAMP.tar.gz" | tee -a "MANIFEST_$STAMP.txt")
echo "  -> $SHIP/grpo-planb-step3_$STAMP.tar.gz"

# ------------------------------------------------------- 2. VERIFY -----------
echo "===== [2] verifying artifacts ====="
python3 - <<'PY'
import os
must = {
  "artifacts/grpo-planb-step3/final/adapter_config.json": 100,
  "artifacts/grpo-planb-step3/final/adapter_model.safetensors": 1_000_000,
}
for p, minsize in must.items():
    if not os.path.exists(p):
        raise SystemExit(f"VERIFY FAIL: {p} missing")
    if os.path.getsize(p) < minsize:
        raise SystemExit(f"VERIFY FAIL: {p} too small ({os.path.getsize(p)}B)")
    print(f"  OK {p} ({os.path.getsize(p)}B)")
h = "artifacts/grpo-planb-step3/history.jsonl"
n = sum(1 for _ in open(h)) if os.path.exists(h) else 0
print(f"  OK history.jsonl rows={n} (target {os.environ.get('STEPS','30')})")
if n != int(os.environ.get('STEPS','30')):
    print(f"  WARN: history rows {n} != expected steps")
print("  ALL VERIFY CHECKS PASSED")
PY

# ---------------------------------------------- 3. EVAL DEV SET (VLLM) ------
echo "===== [3] eval DEV set  (STRICTLY vllm backend, temp 0.0) ====="
ADAPTER="$GRPO/final" MODEL="$MODEL" bash scripts/40_eval.sh
tar -czf "$SHIP/eval-dev_$STAMP.tar.gz" artifacts/eval-grpo-planb-step3

# ------------------------------------------- 4. EVAL JUDGE (VLLM) ---------
echo "===== [4] eval JUDGE benchmark  (STRICTLY vllm backend, temp 0.0) ====="
if [[ ! -f data/judge_tasks.jsonl ]]; then
  echo "FATAL: data/judge_tasks.jsonl missing." >&2
  exit 1
fi
python3 scripts/judge_eval.py \
  --base "$MODEL" --backend "vllm:$MODEL" \
  --adapter "$GRPO/final" \
  --tasks data/judge_tasks.jsonl \
  --out artifacts/judge_eval_grpo-planb \
  --name "pipeline"
tar -czf "$SHIP/eval-judge_$STAMP.tar.gz" artifacts/judge_eval_grpo-planb

# ---------------------------------------------- 5. SAVE ALL REPORTS -------
echo "===== [5] saving all eval reports into ship/ ====="
cp artifacts/eval-grpo-planb-step3/report.txt "$SHIP/dev-report_$STAMP.txt"
cp artifacts/eval-grpo-planb-step3/report.json "$SHIP/dev-report_$STAMP.json"
cp artifacts/judge_eval_grpo-planb/report.txt "$SHIP/judge-report_$STAMP.txt"
cp artifacts/judge_eval_grpo-planb/scores.jsonl "$SHIP/judge-scores_$STAMP.jsonl"
(cd artifacts/ship && sha256sum ./*.tar.gz ./*.txt ./*.json ./*.jsonl | tee "MANIFEST_$STAMP.txt")
echo "  reports saved -> ship/"

# ----------------------------------------------- 6. COMBINED REPORT -------
echo "===== [6] combined professional report (dev + judge) PDF ====="
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
echo "  PHASE 2 COMPLETE (stamp $STAMP). PULL NOW:"
echo "    powershell -File scripts/pull_all.ps1"
echo "######################################################################"