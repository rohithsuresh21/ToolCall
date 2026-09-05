#!/usr/bin/env bash
# Phase 2b: dev eval is DONE (artifacts/eval-final). Archive it, run judge eval
# (vllm), save reports, combined PDF, ship everything.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STEPS="${STEPS:-30}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SHIP="$PWD/artifacts/ship"
GRPO="artifacts/grpo-planb-step3"
mkdir -p "$SHIP"

echo "===== [1] archive dev eval (artifacts/eval-final) ====="
tar -czf "$SHIP/eval-dev_$STAMP.tar.gz" artifacts/eval-final
(cd artifacts/ship && sha256sum "eval-dev_$STAMP.tar.gz" | tee -a "MANIFEST_$STAMP.txt")
echo "  -> $SHIP/eval-dev_$STAMP.tar.gz"

echo "===== [2] eval JUDGE benchmark  (STRICTLY vllm backend, temp 0.0) ====="
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

echo "===== [3] saving all eval reports into ship/ ====="
cp artifacts/eval-final/report.txt "$SHIP/dev-report_$STAMP.txt"
cp artifacts/eval-final/report.json "$SHIP/dev-report_$STAMP.json"
cp artifacts/judge_eval_grpo-planb/report.txt "$SHIP/judge-report_$STAMP.txt"
cp artifacts/judge_eval_grpo-planb/scores.jsonl "$SHIP/judge-scores_$STAMP.jsonl"
(cd artifacts/ship && sha256sum ./*.tar.gz ./*.txt ./*.json ./*.jsonl | tee "MANIFEST_$STAMP.txt")
echo "  reports saved -> ship/"

echo "===== [4] combined professional report (dev + judge) PDF ====="
if python3 scripts/make_combined_report.py \
    --dev artifacts/eval-final/report.json \
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
echo "  PHASE 2B COMPLETE (stamp $STAMP). PULL NOW:"
echo "    powershell -File scripts/pull_all.ps1"
echo "######################################################################"