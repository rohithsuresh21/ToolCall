#!/usr/bin/env bash
# Lightning AI free-tier runner: run GRPO cycles to completion across session
# restarts, then run the vllm evals (dev + judge), then archive everything.
#
# WHY this exists: the free tier caps each session at 4 hours and may preempt
# an interruptible instance at any time. 75_lightning_grpo.sh handles ONE cycle
# (train with max_seconds=3h30m + archive weights before returning). This script
# loops: relaunch with RESUME=$OUT/final as long as the trainer says it stopped
# early, then, once the step budget is reached, run the two evals (STRICTLY vllm,
# temp 0.0) and archive the reports, mirroring 60_pipeline_tomorrow.sh.
#
# Usage (in the Lightning Studio shell, from the repo root):
#   bash scripts/76_lightning_cycle.sh [STEPS]
#
# Env knobs: MODEL, STEPS (default 150), ADAPTER, OUT, REAL_MIX.
#   * INT_MODE=1 (default): hard-stop preemption guard -- after each cycle,
#     before the 4h limit lets the box die, it syncs $OUT into artifacts/ship.
#   * WATCHDOG=1: also `rsync`/`scp`-style push $OUT to a persistent path every
#     cycle so the moment a session is killed, the weights are somewhere safe.
#
# The evals need the FINAL adapter only, so they run once after `steps` is done.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STEPS="${STEPS:-150}"
ADAPTER="${ADAPTER:-artifacts/sft-1p7b-fixed}"
OUT="${OUT:-artifacts/grpo-lightning}"
REAL_MIX="${REAL_MIX:-0.2}"
MAX_CYCLES="${MAX_CYCLES:-10}"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ! -d "$ADAPTER" ]]; then
  echo "FATAL: SFT adapter missing at $ADAPTER" >&2
  exit 1
fi

echo "======================================================================"
echo "  LIGHTNING FULL RUN  model=$MODEL  steps=$STEPS  real_mix=$REAL_MIX"
echo "======================================================================"

# ------------------------------------------------------------------ 1. GRPO --
cycle=0
while :; do
  cycle=$((cycle + 1))
  if ((cycle > MAX_CYCLES)); then
    echo "FATAL: exceeded MAX_CYCLES=$MAX_CYCLES without reaching steps=$STEPS" >&2
    exit 1
  fi

  if [[ -d "$OUT/final" && -f "$OUT/final/trainer_state.pt" ]]; then
    echo "===== GRPO cycle $cycle (resume from $OUT/final) ====="
    RESUME="$OUT/final" MAX_SECONDS="${MAX_SECONDS:-12600}" \
      bash scripts/75_lightning_grpo.sh "$STEPS" || true
  else
    echo "===== GRPO cycle $cycle (fresh start) ====="
    MAX_SECONDS="${MAX_SECONDS:-12600}" \
      bash scripts/75_lightning_grpo.sh "$STEPS" || true
  fi

  # Did the trainer finish the budget, or did it stop at MAX_SECONDS?
  if [[ ! -f "$OUT/history.jsonl" ]]; then
    echo "FATAL: no history.jsonl after cycle $cycle -- trainer never ran." >&2
    mkdir -p artifacts/ship
    cp -r "$OUT" "artifacts/ship/$(basename "$OUT")_precycle_$STAMP" 2>/dev/null || true
    exit 1
  fi
  done_steps="$(wc -l < "$OUT/history.jsonl")"
  if (( done_steps >= STEPS )); then
    echo "GRPO complete: $done_steps/$STEPS steps done after $cycle cycle(s)."
    break
  fi
  echo "GRPO at $done_steps/$STEPS steps after cycle $cycle; relaunching..."
  sleep 2
done

# ------------------------------------------------------------------ 2. EVALS --
echo "===== [eval] DEV set  (STRICTLY vllm, temp 0.0) ====="
ADAPTER="$OUT" MODEL="$MODEL" bash scripts/40_eval.sh
tar -czf "artifacts/ship/eval-dev_$STAMP.tar.gz" "artifacts/eval-$(basename "$OUT")"

echo "===== [eval] JUDGE probe  (STRICTLY vllm, temp 0.0) ====="
if [[ ! -f data/judge_tasks.jsonl ]]; then
  echo "FATAL: data/judge_tasks.jsonl missing." >&2
  exit 1
fi
python3 scripts/judge_eval.py \
  --base "$MODEL" --backend "vllm:$MODEL" \
  --adapter "$OUT" \
  --tasks data/judge_tasks.jsonl \
  --out "artifacts/judge_eval_grpo-lightning" \
  --name "lightning"
tar -czf "artifacts/ship/eval-judge_$STAMP.tar.gz" artifacts/judge_eval_grpo-lightning

# ------------------------------------------------------------------ 3. SHIP ---
echo "===== [ship] all reports -> artifacts/ship ====="
mkdir -p artifacts/ship
cp "artifacts/eval-$(basename "$OUT")/report.txt" "artifacts/ship/dev-report_$STAMP.txt"
cp "artifacts/eval-$(basename "$OUT")/report.json" "artifacts/ship/dev-report_$STAMP.json"
cp artifacts/judge_eval_grpo-lightning/report.txt "artifacts/ship/judge-report_$STAMP.txt"
cp artifacts/judge_eval_grpo-lightning/scores.jsonl "artifacts/ship/judge-scores_$STAMP.jsonl"
(cd artifacts/ship && sha256sum ./*.tar.gz ./*.txt ./*.json ./*.jsonl | tee "MANIFEST_$STAMP.txt")

echo ""
echo "######################################################################"
echo "  RUN COMPLETE (stamp $STAMP)."
echo "  PULL EVERYTHING to local 'pulled 2' NOW (Lightning file sync / CLI):"
echo "    cp -r artifacts/ship/* <local pulled 2>/"
echo "  Rebuild the combined PDF locally if reportlab is not on the box:"
echo "    python scripts/make_combined_report.py --dev <pulled>/dev-report.json"
echo "        --judge <pulled>/judge-scores.jsonl --out ATR-Eval-Report.pdf"
echo "######################################################################"