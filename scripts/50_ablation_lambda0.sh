#!/usr/bin/env bash
# Ablation: isolate whether F7 tool-efficiency (efficiency_lambda) is the dominant
# lever driving the 4-hop "stop at 3 calls" collapse, and whether the new under-call
# penalty (missing_calls) changes the picture.
#
# Design (three arms, A/B/C per the design review):
#   ARM          efficiency_lambda   missing_calls      = Result
#   A (current)  0.15                OFF (baseline)      = collapse (stop at 3)
#   B            0.00                OFF                 = isolates efficiency lambda
#   C            0.00                ON                  = full recommended fix
# Each arm trains GRPO PlanB step3 from the SAME SFT adapter, then evals dev + judge
# at temperature 0.0 (greedy) AND 1.0 (sampled), so a greedy-decoding artifact can be
# separated from a genuine policy collapse.
#
#   - If temp>0 eval still shows early termination / no 4-hop gains, the collapse is
#     policy-level (reward-driven), not a decoding artifact.
#   - If temp>0 "un-collapses" (call variance), part of the dev collapse was decoding.
# ARM="C" runs only that arm; ARM="all" (default) runs A, B and C.
#
# NOTE: judge success on the current checkpoint is largely UNDER-CALL shortcuts on a
# leaky benchmark (3 of 4 judge wins used fewer calls than the hop label) -- the judge
# set should be re-verified for leakage before trusting its numbers as reasoning signal.
#
# Prereqs: run on the reservation node, from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
ADAPTER="${ADAPTER:-artifacts/sft-1p7b-fixed}"       # same SFT base as the baseline run
STEPS="${STEPS:-50}"
STAGE="${STAGE:-step3}"
ARM="${ARM:-all}"                                       # all | A | B | C

# Plan B flags (mirror 31_grpo_planb.sh so this is an apples-to-apples A/B).
PB="--dead-frac-source discarded --advantage-scale mad --advantage-baseline sign --sign-baseline 0.5"
if [ "$STAGE" != "step1" ]; then
  PB="$PB --dqw true --dqw-temp 2.2"
fi
if [ "$STAGE" = "step3" ]; then
  PB="$PB --e2h-curriculum true"
fi

run_arm () {
  local ARM_ID="$1" LAM="$2" UC="$3"
  local OUT="artifacts/grpo-ablation-$ARM_ID"
  local EFF="--efficiency-lambda $LAM"
  local UC_FLAG=""
  if [ "$UC" = "1" ]; then
    UC_FLAG="--under-call-penalty true"
  fi
  echo "==================== ARM $ARM_ID  (efficiency_lambda=$LAM  missing_calls=$UC) ===================="
  python3 -m atr.train.grpo \
    --model-id "$MODEL" \
    --adapter "$ADAPTER" \
    --out-dir "$OUT" \
    --group-size 8 --tasks-per-step 8 --steps "$STEPS" \
    --lr 2e-5 --temperature 1.0 --kl-beta 0.03 --micro-batch 2 \
    --curriculum true --void-turn-filter true --eval-every 50 \
    --log-every 5 \
    $EFF $UC_FLAG $PB

  # ---- EVAL at two temperatures (decode artifact vs policy collapse) ----
  for TEMP in 0.0 1.0; do
    local suffix="$(echo "$TEMP" | tr '.' '_')"
    echo "=== arm $ARM_ID: eval dev  temp=$TEMP ==="
    python3 -m atr.cli eval --dev --n-per-type 20 \
      --backend "hf:$MODEL" --adapter "$OUT/final" \
      --temperature "$TEMP" --max-new-tokens 512 \
      --out "artifacts/eval-ablation-$ARM_ID-dev-t$suffix"

    echo "=== arm $ARM_ID: eval judge  temp=$TEMP ==="
    python3 -m atr.cli eval \
      --backend "hf:$MODEL" --adapter "$OUT/final" \
      --temperature "$TEMP" --max-new-tokens 512 \
      --tasks artifacts/judge_tasks.jsonl \
      --out "artifacts/eval-ablation-$ARM_ID-judge-t$suffix"
  done
}

case "$ARM" in
  all) run_arm A 0.15 0; run_arm B 0.00 0; run_arm C 0.00 1 ;;
  A)   run_arm A 0.15 0 ;;
  B)   run_arm B 0.00 0 ;;
  C)   run_arm C 0.00 1 ;;
  *)   echo "unknown ARM=$ARM (use all|A|B|C)"; exit 2 ;;
esac

# ---- archive for local pull (never skip: the node vanishes) ----
TSTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p artifacts/ship
tar -czf "artifacts/ship/grpo-ablation_$TSTAMP.tar.gz" artifacts/grpo-ablation-*
echo "=== ABLATION DONE ==="
echo "PULL: scp -P 22013 gpu17@10.214.5.55:~/ToolCall/artifacts/ship/grpo-ablation_$TSTAMP.tar.gz ."
