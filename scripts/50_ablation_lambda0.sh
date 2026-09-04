#!/usr/bin/env bash
# Ablation: isolate whether F7 tool-efficiency (efficiency_lambda) is the dominant
# lever driving the 4-hop "stop at 3 calls" collapse.
#
# Design (see session notes):
#   * Train GRPO PlanB step3 with efficiency_lambda=0 (the ablation arm).
#   * Eval the checkpoint at temperature 0.0 (greedy - the current eval default)
#     AND temperature 1.0 (sampled), on BOTH dev and judge, so we can separate a
#     greedy-decoding artifact from a genuine policy collapse.
#       - If temp>0 eval still shows early termination / no 4-hop gains, the
#         collapse is policy-level (reward-driven), not a decoding artifact.
#       - If temp>0 "un-collapses" like the judge set did (variance in calls,
#         some 4-hop wins), part of the dev "collapse" was greedy decoding.
#   * Optional second arm: --efficiency-lambda 0 PLUS the new under-call penalty
#     (p_per_missing_call), to test the full fix. Set UNDER_CALL=1 to enable.
#
# Prereqs: run on the reservation node, from the repo root (defaults below).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
ADAPTER="${ADAPTER:-artifacts/sft-1p7b-fixed}"       # same SFT base as the baseline run
OUT_DIR="${OUT_DIR:-artifacts/grpo-ablation-lambda0}"
STEPS="${STEPS:-50}"
STAGE="${STAGE:-step3}"
UNDER_CALL="${UNDER_CALL:-0}"                          # 1 -> also enable missing_calls penalty

# Plan B flags (mirror 31_grpo_planb.sh so this is an apples-to-apples A/B).
PB="--dead-frac-source discarded --advantage-scale mad --advantage-baseline sign --sign-baseline 0.5"
if [ "$STAGE" != "step1" ]; then
  PB="$PB --dqw true --dqw-temp 2.2"
fi
if [ "$STAGE" = "step3" ]; then
  PB="$PB --e2h-curriculum true"
fi

# The ablation switch: zero out the efficiency multiplier.
EFF_ARG="--efficiency-lambda 0.0"

echo "=== ABLATION Plan B  stage=$STAGE  efficiency_lambda=0  under_call=$UNDER_CALL ==="
echo "=== base: $ADAPTER -> $OUT_DIR ==="

python3 -m atr.train.grpo \
  --model-id "$MODEL" \
  --adapter "$ADAPTER" \
  --out-dir "$OUT_DIR" \
  --group-size 8 --tasks-per-step 8 --steps "$STEPS" \
  --lr 2e-5 --temperature 1.0 --kl-beta 0.03 --micro-batch 2 \
  --curriculum true --void-turn-filter true --eval-every 50 \
  --log-every 5 \
  $EFF_ARG \
  $PB

echo "=== ABLATION TRAIN DONE ==="

# ---- EVAL at two temperatures (separate decode artifact from policy collapse) ----
for TEMP in 0.0 1.0; do
  suffix="$(echo "$TEMP" | tr '.' '_')"
  echo "=== eval dev  temp=$TEMP ==="
  python3 -m atr.cli eval --dev --n-per-type 20 \
    --backend "hf:$MODEL" --adapter "$OUT_DIR/final" \
    --temperature "$TEMP" --max-new-tokens 512 \
    --out "artifacts/eval-ablation-lambda0-dev-t$suffix"

  echo "=== eval judge  temp=$TEMP ==="
  python3 -m atr.cli eval \
    --backend "hf:$MODEL" --adapter "$OUT_DIR/final" \
    --temperature "$TEMP" --max-new-tokens 512 \
    --tasks artifacts/judge_tasks.jsonl \
    --out "artifacts/eval-ablation-lambda0-judge-t$suffix"
done

# ---- archive for local pull (never skip: the node vanishes) ----
TSTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p artifacts/ship
tar -czf "artifacts/ship/grpo-ablation-lambda0_$TSTAMP.tar.gz" "$OUT_DIR"
echo "=== ABLATION DONE ==="
echo "PULL: scp -P 22013 gpu17@10.214.5.55:~/ToolCall/artifacts/ship/grpo-ablation-lambda0_$TSTAMP.tar.gz ."
