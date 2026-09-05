#!/usr/bin/env bash
# Plan B: the 4-hop fix staged GRPO run.
#
# Stages (each is a superset of the previous, all gated + revertible):
#   STAGE=step1   Fix-3 (true dead-frac) + Fix-1 (MAD) + Fix-2 (Sign advantage)
#   STAGE=step2   step1 + Fix-2b (DQW, group difficulty weighting)
#   STAGE=step3   step2 + Fix-4 (E2H-G Gaussian curriculum)
#
# The point of staging: step1 is the safe, theory-proven rescue of the thrown-away
# 4-hop groups. Only after confirming step1 moves 4-hop (and 2/3-hop stay put) do we
# add the tuning-sensitive knobs (DQW temperature, E2H schedule). Each fix can be
# turned off independently via its flag below.
#
# IMPORTANT: run the same stage against the SAME SFT adapter to A/B against the
# original `30_grpo.sh` numbers (2-hop 85 / 3-hop 75 / 4-hop 25, overall 61.7).
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:-step3}"
# Sweet-spot run budget: override with STEPS/EVAL_EVERY to keep the run inside
# the reservation window (HFBackend generation is ~140s/step). e.g.
#   STEPS=36 EVAL_EVERY=36
# ensures exactly ONE end-of-run canary (-> best) and a saved final adapter.
STEPS="${STEPS:-300}"
EVAL_EVERY="${EVAL_EVERY:-50}"
# Wall-clock budget for THIS session (default 3h30m of a 4h reservation), and the
# resume hook. See 30_grpo.sh. With these you no longer have to guess STEPS to fit
# the window: set STEPS high, let MAX_SECONDS end the session, and continue with
#   RESUME=$OUT/final bash scripts/31_grpo_planb.sh
MAX_SECONDS="${MAX_SECONDS:-12600}"

# Plan B knobs (environment-overridable, tunable for the no-degradation guard).
# DQW temperature on the REWARD scale: 2.2 -> hard ~1.6x weight, easy kept ~0.78x.
DQW_TEMP="${DQW_TEMP:-2.2}"
# Sign fixed baseline for a fully-failed group (solved ~1.4-1.7, dead ~0-0.3).
SIGN_BASELINE="${SIGN_BASELINE:-0.5}"

# ---- build the per-stage flags ----
# step1:  Fix-3 (true dead-frac) + Fix-1 (MAD) + Fix-2 (Sign advantage)  [safe core]
# step2:  step1 + Fix-2b (DQW)
# step3:  step2 + Fix-4 (E2H-G)
PB="--dead-frac-source discarded --advantage-scale mad --advantage-baseline sign --sign-baseline $SIGN_BASELINE"
if [ "$STAGE" != "step1" ]; then
  PB="$PB --dqw true --dqw-temp $DQW_TEMP"
fi
if [ "$STAGE" = "step3" ]; then
  PB="$PB --e2h-curriculum true"
fi

echo "=== Plan B GRPO  stage=$STAGE  flags: $PB ==="

python3 -m atr.train.grpo \
  --model-id "${MODEL:-Qwen/Qwen3-1.7B}" \
  --adapter "${ADAPTER:-artifacts/sft-1p7b-fixed}" \
  --out-dir "${OUT:-artifacts/grpo-planb-$STAGE}" \
  --group-size 8 --tasks-per-step 8 --steps "$STEPS" \
  --lr 2e-5 --temperature 1.0 --kl-beta 0.03 --micro-batch 2 \
  --curriculum true --void-turn-filter true --eval-every "$EVAL_EVERY" \
  --log-every 5 \
  --max-seconds "$MAX_SECONDS" \
  ${RESUME:+--resume-from "$RESUME"} \
  $PB

# ---- TRAIN-DONE: archive + advertise for local sync (never skip!) ----
TSTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p artifacts/ship
tar -czf "artifacts/ship/grpo-planb-${STAGE}_$TSTAMP.tar.gz" "${OUT:-artifacts/grpo-planb-$STAGE}"
(cd artifacts/ship && sha256sum ./*.tar.gz | tee "MANIFEST_$TSTAMP.txt")
echo "=== TRAIN DONE ==="
echo "Archived adapter -> artifacts/ship/grpo-planb-${STAGE}_$TSTAMP.tar.gz"
echo "PULL IT TO LOCAL NOW (before reservation ends):"
echo "  scp -P 22013 gpu17@10.214.5.55:~/ToolCall/artifacts/ship/grpo-planb-${STAGE}_$TSTAMP.tar.gz ."
