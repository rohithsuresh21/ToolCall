#!/usr/bin/env bash
# GRPO on top of the SFT checkpoint. Never from base.
# Lane B stabilisers on by default: void-turn filter, entropy + KL monitoring,
# held-out canary every 50 steps, curriculum easy -> default -> hard.
set -euo pipefail
cd "$(dirname "$0")/.."

# Wall-clock budget for THIS session, in seconds. Default 3h30m, which leaves
# ~30min of a 4h reservation for the archive + scp below. The trainer checks the
# clock BEFORE each step and stops one step short rather than being killed
# mid-optimise, then writes a full resume point to $OUT/final.
MAX_SECONDS="${MAX_SECONDS:-12600}"
# RESUME=<dir> continues an interrupted run: optimiser moments, step counter,
# best, history and RNG all come back. --adapter stays the SFT anchor because the
# KL reference must measure drift from SFT, not from the resumed checkpoint.
#   RESUME=artifacts/grpo-1p7b/final bash scripts/30_grpo.sh
python3 -m atr.train.grpo \
  --model-id "${MODEL:-Qwen/Qwen3-1.7B}" \
  --adapter "${ADAPTER:-artifacts/sft-1p7b-fixed}" \
  --out-dir "${OUT:-artifacts/grpo-1p7b}" \
  --group-size 8 --tasks-per-step 8 --steps 300 \
  --lr 2e-5 --temperature 1.0 --kl-beta 0.03 --micro-batch 2 \
  --curriculum true --void-turn-filter true --eval-every 50 \
  --max-seconds "$MAX_SECONDS" \
  ${RESUME:+--resume-from "$RESUME"}

# ---- TRAIN-DONE: archive + advertise for local sync (never skip!) ----
TSTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p artifacts/ship
tar -czf "artifacts/ship/grpo-1p7b_$TSTAMP.tar.gz" "${OUT:-artifacts/grpo-1p7b}"
(cd artifacts/ship && sha256sum ./*.tar.gz | tee "MANIFEST_$TSTAMP.txt")
echo "=== TRAIN DONE ==="
echo "Archived adapter -> artifacts/ship/grpo-1p7b_$TSTAMP.tar.gz"
echo "PULL IT TO LOCAL NOW (before reservation ends):"
echo "  scp -P 22013 gpu17@10.214.5.55:~/ToolCall/artifacts/ship/grpo-1p7b_$TSTAMP.tar.gz ."
