#!/usr/bin/env bash
# GRPO on top of the SFT checkpoint. Never from base.
# Lane B stabilisers on by default: void-turn filter, entropy + KL monitoring,
# held-out canary every 50 steps, curriculum easy -> default -> hard.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m atr.train.grpo \
  --model-id "${MODEL:-Qwen/Qwen3-1.7B}" \
  --adapter "${ADAPTER:-artifacts/sft-1p7b-fixed}" \
  --out-dir "${OUT:-artifacts/grpo-1p7b}" \
  --group-size 8 --tasks-per-step 8 --steps 300 \
  --lr 2e-5 --temperature 1.0 --kl-beta 0.03 --micro-batch 2 \
  --curriculum true --void-turn-filter true --eval-every 50
