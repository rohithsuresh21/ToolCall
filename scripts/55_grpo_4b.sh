#!/usr/bin/env bash
# GRPO on the final model. Starts from the 4B SFT checkpoint (50_sft_4b.sh).
# Same recipe validated on 1.7B; expect the ordering of decisions to transfer,
# not the absolute numbers. Needs A100-80G class GPU with LoRA.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m atr.train.grpo \
  --model-id "${MODEL:-Qwen/Qwen3-4B}" \
  --adapter "${ADAPTER:-artifacts/sft-4b}" \
  --out-dir "${OUT:-artifacts/grpo-4b}" \
  --group-size 8 --tasks-per-step 4 --steps 200 \
  --lr 1e-5 --temperature 1.0 --kl-beta 0.03 --micro-batch 1 --max-len 3072 \
  --curriculum true --void-turn-filter true --eval-every 40
