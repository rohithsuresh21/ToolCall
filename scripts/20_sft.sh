#!/usr/bin/env bash
# Development model first. Only move to 4B once the 1.7B curve has flattened.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
OUT="${OUT:-artifacts/sft-1p7b}"
python3 -m atr.train.sft \
  --model-id "$MODEL" --data artifacts/sft.jsonl --out-dir "$OUT" \
  --lora true --lora-r 32 --lora-alpha 64 --lora-lr 1e-4 \
  --epochs 2 --batch-size 2 --grad-accum 8 --max-len 4096 \
  --late-turn-weight 1.5
