#!/usr/bin/env bash
# SFT warm-up on the FINAL model (Qwen3-4B). The PS requires the submission to
# use Qwen3-4B, so this is mandatory work, not a stretch goal.
# LoRA fits a single A100-80G; full FT of 4B wants H100-80G (see atr/train/sft.py).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT="${OUT:-artifacts/sft-4b}"
DATA="${DATA:-data/sft.jsonl}"

# Same gate as 20_sft.sh -- see scripts/lib_data_gate.sh for why.
source scripts/lib_data_gate.sh
require_clean_dataset "$DATA"

python3 -m atr.train.sft \
  --model-id "$MODEL" --data "$DATA" --out-dir "$OUT" \
  --lora true --lora-r 64 --lora-alpha 128 --lora-lr 1e-4 \
  --epochs 2 --batch-size 2 --grad-accum 8 --max-len 4096 \
  --late-turn-weight 1.5
