#!/usr/bin/env bash
# Development model first. Only move to 4B once the 1.7B curve has flattened.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
OUT="${OUT:-artifacts/sft-1p7b-fixed}"
python3 -m atr.train.sft \
  --model-id "$MODEL" --data artifacts/sft.jsonl --out-dir "$OUT" \
  --lora true --lora-r 32 --lora-alpha 64 --lora-lr 1e-4 \
  --epochs 2 --batch-size 2 --grad-accum 8 --max-len 4096 \
  --step-weight 2.0 --answer-weight 0.5

# ---- TRAIN-DONE: archive + advertise for local sync (never skip!) ----
TSTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p artifacts/ship
tar -czf "artifacts/ship/$(basename "$OUT")_$TSTAMP.tar.gz" "$OUT"
(cd artifacts/ship && sha256sum ./*.tar.gz | tee "MANIFEST_$TSTAMP.txt")
echo "=== TRAIN DONE ==="
echo "Archived adapter -> artifacts/ship/$(basename "$OUT")_$TSTAMP.tar.gz"
echo "PULL IT TO LOCAL NOW (before reservation ends):"
echo "  scp -P 22013 gpu17@10.214.5.55:~/ToolCall/artifacts/ship/$(basename "$OUT")_$TSTAMP.tar.gz ."
