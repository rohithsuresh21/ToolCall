#!/usr/bin/env bash
# F2 (FIX-2): merge the trained LoRA into one submission-ready artifact.
# Produces artifacts/final-<name>/ containing merged weights + tokenizer +
# frozen non-thinking generation config + export_manifest.json.
#
#   bash scripts/60_merge_export.sh                          # defaults below
#   BASE=Qwen/Qwen3-4B ADAPTER=artifacts/grpo-4b/final OUT=artifacts/final-4b bash scripts/60_merge_export.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m atr.export_merge \
  --base "${BASE:-Qwen/Qwen3-4B}" \
  --adapter "${ADAPTER:?set ADAPTER=<peft checkpoint path>}" \
  --out "${OUT:-artifacts/final-merged}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cpu}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}"
