#!/usr/bin/env bash
# Held-out evaluation. Seeds 900000+ are the dev set and are disjoint from the
# training range used in 10_build_data.sh -- keep it that way.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
python3 -m atr.cli eval --dev --n-per-type 20 \
  --backend "vllm:$MODEL" ${ADAPTER:+--adapter "$ADAPTER"} \
  --temperature 0.0 --max-new-tokens 512 \
  --out "artifacts/eval-$(basename "${ADAPTER:-base}")"
