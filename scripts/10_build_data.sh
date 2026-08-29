#!/usr/bin/env bash
# Build the SFT set. Three sources, mixed on purpose (see atr/data/teacher.py).
#   TEACHER=openai:gpt-4.1   -> distillation (rule 9 allows it; training only)
#   TEACHER=oracle           -> free bootstrap, format + tool choice only
set -euo pipefail
cd "$(dirname "$0")/.."
TEACHER="${TEACHER:-oracle}"
N="${N:-4000}"
mkdir -p artifacts

# 1. oracle replays: cheap, perfect, teaches format and tool selection
python3 -m atr.cli collect --n "$N" --seed-start 0 --backend oracle \
    --samples-per-task 1 --out artifacts/raw_oracle.jsonl

# 2. teacher rollouts on the hard slice: this is where deliberation comes from
if [ "$TEACHER" != "oracle" ]; then
  python3 -m atr.cli collect --n "$((N/2))" --seed-start 500000 --backend "$TEACHER" \
      --samples-per-task 4 --temperature 0.8 --out artifacts/raw_teacher.jsonl
fi

# 3. filter + export.  --strict-format only once round 1 has taught the format.
python3 -m atr.cli build artifacts/raw_*.jsonl --rebalance \
    --max-per-task 1 --out artifacts/sft.jsonl

wc -l artifacts/sft.jsonl
