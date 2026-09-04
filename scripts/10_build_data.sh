#!/usr/bin/env bash
# Build the SFT set. Three sources, mixed on purpose (see atr/data/teacher.py).
#   TEACHER=openai:gpt-4.1   -> distillation (rule 9 allows it; training only)
#   TEACHER=oracle           -> free bootstrap, format + tool choice only
set -euo pipefail
cd "$(dirname "$0")/.."
TEACHER="${TEACHER:-oracle}"
N="${N:-40}"            # matches the naturalized seed range (sweet spot); set higher once the cache covers more
CACHE="${CACHE:-artifacts/naturalized_passages.json}"
mkdir -p artifacts

# 1. oracle replays: cheap, perfect, teaches format and tool selection.
#    --cache wires the naturalized passages in so tool-execution prose is varied,
#    not the fixed templates (facts/gold unchanged).
python3 -m atr.cli collect --n "$N" --seed-start 0 --backend oracle \
    --cache "$CACHE" --samples-per-task 1 --out artifacts/raw_oracle.jsonl

# 2. teacher rollouts on the hard slice: this is where deliberation comes from
if [ "$TEACHER" != "oracle" ]; then
  python3 -m atr.cli collect --n "$((N/2))" --seed-start 500000 --backend "$TEACHER" \
      --cache "$CACHE" --samples-per-task 4 --temperature 0.8 --out artifacts/raw_teacher.jsonl
fi

# 3. filter + export.  --strict-format only once round 1 has taught the format.
#    Build to a CANDIDATE path and promote to the committed data/sft.jsonl only
#    after the audit passes. Writing straight to the training path is how a leaky
#    build became a training set: artifacts/ is gitignored, so nothing showed up
#    in `git status` and no one re-audited before the GPU run.
CANDIDATE="${CANDIDATE:-artifacts/sft_candidate.jsonl}"
FINAL="${FINAL:-data/sft.jsonl}"

python3 -m atr.cli build artifacts/raw_*.jsonl --rebalance \
    --max-per-task 1 --out "$CANDIDATE"

echo ""
echo "=== auditing candidate build before promoting to $FINAL ==="
if ! python3 tests/audit_sft.py "$CANDIDATE"; then
  echo "" >&2
  echo "REFUSING TO PROMOTE: $CANDIDATE reports DEFECTS PRESENT." >&2
  echo "  The candidate is left in place for inspection; $FINAL is untouched." >&2
  echo "  Check that the generator is at bd590a5 or later (terminal read +" >&2
  echo "  prefix-leak filter) before rebuilding." >&2
  exit 1
fi

mv -f "$CANDIDATE" "$FINAL"
echo ""
echo "promoted -> $FINAL ($(wc -l < "$FINAL") records)"
echo "COMMIT IT: git add $FINAL && git commit -m 'Rebuild SFT set'"
echo "  (the training scripts read the committed file, not artifacts/)"
