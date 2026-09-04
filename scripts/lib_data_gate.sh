#!/usr/bin/env bash
# Shared training-data gate. Source it, then call require_clean_dataset <path>.
#
# WHY THIS EXISTS
# We trained on prefix-leaky data because a gitignored artifact shadowed the
# committed clean set: every training script read `artifacts/sft.jsonl`, which
# artifacts/ hides from git, so the file that was actually consumed was a stale
# pre-fix build while the audited 1,951-record set sat unread in data/sft.jsonl.
# Nothing downstream could catch it. `eval --backend oracle` scores a leaky set
# 100% by construction -- MockBackend replays the plan and emits oracle_answer
# from a lookup table without ever reading a tool result -- so the sanity gate is
# structurally blind here. tests/audit_sft.py is the only check that reads the
# retrieved passages, and it exits non-zero on DEFECTS PRESENT.
#
# The gate therefore does two things, and both matter:
#   1. refuses to start when the audit reports DEFECTS PRESENT;
#   2. refuses when a gitignored artifacts/ copy could be shadowing the argument,
#      because that is the specific mistake that cost us a training run.

# Resolve a working interpreter. `python3` is the right name on the GPU box but on
# Windows it is often the Microsoft Store stub, which prints "Python was not found"
# and exits non-zero -- indistinguishable from a failed audit unless we check. A
# broken interpreter must never be reported as leaky data.
_gate_python() {
  local candidate
  for candidate in "${PYTHON:-}" python3 python py; do
    [[ -z "$candidate" ]] && continue
    if command -v "$candidate" >/dev/null 2>&1 \
       && "$candidate" -c 'import sys; sys.exit(0)' >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

require_clean_dataset() {
  local data="$1"
  local py rc

  if [[ -z "$data" ]]; then
    echo "FATAL: require_clean_dataset needs a dataset path." >&2
    return 1
  fi
  if [[ ! -f "$data" ]]; then
    echo "FATAL: training set '$data' does not exist." >&2
    echo "  Build it:  bash scripts/10_build_data.sh" >&2
    return 1
  fi

  echo "=== data gate: auditing $data ($(wc -l < "$data") records) ==="

  # Shadowing check: warn loudly if the gitignored twin exists and differs.
  if [[ "$data" != "artifacts/sft.jsonl" && -f "artifacts/sft.jsonl" ]]; then
    if ! cmp -s "$data" artifacts/sft.jsonl; then
      echo "NOTE: artifacts/sft.jsonl exists and DIFFERS from $data." >&2
      echo "      It is gitignored and is NOT what this run trains on. Training on:" >&2
      echo "      $data" >&2
    fi
  fi
  if [[ "$data" == artifacts/* ]]; then
    echo "FATAL: '$data' lives under artifacts/, which is gitignored." >&2
    echo "  Training inputs must come from a committed file so the run is" >&2
    echo "  reproducible and the audit result means something. Use data/sft.jsonl," >&2
    echo "  or promote your build:  bash scripts/10_build_data.sh" >&2
    return 1
  fi

  if ! py="$(_gate_python)"; then
    echo "FATAL: no working Python interpreter found (tried \$PYTHON, python3, python, py)." >&2
    echo "  The gate refuses to pass rather than skip the audit. Set PYTHON=/path/to/python." >&2
    return 1
  fi

  set +e
  "$py" tests/audit_sft.py "$data"
  rc=$?
  set -e

  if [[ $rc -ne 0 && $rc -ne 1 ]]; then
    echo "" >&2
    echo "FATAL: the audit did not run (exit $rc from '$py tests/audit_sft.py')." >&2
    echo "  This is a harness problem, NOT a verdict on the data. Fix it and re-run;" >&2
    echo "  the gate will not let training start on an unaudited set." >&2
    return 1
  fi

  if [[ $rc -eq 1 ]]; then
    echo "" >&2
    echo "########################################################################" >&2
    echo "  REFUSING TO TRAIN: $data reports DEFECTS PRESENT." >&2
    echo "" >&2
    echo "  Any of: psychic first query, gold answer never retrieved, conflicting" >&2
    echo "  labels, or per-hop prefix leakage. A model trained on this scores fine" >&2
    echo "  under the oracle and teaches truncated chains. Rebuild from the current" >&2
    echo "  generator before spending GPU time:" >&2
    echo "    bash scripts/10_build_data.sh" >&2
    echo "########################################################################" >&2
    return 1
  fi

  echo "=== data gate: PASS -- $data is clean, starting training ==="
}
