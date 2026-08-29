#!/usr/bin/env bash
# Run this FIRST, and after every change to the tools or the verifiers.
# The oracle must score 100%. If it does not, a verifier is wrong and every
# number downstream -- including your RL reward -- is lying to you.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m atr.cli eval --dev --n-per-type 6 --backend oracle
python3 -m atr.cli ablate --dev --n-per-type 10
