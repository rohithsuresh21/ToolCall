#!/usr/bin/env bash
# Lightning AI runner: 4-hop-fix GRPO (20% real MuSiQue-Ans + 80% synthetic),
# resume-aware for the free tier's hard 4-hour session cap + interruptible
# preemption.
#
# One session-cycle of GRPO. It never finishes the run on its own -- Lightning
# free tier kills/restarts the session every 4h and can preempt an interruptible
# instance at any moment, so you run this repeatedly via 76_lightning_cycle.sh,
# which relaunches it with RESUME set until the step budget is reached.
#
# Free-tier constraints this script is built around:
#   * 4h hard session cap  -> MAX_SECONDS default 12600 (3h30m) leaves ~30min for
#     the archive + pull inside the same session window.
#   * preemption           -> every cycle archives + advertises the intermediate
#     weights BEFORE returning, and the trainer writes a resume point to
#     $OUT/final with optimiser moments + step counter + RNG (see grpo.py).
#   * RAM/disk are recycled on restart -> the model + data normally live in the
#     Studio's persistent volume; pull them only after upload (see NOTES below).
#
# NEW for the 4-hop fix: --real-tasks-path data/musique_train_tasks.jsonl draws
# 20% of each step's tasks from the real MuSiQue-Ans train pool (built by
# scripts/make_musique_train_tasks.py, double-checked disjoint from the 54-probe
# judge set). That pool is hop-stratified 40/30/30, so real 4-hop gets real
# exposure instead of the 1.2%/step the natural 72/22/6 split would allow.
#
# Usage (in the Lightning Studio shell, from the repo root):
#   bash scripts/75_lightning_grpo.sh [STEPS]
# Env knobs: MODEL, STEPS (default 150), ADAPTER (default SFT adapter),
#            OUT, MAX_SECONDS, RESUME (set by the cycler), REAL_MIX (default 0.2)
#
# NOTES on getting the pieces onto Lightning first:
#   * git clone the repo; the box needs the SFT adapter unpacked at
#     artifacts/sft-1p7b-fixed (adapter_config.json + adapter_model.safetensors)
#   * data/musique_train_tasks.jsonl is COMMITTED data (like judge_tasks.jsonl) ->
#     just git-pull. Regenerate with make_musique_train_tasks.py if you changed it.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STEPS="${STEPS:-150}"
ADAPTER="${ADAPTER:-artifacts/sft-1p7b-fixed}"
OUT="${OUT:-artifacts/grpo-lightning}"
MAX_SECONDS="${MAX_SECONDS:-12600}"     # 3h30m inside the 4h session cap
REAL_MIX="${REAL_MIX:-0.2}"
REAL_TASKS="data/musique_train_tasks.jsonl"

if [[ ! -f "$REAL_TASKS" ]]; then
  echo "FATAL: $REAL_TASKS missing. git pull (it is committed) or run:" >&2
  echo "  python3 scripts/make_musique_train_tasks.py" >&2
  exit 1
fi
if [[ ! -f "$ADAPTER/adapter_config.json" || ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "FATAL: SFT adapter missing at $ADAPTER" >&2
  exit 1
fi

echo "======================================================================"
echo "  LIGHTNING GRPO cycle  model=$MODEL  steps=$STEPS  real_mix=$REAL_MIX"
echo "  out=$OUT  max_seconds=$MAX_SECONDS  resume=${RESUME:-<fresh>}"
echo "======================================================================"

python3 -m atr.train.grpo \
  --model-id "$MODEL" \
  --adapter "$ADAPTER" \
  --out-dir "$OUT" \
  --group-size 8 --tasks-per-step 8 --steps "$STEPS" \
  --lr 2e-5 --temperature 1.0 --kl-beta 0.03 --micro-batch 2 \
  --curriculum true --void-turn-filter true --eval-every 50 \
  --log-every 5 \
  --dead-frac-source discarded --advantage-scale mad \
  --advantage-baseline sign --sign-baseline 0.5 \
  --dqw true --dqw-temp 2.2 --e2h-curriculum true \
  --efficiency-lambda 0.0 --under-call-penalty true \
  --max-seconds "$MAX_SECONDS" \
  --real-tasks-path "$REAL_TASKS" --real-fraction "$REAL_MIX" \
  ${RESUME:+--resume-from "$RESUME"}

# ---- TRAIN-DONE: archive + advertise for local pull (never skip!) ---------
# Even on "done early" the free tier can die at the 4h mark mid-next-cycle, so
# the weights land in ship/ BEFORE this returns.
TSTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p artifacts/ship
tar -czf "artifacts/ship/grpo-lightning_$TSTAMP.tar.gz" "$OUT"
(cd artifacts/ship && sha256sum ./*.tar.gz | tee "MANIFEST_$TSTAMP.txt")
echo "=== CYCLE DONE ==="
echo "Archived adapter -> artifacts/ship/grpo-lightning_$TSTAMP.tar.gz"
echo "PULL VIA STUDIO FILE SYNC / CLI TO 'pulled 2' LOCAL BEFORE RESTART."
echo ""
echo "Next session-cycle (auto): RESUME=$OUT/final bash scripts/75_lightning_grpo.sh"