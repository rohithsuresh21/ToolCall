#!/usr/bin/env bash
# =============================================================================
# run_all.sh - Cynaptics container is a THROWAWAY sandbox: EVERYTHING VANISHES
# when the reservation ends. Trained weights exist ONLY here and ONLY until then.
#
# THEREFORE: we do NOT fire-and-forget all phases. We run ONE phase, archive +
# advertise its artifact, then STOP and demand you pull it to your LOCAL machine
# and confirm, before the next phase starts. Nothing is left un-pulled.
#
# Usage:
#   STAGE=sft      -> run SFT, then STOP for pull
#   STAGE=grpo     -> run GRPO, then STOP for pull
#   STAGE=deveval  -> run dev-set eval, then STOP for pull
#   STAGE=judge    -> run judge eval, then STOP for pull
#
#   There is NO "all". Run each phase, pull, then start the next.
#   Repo assumed at ${REPO:-$HOME/ToolCall/atr}. Set REPO if it differs.
# =============================================================================
set -euo pipefail

REPO="${REPO:-$HOME/ToolCall/atr}"
STAGE="${STAGE:?set STAGE= to run ONE phase at a time: sft | grpo | deveval | judge}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SHIP="$REPO/artifacts/ship"
mkdir -p "$SHIP"

echo "=== run_all  stage=$STAGE  repo=$REPO  stamp=$STAMP ==="
cd "$REPO"

# --------------------------------------------------------------------------- #
wait_for_pull() {
  # args: <tarball path> <human description>
  local tarball="$1" what="$2"
  echo ""
  echo "####################################################################"
  echo "  PHASE DONE - DO NOT START THE NEXT PHASE YET."
  echo "  The remote is a THROWAWAY sandbox; these files will VANISH."
  echo ""
  echo "  PULL THIS TO YOUR LOCAL MACHINE NOW:"
  echo "    scp -P 22013 gpu17@10.214.5.55:$tarball ."
  echo ""
  echo "  Then verify on local, and only then continue."
  echo "####################################################################"
}

# --------------------------------------------------------------------------- #
if [[ "$STAGE" == "sft" ]]; then
  echo "===== [SFT] 1.5k examples -> artifacts/sft-1p7b-fixed ====="
  OUT="$REPO/artifacts/sft-1p7b-fixed" MODEL="${MODEL:-Qwen/Qwen3-1.7B}" \
    bash scripts/20_sft.sh
  wait_for_pull "$SHIP/sft-1p7b-fixed_*.tar.gz" "SFT adapter"
  exit 0
fi

if [[ "$STAGE" == "grpo" ]]; then
  echo "===== [GRPO] Plan B all-on, safe E2H MU=0.70 ====="
  [[ -d "$REPO/artifacts/sft-1p7b-fixed" ]] || { echo "FATAL: no SFT adapter"; exit 1; }
  STAGE=step3 bash "$REPO/scripts/31_grpo_planb.sh" 2>&1 | tee "$SHIP/grpo_planb_$STAMP.log"
  wait_for_pull "$SHIP/grpo-planb-step3_*.tar.gz" "GRPO adapter (incl. best/final)"
  exit 0
fi

if [[ "$STAGE" == "deveval" ]]; then
  echo "===== [DEV-EVAL] held-out dev set -> report ====="
  [[ -d "$REPO/artifacts/grpo-planb-step3" ]] || { echo "FATAL: no GRPO adapter"; exit 1; }
  ADAPTER="$REPO/artifacts/grpo-planb-step3" MODEL="${MODEL:-Qwen/Qwen3-1.7B}" \
    bash scripts/40_eval.sh
  tar -czf "$SHIP/eval-grpo-planb-step3_$STAMP.tar.gz" "$REPO/artifacts/eval-grpo-planb-step3"
  tar -czf "$SHIP/grpo-history_$STAMP.tar.gz" "$REPO/artifacts/grpo-planb-step3/history.jsonl"
  wait_for_pull "$SHIP/eval-grpo-planb-step3_*.tar.gz" "dev-set eval report + history"
  exit 0
fi

if [[ "$STAGE" == "judge" ]]; then
  echo "===== [JUDGE] benchmark eval -> report ====="
  [[ -d "$REPO/artifacts/grpo-planb-step3" ]] || { echo "FATAL: no GRPO adapter"; exit 1; }
  if [[ ! -f "$REPO/artifacts/judge_tasks.jsonl" ]]; then
    echo "FATAL: $REPO/artifacts/judge_tasks.jsonl not found. Push it from local first:"
    echo "  powershell -File scripts/push_judge_tasks.ps1"
    exit 1
  fi
  python3 "$REPO/scripts/judge_eval.py" \
    --base "${MODEL:-Qwen/Qwen3-1.7B}" \
    --adapter "$REPO/artifacts/grpo-planb-step3" \
    --tasks "$REPO/artifacts/judge_tasks.jsonl" \
    --out "$REPO/artifacts/judge_eval_grpo-planb"
  tar -czf "$SHIP/judge-eval-grpo-planb_$STAMP.tar.gz" "$REPO/artifacts/judge_eval_grpo-planb"
  wait_for_pull "$SHIP/judge-eval-grpo-planb_*.tar.gz" "judge benchmark eval report"
  exit 0
fi

echo "FATAL: unknown STAGE='$STAGE'. Use one of: sft, grpo, deveval, judge"
exit 2
