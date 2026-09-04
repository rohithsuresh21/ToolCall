# =============================================================================
# pull_all.ps1 - fetch every Plan B artifact from the remote node to THIS local
# machine. Run from the directory you want the files in.
#
# WARNING: the Cynaptics node is a THROWAWAY container - checkpoints/files
# VANISH when the reservation ends. Trained weights exist ONLY there and ONLY
# until expiry. Pull each artifact the INSTANT a phase reports it ready, and run
# this script as a final sweep before the window closes. Do not rely on anything
# staying on the remote.
#
# Local shell is PowerShell 5.1. Run with:
#   powershell -File pull_all.ps1
# or paste the scp lines printed by run_all.sh one at a time.
# =============================================================================

$Host = "gpu17@10.214.5.55"
$Port = 22013
$Base = "~/ToolCall/artifacts"
$Dest = Split-Path -Parent $MyInvocation.MyCommand.Path   # this script's folder

# The set of remote artifact names to try to fetch (missing ones are skipped).
$items = @(
  "ship/sft-1p7b-fixed_*.tar.gz",
  "ship/grpo-planb-step3_*.tar.gz",
  "ship/eval-grpo-planb-step3_*.tar.gz",
  "ship/grpo-history_*.tar.gz",
  "ship/judge-eval-grpo-planb_*.tar.gz",
  "sft-1p7b-fixed",
  "grpo-planb-step3",
  "eval-grpo-planb-step3",
  "judge_eval_grpo-planb",
  "grpo-planb-step3/history.jsonl"
)

foreach ($it in $items) {
  $src = "$Base/$it"
  # quote for scp; directories need -r
  $isDir = $it -notmatch '\.(jsonl|json|txt|log|gz)$'
  Write-Host "`n=== pulling $src (dir=$isDir) ===" -ForegroundColor Cyan
  if ($isDir) {
    scp -r -P $Port "$($Host):$src" $Dest
  } else {
    scp -P $Port "$($Host):$src" $Dest
  }
  $code = $LASTEXITCODE
  if ($code -eq 0) {
    Write-Host "   OK -> $Dest" -ForegroundColor Green
  } else {
    Write-Host "   skipped/failed (exit $code) - likely not ready yet or name differs" -ForegroundColor Yellow
  }
}

Write-Host "`n===== pull_all done. Check $Dest for: sft adapter, grpo adapter (best/final), eval report, judge report, history =====" -ForegroundColor Green
