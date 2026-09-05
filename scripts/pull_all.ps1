# =============================================================================
# pull_all.ps1 - fetch every artifact produced by 60_pipeline_tomorrow.sh from
# the remote node to THIS local machine's "pulled 2" folder.
#
# WARNING: the Cynaptics node is a THROWAWAY container - checkpoints/files
# VANISH when the reservation ends. Trained weights exist ONLY there and ONLY
# until expiry. Run this as the FINAL sweep before the window closes.
#
# Local shell is PowerShell 5.1. Run with:
#   powershell -File pull_all.ps1
# =============================================================================

$Host_ = "gpu17@10.214.5.55"
$Port = 22013
$Base = "~/ToolCall/atr/artifacts"
$Dest = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "pulled 2"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$items = @(
  "ship/sft-1p7b-fixed_*.tar.gz",
  "ship/grpo-planb-step3_*.tar.gz",
  "ship/eval-dev_*.tar.gz",
  "ship/eval-judge_*.tar.gz",
  "ship/ATR-Eval-Report_*.pdf",
  "ship/dev-report_*.txt",
  "ship/dev-report_*.json",
  "ship/judge-report_*.txt",
  "ship/judge-scores_*.jsonl",
  "ship/MANIFEST_*.txt",
  "sft-1p7b-fixed",
  "grpo-planb-step3",
  "eval-grpo-planb-step3",
  "judge_eval_grpo-planb"
)

foreach ($it in $items) {
  $src = "$Base/$it"
  $isDir = $it -notmatch '\.(jsonl|json|txt|pdf|gz|log)$'
  Write-Host "`n=== pulling $src (dir=$isDir) ===" -ForegroundColor Cyan
  if ($isDir) {
    scp -r -P $Port "$($Host_):$src" $Dest
  } else {
    scp -P $Port "$($Host_):$src" $Dest
  }
  $code = $LASTEXITCODE
  if ($code -eq 0) {
    Write-Host "   OK -> $Dest" -ForegroundColor Green
  } else {
    Write-Host "   skipped/failed (exit $code) - likely not ready yet or name differs" -ForegroundColor Yellow
  }
}

Write-Host "`n===== pull_all done. Check '$Dest' for: sft adapter, grpo adapter, dev eval, judge eval, combined report =====" -ForegroundColor Green