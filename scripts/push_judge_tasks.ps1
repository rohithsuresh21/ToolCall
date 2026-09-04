# =============================================================================
# push_judge_tasks.ps1 - build judge_tasks.jsonl locally (the parquet is only
# on this machine) and push it to the remote node where judge_eval.py will run.
#
# Usage (PowerShell):
#   powershell -File scripts\push_judge_tasks.ps1
# =============================================================================

$Host  = "gpu17@10.214.5.55"
$Port  = 22013
$Repo  = "E:\Multi modal reasoning tool\atr"
$local = Join-Path $Repo "artifacts\judge_tasks.jsonl"
$remoteSrc = "~/ToolCall/atr/artifacts"

# 1) build the jsonl from the judge parquet (local only)
Write-Host "=== building judge_tasks.jsonl locally ===" -ForegroundColor Cyan
Push-Location $Repo
python scripts\make_judge_tasks.py --out "$local"
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { Write-Host "build FAILED (exit $code)" -ForegroundColor Red; exit 1 }
Write-Host "built: $local" -ForegroundColor Green

# 2) ensure remote dir exists, then push
Write-Host "`n=== creating remote dir (if needed) ===" -ForegroundColor Cyan
ssh -p $Port $Host "mkdir -p $remoteSrc"

Write-Host "`n=== pushing judge_tasks.jsonl to remote ===" -ForegroundColor Cyan
scp -P $Port "$local" "$($Host):$remoteSrc/judge_tasks.jsonl"
if ($LASTEXITCODE -eq 0) {
  Write-Host "OK -> $($Host):$remoteSrc/judge_tasks.jsonl" -ForegroundColor Green
  Write-Host "Next: on the node run the judge phase of run_all.sh or:" -ForegroundColor Green
  Write-Host "  cd ~/ToolCall/atr && STAGE=judge bash scripts/run_all.sh"
} else {
  Write-Host "scp FAILED (exit $LASTEXITCODE) - check the node is up / path (REPO=$remoteSrc)" -ForegroundColor Red
}
