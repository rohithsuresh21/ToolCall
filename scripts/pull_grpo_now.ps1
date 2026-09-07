# =============================================================================
# pull_grpo_now.ps1 - IMMEDIATE weight pull from the Cynaptics opengpu node.
#
# The user's rule for the opengpu run: the trained GRPO adapter is pulled to this
# machine the MOMENT it exists on the node, BEFORE anything else (the node is a
# throwaway container: weights vanish when the reservation ends). Then the eval
# reports / judge scores / PDF are pulled, and the combined report is rebuilt
# locally if reportlab works here.
#
# Behaviour:
#   * keeps polling the node every 15s until it is reachable;
#   * then polls ship/ for the first <OUT_NAME>_*.tar.gz (default grpo-real) --
#     tiny tarball gate: it must be > 5 MB so a half-written archive is skipped;
#   * scp's the weights tarball FIRST to "pulled 2", prints size + sha256;
#   * then pulls the eval tarballs, reports, manifest and PDF for that run;
#   * optionally rebuilds the combined PDF locally (REBUILD_PDF=1 default).
#
# Run with:
#   powershell -File scripts/pull_grpo_now.ps1
# Env knobs: HOST, PORT, OUT_NAME (default grpo-real), PULL_ALL (0/1, default 1)
# =============================================================================

param(
  [string]$Host_    = (Get-Content Env:HOST -ErrorAction SilentlyContinue),
  [int]$Port        = 22013,
  [string]$OutName  = "grpo-real",
  [switch]$Poll     = $true,
  [switch]$PullOnlyWeights = $false,
  [int]$MaxWaitMin  = 240
)
if (-not $Host_) { $Host_ = "gpu17@10.214.5.55" }

$Base = "~/ToolCall/artifacts/ship"
$Dest = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "pulled 2"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

function Test-Port($addr, $p) {
  try {
    $c = New-Object System.Net.Sockets.TcpClient
    $iar = $c.BeginConnect($addr, $p, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne(4000, $false)
    if ($ok -and $c.Connected) { $c.Close(); return $true }
    $c.Close(); return $false
  } catch { return $false }
}

# Split "gpu17@10.214.5.55" into ssh host + connect host for the TCP probe.
$sshHost = $Host_
$connHost = $Host_
if ($Host_ -match '@') {
  $connHost = ($Host_ -split '@')[1]
}

# ---------------------------------------------------------------- 1. WAIT ---
Write-Host "Waiting for node ${connHost}:$Port ..." -ForegroundColor Cyan
$deadline = (Get-Date).AddMinutes($MaxWaitMin)
while (-not (Test-Port $connHost $Port)) {
  if ((Get-Date) -gt $deadline) {
    Write-Host "Node not reachable within $MaxWaitMin min. Exiting." -ForegroundColor Red
    exit 2
  }
  Start-Sleep -Seconds 15
}
Write-Host "Node is UP." -ForegroundColor Green

# ------------------------------------------------- 2. WAIT FOR WEIGHTS ----
Write-Host "Waiting for ship/${OutName}_*.tar.gz (weights) on node ..." -ForegroundColor Cyan
$target = $null
$lastSize = -1
$stableCount = 0
while ($true) {
  if ((Get-Date) -gt $deadline) {
    Write-Host "Timed out waiting for weights tarball." -ForegroundColor Red
    exit 3
  }
  # Find the newest matching tarball and its size (Linux stat). Remote glob.
  $probe = ssh -p $Port "${Host_}" "ls -t $Base/${OutName}_*.tar.gz 2>/dev/null | head -1"
  if ($LASTEXITCODE -eq 0 -and $probe) {
    $probe = ($probe | Select-Object -Last 1).Trim()
    if ($probe) {
      $sz = (ssh -p $Port "${Host_}" "stat -c %s '$probe' 2>/dev/null").Trim()
      try { $sz = [int64]$sz } catch { $sz = -1 }
      if ($sz -eq $lastSize -and $sz -gt 5MB) {
        $stableCount++
      } else {
        $stableCount = 0
      }
      $lastSize = $sz
      Write-Host "  remote size: $sz bytes (stable x$stableCount)" -ForegroundColor DarkGray
      # stable across 3 polls (>=60s) and > 5MB => archive fully written
      if ($stableCount -ge 3 -and $sz -gt 5MB) {
        $target = "$probe"
        break
      }
    }
  }
  Start-Sleep -Seconds 20
}

# --------------------------------------------------- 3. PULL WEIGHTS FIRST --
Write-Host "`n=== PULL WEIGHTS (weights first, then evals) ===" -ForegroundColor Cyan
scp -P $Port "${Host_}:$target" $Dest
if ($LASTEXITCODE -ne 0) {
  Write-Host "scp failed for weights." -ForegroundColor Red
  exit 4
}
$w = Get-ChildItem -LiteralPath $Dest -Filter "${OutName}_*.tar.gz" |
     Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "  weights OK: $($w.Name)  ($([math]::Round($w.Length/1MB,1)) MB)" -ForegroundColor Green
$null = & certutil -hashfile $w.FullName SHA256 | Select-Object -Skip 1 -First 1

if ($PullOnlyWeights) {
  Write-Host "`nPullOnlyWeights set - stopping after weights." -ForegroundColor Green
  exit 0
}

# ---------------------------------------------------- 4. PULL THE REST -----
# Evals run on the node AFTER weights are archived, so wait for the dev-report
# and judge-scores to appear (that signals both evals finished) before pulling.
Write-Host "`n=== Waiting for node eval reports (weights already safe locally) ===" -ForegroundColor Cyan
$done = $false
while (-not $done) {
  if ((Get-Date) -gt $deadline) {
    Write-Host "Timed out waiting for eval reports; pulling what exists." -ForegroundColor Yellow
    break
  }
  $d1 = ssh -p $Port "${Host_}" "ls $Base/dev-report_*.json 2>/dev/null | head -1"
  $d2 = ssh -p $Port "${Host_}" "ls $Base/judge-scores_*.jsonl 2>/dev/null | head -1"
  if ($LASTEXITCODE -eq 0 -and $d1 -and $d2) { $done = $true }
  if (-not $done) { Start-Sleep -Seconds 30 }
}
Write-Host "  reports present on node." -ForegroundColor Green

Write-Host "`n=== PULL eval reports / scores / PDF ===" -ForegroundColor Cyan
$rest = @(
  "eval-dev_*.tar.gz",
  "eval-judge_*.tar.gz",
  "dev-report_*.txt",
  "dev-report_*.json",
  "judge-report_*.txt",
  "judge-scores_*.jsonl",
  "ATR-Eval-Report_*.pdf",
  "MANIFEST_*.txt"
)
foreach ($it in $rest) {
  scp -P $Port "${Host_}:$Base/$it" $Dest 2>$null | Out-Null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK -> $it" -ForegroundColor Green
  } else {
    Write-Host "  not yet: $it" -ForegroundColor Yellow
  }
}

# -------------------------------------------------- 5. REBUILD PDF LOCALLY --
if ($env:REBUILD_PDF -ne "0") {
  $dev  = Get-ChildItem -LiteralPath $Dest -Filter "dev-report_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $jud  = Get-ChildItem -LiteralPath $Dest -Filter "judge-scores_*.jsonl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($dev -and $jud) {
    Write-Host "`n=== Rebuilding combined PDF locally ===" -ForegroundColor Cyan
    python (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "make_combined_report.py") `
      --dev $dev.FullName --judge $jud.FullName `
      --out (Join-Path $Dest "ATR-Eval-Report_rebuilt.pdf") --model "Qwen/Qwen3-1.7B"
    if ($LASTEXITCODE -eq 0) {
      Write-Host "  local PDF -> $Dest\ATR-Eval-Report_rebuilt.pdf" -ForegroundColor Green
    } else {
      Write-Host "  local PDF build skipped (reportlab/env issue); the node PDF (if any) was already pulled." -ForegroundColor Yellow
    }
  }
}

Write-Host "`n===== pull_grpo_now done. Check '$Dest' =====" -ForegroundColor Green