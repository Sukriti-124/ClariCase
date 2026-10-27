# Live training tracker for src/improved_model_3
# Usage:
#   .\scripts\watch_training.ps1
#   .\scripts\watch_training.ps1 -LogPath docs\metrics\improved_model_3\train.log
#
# Shows: GPU memory + utilization, last 15 log lines, refreshes every 3 s.
# Press Ctrl+C to exit.

param(
    [string]$LogPath = "docs\metrics\improved_model_3\train.log",
    [int]$IntervalSec = 3,
    [int]$TailLines = 15
)

$ErrorActionPreference = "SilentlyContinue"

while ($true) {
    Clear-Host
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  FLAN-T5-base Training Tracker   $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    # GPU snapshot via nvidia-smi
    Write-Host ""
    Write-Host "[GPU]" -ForegroundColor Yellow
    $gpu = & nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null
    if ($gpu) {
        $parts = $gpu -split ','
        $name = $parts[0].Trim()
        $memUsed = [int]$parts[1].Trim()
        $memTotal = [int]$parts[2].Trim()
        $util = [int]$parts[3].Trim()
        $temp = [int]$parts[4].Trim()
        $pwr = $parts[5].Trim()

        $memPct = if ($memTotal -gt 0) { [math]::Round(100 * $memUsed / $memTotal, 1) } else { 0 }
        $barLen = 30
        $filled = [int]($memPct / 100 * $barLen)
        $bar = ('#' * $filled) + ('.' * ($barLen - $filled))

        Write-Host "  $name"
        $color = if ($memPct -gt 92) {"Red"} elseif ($memPct -gt 80) {"Yellow"} else {"Green"}
        Write-Host "  VRAM : [$bar] $memUsed MB / $memTotal MB ($memPct%)" -ForegroundColor $color
        Write-Host "  Util : $util%   Temp: $temp C   Power: $pwr W"
    } else {
        Write-Host "  nvidia-smi not found (training must be on CPU — abort and fix)" -ForegroundColor Red
    }

    # Log tail
    Write-Host ""
    Write-Host "[LOG] $LogPath" -ForegroundColor Yellow
    if (Test-Path $LogPath) {
        $size = (Get-Item $LogPath).Length
        $sizeKb = [math]::Round($size / 1024, 1)
        Write-Host "  size: $sizeKb KB   last modified: $((Get-Item $LogPath).LastWriteTime.ToString('HH:mm:ss'))"
        Write-Host ""
        # Carriage returns in HF progress bars make Get-Content noisy; split
        $raw = Get-Content $LogPath -Raw -ErrorAction SilentlyContinue
        if ($raw) {
            $lines = $raw -split "[`r`n]+" | Where-Object { $_.Trim().Length -gt 0 }
            $lines | Select-Object -Last $TailLines | ForEach-Object {
                # Color-code recognizable patterns
                if ($_ -match "Traceback|Error|OOM|OutOfMemory|CUDA out of memory|RuntimeError") {
                    Write-Host "  $_" -ForegroundColor Red
                } elseif ($_ -match "loss|team_macro|epoch|eval_") {
                    Write-Host "  $_" -ForegroundColor Green
                } elseif ($_ -match "\[train\]") {
                    Write-Host "  $_" -ForegroundColor Cyan
                } else {
                    Write-Host "  $_" -ForegroundColor DarkGray
                }
            }
        }
    } else {
        Write-Host "  (log file does not exist yet)" -ForegroundColor DarkGray
    }

    # Metrics files
    Write-Host ""
    Write-Host "[OUTPUTS]" -ForegroundColor Yellow
    $metricsDir = Split-Path $LogPath
    foreach ($f in @("run_info.json", "metrics_smoke.json", "metrics_eval_subset.json")) {
        $p = Join-Path $metricsDir $f
        if (Test-Path $p) {
            Write-Host "  OK   $f" -ForegroundColor Green
        } else {
            Write-Host "  --   $f (not written yet)" -ForegroundColor DarkGray
        }
    }

    Write-Host ""
    Write-Host "Refreshing every ${IntervalSec}s. Ctrl+C to exit." -ForegroundColor DarkGray

    Start-Sleep -Seconds $IntervalSec
}
