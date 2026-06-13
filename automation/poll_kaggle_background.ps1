$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$KernelId = "tristianyosa/face-mask-from-scratch-cv"
$ResultsDir = "results"
$LogPath = Join-Path $ResultsDir "kaggle_background_poll.log"
$PollSeconds = 180
$DeleteLogOnExit = $false
$PythonExe = "python"

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ResultsDir "figures") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ResultsDir "metrics") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ResultsDir "models") | Out-Null

function Write-Log {
    param([string]$Message)
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss UTC")
    "$timestamp $Message" | Tee-Object -FilePath $LogPath -Append
}

Write-Log "Starting Kaggle background poll for $KernelId"

while ($true) {
    try {
        $statusOutput = & $PythonExe -m kaggle kernels status $KernelId 2>&1
        Write-Log $statusOutput

        if ($statusOutput -match "COMPLETE") {
            Write-Log "Kernel complete. Downloading output to $ResultsDir"
            & $PythonExe -m kaggle kernels output $KernelId -p $ResultsDir 2>&1 | Tee-Object -FilePath $LogPath -Append

            Get-ChildItem -Path $ResultsDir -File -Filter "metrics.json" | Move-Item -Destination (Join-Path $ResultsDir "metrics") -Force
            Get-ChildItem -Path $ResultsDir -File -Filter "*.png" | Move-Item -Destination (Join-Path $ResultsDir "figures") -Force
            Get-ChildItem -Path $ResultsDir -File |
                Where-Object { $_.Extension -in ".keras", ".h5", ".tflite", ".joblib", ".pkl" } |
                Move-Item -Destination (Join-Path $ResultsDir "models") -Force

            Write-Log "Download and organization complete."
            Get-ChildItem -Path $ResultsDir -File -Filter "*.log" | Remove-Item -Force
            $pullDir = Join-Path $ResultsDir "kaggle_pull"
            if (Test-Path $pullDir) {
                Remove-Item -LiteralPath $pullDir -Recurse -Force
            }
            $DeleteLogOnExit = $true
            break
        }

        if ($statusOutput -match "ERROR|FAILED|CANCELED|CANCELLED") {
            Write-Log "Kernel ended with non-complete status. Downloading logs/output if available."
            & $PythonExe -m kaggle kernels output $KernelId -p $ResultsDir 2>&1 | Tee-Object -FilePath $LogPath -Append
            break
        }
    }
    catch {
        Write-Log "Polling error: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds $PollSeconds
}

if (-not $DeleteLogOnExit) {
    Write-Log "Background poller stopped."
}
