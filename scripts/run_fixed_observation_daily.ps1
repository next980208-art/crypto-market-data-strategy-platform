param(
    [string]$Observation = "runtime_reports/fixed_breakout_forward_20260922"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $projectRoot "$Observation/scheduler_logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logRoot "run_$timestamp.log"
$latestPath = Join-Path $logRoot "latest.log"
$python = (Get-Command python -ErrorAction Stop).Source

Push-Location $projectRoot
try {
    "[$(Get-Date -Format o)] fixed observation started" | Tee-Object -FilePath $logPath
    & $python -m scripts.observe_fixed_breakout --output $Observation --resume 2>&1 |
        Tee-Object -FilePath $logPath -Append
    $observerExitCode = $LASTEXITCODE
    "[$(Get-Date -Format o)] fixed observation finished exit=$observerExitCode" |
        Tee-Object -FilePath $logPath -Append

    "[$(Get-Date -Format o)] weekly monitoring started" |
        Tee-Object -FilePath $logPath -Append
    & $python -m scripts.evaluate_fixed_observation `
        --output $Observation `
        --observer-exit-code $observerExitCode 2>&1 |
        Tee-Object -FilePath $logPath -Append
    $monitorExitCode = $LASTEXITCODE
    "[$(Get-Date -Format o)] weekly monitoring finished exit=$monitorExitCode" |
        Tee-Object -FilePath $logPath -Append

    Copy-Item -LiteralPath $logPath -Destination $latestPath -Force
    if ($observerExitCode -ne 0) {
        exit $observerExitCode
    }
    exit $monitorExitCode
}
catch {
    "[$(Get-Date -Format o)] fixed observation failed: $($_.Exception.Message)" |
        Tee-Object -FilePath $logPath -Append
    try {
        & $python -m scripts.evaluate_fixed_observation `
            --output $Observation `
            --observer-exit-code 1 2>&1 |
            Tee-Object -FilePath $logPath -Append
    }
    catch {
        "[$(Get-Date -Format o)] monitoring fallback failed: $($_.Exception.Message)" |
            Tee-Object -FilePath $logPath -Append
    }
    Copy-Item -LiteralPath $logPath -Destination $latestPath -Force
    exit 1
}
finally {
    Pop-Location
}
