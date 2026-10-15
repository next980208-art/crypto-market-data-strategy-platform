param(
    [string]$DatasetFile = "ml_dataset_v2_3m/market=usdm/symbol=BTCUSDT/timeframe=1m/ml_dataset_BTCUSDT_USDM_1m.parquet",
    [string]$ReportPath = "docs/selective_direction_model_experiment_2026-08-23.md"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

function Assert-LastExitCode([string]$Message) {
    if ($LASTEXITCODE -ne 0) {
        throw $Message
    }
}

Push-Location $projectRoot
try {
    Write-Output "[1/3] Train the causal selective direction model."
    python 13_train_selective_direction_model.py `
        --dataset-file $DatasetFile `
        --model-folder models_selective_v3 `
        --prediction-folder predictions_selective_v3 `
        --symbol BTCUSDT `
        --market usdm `
        --timeframe 1m `
        --train-fraction 0.60 `
        --validation-fraction 0.20 `
        --ambiguous-return-r 0.20 `
        --threshold-candidates "0.52,0.55,0.58,0.60,0.62,0.65" `
        --minimum-validation-trades 20
    Assert-LastExitCode "Selective direction model training failed."

    $metadataPath = "models_selective_v3/selective_direction_model_BTCUSDT_USDM_1m_metadata.json"
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    $threshold = $metadata.selected_probability_threshold.ToString(
        [System.Globalization.CultureInfo]::InvariantCulture
    )

    Write-Output "[2/3] Backtest the untouched final time partition with threshold $threshold."
    python 8_model_signal_backtest.py `
        --prediction-folder predictions_selective_v3 `
        --symbol BTCUSDT `
        --market usdm `
        --timeframe 1m `
        --take-profit-r 1.0 `
        --risk-per-trade-pct 0.02 `
        --min-confidence $threshold `
        --min-minutes-between-entries 15 `
        --max-new-entries-per-day 5 `
        --daily-stop-pct 0.04 `
        --hard-daily-stop-pct 0.06 `
        --max-consecutive-losses 3 `
        --extra-slippage-bps 5 `
        --report-path runtime_reports/model_signal_backtest_report_selective_v3_2026-08-23.json `
        --skip-markdown
    Assert-LastExitCode "Selective direction model backtest failed."

    Write-Output "[3/3] Build the comparison and next-step report."
    python scripts/build_selective_model_report.py --report-path $ReportPath
    Assert-LastExitCode "Selective direction model report generation failed."

    Write-Output "Selective model experiment completed: $ReportPath"
}
finally {
    Pop-Location
}
