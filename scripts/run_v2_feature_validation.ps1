param(
    [string]$StartDate = "2026-08-20",
    [ValidateRange(2, 14)]
    [int]$Days = 2,
    [string]$ReportPath = "runtime_reports/v2_batch_pipeline_execution.md"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$start = [datetime]::ParseExact(
    $StartDate,
    "yyyy-MM-dd",
    [System.Globalization.CultureInfo]::InvariantCulture
)
$endDate = $start.AddDays($Days).ToString("yyyy-MM-dd")

Push-Location $projectRoot
try {
    docker info | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Desktop이 실행 중이 아닙니다. Docker Desktop을 먼저 켜세요."
    }

    Write-Output "[1/3] Flink와 Airflow 컨테이너를 시작합니다."
    docker compose up -d flink-staging-init jobmanager taskmanager airflow
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose 서비스 시작에 실패했습니다."
    }

    Write-Output "[2/3] Binance USDT-M 실제 1분봉을 수집하고 PyFlink로 가공합니다."
    docker compose exec -T -u 0 -w /opt/airflow/project airflow python /opt/airflow/project/backfill_runner.py `
        --symbol BTC/USDT `
        --market usdm `
        --timeframe 1m `
        --start-date $StartDate `
        --end-date $endDate `
        --chunk-days 1 `
        --temp-folder /opt/airflow/project/temp_raw_data_v2 `
        --feature-folder /opt/airflow/project/feature_store_v2 `
        --processor flink
    if ($LASTEXITCODE -ne 0) {
        throw "Feature Store v2 백필 실행에 실패했습니다."
    }

    Write-Output "[3/3] 성공 마커, Parquet 행 수, 청크 경계와 원천 삭제를 검증합니다."
    python scripts/build_v2_validation_report.py `
        --start-date $StartDate `
        --end-date $endDate `
        --feature-folder feature_store_v2 `
        --raw-folder temp_raw_data_v2 `
        --report-path $ReportPath
    if ($LASTEXITCODE -ne 0) {
        throw "v2 실행 결과 검증에 실패했습니다. 보고서를 확인하세요: $ReportPath"
    }

    Write-Output "Feature Store v2 검증이 완료되었습니다."
    Write-Output "실행 보고서: $ReportPath"
}
finally {
    Pop-Location
}
