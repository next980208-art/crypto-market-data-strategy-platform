param(
    [switch]$WithRealtime
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    if ($WithRealtime) {
        docker compose --profile streaming --profile realtime up -d airflow jobmanager taskmanager kafka zookeeper realtime-collector realtime-inference realtime-risk-paper
    }
    else {
        docker compose up -d airflow jobmanager taskmanager
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Docker services could not be started."
    }

    docker compose exec -T airflow airflow dags unpause btcusdt_usdm_historical_backfill
    if ($LASTEXITCODE -ne 0) {
        throw "The historical backfill DAG could not be enabled."
    }

    Write-Host "Historical backfill automation is enabled. Airflow processes up to 56 days every hour while this PC is running."
}
finally {
    Pop-Location
}
