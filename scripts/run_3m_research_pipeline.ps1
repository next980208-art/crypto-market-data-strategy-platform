param(
    [string]$StartDate = "2026-05-22",
    [string]$EndDate = "2026-08-20",
    [string]$ReportPath = "docs/three_month_v2_research_pipeline_execution_2026-08-23.md"
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
    docker info | Out-Null
    Assert-LastExitCode "Docker Desktop이 실행 중이 아닙니다."

    Write-Output "[1/8] Flink와 Airflow 컨테이너를 시작합니다."
    docker compose up -d flink-staging-init jobmanager taskmanager airflow
    Assert-LastExitCode "Docker Compose 서비스 시작에 실패했습니다."

    Write-Output "[2/8] 실제 Binance USDT-M 90일 1분봉을 14일 청크로 수집·가공합니다."
    docker compose exec -T -u 0 -w /opt/airflow/project airflow python /opt/airflow/project/backfill_runner.py `
        --symbol BTC/USDT `
        --market usdm `
        --timeframe 1m `
        --start-date $StartDate `
        --end-date $EndDate `
        --chunk-days 14 `
        --temp-folder /opt/airflow/project/temp_raw_data_v2_3m `
        --feature-folder /opt/airflow/project/feature_store_v2_3m `
        --processor flink `
        --recover-stale-staging
    Assert-LastExitCode "3개월 Feature Store v2 생성에 실패했습니다."

    Write-Output "[3/8] 같은 90일 구간의 mark price·funding·open interest 문맥을 수집합니다."
    python 9_futures_context_collector.py `
        --symbol BTC/USDT `
        --timeframe 1m `
        --open-interest-timeframe 5m `
        --start-date $StartDate `
        --end-date $EndDate `
        --context-folder futures_context_store_v2_3m
    Assert-LastExitCode "3개월 선물 문맥 수집에 실패했습니다."

    Write-Output "[4/8] 다음 봉 시가 기준 Triple Barrier 라벨을 생성합니다."
    python 4_triple_barrier_labeler.py `
        --feature-folder feature_store_v2_3m `
        --label-folder label_store_v2_3m `
        --symbol BTCUSDT `
        --market usdm `
        --timeframe 1m `
        --stop-distance-pct 0.005 `
        --take-profit-r "1,1.5,3" `
        --max-holding-minutes 240 `
        --round-trip-cost-bps 10 `
        --batch-size 50000 `
        --verify-sample-size 100
    Assert-LastExitCode "3개월 Triple Barrier 라벨 생성에 실패했습니다."

    Write-Output "[5/8] Feature·Label·선물 문맥을 결합한 ML 데이터셋을 생성합니다."
    python 6_build_ml_dataset.py `
        --feature-folder feature_store_v2_3m `
        --label-folder label_store_v2_3m `
        --context-folder futures_context_store_v2_3m `
        --dataset-folder ml_dataset_v2_3m `
        --symbol BTCUSDT `
        --market usdm `
        --timeframe 1m `
        --min-expected-r 0.0
    Assert-LastExitCode "3개월 ML 데이터셋 생성에 실패했습니다."

    Write-Output "[6/8] 시간 순서 분리와 라벨 구간 purge를 적용해 방향 모델을 학습합니다."
    python 7_train_direction_model.py `
        --dataset-folder ml_dataset_v2_3m `
        --model-folder models_v2_3m `
        --prediction-folder predictions_v2_3m `
        --symbol BTCUSDT `
        --market usdm `
        --timeframe 1m `
        --test-size 0.2
    Assert-LastExitCode "3개월 방향 모델 학습에 실패했습니다."

    Write-Output "[7/8] 거래당 계좌 손실 2% 상한과 비용을 반영한 연구용 백테스트를 실행합니다."
    python 8_model_signal_backtest.py `
        --prediction-folder predictions_v2_3m `
        --symbol BTCUSDT `
        --market usdm `
        --timeframe 1m `
        --take-profit-r 1.0 `
        --risk-per-trade-pct 0.02 `
        --min-confidence 0.45 `
        --min-minutes-between-entries 15 `
        --max-new-entries-per-day 5 `
        --daily-stop-pct 0.04 `
        --hard-daily-stop-pct 0.06 `
        --max-consecutive-losses 3 `
        --extra-slippage-bps 5 `
        --report-path runtime_reports/model_signal_backtest_report_v2_3m_2026-08-23.json `
        --skip-markdown
    Assert-LastExitCode "3개월 모델 신호 백테스트에 실패했습니다."

    Write-Output "[8/8] 전체 산출물과 위험 한도를 검증하고 통합 Markdown을 생성합니다."
    python scripts/build_3m_research_report.py `
        --start-date $StartDate `
        --end-date $EndDate `
        --report-path $ReportPath
    Assert-LastExitCode "3개월 연구 파이프라인 최종 검증에 실패했습니다."

    Write-Output "3개월 v2 연구 파이프라인이 완료되었습니다."
    Write-Output "통합 실행 보고서: $ReportPath"
}
finally {
    Pop-Location
}
