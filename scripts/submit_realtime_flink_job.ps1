param(
    [string]$ConnectorJar = "connectors/flink-connector-kafka.jar",
    [string]$KafkaClientsJar = "connectors/kafka-clients.jar",
    [string]$BootstrapServers = "kafka:29092",
    [int]$MaxOutOfOrdernessSeconds = 5,
    [int]$AllowedLatenessSeconds = 10,
    [int]$IdleTimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$connectorPath = Join-Path $projectRoot $ConnectorJar
$kafkaClientsPath = Join-Path $projectRoot $KafkaClientsJar

if ((-not (Test-Path -LiteralPath $connectorPath -PathType Leaf)) -or
    (-not (Test-Path -LiteralPath $kafkaClientsPath -PathType Leaf))) {
    throw "Kafka connector dependencies are missing. Run scripts/install_flink_kafka_connector.ps1 first."
}

Push-Location $projectRoot
try {
    docker compose --profile streaming up -d flink-staging-init jobmanager taskmanager zookeeper kafka
    if ($LASTEXITCODE -ne 0) { throw "Could not start Flink and Kafka services." }

    docker compose --profile streaming --profile realtime stop realtime-feature-worker
    if ($LASTEXITCODE -ne 0) { throw "Could not stop the development feature worker." }

    docker compose exec -T jobmanager flink run `
        -C "file:///opt/flink/project/$ConnectorJar" `
        -C "file:///opt/flink/project/$KafkaClientsJar" `
        -py /opt/flink/project/flink_jobs/realtime_kafka_feature_job.py `
        -pyfs /opt/flink/project/market_metadata.py `
        --bootstrap-servers $BootstrapServers `
        --max-out-of-orderness-seconds $MaxOutOfOrdernessSeconds `
        --allowed-lateness-seconds $AllowedLatenessSeconds `
        --idle-timeout-seconds $IdleTimeoutSeconds
    if ($LASTEXITCODE -ne 0) { throw "Flink realtime feature job submission failed." }
}
finally {
    Pop-Location
}
