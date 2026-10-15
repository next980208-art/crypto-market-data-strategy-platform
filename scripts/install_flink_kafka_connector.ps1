param(
    [string]$Destination = "connectors/flink-connector-kafka.jar",
    [string]$KafkaClientsDestination = "connectors/kafka-clients.jar"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$url = "https://repo.maven.apache.org/maven2/org/apache/flink/flink-connector-kafka/5.0.0-2.2/flink-connector-kafka-5.0.0-2.2.jar"
$kafkaClientsUrl = "https://repo.maven.apache.org/maven2/org/apache/kafka/kafka-clients/4.2.0/kafka-clients-4.2.0.jar"
$destinationPath = Join-Path $projectRoot $Destination
$kafkaClientsPath = Join-Path $projectRoot $KafkaClientsDestination
$destinationFolder = Split-Path -Parent $destinationPath

New-Item -ItemType Directory -Force -Path $destinationFolder | Out-Null
Invoke-WebRequest -Uri $url -OutFile $destinationPath -UseBasicParsing
Invoke-WebRequest -Uri $kafkaClientsUrl -OutFile $kafkaClientsPath -UseBasicParsing

if ((Get-Item -LiteralPath $destinationPath).Length -lt 100000) {
    throw "Downloaded Kafka connector JAR is unexpectedly small: $destinationPath"
}

if ((Get-Item -LiteralPath $kafkaClientsPath).Length -lt 1000000) {
    throw "Downloaded Kafka clients JAR is unexpectedly small: $kafkaClientsPath"
}

Write-Output "Installed Flink 2.2 Kafka connector: $destinationPath"
Write-Output "Installed Kafka client dependency: $kafkaClientsPath"
