$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    docker compose --profile streaming --profile realtime stop `
        realtime-collector realtime-feature-worker realtime-inference realtime-risk-paper kafka zookeeper
    if ($LASTEXITCODE -ne 0) { throw "Could not stop the realtime paper stack." }
}
finally {
    Pop-Location
}
