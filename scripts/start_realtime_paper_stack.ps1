$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    function Test-DockerReady {
        try {
            & docker info 2>$null | Out-Null
            return $LASTEXITCODE -eq 0
        }
        catch {
            return $false
        }
    }

    if (-not (Test-DockerReady)) { throw "Docker Desktop is not running." }

    docker compose --profile streaming --profile realtime up -d --build `
        zookeeper kafka realtime-collector realtime-feature-worker realtime-inference realtime-risk-paper
    if ($LASTEXITCODE -ne 0) { throw "Could not start the realtime paper stack." }

    Write-Output "Realtime paper stack started."
    Write-Output "Safety: inference publishes no_trade until an approved model path is implemented."
    Write-Output "Safety: risk service never calls an exchange order API."
    docker compose ps
}
finally {
    Pop-Location
}
