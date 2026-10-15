param(
    [switch]$Build
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

function Test-DockerReady {
    try {
        & docker info 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

if (-not (Test-DockerReady)) {
    throw "Docker Desktop is not ready. Start Docker Desktop and run this script again."
}

$composeArgs = @("compose", "--profile", "streaming", "--profile", "realtime", "up", "-d")
if ($Build) {
    $composeArgs += "--build"
}

& docker @composeArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

docker compose ps
Write-Output "Airflow: http://localhost:8080"
Write-Output "Flink: http://localhost:8081"
Write-Output "Realtime services are paper-only and cannot call an exchange order API."
