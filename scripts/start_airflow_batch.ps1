param(
    [switch]$Build,
    [switch]$Streaming
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

for ($attempt = 1; $attempt -le 30; $attempt++) {
    if (Test-DockerReady) {
        $composeArgs = @("compose")
        if ($Streaming) {
            $composeArgs += @("--profile", "streaming")
        }
        $composeArgs += @("up", "-d")
        if ($Build) {
            $composeArgs += "--build"
        }
        & docker @composeArgs
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        docker compose ps
        exit $LASTEXITCODE
    }
    Start-Sleep -Seconds 10
}

throw "Docker Desktop did not become ready within 5 minutes. Start Docker Desktop and run this script again."
