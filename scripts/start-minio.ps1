Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$composeFile = Join-Path $root "docker-compose.yml"
$containerName = "dashcam-minio"

if (-not (Test-Path $composeFile)) {
    throw "docker-compose.yml was not found at $composeFile"
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
docker info --format '{{.ServerVersion}}' 2>$null | Out-Null
$ErrorActionPreference = $previousErrorActionPreference
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is not running. Start Docker Desktop, then rerun this script."
}

$existingState = docker ps -a --filter "name=^$containerName$" --format "{{.State}}"

if ($existingState -eq "running") {
    Write-Host "MinIO container '$containerName' is already running."
} elseif ($existingState) {
    Write-Host "Starting existing MinIO container '$containerName'..."
    docker start $containerName | Out-Null
} else {
    Write-Host "Starting MinIO with docker compose..."
    docker compose -f $composeFile up -d minio | Out-Null
}

Write-Host "Waiting for MinIO API and console to become ready..."
for ($i = 0; $i -lt 60; $i++) {
    try {
        $api = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:19000/minio/health/live -TimeoutSec 2
        $console = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:19001 -TimeoutSec 2
        if ($api.StatusCode -eq 200 -and $console.StatusCode -ge 200) {
            Write-Host "MinIO is ready on localhost:19000 and console on localhost:19001"
            exit 0
        }
    } catch {
    }
    Start-Sleep -Seconds 2
}

Write-Error "MinIO did not become ready in time. Check: docker logs $containerName --tail 100"
exit 1
