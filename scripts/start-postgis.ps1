Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$composeFile = Join-Path $root "docker-compose.yml"
$containerName = "dashcam-postgis"

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
    Write-Host "PostGIS container '$containerName' is already running."
} elseif ($existingState) {
    Write-Host "Starting existing PostGIS container '$containerName'..."
    docker start $containerName | Out-Null
} else {
    Write-Host "Starting PostGIS with docker compose..."
    docker compose -f $composeFile up -d postgis | Out-Null
}

Write-Host "Waiting for PostgreSQL/PostGIS to become ready..."
for ($i = 0; $i -lt 60; $i++) {
    docker exec $containerName pg_isready -U postgres -d dashcam 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PostGIS is ready on localhost:5432"
        exit 0
    }
    Start-Sleep -Seconds 2
}

Write-Error "PostGIS did not become ready in time. Check: docker logs $containerName --tail 100"
exit 1
