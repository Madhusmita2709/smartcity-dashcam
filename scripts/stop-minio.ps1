Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$containerName = "dashcam-minio"
$state = docker ps -a --filter "name=^$containerName$" --format "{{.State}}"

if (-not $state) {
    Write-Host "No MinIO container named '$containerName' exists."
    exit 0
}

if ($state -eq "running") {
    docker stop $containerName | Out-Null
    Write-Host "Stopped '$containerName'."
} else {
    Write-Host "Container '$containerName' is already stopped."
}
