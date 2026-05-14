param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoReload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$startPostgisScript = Join-Path $PSScriptRoot "start-postgis.ps1"
$startMinioScript = Join-Path $PSScriptRoot "start-minio.ps1"
$ffmpegBin = "C:\Users\Chinmaya\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment Python was not found at $venvPython"
}

if (-not (Test-Path $startPostgisScript)) {
    throw "start-postgis.ps1 was not found at $startPostgisScript"
}

if (-not (Test-Path $startMinioScript)) {
    throw "start-minio.ps1 was not found at $startMinioScript"
}

Write-Host "Ensuring PostGIS is running..."
powershell -ExecutionPolicy Bypass -File $startPostgisScript
Write-Host "Ensuring MinIO is running..."
powershell -ExecutionPolicy Bypass -File $startMinioScript

if (Test-Path $ffmpegBin) {
    $env:PATH = "$ffmpegBin;$env:PATH"
}

$uvicornArgs = @(
    "-m",
    "uvicorn",
    "backend.app.main:app",
    "--host",
    $BindHost,
    "--port",
    $Port
)

if (-not $NoReload) {
    $uvicornArgs += "--reload"
}

Write-Host "Starting FastAPI app on http://$BindHost`:$Port ..."
Write-Host "Press Ctrl+C to stop the app. PostGIS and MinIO will remain running."

Push-Location $root
try {
    & $venvPython @uvicornArgs
} finally {
    Pop-Location
}
