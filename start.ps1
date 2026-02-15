# Kiwi Voice Service Launcher (PowerShell)

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "   Kiwi Voice Service" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# Переходим в директорию скрипта
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Проверяем venv
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Активируем venv
& ".\venv\Scripts\Activate.ps1"

# Проверяем зависимости
$installed = pip show faster-whisper 2>$null
if (-not $installed) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Проверяем env переменные
if (-not $env:RUNPOD_API_KEY) {
    Write-Host "WARNING: RUNPOD_API_KEY not set" -ForegroundColor Yellow
    Write-Host "Voice responses will be disabled" -ForegroundColor Yellow
}

if (-not $env:RUNPOD_TTS_ENDPOINT_ID) {
    Write-Host "WARNING: RUNPOD_TTS_ENDPOINT_ID not set" -ForegroundColor Yellow
    Write-Host "Voice responses will be disabled" -ForegroundColor Yellow
}

# Запускаем
Write-Host ""
Write-Host "Starting Kiwi Voice Service..." -ForegroundColor Green
python -m kiwi
