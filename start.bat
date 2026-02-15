@echo off
REM Kiwi Voice Service Launcher

echo ====================================
echo   Kiwi Voice Service
echo ====================================

REM Проверяем Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

REM Проверяем venv
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Активируем venv
call venv\Scripts\activate.bat

REM Устанавливаем зависимости если нужно
pip show faster-whisper >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Проверяем env переменные
if "%RUNPOD_API_KEY%"=="" (
    echo WARNING: RUNPOD_API_KEY not set
    echo Voice responses will be disabled
)

if "%RUNPOD_TTS_ENDPOINT_ID%"=="" (
    echo WARNING: RUNPOD_TTS_ENDPOINT_ID not set
    echo Voice responses will be disabled
)

REM Запускаем
echo.
echo Starting Kiwi Voice Service...
python -m kiwi

pause
