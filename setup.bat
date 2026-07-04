@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo  ========================================
echo   EventSync AI - First-time Setup
echo  ========================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    where py >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
        echo         During install, check "Add Python to PATH".
        pause
        exit /b 1
    )
    set "PYTHON=py -3"
) else (
    set "PYTHON=python"
)

echo [1/3] Checking Python...
%PYTHON% --version
if errorlevel 1 (
    echo [ERROR] Could not run Python.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    echo [2/3] Virtual environment already exists at .venv
) else (
    echo [2/3] Creating virtual environment in .venv ...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [3/3] Installing dependencies from requirements.txt ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo  Setup complete! Run start.bat to launch EventSync AI.
echo.
if /i not "%~1"=="--nopause" pause
endlocal
