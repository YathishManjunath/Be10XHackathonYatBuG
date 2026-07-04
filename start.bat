@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo  ========================================
echo   EventSync AI - Starting Application
echo  ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Running setup first...
    echo.
    call "%~dp0setup.bat" --nopause
    if errorlevel 1 exit /b 1
)

if not exist ".venv\Scripts\streamlit.exe" (
    echo Streamlit not installed. Running setup...
    echo.
    call "%~dp0setup.bat" --nopause
    if errorlevel 1 exit /b 1
)

echo Launching EventSync AI in your browser...
echo Press Ctrl+C in this window to stop the server.
echo.

".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false

if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit exited with an error.
    pause
)

endlocal
