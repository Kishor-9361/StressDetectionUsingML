@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   Multimodal Stress Detection - Production Launcher
echo ========================================================
echo.

cd /d "%~dp0backend"

set PORT=5000
set FRONTEND_URL=*

:: Detect Python executable
set "PYTHON_EXE="
where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=py"
    )
)

if not defined PYTHON_EXE (
    if exist "C:\ProgramData\anaconda3\python.exe" (
        set "PYTHON_EXE=C:\ProgramData\anaconda3\python.exe"
    )
)

if not defined PYTHON_EXE (
    for /d %%D in ("C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python*") do (
        if exist "%%D\python.exe" set "PYTHON_EXE=%%D\python.exe"
    )
)

if not defined PYTHON_EXE (
    for /d %%D in ("C:\Program Files\Python*") do (
        if exist "%%D\python.exe" set "PYTHON_EXE=%%D\python.exe"
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python not found on your system.
    echo Please install Python 3.10+ from python.org or add your existing Python to PATH.
    pause
    exit /b 1
)

echo [Found Python]: %PYTHON_EXE%
%PYTHON_EXE% --version
echo.

echo [Starting Backend + Frontend Server]
echo Access the full web application and API at:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server.
echo --------------------------------------------------------

%PYTHON_EXE% app.py

pause
