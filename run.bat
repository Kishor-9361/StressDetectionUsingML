@echo off
echo ===================================================
echo   Starting Multimodal Stress Detection System
echo ===================================================

echo.
echo [0/2] Cleaning up existing processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1
echo Done.

echo.
echo [1/2] Starting Backend Server (Flask)...
start "Backend Server" cmd /k "cd backend && python app.py"

echo.
echo [2/2] Starting Frontend Application (React)...
start "Frontend App" cmd /k "cd frontend && npm start"

echo.
echo ===================================================
echo   System Started Successfully!
echo   ---------------------------------------------
echo   - Frontend: http://localhost:3000
echo   - Backend:  http://localhost:5000
echo ===================================================
pause
