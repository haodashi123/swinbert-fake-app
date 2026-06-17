@echo off
title IGP Platform

echo ========================================
echo   IGP Platform - Starting...
echo ========================================

cd /d "%~dp0backend_api"
start "IGP-Backend" cmd /k "python -m uvicorn main:app --reload --port 8000"

timeout /t 3 /nobreak >nul

cd /d "%~dp0frontend_web"
start "IGP-Frontend" cmd /k "npm.cmd run build && npm.cmd run preview -- --port 5173"

timeout /t 4 /nobreak >nul

start http://localhost:5173

echo.
echo ========================================
echo   Server Started!
echo   Backend: http://127.0.0.1:8000
echo   Frontend: http://localhost:5173
echo   Close all black cmd windows to stop
echo ========================================
pause
