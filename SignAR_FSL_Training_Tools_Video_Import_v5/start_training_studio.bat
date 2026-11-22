@echo off
setlocal
cd /d "%~dp0\.."
where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher ^(py^) was not found. Install Python 3.10, 3.11, or 3.12 and retry.
    pause
    exit /b 1
)
py -3 SignAR_FSL_Training_Tools_Video_Import_v5\training\studio_server.py
pause
