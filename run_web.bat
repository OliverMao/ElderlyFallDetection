@echo off
setlocal enabledelayedexpansion
title AI Fall Detection Web Preview

echo ====================================================================
echo   Starting AI Fall Detection Web Preview Server...
echo   Open http://127.0.0.1:8000 in your browser
echo ====================================================================

:: Navigate to script directory
cd /d "%~dp0"

:: 1. Prefer the project virtual environment
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    goto :RUN
)

:: 2. Check if Anaconda Python exists
if exist "C:\ProgramData\anaconda3\python.exe" (
    set "PYTHON_EXE=C:\ProgramData\anaconda3\python.exe"
    goto :RUN
)

:: 3. Check if py launcher exists
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=py -3.11"
    goto :RUN
)

:: 4. Check default python in PATH
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python"
    goto :RUN
)

echo [ERROR] No compatible Python executable found.
echo Please ensure Python 3.10+ or Anaconda is installed.
pause
exit /b 1

:RUN
echo Using Python: !PYTHON_EXE!
echo.
!PYTHON_EXE! server.py
echo.
pause
