@echo off
REM Windows Batch Script for Quick Setup
REM This is a wrapper around setup.py for Windows users

echo ======================================
echo Drone Detection System - Quick Setup
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.12 is not installed or not in PATH
    echo Please install Python 3.12 from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check Python version
python --version
echo.

REM Run the setup script
python setup.py

if errorlevel 1 (
    echo.
    echo Setup failed! Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ======================================
echo Setup completed successfully!
echo ======================================
echo.
pause

