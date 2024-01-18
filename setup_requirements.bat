@echo off
setlocal

:: Check if Python is installed
where python >nul 2>nul
if errorlevel 1 (
    echo Python is not installed. Please install Python first.
    exit /b 1
)

:: Set the virtual environment name
set VENV_NAME=.venv

:: Check if virtual environment exists
if not exist %VENV_NAME% (
    echo Creating virtual environment...
    python -m venv %VENV_NAME%
)

:: Activate the virtual environment
call %VENV_NAME%\Scripts\activate

:: Install dependencies
if exist requirements.txt (
    echo Installing dependencies...
    pip install -r requirements.txt
)

endlocal
