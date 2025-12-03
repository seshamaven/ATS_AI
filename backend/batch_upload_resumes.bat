@echo off
REM Batch Upload Resumes Script
REM Recursively scans directories and uploads all resume files to the ATS API

echo ==========================================
echo Batch Resume Upload Script
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python to use this script.
    echo.
    pause
    exit /b 1
)

REM Check if API is running (optional check)
echo Checking API availability...
python -c "import requests; requests.get('http://localhost:5002/health', timeout=2)" >nul 2>&1
if errorlevel 1 (
    echo WARNING: API health check failed!
    echo The API may not be running at http://localhost:5002
    echo.
    set /p CONTINUE="Continue anyway? (Y/N): "
    if /i not "%CONTINUE%"=="Y" (
        echo Aborted.
        pause
        exit /b 1
    )
    echo.
) else (
    echo ✓ API is running
    echo.
)

REM Prompt for directory
set /p UPLOAD_DIR="Enter directory path to scan (e.g., uploads/): "

if not exist "%UPLOAD_DIR%" (
    echo.
    echo ERROR: Directory does not exist: %UPLOAD_DIR%
    pause
    exit /b 1
)

REM Prompt for API URL (optional)
set /p API_URL="Enter API URL (press Enter for default http://localhost:5002/api/processResume): "
if "%API_URL%"=="" (
    set API_URL=http://localhost:5002/api/processResume
)

REM Validate API URL format
echo %API_URL% | findstr /R "^http:// ^https://" >nul
if errorlevel 1 (
    echo.
    echo ERROR: Invalid API URL format!
    echo API URL must start with http:// or https://
    echo You entered: %API_URL%
    echo.
    echo Using default: http://localhost:5002/api/processResume
    set API_URL=http://localhost:5002/api/processResume
    echo.
)

REM Ask for dry run
set /p DRY_RUN="Dry run (scan only, don't upload)? (Y/N): "
if /i "%DRY_RUN%"=="Y" (
    set DRY_FLAG=--dry-run
) else (
    set DRY_FLAG=
)

echo.
echo ==========================================
echo Starting batch upload...
echo Directory: %UPLOAD_DIR%
echo API URL: %API_URL%
echo ==========================================
echo.

REM Run the Python script
cd /d "%~dp0"
if "%DRY_FLAG%"=="" (
    python batch_upload_resumes.py "%UPLOAD_DIR%" --api-url "%API_URL%"
) else (
    python batch_upload_resumes.py "%UPLOAD_DIR%" --api-url "%API_URL%" --dry-run
)

if errorlevel 1 (
    echo.
    echo ==========================================
    echo Upload completed with errors
    echo ==========================================
    echo Check batch_upload.log for details
) else (
    echo.
    echo ==========================================
    echo Upload completed successfully
    echo ==========================================
)

echo.
pause

