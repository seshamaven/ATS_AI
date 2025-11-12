@echo off
REM 4. Upload Resume to ATS

echo ==========================================
echo Step 4: Upload Resume to ATS
echo ==========================================
echo.

REM Check if API is running
python -c "import requests; requests.get('http://localhost:5002/health', timeout=2)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: ATS API is not running!
    echo Please start the API first using 2_start_api.bat
    echo.
    pause
    exit /b 1
)

echo ✓ API is running
echo.

REM Prompt for resume file
set /p RESUME_FILE="Enter path to resume file (PDF, DOCX, or DOC): "

if not exist "%RESUME_FILE%" (
    echo.
    echo ERROR: File not found: %RESUME_FILE%
    pause
    exit /b 1
)

echo.
echo Uploading resume: %RESUME_FILE%
echo.

REM Upload using curl if available, otherwise use Python
where curl >nul 2>&1
if errorlevel 1 (
    echo Using Python to upload...
    python -c "import requests; r=requests.post('http://localhost:5002/api/processResume', files={'file': open('%RESUME_FILE%', 'rb')}); print('Status:', r.status_code); print(r.json())"
) else (
    echo Using curl to upload...
    curl -X POST http://localhost:5002/api/processResume -F "file=@%RESUME_FILE%"
)

echo.
echo ==========================================
echo Upload Complete
echo ==========================================
echo.
pause

