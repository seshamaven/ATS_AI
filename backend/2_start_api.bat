@echo off
REM 2. Start ATS API Server

echo ==========================================
echo Step 2: Starting ATS API Server
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if requirements are installed
echo Checking dependencies...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements_ats.txt
    echo.
    echo Downloading spaCy NLP model...
    python -m spacy download en_core_web_sm
)

REM Create uploads directory if not exists
if not exist "uploads\" (
    echo Creating uploads directory...
    mkdir uploads
)

REM Check if .env exists
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please create .env file with your configuration.
    echo You can copy from env_ats_template.txt
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" (
        pause
        exit /b 1
    )
)

REM Start the API
echo.
echo ==========================================
echo Starting ATS API on http://localhost:5002
echo ==========================================
echo.
echo Available endpoints:
echo   - POST /api/processResume
echo   - POST /api/profileRankingByJD
echo   - GET  /api/candidate/^<id^>
echo   - GET  /health
echo.
echo Press Ctrl+C to stop the server
echo.

python ats_api.py

pause

