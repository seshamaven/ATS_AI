@echo off
REM 3. Test ATS API Endpoints

echo ==========================================
echo Step 3: Testing ATS API
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Check if API is running
echo Checking if API is running...
python -c "import requests; requests.get('http://localhost:5002/health', timeout=2)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: ATS API is not running!
    echo Please start the API first using 2_start_api.bat
    echo.
    pause
    exit /b 1
)

echo ✓ API is running
echo.
echo ==========================================
echo Running ATS API Test Suite
echo ==========================================
echo.

python test_ats_api.py

echo.
echo ==========================================
echo Test Complete
echo ==========================================
echo.
pause

