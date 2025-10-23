@echo off
REM 5. Rank Candidates Against Job Description

echo ==========================================
echo Step 5: Rank Candidates Against JD
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

REM Get job details
set /p JOB_ID="Enter Job ID (e.g., JD001): "
if "%JOB_ID%"=="" set JOB_ID=JD001

echo.
echo Enter Job Description (press Ctrl+Z then Enter when done):
echo ----------------------------------------

REM Create a temporary file for job description
set TEMP_JD_FILE=%TEMP%\ats_jd_%RANDOM%.txt
copy con "%TEMP_JD_FILE%" >nul

echo.
set /p REQUIRED_SKILLS="Enter required skills (comma-separated): "
set /p MIN_EXP="Enter minimum experience (years, e.g., 3): "
if "%MIN_EXP%"=="" set MIN_EXP=0

set /p DOMAIN="Enter domain (e.g., fintech, technology): "

echo.
echo Ranking candidates...
echo.

REM Create JSON and call API using Python
python -c "import requests, json; jd = open(r'%TEMP_JD_FILE%', 'r', encoding='utf-8').read(); data = {'job_id': '%JOB_ID%', 'job_description': jd, 'required_skills': '%REQUIRED_SKILLS%', 'min_experience': %MIN_EXP%, 'domain': '%DOMAIN%'}; r = requests.post('http://localhost:5002/api/profileRankingByJD', json=data); print('Status:', r.status_code); result = r.json(); print(json.dumps(result, indent=2)); print('\n=== Top 5 Candidates ==='); [print(f\"{p['rank']}. {p['name']} - Score: {p['total_score']:.1f} - Match: {p['match_percent']:.1f}%%\") for p in result.get('ranked_profiles', [])[:5]]"

REM Clean up temp file
del "%TEMP_JD_FILE%" >nul 2>&1

echo.
echo ==========================================
echo Ranking Complete
echo ==========================================
echo.
pause

