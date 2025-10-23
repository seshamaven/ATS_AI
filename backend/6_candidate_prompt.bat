@echo off
REM 6. Candidate Search with Prompt

echo ==========================================
echo Step 6: Candidate Search with Prompt
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

echo Search Examples:
echo - Skills: "python, java, sql"
echo - Name: "john smith"
echo - Domain: "fintech"
echo - Experience: "senior developer"
echo.

set /p CANDIDATE_QUERY="Enter candidate name or search query: "
if "%CANDIDATE_QUERY%"=="" (
    echo ERROR: Search query cannot be empty!
    pause
    exit /b 1
)

echo.
echo Searching for: %CANDIDATE_QUERY%
echo.

REM Create JSON payload and call API using Python with better formatting
python -c "import requests, json; data = {'query': '%CANDIDATE_QUERY%', 'top_n': 10, 'min_score': 15}; r = requests.post('http://localhost:5002/api/candidateSearch', json=data); print('Status:', r.status_code); result = r.json(); print('\n=== SEARCH RESULTS ==='); print(f'Query: {result.get(\"query\", \"\")}'); print(f'Total candidates found: {len(result.get(\"top_candidates\", []))}'); print('\n=== CANDIDATE DETAILS ==='); [print(f'\n{i+1}. {c[\"name\"]} - Match: {c[\"match_score\"]}%%'); print(f'   Email: {c[\"email\"]}'); print(f'   Phone: {c[\"phone\"]}'); print(f'   Skills: {c[\"skills\"]}'); print(f'   Experience: {c[\"experience\"]} years'); print(f'   Company: {c[\"current_company\"]}'); print(f'   Domain: {c[\"domain\"]}'); print(f'   Location: {c[\"location\"]}') for i, c in enumerate(result.get('top_candidates', []))]"

echo.
echo ==========================================
echo Search Complete
echo ==========================================
echo.
pause
