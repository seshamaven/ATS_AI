@echo off
echo ========================================
echo Creating Main Pinecone Index
echo ========================================
echo.
echo With namespace implementation, only ONE index is needed.
echo Namespaces will be created automatically when resumes are indexed.
echo.

REM Check if API key is provided as argument
if "%1"=="" (
    echo Running with API key from environment variable...
    python create_main_index.py
) else (
    echo Running with provided API key...
    python create_main_index.py --api-key %1
)

pause

