@echo off
REM Batch script to create all Pinecone indexes
REM Usage: create_all_indexes.bat [API_KEY]

echo ============================================================
echo Creating Multiple Pinecone Indexes
echo ============================================================
echo.

REM Check if API key provided as argument
if "%1"=="" (
    echo No API key provided as argument.
    echo Checking environment variable...
    if "%PINECONE_API_KEY%"=="" (
        echo ERROR: PINECONE_API_KEY not set!
        echo.
        echo Usage:
        echo   1. Set environment variable: set PINECONE_API_KEY=your-key
        echo   2. Or pass as argument: create_all_indexes.bat your-key
        echo   3. Or run: python create_multiple_indexes.py --api-key your-key
        exit /b 1
    )
    echo Using PINECONE_API_KEY from environment...
    python create_multiple_indexes.py
) else (
    echo Using API key from argument...
    python create_multiple_indexes.py --api-key %1
)

echo.
echo ============================================================
echo Done!
echo ============================================================
pause

