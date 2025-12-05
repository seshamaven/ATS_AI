@echo off
REM Script to insert role mappings into Role_Processor table

echo ==========================================
echo Inserting Role Mappings
echo ==========================================
echo.

cd /d "%~dp0"

python insert_role_mappings.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to insert role mappings.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Role mappings inserted successfully!
echo ==========================================
echo.
pause

