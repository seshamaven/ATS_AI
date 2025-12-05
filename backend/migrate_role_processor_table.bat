@echo off
REM Script to migrate Role_Processor table to new structure

echo ==========================================
echo Migrating Role_Processor Table
echo ==========================================
echo.
echo This will:
echo   1. Backup existing data
echo   2. Restructure table to store unique normalized roles
echo   3. Aggregate all original roles into JSON array per normalized role
echo.
echo WARNING: This will modify the table structure!
echo.

set /p MYSQL_PASSWORD="Enter MySQL root password (default: Reset@123): "
if "%MYSQL_PASSWORD%"=="" set MYSQL_PASSWORD=Reset@123

echo.
echo Starting migration...
cmd /c "mysql -u root -p%MYSQL_PASSWORD% ats_db < migrate_role_processor_table.sql"

if errorlevel 1 (
    echo.
    echo ERROR: Migration failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ✓ Migration completed successfully!
echo.
echo Verifying new table structure...
mysql -u root -p%MYSQL_PASSWORD% -e "USE ats_db; DESCRIBE role_processor;"

echo.
echo ==========================================
echo ✓ Migration Complete!
echo ==========================================
echo.
pause

