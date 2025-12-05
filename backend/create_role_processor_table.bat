@echo off
REM Script to create Role_Processor table

echo ==========================================
echo Creating Role_Processor Table
echo ==========================================
echo.

set /p MYSQL_PASSWORD="Enter MySQL root password (default: Reset@123): "
if "%MYSQL_PASSWORD%"=="" set MYSQL_PASSWORD=Reset@123

echo.
echo Creating role_processor table...
cmd /c "mysql -u root -p%MYSQL_PASSWORD% ats_db < create_role_processor_table.sql"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to create table. Please check your password and database connection.
    pause
    exit /b 1
)

echo.
echo ✓ Table created successfully!
echo.
echo Verifying table...
mysql -u root -p%MYSQL_PASSWORD% -e "USE ats_db; DESCRIBE role_processor;"

echo.
echo ==========================================
echo ✓ Role_Processor Table Setup Complete!
echo ==========================================
echo.
echo Next step: Run insert_role_mappings.bat to insert role mappings
echo.
pause

