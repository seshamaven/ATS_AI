@echo off
REM Script to create ATS database and run schema

echo ==========================================
echo ATS Database Setup
echo ==========================================
echo.

set /p MYSQL_PASSWORD="Enter MySQL root password: "

echo.
echo Creating ats_db database...
mysql -u root -p%MYSQL_PASSWORD% -e "CREATE DATABASE IF NOT EXISTS ats_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to create database. Please check your password.
    pause
    exit /b 1
)

echo ✓ Database created successfully!
echo.
echo Running schema file...
mysql -u root -p%MYSQL_PASSWORD% ats_db < ats_schema.sql

if errorlevel 1 (
    echo.
    echo ERROR: Failed to run schema file.
    pause
    exit /b 1
)

echo ✓ Schema executed successfully!
echo.
echo Verifying tables...
mysql -u root -p%MYSQL_PASSWORD% -e "USE ats_db; SHOW TABLES;"

echo.
echo ==========================================
echo ✓ ATS Database Setup Complete!
echo ==========================================
echo.
echo Database: ats_db
echo Tables created:
echo   - resume_metadata
echo   - job_descriptions
echo   - ranking_history
echo   - skills_master
echo   - applications
echo.
pause

