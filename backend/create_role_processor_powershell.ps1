# PowerShell script to create Role_Processor table
# Usage: .\create_role_processor_table_powershell.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Creating Role_Processor Table" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$mysqlPassword = Read-Host "Enter MySQL root password (default: Reset@123)"
if ([string]::IsNullOrWhiteSpace($mysqlPassword)) {
    $mysqlPassword = "Reset@123"
}

Write-Host ""
Write-Host "Creating role_processor table..." -ForegroundColor Yellow

# Read SQL file content
$sqlContent = Get-Content -Path "create_role_processor_table.sql" -Raw

# Execute SQL using mysql command
$sqlContent | & mysql -u root -p$mysqlPassword ats_db

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to create table. Please check your password and database connection." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "✓ Table created successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Verifying table..." -ForegroundColor Yellow
& mysql -u root -p$mysqlPassword -e "USE ats_db; DESCRIBE role_processor;"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✓ Role_Processor Table Setup Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next step: Run insert_role_mappings.bat to insert role mappings" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit"

