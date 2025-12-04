# Batch Upload Resumes Script (PowerShell)
# Recursively scans directories and uploads all resume files to the ATS API

param(
    [Parameter(Mandatory=$false)]
    [string]$Directory = "",
    
    [Parameter(Mandatory=$false)]
    [string]$ApiUrl = "http://127.0.0.1:8000/api/upload_resume",
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun
)

# Resume file extensions
$ResumeExtensions = @('.pdf', '.doc', '.docx', '.docu')

# Log file
$LogFile = "batch_upload.log"
$ScriptStartTime = Get-Date

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "$Timestamp - $Level - $Message"
    Add-Content -Path $LogFile -Value $LogMessage
    Write-Host $LogMessage
}

function Find-ResumeFiles {
    param([string]$RootDir)
    
    Write-Log "Scanning directory: $RootDir"
    
    if (-not (Test-Path $RootDir)) {
        Write-Log "Directory does not exist: $RootDir" "ERROR"
        return @()
    }
    
    $resumeFiles = @()
    
    # Recursively get all files
    Get-ChildItem -Path $RootDir -Recurse -File | ForEach-Object {
        if ($ResumeExtensions -contains $_.Extension.ToLower()) {
            $resumeFiles += $_.FullName
            Write-Log "Found resume: $($_.FullName)" "DEBUG"
        }
    }
    
    Write-Log "Found $($resumeFiles.Count) resume file(s)"
    return $resumeFiles
}

function Test-ApiHealth {
    param([string]$Url)
    
    try {
        $baseUrl = $Url -replace '/api/.*$', ''
        $healthUrl = "$baseUrl/health"
        
        $response = Invoke-WebRequest -Uri $healthUrl -Method Get -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Log "API is running and accessible"
            return $true
        }
    } catch {
        # Try the actual endpoint
        try {
            $response = Invoke-WebRequest -Uri $Url -Method Head -TimeoutSec 5 -ErrorAction SilentlyContinue
            return $true
        } catch {
            Write-Log "Could not verify API health, proceeding anyway..." "WARNING"
            return $true
        }
    }
    
    return $false
}

function Upload-Resume {
    param(
        [string]$FilePath,
        [string]$ApiUrl
    )
    
    try {
        Write-Log "Uploading: $FilePath"
        
        $fileName = Split-Path -Leaf $FilePath
        
        # Create multipart form data
        $fileBytes = [System.IO.File]::ReadAllBytes($FilePath)
        $boundary = [System.Guid]::NewGuid().ToString()
        $LF = "`r`n"
        
        $bodyLines = @(
            "--$boundary",
            "Content-Disposition: form-data; name=`"file`"; filename=`"$fileName`"",
            "Content-Type: application/octet-stream$LF",
            [System.Text.Encoding]::GetEncoding('iso-8859-1').GetString($fileBytes),
            "$LF--$boundary--$LF"
        ) -join $LF
        
        $bodyBytes = [System.Text.Encoding]::GetEncoding('iso-8859-1').GetBytes($bodyLines)
        
        # Make request
        $headers = @{
            "Content-Type" = "multipart/form-data; boundary=$boundary"
        }
        
        $response = Invoke-RestMethod -Uri $ApiUrl -Method Post -Body $bodyBytes -Headers $headers -TimeoutSec 60
        
        if ($response) {
            $candidateId = $response.candidate_id
            if ($candidateId) {
                Write-Log "✓ Success: $fileName (Candidate ID: $candidateId)"
                return $true, "Success - Candidate ID: $candidateId"
            } else {
                Write-Log "✓ Success: $fileName"
                return $true, "Success"
            }
        } else {
            Write-Log "✗ Failed: $fileName - Unknown error" "ERROR"
            return $false, "Unknown error"
        }
        
    } catch {
        $errorMsg = $_.Exception.Message
        Write-Log "✗ Failed: $fileName - $errorMsg" "ERROR"
        return $false, $errorMsg
    }
}

# Main execution
Write-Host "=========================================="
Write-Host "Batch Resume Upload Script"
Write-Host "=========================================="
Write-Host ""

# Initialize log file
"Batch upload started at $ScriptStartTime" | Out-File -FilePath $LogFile

# Get directory if not provided
if ([string]::IsNullOrEmpty($Directory)) {
    $Directory = Read-Host "Enter directory path to scan (e.g., uploads/)"
}

if (-not (Test-Path $Directory)) {
    Write-Log "Directory does not exist: $Directory" "ERROR"
    exit 1
}

# Get API URL if not provided
if ([string]::IsNullOrEmpty($ApiUrl)) {
    $ApiUrl = Read-Host "Enter API URL (press Enter for default http://127.0.0.1:8000/api/upload_resume)"
    if ([string]::IsNullOrEmpty($ApiUrl)) {
        $ApiUrl = "http://127.0.0.1:8000/api/upload_resume"
    }
}

# Check API health
Write-Log "Checking API availability at: $ApiUrl"
if (-not $DryRun) {
    Test-ApiHealth -Url $ApiUrl | Out-Null
}

# Find all resume files
$resumeFiles = Find-ResumeFiles -RootDir $Directory

if ($resumeFiles.Count -eq 0) {
    Write-Log "No resume files found!" "WARNING"
    exit 0
}

# Statistics
$stats = @{
    Total = $resumeFiles.Count
    Success = 0
    Failed = 0
    Results = @()
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Starting batch upload: $($stats.Total) file(s)"
Write-Host "=========================================="
Write-Host ""

# Upload each file
$idx = 0
foreach ($filePath in $resumeFiles) {
    $idx++
    Write-Host "[$idx/$($stats.Total)] Processing: $filePath"
    
    if ($DryRun) {
        Write-Log "[DRY RUN] Would upload: $filePath"
        $stats.Results += @{
            File = $filePath
            Success = $null
            Message = "Dry run - not uploaded"
        }
    } else {
        $success, $message = Upload-Resume -FilePath $filePath -ApiUrl $ApiUrl
        
        $stats.Results += @{
            File = $filePath
            Success = $success
            Message = $message
        }
        
        if ($success) {
            $stats.Success++
        } else {
            $stats.Failed++
        }
    }
    
    Write-Host ""
}

# Print summary
Write-Host ""
Write-Host "=========================================="
Write-Host "UPLOAD SUMMARY"
Write-Host "=========================================="
Write-Host "Total files: $($stats.Total)"
if (-not $DryRun) {
    Write-Host "Successful: $($stats.Success)"
    Write-Host "Failed: $($stats.Failed)"
    $successRate = if ($stats.Total -gt 0) { ($stats.Success / $stats.Total * 100) } else { 0 }
    Write-Host "Success rate: $([math]::Round($successRate, 1))%"
}
Write-Host "=========================================="
Write-Host ""

Write-Log "Batch upload completed. Check $LogFile for details."

if ($stats.Failed -gt 0) {
    exit 1
} else {
    exit 0
}


