# PowerShell script to upload profiles to Railway
# Run this script after redeploying to Railway

$BASE_URL = "https://atsai-production.up.railway.app"
$PROFILES_DIR = "D:\profiles"

Write-Host "=" * 80
Write-Host "UPLOADING PROFILES TO RAILWAY"
Write-Host "=" * 80
Write-Host "Base URL: $BASE_URL"
Write-Host "Profiles Directory: $PROFILES_DIR"
Write-Host ""

# Check if profiles directory exists
if (-not (Test-Path $PROFILES_DIR)) {
    Write-Host "❌ Profiles directory $PROFILES_DIR does not exist!" -ForegroundColor Red
    exit 1
}

# Get all PDF files
$pdfFiles = Get-ChildItem -Path $PROFILES_DIR -Filter "*.pdf"

if ($pdfFiles.Count -eq 0) {
    Write-Host "❌ No PDF files found in $PROFILES_DIR" -ForegroundColor Red
    exit 1
}

Write-Host "📁 Found $($pdfFiles.Count) PDF files to upload:"
foreach ($file in $pdfFiles) {
    Write-Host "  - $($file.Name)"
}
Write-Host ""

# Upload each file
$uploadedCount = 0
foreach ($file in $pdfFiles) {
    Write-Host "📤 Uploading $($file.Name)..." -ForegroundColor Yellow
    
    try {
        $response = Invoke-RestMethod -Uri "$BASE_URL/api/upload-profile" -Method Post -Form @{
            file = Get-Item $file.FullName
        } -TimeoutSec 60
        
        if ($response.status -eq "success") {
            Write-Host "✅ Successfully uploaded $($file.Name)" -ForegroundColor Green
            Write-Host "   File path: $($response.file_path)"
            $uploadedCount++
        } else {
            Write-Host "❌ Failed to upload $($file.Name): $($response.message)" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "❌ Error uploading $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ""
}

Write-Host "📊 Upload Summary: $uploadedCount/$($pdfFiles.Count) files uploaded successfully" -ForegroundColor Cyan

# List uploaded profiles
Write-Host ""
Write-Host "=" * 80
Write-Host "LISTING UPLOADED PROFILES"
Write-Host "=" * 80

try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/list-uploaded-profiles" -Method Get -TimeoutSec 30
    
    if ($response.status -eq "success") {
        Write-Host "✅ Success!" -ForegroundColor Green
        Write-Host "Profiles Directory: $($response.profiles_directory)"
        Write-Host "Total Profiles: $($response.total_profiles)"
        Write-Host ""
        
        if ($response.profiles.Count -gt 0) {
            Write-Host "📁 UPLOADED PROFILES:" -ForegroundColor Cyan
            foreach ($profile in $response.profiles) {
                Write-Host "  - $($profile.filename) (ID: $($profile.candidate_id))"
                Write-Host "    Size: $($profile.file_size) bytes"
            }
        } else {
            Write-Host "No profiles found" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Error: $($response.error)" -ForegroundColor Red
    }
}
catch {
    Write-Host "❌ Error listing profiles: $($_.Exception.Message)" -ForegroundColor Red
}

# Test comprehensive ranking
Write-Host ""
Write-Host "=" * 80
Write-Host "TESTING COMPREHENSIVE PROFILE RANKING"
Write-Host "=" * 80

$jobRequirements = @{
    job_requirements = @{
        required_skills = @("python")
        preferred_skills = @()
        min_experience = 4
        max_experience = 7
        domain = "Computer Science Engineering"
        education_required = "Computer Science"
    }
    top_k = 10
} | ConvertTo-Json -Depth 3

try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/comprehensive-profile-ranking" -Method Post -Body $jobRequirements -ContentType "application/json" -TimeoutSec 60
    
    if ($response.status -eq "success") {
        Write-Host "✅ SUCCESS!" -ForegroundColor Green
        Write-Host "Status: $($response.status)"
        Write-Host "Message: $($response.message)"
        Write-Host "Total Candidates Evaluated: $($response.total_candidates_evaluated)"
        Write-Host "Processing Time: $($response.processing_time_ms) ms"
        Write-Host ""
        
        if ($response.ranked_profiles.Count -gt 0) {
            Write-Host "🏆 TOP CANDIDATES:" -ForegroundColor Cyan
            foreach ($profile in $response.ranked_profiles) {
                Write-Host "Rank $($profile.rank): $($profile.name) (ID: $($profile.candidate_id))"
                Write-Host "   Total Score: $($profile.total_score)%"
                Write-Host "   Skills Score: $($profile.skills_score)%"
                Write-Host "   Experience Score: $($profile.experience_score)%"
                Write-Host "   Matched Skills: $($profile.matched_skills -join ', ')"
                Write-Host ""
            }
        } else {
            Write-Host "No candidates found" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Error: $($response.error)" -ForegroundColor Red
    }
}
catch {
    Write-Host "❌ Error testing ranking: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "=" * 80
Write-Host "UPLOAD AND TEST COMPLETED"
Write-Host "=" * 80

