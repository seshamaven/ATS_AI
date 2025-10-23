# PowerShell script to test the Embedding API
# This script calls the embedding API to process regulations and create embeddings

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Regulatory RAG - Embedding API Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$apiUrl = "http://localhost:5000/embed"
$timeout = 300

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "- API Endpoint: $apiUrl" -ForegroundColor White
Write-Host "- Timeout: $timeout seconds" -ForegroundColor White
Write-Host "- Test Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White
Write-Host ""

try {
    Write-Host "Making API call to embedding endpoint..." -ForegroundColor Green
    Write-Host ""
    
    # Start timing
    $startTime = Get-Date
    
    # Make the API call
    $response = Invoke-RestMethod -Uri $apiUrl -Method POST -ContentType "application/json" -Body "{}" -TimeoutSec $timeout
    
    # Calculate processing time
    $processingTime = (Get-Date) - $startTime
    
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "API Response" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Processing Time: $($processingTime.TotalSeconds.ToString('F2')) seconds" -ForegroundColor White
    Write-Host ""
    
    Write-Host "SUCCESS: Embedding API call completed successfully!" -ForegroundColor Green
    Write-Host ""
    
    # Display response data
    Write-Host "Response Data:" -ForegroundColor Yellow
    $response | ConvertTo-Json -Depth 10 | Write-Host -ForegroundColor White
    
    # Extract key information
    if ($response -is [PSCustomObject]) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "Summary" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        
        if ($response.message) {
            Write-Host "Message: $($response.message)" -ForegroundColor White
        }
        
        if ($response.processed_regulations) {
            Write-Host "Processed Regulations: $($response.processed_regulations)" -ForegroundColor White
        }
        
        if ($response.total_vectors_created) {
            Write-Host "Total Vectors Created: $($response.total_vectors_created)" -ForegroundColor White
        }
        
        if ($response.token_usage) {
            Write-Host "Token Usage:" -ForegroundColor White
            Write-Host "  - Total Tokens: $($response.token_usage.total_tokens)" -ForegroundColor White
            Write-Host "  - Total Cost: `$$($response.token_usage.total_cost_usd)" -ForegroundColor White
            Write-Host "  - Model Used: $($response.token_usage.model_used)" -ForegroundColor White
            Write-Host "  - Avg Tokens per Chunk: $($response.token_usage.avg_tokens_per_chunk)" -ForegroundColor White
        }
    }
    
} catch {
    Write-Host "ERROR: API call failed" -ForegroundColor Red
    Write-Host "Error Details: $($_.Exception.Message)" -ForegroundColor Red
    
    if ($_.Exception.Message -like "*connection*") {
        Write-Host ""
        Write-Host "The embedding API may not be running." -ForegroundColor Yellow
        Write-Host "To start the embedding API:" -ForegroundColor Yellow
        Write-Host "1. Open a terminal/command prompt" -ForegroundColor White
        Write-Host "2. Navigate to the regaiagent directory" -ForegroundColor White
        Write-Host "3. Run: python embed_api.py" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test completed" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Read-Host "Press Enter to exit"
