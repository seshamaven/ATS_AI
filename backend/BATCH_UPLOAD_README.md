# Batch Resume Upload Scripts

This directory contains scripts to batch upload resume files from a directory structure to the ATS API.

## Files

- **`batch_upload_resumes.py`** - Python script (recommended)
- **`batch_upload_resumes.bat`** - Windows batch script wrapper
- **`batch_upload_resumes.ps1`** - PowerShell script (alternative)

## Features

- ✅ Recursively scans all subdirectories
- ✅ Finds files with extensions: `.pdf`, `.doc`, `.docx`, `.docu`
- ✅ Traverses year/month folder structures automatically
- ✅ Uploads each file via POST request
- ✅ Logs success/failure for each upload
- ✅ Detailed logging to console and log file
- ✅ Dry-run mode for testing
- ✅ Configurable API endpoint

## Quick Start

### Python Script (Recommended)

```bash
# Basic usage - scan uploads/ directory
python batch_upload_resumes.py uploads/

# With custom API URL
python batch_upload_resumes.py uploads/ --api-url http://localhost:5002/api/processResume

# Dry run (scan only, don't upload)
python batch_upload_resumes.py uploads/ --dry-run
```

### Batch Script (Windows)

```cmd
# Run the batch file and follow prompts
batch_upload_resumes.bat
```

### PowerShell Script

```powershell
# Basic usage
.\batch_upload_resumes.ps1 -Directory "uploads/"

# With custom API URL
.\batch_upload_resumes.ps1 -Directory "uploads/" -ApiUrl "http://localhost:5002/api/processResume"

# Dry run
.\batch_upload_resumes.ps1 -Directory "uploads/" -DryRun
```

## Directory Structure

The script works with any directory structure, including:

```
uploads/
├── 2025/
│   ├── 1/
│   │   ├── resume1.pdf
│   │   └── resume2.docx
│   ├── 2/
│   │   └── resume3.pdf
│   └── 3/
│       └── resume4.doc
├── 2024/
│   ├── 12/
│   │   └── resume5.pdf
│   └── 11/
│       └── resume6.docx
└── other_folder/
    └── resume7.pdf
```

The script will find and upload all resume files regardless of folder structure.

## API Endpoints

The script supports different API endpoints:

- **Default**: `http://127.0.0.1:8000/api/upload_resume`
- **Fallback**: `http://localhost:5002/api/processResume` (existing ATS endpoint)

You can specify any endpoint using the `--api-url` parameter.

## Output

### Console Output

The script displays:
- Progress for each file (`[1/10] Processing: ...`)
- Success/failure status for each upload
- Summary statistics at the end

### Log File

All operations are logged to `batch_upload.log` with timestamps:
```
2025-01-15 10:30:45 - INFO - Scanning directory: uploads/
2025-01-15 10:30:46 - INFO - Found 15 resume file(s)
2025-01-15 10:30:47 - INFO - Uploading: uploads/2025/1/resume1.pdf
2025-01-15 10:30:48 - INFO - ✓ Success: resume1.pdf (Candidate ID: 12345)
```

## Error Handling

- **Connection errors**: Logged and reported, script continues with next file
- **API errors**: Status code and error message logged
- **File errors**: File path and error logged
- **Summary**: Final report shows total, successful, and failed uploads

## Examples

### Example 1: Upload all resumes from uploads folder

```bash
python batch_upload_resumes.py uploads/
```

### Example 2: Test scan without uploading

```bash
python batch_upload_resumes.py uploads/ --dry-run
```

### Example 3: Upload to custom API endpoint

```bash
python batch_upload_resumes.py uploads/ --api-url http://localhost:5002/api/processResume
```

### Example 4: Upload from year/month structure

```bash
# Works automatically - no special configuration needed
python batch_upload_resumes.py uploads/2025/
```

## Requirements

### Python Script
- Python 3.6+
- `requests` library: `pip install requests`

### Batch Script
- Python (for the underlying script)
- Windows OS

### PowerShell Script
- PowerShell 5.1+ (Windows)
- No additional dependencies

## Troubleshooting

### API Not Running
```
ERROR: Connection error - API may not be running
```
**Solution**: Start the ATS API first using `2_start_api.bat`

### No Files Found
```
WARNING: No resume files found!
```
**Solution**: 
- Check the directory path
- Verify files have correct extensions (.pdf, .doc, .docx, .docu)
- Check file permissions

### Permission Errors
```
ERROR: Permission denied
```
**Solution**: Run with appropriate permissions or as administrator

## Notes

- The script processes files sequentially (one at a time)
- Large batches may take time - be patient
- Log file (`batch_upload.log`) is created in the script directory
- Failed uploads don't stop the batch - all files are attempted


