# Test Instructions for Resume Upload Endpoints

## Prerequisites

1. **Install pytest** (if not already installed):
   ```bash
   pip install pytest pytest-cov
   ```
   
   Or install from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

### Option 1: Run all tests using pytest directly
```bash
cd backend
pytest test_resume_upload.py -v
```

### Option 2: Run with detailed output
```bash
cd backend
pytest test_resume_upload.py -v --tb=short
```

### Option 3: Run specific test class
```bash
cd backend
pytest test_resume_upload.py::TestProcessResume -v
```

### Option 4: Run specific test method
```bash
cd backend
pytest test_resume_upload.py::TestProcessResume::test_upload_pdf_success -v
```

### Option 5: Run with coverage report
```bash
cd backend
pytest test_resume_upload.py --cov=ats_api --cov-report=html
```

### Option 6: Use the test runner script
```bash
cd backend
python run_tests.py
```

## Test Coverage

The test suite covers:

### `/api/processResume` Endpoint:
- ✅ Successful PDF upload
- ✅ Successful DOCX upload
- ✅ Upload with Pinecone indexing
- ✅ Error: No file provided
- ✅ Error: Empty filename
- ✅ Error: Invalid file type
- ✅ Error: Database insert failure
- ✅ Error: Embedding generation failure

### `/api/processResumeBase64` Endpoint:
- ✅ Successful base64 PDF upload
- ✅ Successful base64 DOCX upload
- ✅ Base64 upload with Pinecone indexing
- ✅ Error: Non-JSON request
- ✅ Error: Missing filename
- ✅ Error: Missing fileBase64
- ✅ Error: Invalid base64 data
- ✅ Error: Invalid file type
- ✅ Error: Database insert failure

### Integration Tests:
- ✅ Full upload flow with all components
- ✅ Response format completeness
- ✅ Profile scores calculation

## Expected Output

When tests pass, you should see output like:
```
test_resume_upload.py::TestProcessResume::test_upload_pdf_success PASSED
test_resume_upload.py::TestProcessResume::test_upload_docx_success PASSED
...
========================= X passed in Y.YYs =========================
```

## Troubleshooting

### Issue: ModuleNotFoundError
If you get import errors, make sure you're in the `backend` directory:
```bash
cd backend
pytest test_resume_upload.py -v
```

### Issue: pytest not found
Install pytest:
```bash
pip install pytest
```

### Issue: Tests fail due to missing mocks
All external dependencies (database, embedding service, Pinecone) are mocked in `conftest.py`. 
The tests should run without actual database or API connections.

## Test Files

- `conftest.py` - Pytest fixtures and configuration
- `test_resume_upload.py` - Test cases for resume upload endpoints

