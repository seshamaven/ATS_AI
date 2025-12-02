# ATS Quick Start Guide

Get your ATS system up and running in 5 minutes.

## Prerequisites

- Python 3.8+
- MySQL 8.0+ running
- Azure OpenAI API key (or OpenAI API key)

## Step 1: Install Dependencies

```bash
pip install -r requirements_ats.txt
python -m spacy download en_core_web_sm
```

## Step 2: Setup Database

```bash
# Login to MySQL
mysql -u root -p

# Run schema
mysql -u root -p < ats_schema.sql
```

Or manually:
```sql
CREATE DATABASE ats_db;
USE ats_db;
-- Copy contents from ats_schema.sql
```

## Step 3: Configure Environment

Create `.env` file in the backend directory:

```env
# MySQL
ATS_MYSQL_HOST=localhost
ATS_MYSQL_USER=root
ATS_MYSQL_PASSWORD=your_password
ATS_MYSQL_DATABASE=ats_db
ATS_MYSQL_PORT=3306

# Azure OpenAI (or use OPENAI_API_KEY)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=text-embedding-ada-002

# API
ATS_API_PORT=5002
FLASK_DEBUG=True
```

## Step 4: Create Upload Directory

```bash
mkdir uploads
```

## Step 5: Start the API

**Windows:**
```bash
start_ats_api.bat
```

**Linux/Mac:**
```bash
chmod +x start_ats_api.sh
./start_ats_api.sh
```

**Or manually:**
```bash
python ats_api.py
```

## Step 6: Test the API

### Health Check
```bash
curl http://localhost:5002/health
```

### Upload Resume
```bash
curl -X POST http://localhost:5002/api/processResume \
  -F "file=@your_resume.pdf"
```

### Rank Candidates
```bash
curl -X POST http://localhost:5002/api/profileRankingByJD \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "JD001",
    "job_description": "Looking for Python developer with 5+ years experience...",
    "required_skills": "python, flask, sql",
    "min_experience": 5,
    "domain": "technology"
  }'
```

## Test Script

Run the automated test suite:

```bash
python test_ats_api.py
```

## Troubleshooting

### Issue: Database connection failed
- Ensure MySQL is running: `systemctl status mysql` (Linux) or check Services (Windows)
- Verify credentials in `.env`
- Test connection: `mysql -u root -p -h localhost`

### Issue: OpenAI API error
- Verify API key is correct in `.env`
- Check Azure endpoint URL format
- Ensure deployment name matches your Azure resource

### Issue: spaCy model not found
- Install model: `python -m spacy download en_core_web_sm`

### Issue: File upload fails
- Check `uploads` directory exists and is writable
- Verify file size is under 10MB (configurable in `.env`)
- Only PDF and DOCX files are supported

### Issue: Port already in use
- Change port in `.env`: `ATS_API_PORT=5003`
- Or kill process using port 5002

## Next Steps

1. **Upload multiple resumes** to build your candidate database
2. **Create job descriptions** and rank candidates
3. **Customize ranking weights** in `.env`
4. **Integrate with your frontend** application
5. **Deploy to production** using Docker

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/processResume` | POST | Upload resume |
| `/api/profileRankingByJD` | POST | Rank candidates |
| `/api/candidate/<id>` | GET | Get candidate details |
| `/api/job/<id>/rankings` | GET | Get job rankings |
| `/api/statistics` | GET | Get system statistics |

## Architecture Overview

```
Client → Flask API → Resume Parser → Azure OpenAI → MySQL
                  → Ranking Engine →
```

**Key Features:**
- Automatic skill extraction
- Semantic similarity matching
- Weighted scoring algorithm
- Production-ready error handling

---

For detailed documentation, see `README_ATS.md`

