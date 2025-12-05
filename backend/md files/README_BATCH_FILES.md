# ATS Batch Files Guide

## Quick Start - Run in Order

Execute these batch files in sequence to set up and use the ATS system:

### 1️⃣ `1_setup_database.bat`
**Purpose:** Create database and tables

**What it does:**
- Creates `ats_db` database
- Runs `ats_schema.sql`
- Creates 5 tables (resume_metadata, job_descriptions, etc.)
- Verifies setup

**Requirements:**
- MySQL installed and running
- MySQL root credentials (default: root/root)

**Usage:**
```cmd
1_setup_database.bat
```

---

### 2️⃣ `2_start_api.bat`
**Purpose:** Start the ATS API server

**What it does:**
- Creates/activates virtual environment
- Installs dependencies (Flask, spaCy, etc.)
- Downloads NLP models
- Creates uploads directory
- Starts API on http://localhost:5002

**Requirements:**
- Python 3.8+ installed
- `.env` file configured

**Usage:**
```cmd
2_start_api.bat
```

**API will run at:** http://localhost:5002

**Keep this window open** - the API runs in foreground

---

### 3️⃣ `3_test_api.bat`
**Purpose:** Test all API endpoints

**What it does:**
- Checks if API is running
- Runs comprehensive test suite
- Tests health, statistics, resume upload, ranking

**Requirements:**
- API must be running (step 2)

**Usage:**
```cmd
3_test_api.bat
```

**Note:** Open a NEW command window while API is running

---

### 4️⃣ `4_upload_resume.bat`
**Purpose:** Upload a single resume

**What it does:**
- Prompts for resume file path
- Uploads to API
- Displays extracted metadata
- Returns candidate ID

**Requirements:**
- API must be running
- PDF or DOCX resume file

**Usage:**
```cmd
4_upload_resume.bat
```

**Example:**
```
Enter path to resume file: C:\resumes\john_doe.pdf
```

---

### 5️⃣ `5_rank_candidates.bat`
**Purpose:** Rank candidates against a job description

**What it does:**
- Prompts for job details
- Collects JD text
- Calls ranking API
- Displays top candidates with scores

**Requirements:**
- API must be running
- At least one resume uploaded

**Usage:**
```cmd
5_rank_candidates.bat
```

**Example:**
```
Enter Job ID: JD001
Enter Job Description: [paste JD text, press Ctrl+Z, Enter]
Enter required skills: python, flask, sql
Enter minimum experience: 5
Enter domain: fintech
```

---

## Other Utility Files

### `setup_ats_database.bat`
Alternative database setup script (same as #1)

### `start_ats_api.bat`
Alternative API start script (same as #2)

---

## Typical Workflow

### First Time Setup
```cmd
REM 1. Setup database (one time only)
1_setup_database.bat

REM 2. Start API (in one window)
2_start_api.bat

REM 3. Test API (in another window)
3_test_api.bat
```

### Daily Usage
```cmd
REM 1. Start API
2_start_api.bat

REM 2. Upload resumes (in new window)
4_upload_resume.bat

REM 3. Rank candidates (in new window)
5_rank_candidates.bat
```

---

## Troubleshooting

### Issue: "Python not found"
**Solution:** Install Python 3.8+ and add to PATH

### Issue: "MySQL command not found"
**Solution:** Add MySQL bin directory to PATH
```
C:\Program Files\MySQL\MySQL Server 8.X\bin
```

### Issue: "API not running" in test/upload/rank scripts
**Solution:** Start API using `2_start_api.bat` in a separate window

### Issue: ".env file not found"
**Solution:** 
1. Copy `env_ats_template.txt`
2. Create `.env` file
3. Add your configuration

### Issue: "Dependencies not installed"
**Solution:** Scripts auto-install dependencies. If fails:
```cmd
pip install -r requirements_ats.txt
python -m spacy download en_core_web_sm
```

---

## Environment Variables (.env)

Ensure your `.env` file contains:

```env
# MySQL
ATS_MYSQL_HOST=localhost
ATS_MYSQL_USER=root
ATS_MYSQL_PASSWORD=Reset@123
ATS_MYSQL_DATABASE=ats_db
ATS_MYSQL_PORT=3306

# Azure OpenAI or OpenAI
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Or
OPENAI_API_KEY=your_key

# API Settings
ATS_API_PORT=5002
FLASK_DEBUG=True
```

---

## Windows Task Scheduler (Optional)

To auto-start API on Windows startup:

1. Open Task Scheduler
2. Create Basic Task
3. Trigger: At startup
4. Action: Start a program
5. Program: `C:\ATS\backend\2_start_api.bat`
6. Start in: `C:\ATS\backend`

---

## Command Line Shortcuts

### Quick Test
```cmd
curl http://localhost:5002/health
```

### Quick Upload
```cmd
curl -X POST http://localhost:5002/api/processResume -F "file=@resume.pdf"
```

### View Statistics
```cmd
curl http://localhost:5002/api/statistics
```

---

## File Execution Order

```
Setup (Once):
  1_setup_database.bat

Daily Operations:
  2_start_api.bat (keep running)
    ↓
  3_test_api.bat (verify)
    ↓
  4_upload_resume.bat (upload resumes)
    ↓
  5_rank_candidates.bat (rank against JD)
```

---

## Support

For issues or questions, refer to:
- `README_ATS.md` - Complete documentation
- `QUICKSTART_ATS.md` - Quick start guide
- `API_EXAMPLES_ATS.md` - API usage examples

---

**All batch files are located in:** `C:\ATS\backend\`

