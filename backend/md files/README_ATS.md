# ATS (Application Tracking System) - Profile Ranking API

A production-ready **Application Tracking System** that ranks candidate profiles against Job Descriptions using **Azure OpenAI embeddings** and **MySQL** for data persistence.

## 🎯 Features

- **Resume Processing API** - Upload and parse PDF/DOCX resumes with automatic metadata extraction
- **Profile Ranking API** - Semantic matching of candidates to job descriptions
- **Weighted Scoring Algorithm** - Skills (40%), Experience (30%), Domain (20%), Education (10%)
- **Intelligent Resume Parser** - Extracts skills, experience, education, contact info using NLP
- **Vector Embeddings** - Azure OpenAI text-embedding-ada-002 for semantic similarity
- **MySQL Storage** - Structured metadata with JSON-embedded vectors
- **Production-Ready** - Comprehensive error handling, logging, and monitoring
- **Docker-Ready** - Easy containerization for deployment

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Setup](#setup)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Database Schema](#database-schema)
- [Deployment](#deployment)

---

## 🏗 Architecture

```
┌─────────────────┐
│   Frontend/     │
│   Client App    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│       ATS Flask API                 │
│  ┌──────────────────────────────┐  │
│  │  /api/processResume          │  │
│  │  /api/profileRankingByJD     │  │
│  └──────────────────────────────┘  │
└──────┬─────────────────────┬───────┘
       │                     │
       ▼                     ▼
┌──────────────┐    ┌─────────────────┐
│ Resume Parser│    │ Ranking Engine  │
│  - PDF/DOCX  │    │  - Weighted     │
│  - NLP       │    │  - Semantic     │
└──────┬───────┘    └────────┬────────┘
       │                     │
       ▼                     ▼
┌──────────────────────────────────────┐
│      Azure OpenAI Embeddings         │
│      text-embedding-ada-002          │
└──────────────────┬───────────────────┘
                   │
                   ▼
           ┌──────────────┐
           │   MySQL DB   │
           │  - Metadata  │
           │  - Embeddings│
           └──────────────┘
```

### Key Components

1. **ats_api.py** - Main Flask API with endpoints
2. **resume_parser.py** - Intelligent resume parsing with NLP
3. **ranking_engine.py** - Weighted scoring algorithm
4. **ats_database.py** - MySQL operations manager
5. **ats_config.py** - Configuration management
6. **ats_schema.sql** - Database schema

---

## 🚀 Setup

### Prerequisites

- Python 3.8+
- MySQL 8.0+
- Azure OpenAI account (or OpenAI API key)

### Installation

1. **Clone the repository**
```bash
cd backend
```

2. **Install dependencies**
```bash
pip install -r requirements_ats.txt
```

3. **Download spaCy NLP model**
```bash
python -m spacy download en_core_web_sm
```

4. **Setup MySQL database**
```bash
mysql -u root -p < ats_schema.sql
```

5. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your credentials
```

6. **Create upload directory**
```bash
mkdir uploads
```

### Environment Variables

Key variables to configure in `.env`:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# MySQL
ATS_MYSQL_HOST=localhost
ATS_MYSQL_USER=root
ATS_MYSQL_PASSWORD=your_password
ATS_MYSQL_DATABASE=ats_db

# API Port
ATS_API_PORT=5002
```

---

## 📡 API Endpoints

### 1. Process Resume

**Endpoint:** `POST /api/processResume`

**Description:** Upload and parse resume, extract metadata, generate embeddings, store in MySQL.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** 
  - `file`: PDF or DOCX resume file

**Response:**
```json
{
  "status": "success",
  "message": "Resume processed successfully",
  "candidate_id": 12345,
  "candidate_name": "John Doe",
  "email": "john.doe@example.com",
  "total_experience": 5.0,
  "primary_skills": "python, flask, sql, react",
  "domain": "finance",
  "education": "Bachelors",
  "embedding_dimensions": 1536,
  "timestamp": "2025-10-15T10:30:00"
}
```

**Example:**
```bash
curl -X POST http://localhost:5002/api/processResume \
  -F "file=@resume.pdf"
```

---

### 2. Profile Ranking by Job Description

**Endpoint:** `POST /api/profileRankingByJD`

**Description:** Rank all candidate profiles against a job description using weighted scoring.

**Request:**
- **Content-Type:** `application/json`
- **Body:**
```json
{
  "job_id": "JD001",
  "job_description": "We are hiring a Python backend engineer with 3+ years of experience...",
  "required_skills": "python, flask, sql, rest api",
  "preferred_skills": "docker, kubernetes, aws",
  "min_experience": 3,
  "max_experience": 7,
  "domain": "finance",
  "education_required": "Bachelors"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Profile ranking completed successfully",
  "job_id": "JD001",
  "ranked_profiles": [
    {
      "rank": 1,
      "candidate_id": 12345,
      "name": "John Doe",
      "email": "john.doe@example.com",
      "total_score": 92.5,
      "match_percent": 92.5,
      "skills_score": 95.0,
      "experience_score": 100.0,
      "domain_score": 85.0,
      "education_score": 100.0,
      "matched_skills": ["python", "flask", "sql", "docker"],
      "missing_skills": ["kubernetes"],
      "experience_match": "High",
      "domain_match": "High"
    }
  ],
  "total_candidates_evaluated": 150,
  "top_candidates_returned": 50,
  "ranking_criteria": {
    "weights": {
      "skills": 0.4,
      "experience": 0.3,
      "domain": 0.2,
      "education": 0.1
    },
    "semantic_similarity": "enabled"
  },
  "timestamp": "2025-10-15T10:35:00"
}
```

**Example:**
```bash
curl -X POST http://localhost:5002/api/profileRankingByJD \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "JD001",
    "job_description": "Looking for a Python backend engineer..."
  }'
```

---

### 3. Get Candidate Details

**Endpoint:** `GET /api/candidate/<candidate_id>`

**Example:**
```bash
curl http://localhost:5002/api/candidate/12345
```

---

### 4. Get Job Rankings

**Endpoint:** `GET /api/job/<job_id>/rankings?limit=50`

**Example:**
```bash
curl http://localhost:5002/api/job/JD001/rankings?limit=20
```

---

### 5. Health Check

**Endpoint:** `GET /health`

**Example:**
```bash
curl http://localhost:5002/health
```

---

## ⚙️ Configuration

### Ranking Weights

Customize the scoring weights in `.env`:

```env
RANKING_WEIGHT_SKILLS=0.4      # 40% weight on skills match
RANKING_WEIGHT_EXPERIENCE=0.3  # 30% weight on experience
RANKING_WEIGHT_DOMAIN=0.2      # 20% weight on domain/industry
RANKING_WEIGHT_EDUCATION=0.1   # 10% weight on education
```

**Note:** Weights must sum to 1.0

### Score Matching Thresholds

```env
EXP_MATCH_HIGH=0.9        # 90%+ match = "High"
EXP_MATCH_MEDIUM=0.7      # 70%+ match = "Medium"
DOMAIN_MATCH_HIGH=0.85    # 85%+ match = "High"
DOMAIN_MATCH_MEDIUM=0.65  # 65%+ match = "Medium"
```

---

## 💼 Usage Examples

### Python Client

```python
import requests

# 1. Upload Resume
with open('resume.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5002/api/processResume', files=files)
    result = response.json()
    candidate_id = result['candidate_id']
    print(f"Resume processed: {candidate_id}")

# 2. Rank Profiles
jd_data = {
    "job_id": "JD001",
    "job_description": "We need a senior Python developer with 5+ years...",
    "required_skills": "python, flask, sql, docker",
    "min_experience": 5,
    "domain": "fintech"
}

response = requests.post('http://localhost:5002/api/profileRankingByJD', json=jd_data)
rankings = response.json()

for profile in rankings['ranked_profiles'][:5]:
    print(f"{profile['rank']}. {profile['name']} - Score: {profile['total_score']}")
```

### JavaScript Client

```javascript
// Upload Resume
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const uploadResponse = await fetch('http://localhost:5002/api/processResume', {
  method: 'POST',
  body: formData
});

const uploadResult = await uploadResponse.json();
console.log('Candidate ID:', uploadResult.candidate_id);

// Rank Profiles
const jdData = {
  job_id: 'JD001',
  job_description: 'Looking for React developer...',
  required_skills: 'react, javascript, node.js',
  min_experience: 3
};

const rankResponse = await fetch('http://localhost:5002/api/profileRankingByJD', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(jdData)
});

const rankings = await rankResponse.json();
console.log('Top candidate:', rankings.ranked_profiles[0]);
```

---

## 🗄 Database Schema

### Tables

1. **resume_metadata** - Stores parsed resume data and embeddings
2. **job_descriptions** - Stores job postings and JD embeddings
3. **ranking_history** - Stores ranking results for analytics
4. **skills_master** - Optional skill standardization table
5. **applications** - Optional workflow tracking

See `ats_schema.sql` for complete schema.

---

## 🐳 Deployment

### Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    default-mysql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_ats.txt .
RUN pip install --no-cache-dir -r requirements_ats.txt
RUN python -m spacy download en_core_web_sm

# Copy application files
COPY . .

# Create upload directory
RUN mkdir -p uploads

EXPOSE 5002

CMD ["python", "ats_api.py"]
```

Build and run:
```bash
docker build -t ats-api .
docker run -p 5002:5002 --env-file .env ats-api
```

### Docker Compose

```yaml
version: '3.8'

services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: ats_db
    ports:
      - "3306:3306"
    volumes:
      - ./ats_schema.sql:/docker-entrypoint-initdb.d/schema.sql

  ats-api:
    build: .
    ports:
      - "5002:5002"
    environment:
      ATS_MYSQL_HOST: mysql
      ATS_MYSQL_USER: root
      ATS_MYSQL_PASSWORD: rootpassword
      ATS_MYSQL_DATABASE: ats_db
    depends_on:
      - mysql
```

Run with:
```bash
docker-compose up
```

---

## 🔧 Development

### Run API locally

```bash
python ats_api.py
```

### Run tests

```bash
python -m pytest tests/
```

---

## 📊 Monitoring & Logging

The API includes comprehensive logging:

- **Request/Response logging** for all API calls
- **Error tracking** with stack traces
- **Performance metrics** for embedding generation and ranking
- **Database operation logging**

Logs are written to console (stdout) by default. For production, configure logging to files or external services.

---

## 🔐 Security Considerations

1. **API Authentication** - Add JWT or API key authentication for production
2. **File Validation** - Only PDF/DOCX allowed, with size limits
3. **SQL Injection** - All queries use parameterized statements
4. **Environment Variables** - Never commit `.env` to version control
5. **HTTPS** - Use SSL/TLS in production
6. **Rate Limiting** - Implement rate limiting for public APIs

---

## 🎯 Ranking Algorithm

### Weighted Scoring Formula

```
Total Score = (Skills × 0.4) + (Experience × 0.3) + (Domain × 0.2) + (Education × 0.1)
```

### Skills Matching (40%)
- **Exact matches** for required and preferred skills
- **Partial matching** using skill aliases
- **Missing skills** identified for feedback

### Experience Matching (30%)
- **Perfect match** if within min-max range
- **Penalty** for under-qualified candidates
- **Minor penalty** for over-qualified (to reduce attrition risk)

### Domain Matching (20%)
- **Exact domain** = 100% score
- **Related domains** (e.g., finance/fintech) = 70-80%
- **Different domains** = 30%

### Education Matching (10%)
- **Meets or exceeds** required education = 100%
- **One level below** = 70%
- **Two+ levels below** = 40%

### Semantic Boost
- **Vector similarity** between resume and JD adds up to 5% bonus
- Uses cosine similarity of embeddings

---

## 🚀 Scaling for 1M+ Resumes

For large-scale deployments:

1. **Enable Pinecone** for vector search:
   ```env
   USE_PINECONE=True
   ATS_PINECONE_API_KEY=your_key
   ```

2. **Database optimization**:
   - Add indexes on frequently queried fields
   - Consider read replicas for MySQL
   - Use connection pooling

3. **Caching**:
   - Cache JD embeddings with Redis
   - Cache frequently accessed candidate profiles

4. **Async processing**:
   - Use Celery for background resume processing
   - Queue-based architecture for bulk operations

---

## 📝 License

This project is proprietary. All rights reserved.

---

## 🤝 Support

For issues, questions, or feature requests, contact the development team.

---

## 📚 Additional Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [spaCy Documentation](https://spacy.io/)

---

**Built with ❤️ for efficient recruitment**

