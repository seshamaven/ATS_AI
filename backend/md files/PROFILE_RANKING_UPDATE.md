# Profile Ranking by JD - MySQL Metadata Integration

## Update Summary

The `/api/profileRankingByJD` endpoint has been updated to **use MySQL metadata as input** when available. This allows the API to use pre-stored job requirements from the database instead of always extracting from text.

## What Changed

### Before
- Only accepted `job_description` text
- Always extracted skills, experience, domain from text
- No way to use existing metadata

### After
- Accepts **both `job_id` and `job_description`**
- Uses MySQL metadata when `job_id` is provided
- Allows explicit override via request parameters
- Falls back to extraction if metadata not available

## Usage Examples

### 1. Using Existing Job from Database

```json
{
  "job_id": "JD_123456"
}
```

**Behavior:**
- Fetches job metadata from MySQL for `JD_123456`
- Uses stored `required_skills`, `preferred_skills`, `min_experience`, etc.
- Only generates embedding for ranking

### 2. New Job with Metadata Override

```json
{
  "job_description": "We are looking for a Python developer...",
  "job_title": "Senior Python Developer",
  "required_skills": "Python, Django, Flask, PostgreSQL",
  "preferred_skills": "AWS, Docker, Kubernetes",
  "min_experience": 5,
  "max_experience": 10,
  "domain": "Technology",
  "education_required": "Bachelor's in Computer Science"
}
```

**Behavior:**
- Uses explicit parameters (priority 1)
- Extracts from text only if parameters not provided
- Stores everything in MySQL for future use

### 3. New Job Without Explicit Metadata

```json
{
  "job_description": "We are looking for a Python developer with 5+ years experience..."
}
```

**Behavior:**
- Extracts skills, experience, domain from text
- Stores extracted data in MySQL
- Future requests with same `job_id` will use stored data

## Priority Order

When processing a request, the system uses this priority:

1. **Explicit request parameters** - If provided in request body
2. **MySQL metadata** - If `job_id` exists in database
3. **Text extraction** - Parse and extract from `job_description`

## Supported Parameters

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | string | No | ID of existing job in database |
| `job_description` | string | Conditional* | Job description text |
| `job_title` | string | No | Job title |
| `required_skills` | string/list | No | Required skills (comma-separated or array) |
| `preferred_skills` | string/list | No | Preferred skills (comma-separated or array) |
| `min_experience` | number | No | Minimum years of experience |
| `max_experience` | number | No | Maximum years of experience |
| `domain` | string | No | Industry domain |
| `education_required` | string | No | Required education level |
| `location` | string | No | Job location |
| `employment_type` | string | No | Full-time, Part-time, Contract |
| `salary_range` | string | No | Salary range |
| `top_k` | number | No | Number of results to return (default: 50) |

*`job_description` is required if `job_id` doesn't exist in database.

## Response Format

The response includes both the ranked profiles and the job requirements used:

```json
{
  "status": "success",
  "message": "Profile ranking completed successfully",
  "job_id": "JD_123456",
  "ranked_profiles": [
    {
      "candidate_id": 5,
      "name": "John Doe",
      "total_score": 85.5,
      "match_percent": 85.5,
      "skills_score": 90.0,
      "experience_score": 85.0,
      "domain_score": 80.0,
      "education_score": 100.0,
      ...
    }
  ],
  "total_candidates_evaluated": 150,
  "top_candidates_returned": 50,
  "extracted_job_requirements": {
    "required_skills": ["Python", "Django", "PostgreSQL"],
    "preferred_skills": ["AWS", "Docker"],
    "min_experience": 5,
    "max_experience": 10,
    "domain": "Technology",
    "education_required": "Bachelor's"
  },
  "ranking_criteria": {
    "weights": {
      "skills": 0.4,
      "experience": 0.3,
      "domain": 0.2,
      "education": 0.1
    }
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

## Benefits

### 1. **Performance**
- No need to re-extract metadata if job already exists
- Faster ranking for repeated queries

### 2. **Consistency**
- Same job always uses same requirements
- Metadata can be manually reviewed and corrected

### 3. **Flexibility**
- Can use database metadata as baseline
- Can override specific parameters per request
- Can still use new jobs without metadata

### 4. **Workflow Support**
- Create jobs with structured metadata
- Use same job for multiple ranking requests
- Maintain job requirements separately from description

## Example Workflow

### Step 1: Create/Store Job with Metadata

```bash
POST /api/profileRankingByJD
{
  "job_id": "JD_SENIOR_PYTHON_2024",
  "job_description": "...",
  "job_title": "Senior Python Developer",
  "required_skills": "Python, Django, PostgreSQL, REST APIs",
  "preferred_skills": "AWS, Docker, Kubernetes, CI/CD",
  "min_experience": 5,
  "domain": "Technology"
}
```

This stores the job in MySQL with metadata.

### Step 2: Reuse Same Job for Ranking

```bash
POST /api/profileRankingByJD
{
  "job_id": "JD_SENIOR_PYTHON_2024",
  "top_k": 10
}
```

This uses stored metadata, no re-extraction needed!

### Step 3: Override Specific Requirements

```bash
POST /api/profileRankingByJD
{
  "job_id": "JD_SENIOR_PYTHON_2024",
  "min_experience": 7,  // Override: looking for more senior
  "top_k": 5
}
```

Uses MySQL data but overrides `min_experience`.

## Database Schema

Job descriptions are stored in `job_descriptions` table:

```sql
CREATE TABLE job_descriptions (
    job_id VARCHAR(100) PRIMARY KEY,
    job_title VARCHAR(255),
    job_description LONGTEXT,
    required_skills TEXT,
    preferred_skills TEXT,
    min_experience FLOAT,
    max_experience FLOAT,
    domain VARCHAR(255),
    education_required VARCHAR(500),
    location VARCHAR(255),
    employment_type VARCHAR(50),
    salary_range VARCHAR(100),
    embedding JSON,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

## Migration Notes

### Existing Code
- Still works as before if you provide only `job_description`
- Backwards compatible

### New Usage
- To leverage MySQL metadata, provide `job_id`
- Ensure job exists in database (created via first ranking or manually)

## Error Handling

### Job ID Not Found
```json
{
  "error": "Either job_description or valid job_id is required"
}
```
→ Provide `job_description` or use valid `job_id`

### Empty Metadata
If `job_id` exists but has no metadata:
→ Falls back to extracting from `job_description` text

## Testing

Test the new functionality:

```bash
# 1. First request - creates and stores job
curl -X POST https://your-api.com/api/profileRankingByJD \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "TEST_JOB",
    "job_description": "Looking for Python developer with 5 years experience...",
    "min_experience": 5
  }'

# 2. Second request - uses stored metadata
curl -X POST https://your-api.com/api/profileRankingByJD \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "TEST_JOB"
  }'
```

