# ATS API Examples

Complete examples for all API endpoints with various programming languages.

## Table of Contents
- [Python Examples](#python-examples)
- [JavaScript Examples](#javascript-examples)
- [cURL Examples](#curl-examples)
- [Postman Collection](#postman-collection)

---

## Python Examples

### 1. Upload Resume

```python
import requests

def upload_resume(file_path):
    """Upload and process a resume."""
    url = "http://localhost:5002/api/processResume"
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Resume processed successfully!")
        print(f"  Candidate ID: {result['candidate_id']}")
        print(f"  Name: {result['candidate_name']}")
        print(f"  Experience: {result['total_experience']} years")
        print(f"  Skills: {result['primary_skills']}")
        return result['candidate_id']
    else:
        print(f"✗ Error: {response.json()}")
        return None

# Usage
candidate_id = upload_resume("resume.pdf")
```

### 2. Rank Candidates Against Job

```python
import requests
import json

def rank_candidates(job_data):
    """Rank all candidates against a job description."""
    url = "http://localhost:5002/api/profileRankingByJD"
    
    response = requests.post(
        url,
        json=job_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Ranking completed!")
        print(f"  Candidates evaluated: {result['total_candidates_evaluated']}")
        print(f"\nTop 5 Candidates:")
        
        for profile in result['ranked_profiles'][:5]:
            print(f"\n  {profile['rank']}. {profile['name']}")
            print(f"     Score: {profile['total_score']:.1f}/100")
            print(f"     Match: {profile['match_percent']:.1f}%")
            print(f"     Skills: {', '.join(profile['matched_skills'][:3])}")
            print(f"     Experience: {profile['experience_match']}")
        
        return result['ranked_profiles']
    else:
        print(f"✗ Error: {response.json()}")
        return None

# Usage
job_data = {
    "job_id": "JD001",
    "job_description": """
        Senior Python Backend Engineer
        
        Requirements:
        - 5+ years of Python development
        - Strong experience with Flask/FastAPI
        - MySQL/PostgreSQL expertise
        - REST API design
        - Cloud platforms (AWS/Azure)
        - Docker/Kubernetes
        - Fintech experience preferred
        - Bachelor's degree required
    """,
    "required_skills": "python, flask, sql, rest api, mysql",
    "preferred_skills": "docker, aws, fastapi, kubernetes",
    "min_experience": 5,
    "max_experience": 10,
    "domain": "fintech",
    "education_required": "Bachelors"
}

rankings = rank_candidates(job_data)
```

### 3. Get Candidate Details

```python
import requests

def get_candidate_details(candidate_id):
    """Get detailed information about a candidate."""
    url = f"http://localhost:5002/api/candidate/{candidate_id}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        candidate = response.json()
        print(f"\nCandidate Profile:")
        print(f"  ID: {candidate['candidate_id']}")
        print(f"  Name: {candidate['name']}")
        print(f"  Email: {candidate['email']}")
        print(f"  Phone: {candidate.get('phone', 'N/A')}")
        print(f"  Experience: {candidate['total_experience']} years")
        print(f"  Domain: {candidate.get('domain', 'N/A')}")
        print(f"  Education: {candidate.get('education', 'N/A')}")
        print(f"  Skills: {candidate['primary_skills'][:100]}...")
        return candidate
    else:
        print(f"✗ Error: {response.json()}")
        return None

# Usage
candidate = get_candidate_details(12345)
```

### 4. Batch Upload Resumes

```python
import os
import requests
from pathlib import Path

def batch_upload_resumes(directory):
    """Upload all resumes from a directory."""
    url = "http://localhost:5002/api/processResume"
    
    resume_files = list(Path(directory).glob("*.pdf")) + \
                   list(Path(directory).glob("*.docx"))
    
    results = []
    
    for i, file_path in enumerate(resume_files, 1):
        print(f"\nProcessing {i}/{len(resume_files)}: {file_path.name}")
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ Success: {result['candidate_name']}")
                results.append(result)
            else:
                print(f"  ✗ Failed: {response.json().get('error')}")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"Uploaded {len(results)}/{len(resume_files)} resumes successfully")
    return results

# Usage
results = batch_upload_resumes("./resumes")
```

### 5. Complete Workflow

```python
import requests
import time

class ATSClient:
    """Complete ATS API client."""
    
    def __init__(self, base_url="http://localhost:5002"):
        self.base_url = base_url
    
    def health_check(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        return response.status_code == 200
    
    def upload_resume(self, file_path):
        """Upload resume."""
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/api/processResume",
                files={'file': f}
            )
        return response.json()
    
    def rank_candidates(self, job_data):
        """Rank candidates."""
        response = requests.post(
            f"{self.base_url}/api/profileRankingByJD",
            json=job_data
        )
        return response.json()
    
    def get_candidate(self, candidate_id):
        """Get candidate details."""
        response = requests.get(
            f"{self.base_url}/api/candidate/{candidate_id}"
        )
        return response.json()
    
    def get_job_rankings(self, job_id, limit=50):
        """Get rankings for a job."""
        response = requests.get(
            f"{self.base_url}/api/job/{job_id}/rankings?limit={limit}"
        )
        return response.json()
    
    def get_statistics(self):
        """Get system statistics."""
        response = requests.get(f"{self.base_url}/api/statistics")
        return response.json()

# Usage
client = ATSClient()

# Check health
if not client.health_check():
    print("API is not healthy!")
    exit(1)

# Upload resume
result = client.upload_resume("resume.pdf")
candidate_id = result['candidate_id']

# Rank candidates
job_data = {
    "job_id": "JD001",
    "job_description": "Python developer needed...",
    "required_skills": "python, flask, sql"
}
rankings = client.rank_candidates(job_data)

# Get statistics
stats = client.get_statistics()
print(f"Total resumes: {stats['statistics']['total_resumes']}")
```

---

## JavaScript Examples

### 1. Upload Resume (Node.js)

```javascript
const fs = require('fs');
const FormData = require('form-data');
const axios = require('axios');

async function uploadResume(filePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    
    try {
        const response = await axios.post(
            'http://localhost:5002/api/processResume',
            form,
            { headers: form.getHeaders() }
        );
        
        console.log('✓ Resume processed successfully!');
        console.log('  Candidate ID:', response.data.candidate_id);
        console.log('  Name:', response.data.candidate_name);
        return response.data.candidate_id;
    } catch (error) {
        console.error('✗ Error:', error.response?.data || error.message);
        return null;
    }
}

// Usage
uploadResume('resume.pdf');
```

### 2. Rank Candidates (Browser)

```javascript
async function rankCandidates() {
    const jobData = {
        job_id: 'JD001',
        job_description: 'Senior Python developer needed...',
        required_skills: 'python, flask, sql',
        min_experience: 5,
        domain: 'technology'
    };
    
    try {
        const response = await fetch('http://localhost:5002/api/profileRankingByJD', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(jobData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            console.log('✓ Ranking completed!');
            console.log('Top candidates:', result.ranked_profiles.slice(0, 5));
            return result.ranked_profiles;
        } else {
            console.error('✗ Error:', result.error);
            return null;
        }
    } catch (error) {
        console.error('✗ Network error:', error);
        return null;
    }
}

// Usage
rankCandidates().then(profiles => {
    profiles.forEach(profile => {
        console.log(`${profile.rank}. ${profile.name} - Score: ${profile.total_score}`);
    });
});
```

### 3. React Component

```javascript
import React, { useState } from 'react';
import axios from 'axios';

function ResumeUploader() {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    
    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };
    
    const handleUpload = async () => {
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        setLoading(true);
        
        try {
            const response = await axios.post(
                'http://localhost:5002/api/processResume',
                formData
            );
            setResult(response.data);
        } catch (error) {
            console.error('Upload failed:', error);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div>
            <input type="file" onChange={handleFileChange} accept=".pdf,.docx" />
            <button onClick={handleUpload} disabled={!file || loading}>
                {loading ? 'Processing...' : 'Upload Resume'}
            </button>
            
            {result && (
                <div>
                    <h3>Resume Processed!</h3>
                    <p>Candidate ID: {result.candidate_id}</p>
                    <p>Name: {result.candidate_name}</p>
                    <p>Experience: {result.total_experience} years</p>
                </div>
            )}
        </div>
    );
}

export default ResumeUploader;
```

---

## cURL Examples

### 1. Health Check

```bash
curl http://localhost:5002/health
```

### 2. Upload Resume

```bash
curl -X POST http://localhost:5002/api/processResume \
  -F "file=@resume.pdf"
```

### 3. Rank Candidates

```bash
curl -X POST http://localhost:5002/api/profileRankingByJD \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "JD001",
    "job_description": "Senior Python Backend Engineer with 5+ years...",
    "required_skills": "python, flask, sql, rest api",
    "preferred_skills": "docker, aws",
    "min_experience": 5,
    "max_experience": 10,
    "domain": "fintech",
    "education_required": "Bachelors"
  }'
```

### 4. Get Candidate

```bash
curl http://localhost:5002/api/candidate/12345
```

### 5. Get Job Rankings

```bash
curl "http://localhost:5002/api/job/JD001/rankings?limit=20"
```

### 6. Get Statistics

```bash
curl http://localhost:5002/api/statistics
```

---

## Postman Collection

### Import this JSON into Postman

```json
{
    "info": {
        "name": "ATS API",
        "description": "Application Tracking System API endpoints",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    },
    "item": [
        {
            "name": "Health Check",
            "request": {
                "method": "GET",
                "header": [],
                "url": {
                    "raw": "http://localhost:5002/health",
                    "protocol": "http",
                    "host": ["localhost"],
                    "port": "5002",
                    "path": ["health"]
                }
            }
        },
        {
            "name": "Upload Resume",
            "request": {
                "method": "POST",
                "header": [],
                "body": {
                    "mode": "formdata",
                    "formdata": [
                        {
                            "key": "file",
                            "type": "file",
                            "src": "resume.pdf"
                        }
                    ]
                },
                "url": {
                    "raw": "http://localhost:5002/api/processResume",
                    "protocol": "http",
                    "host": ["localhost"],
                    "port": "5002",
                    "path": ["api", "processResume"]
                }
            }
        },
        {
            "name": "Rank Candidates",
            "request": {
                "method": "POST",
                "header": [
                    {
                        "key": "Content-Type",
                        "value": "application/json"
                    }
                ],
                "body": {
                    "mode": "raw",
                    "raw": "{\n  \"job_id\": \"JD001\",\n  \"job_description\": \"Senior Python developer...\",\n  \"required_skills\": \"python, flask, sql\",\n  \"min_experience\": 5\n}"
                },
                "url": {
                    "raw": "http://localhost:5002/api/profileRankingByJD",
                    "protocol": "http",
                    "host": ["localhost"],
                    "port": "5002",
                    "path": ["api", "profileRankingByJD"]
                }
            }
        }
    ]
}
```

---

## Error Handling

### Python Error Handling

```python
import requests
from requests.exceptions import RequestException

def safe_api_call(url, method='GET', **kwargs):
    """Make API call with error handling."""
    try:
        if method == 'GET':
            response = requests.get(url, **kwargs)
        elif method == 'POST':
            response = requests.post(url, **kwargs)
        
        response.raise_for_status()
        return response.json(), None
    
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP Error: {e.response.status_code} - {e.response.json()}"
    except requests.exceptions.ConnectionError:
        return None, "Connection Error: Cannot reach API server"
    except requests.exceptions.Timeout:
        return None, "Timeout Error: Request took too long"
    except RequestException as e:
        return None, f"Request Error: {str(e)}"

# Usage
result, error = safe_api_call('http://localhost:5002/health')
if error:
    print(f"Error: {error}")
else:
    print(f"Success: {result}")
```

---

For more examples and detailed documentation, see `README_ATS.md`

