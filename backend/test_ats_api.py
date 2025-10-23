"""
Test script for ATS API endpoints.
"""

import requests
import json
import os

# API Base URL
BASE_URL = "http://localhost:5002"


def test_health_check():
    """Test health check endpoint."""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_process_resume(resume_file_path):
    """Test resume processing endpoint."""
    print("\n" + "="*60)
    print("Testing Process Resume Endpoint")
    print("="*60)
    
    if not os.path.exists(resume_file_path):
        print(f"ERROR: Resume file not found: {resume_file_path}")
        return None
    
    with open(resume_file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/api/processResume", files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        candidate_id = response.json().get('candidate_id')
        print(f"\n✓ Resume processed successfully! Candidate ID: {candidate_id}")
        return candidate_id
    else:
        print(f"\n✗ Failed to process resume")
        return None


def test_profile_ranking(job_id="JD001"):
    """Test profile ranking endpoint."""
    print("\n" + "="*60)
    print("Testing Profile Ranking Endpoint")
    print("="*60)
    
    jd_data = {
        "job_id": job_id,
        "job_description": """
        We are looking for a Senior Python Backend Engineer with strong experience in building 
        scalable APIs and working with databases. The ideal candidate should have:
        
        - 5+ years of professional experience in Python development
        - Strong knowledge of Flask or FastAPI frameworks
        - Experience with MySQL, PostgreSQL, or other relational databases
        - Understanding of REST API design principles
        - Experience with cloud platforms (AWS/Azure/GCP)
        - Knowledge of Docker and containerization
        - Experience in fintech or financial services domain is a plus
        - Bachelor's degree in Computer Science or related field
        
        Responsibilities:
        - Design and develop RESTful APIs
        - Optimize database queries and performance
        - Collaborate with frontend teams
        - Write clean, maintainable code
        - Participate in code reviews
        """,
        "required_skills": "python, flask, sql, rest api, mysql",
        "preferred_skills": "docker, aws, fastapi, postgresql",
        "min_experience": 5,
        "max_experience": 10,
        "domain": "fintech",
        "education_required": "Bachelors"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/profileRankingByJD",
        json=jd_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nTotal Candidates Evaluated: {result.get('total_candidates_evaluated')}")
        print(f"Top Candidates Returned: {result.get('top_candidates_returned')}")
        
        print(f"\n{'='*60}")
        print("Top 5 Ranked Candidates:")
        print(f"{'='*60}")
        
        for profile in result['ranked_profiles'][:5]:
            print(f"\nRank {profile['rank']}:")
            print(f"  Name: {profile['name']}")
            print(f"  Email: {profile['email']}")
            print(f"  Total Score: {profile['total_score']:.2f}")
            print(f"  Match Percent: {profile['match_percent']:.1f}%")
            print(f"  Skills Score: {profile['skills_score']:.2f}")
            print(f"  Experience Score: {profile['experience_score']:.2f}")
            print(f"  Domain Score: {profile['domain_score']:.2f}")
            print(f"  Education Score: {profile['education_score']:.2f}")
            print(f"  Matched Skills: {', '.join(profile['matched_skills'][:5])}")
            print(f"  Missing Skills: {', '.join(profile['missing_skills'][:5])}")
            print(f"  Experience Match: {profile['experience_match']}")
            print(f"  Domain Match: {profile['domain_match']}")
        
        print(f"\n✓ Profile ranking completed successfully!")
        return True
    else:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print(f"\n✗ Failed to rank profiles")
        return False


def test_get_candidate(candidate_id):
    """Test get candidate endpoint."""
    print("\n" + "="*60)
    print(f"Testing Get Candidate Endpoint (ID: {candidate_id})")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/candidate/{candidate_id}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        candidate = response.json()
        print(f"\nCandidate Details:")
        print(f"  Name: {candidate.get('name')}")
        print(f"  Email: {candidate.get('email')}")
        print(f"  Experience: {candidate.get('total_experience')} years")
        print(f"  Domain: {candidate.get('domain')}")
        print(f"  Education: {candidate.get('education')}")
        print(f"  Primary Skills: {candidate.get('primary_skills', '')[:100]}...")
        print(f"\n✓ Candidate details retrieved successfully!")
        return True
    else:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print(f"\n✗ Failed to get candidate details")
        return False


def test_get_statistics():
    """Test get statistics endpoint."""
    print("\n" + "="*60)
    print("Testing Statistics Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/statistics")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        stats = response.json()['statistics']
        print(f"\nATS Statistics:")
        print(f"  Total Resumes: {stats.get('total_resumes', 0)}")
        print(f"  Total Jobs: {stats.get('total_jobs', 0)}")
        print(f"  Total Rankings: {stats.get('total_rankings', 0)}")
        print(f"  Average Experience: {stats.get('avg_experience', 0)} years")
        print(f"\n✓ Statistics retrieved successfully!")
        return True
    else:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print(f"\n✗ Failed to get statistics")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ATS API Test Suite")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print("="*60)
    
    # Test 1: Health Check
    health_ok = test_health_check()
    if not health_ok:
        print("\n✗ Health check failed. Make sure API is running.")
        return
    
    # Test 2: Statistics
    test_get_statistics()
    
    # Test 3: Process Resume (if file exists)
    resume_file = "sample_resume.pdf"
    candidate_id = None
    
    if os.path.exists(resume_file):
        candidate_id = test_process_resume(resume_file)
        
        if candidate_id:
            # Test 4: Get Candidate
            test_get_candidate(candidate_id)
    else:
        print(f"\nSkipping resume processing test (file not found: {resume_file})")
    
    # Test 5: Profile Ranking
    test_profile_ranking()
    
    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60)


if __name__ == "__main__":
    main()

