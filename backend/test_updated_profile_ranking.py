#!/usr/bin/env python3
"""
Test script for the updated /api/profileRankingByJD endpoint
Tests with general job descriptions instead of structured input
"""

import requests
import json
import time

# Configuration
BASE_URL = "https://atsai-production.up.railway.app"
# BASE_URL = "http://localhost:5002"  # For local testing

def test_python_developer_jd():
    """Test with the Python Developer job description"""
    
    python_dev_jd = """About the job
Purpose of the role: 

As Python Developer, the person will be responsible for designing, developing, and maintaining high-performance, scalable applications and services using Python programming language. The role requires expertise in Python and its ecosystem to deliver efficient, reliable, and maintainable software solutions. 



KEY RESPONSIBILITIES: In this role, you will be responsible for: 

Designing and implementing backend services and applications using Python
Building high-performance, concurrent, and scalable systems.
Building the server-side logic of web applications, including APIs and database interactions.
Connecting applications with other services, APIs, and databases. 
Ensuring code quality through testing, debugging, and troubleshooting. 


Experience: 

Graduate or postgraduate in Computer Science Engineering Specialization. 
4-7 years hands-on experience in software development with significant focus on Python


Skills & Competencies: 

Must Have: 

Strong proficiency in Python programming language and its standard library
Experience with web frameworks like Django or Flask.
Knowledge of databases (SQL and NoSQL).
Understanding of software development methodologies (e.g., Agile).
Experience with version control systems (e.g., Git).
Familiarity with cloud platforms (e.g., AWS, Azure, GCP) is often preferred.
Excellent problem-solving and communication skills."""

    payload = {
        "job_id": "PYTHON_DEV_001",
        "job_description": python_dev_jd,
        "job_title": "Python Developer",
        "company_name": "Tech Company",
        "location": "Remote"
    }
    
    print("=" * 80)
    print("TESTING UPDATED /api/profileRankingByJD ENDPOINT")
    print("=" * 80)
    print(f"URL: {BASE_URL}/api/profileRankingByJD")
    print(f"Job Title: {payload['job_title']}")
    print(f"Job Description Length: {len(payload['job_description'])} characters")
    print()
    
    try:
        print("Sending request...")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/api/profileRankingByJD",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        response_time = time.time() - start_time
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Time: {response_time:.2f} seconds")
        print()
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ SUCCESS!")
            print(f"Status: {data.get('status')}")
            print(f"Message: {data.get('message')}")
            print(f"Job ID: {data.get('job_id')}")
            print(f"Total Candidates Evaluated: {data.get('total_candidates_evaluated')}")
            print(f"Top Candidates Returned: {data.get('top_candidates_returned')}")
            print()
            
            # Display extracted job requirements
            extracted_reqs = data.get('extracted_job_requirements', {})
            print("📋 EXTRACTED JOB REQUIREMENTS:")
            print(f"Required Skills: {extracted_reqs.get('required_skills', [])}")
            print(f"Preferred Skills: {extracted_reqs.get('preferred_skills', [])}")
            print(f"Experience Range: {extracted_reqs.get('min_experience')}-{extracted_reqs.get('max_experience')} years")
            print(f"Domain: {extracted_reqs.get('domain')}")
            print(f"Education Required: {extracted_reqs.get('education_required')}")
            print(f"All Extracted Skills: {extracted_reqs.get('all_extracted_skills', [])}")
            print()
            
            # Display top candidates
            ranked_profiles = data.get('ranked_profiles', [])
            if ranked_profiles:
                print("🏆 TOP CANDIDATES:")
                for i, profile in enumerate(ranked_profiles[:5], 1):
                    print(f"{i}. {profile.get('name', 'Unknown')} - Score: {profile.get('total_score', 0):.3f}")
                    print(f"   Skills Match: {profile.get('skills_score', 0):.3f}")
                    print(f"   Experience Match: {profile.get('experience_score', 0):.3f}")
                    print(f"   Domain Match: {profile.get('domain_score', 0):.3f}")
                    print(f"   Education Match: {profile.get('education_score', 0):.3f}")
                    print(f"   Matched Skills: {', '.join(profile.get('matched_skills', []))}")
                    print()
            else:
                print("No candidates found in database")
            
            # Display ranking criteria
            ranking_criteria = data.get('ranking_criteria', {})
            print("⚖️ RANKING CRITERIA:")
            print(f"Weights: {ranking_criteria.get('weights', {})}")
            print(f"Semantic Similarity: {ranking_criteria.get('semantic_similarity')}")
            
        else:
            print("❌ ERROR!")
            print(f"Status Code: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Response Text: {response.text}")
                
    except requests.exceptions.Timeout:
        print("❌ REQUEST TIMEOUT!")
        print("The request took longer than 60 seconds")
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR!")
        print("Could not connect to the API server")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")

def test_minimal_jd():
    """Test with minimal job description"""
    
    minimal_jd = "We are looking for a Python developer with 3-5 years experience in Django and Flask."
    
    payload = {
        "job_id": "MINIMAL_JD_001",
        "job_description": minimal_jd
    }
    
    print("\n" + "=" * 80)
    print("TESTING WITH MINIMAL JOB DESCRIPTION")
    print("=" * 80)
    print(f"Job Description: {minimal_jd}")
    print()
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/profileRankingByJD",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")
            
            extracted_reqs = data.get('extracted_job_requirements', {})
            print("📋 EXTRACTED REQUIREMENTS:")
            print(f"Required Skills: {extracted_reqs.get('required_skills', [])}")
            print(f"Experience Range: {extracted_reqs.get('min_experience')}-{extracted_reqs.get('max_experience')} years")
            print(f"All Skills: {extracted_reqs.get('all_extracted_skills', [])}")
            
        else:
            print("❌ ERROR!")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Response Text: {response.text}")
                
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    print("Starting tests for updated /api/profileRankingByJD endpoint...")
    print(f"Testing against: {BASE_URL}")
    print()
    
    # Test with full Python Developer job description
    test_python_developer_jd()
    
    # Test with minimal job description
    test_minimal_jd()
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETED")
    print("=" * 80)
