#!/usr/bin/env python3
"""
Test script to read PDF files from local profiles directory and test comprehensive ranking
"""

import requests
import json
import time
import os

# Configuration
BASE_URL = "https://atsai-production.up.railway.app"
LOCAL_PROFILES_DIR = "./profiles"

def test_local_profiles_directory():
    """Test reading profiles from local profiles directory"""
    
    print("=" * 80)
    print("TESTING LOCAL PROFILES DIRECTORY")
    print("=" * 80)
    print(f"Local Profiles Directory: {LOCAL_PROFILES_DIR}")
    
    # Check if profiles directory exists
    if not os.path.exists(LOCAL_PROFILES_DIR):
        print(f"❌ Profiles directory {LOCAL_PROFILES_DIR} does not exist!")
        return False
    
    # List files in directory
    files = os.listdir(LOCAL_PROFILES_DIR)
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    
    print(f"📁 Found {len(files)} files in directory:")
    for file in files:
        print(f"  - {file}")
    
    print(f"\n📄 PDF Files ({len(pdf_files)}):")
    for pdf_file in pdf_files:
        file_path = os.path.join(LOCAL_PROFILES_DIR, pdf_file)
        file_size = os.path.getsize(file_path)
        print(f"  - {pdf_file} ({file_size:,} bytes)")
    
    return len(pdf_files) > 0

def test_comprehensive_ranking_with_local_profiles():
    """Test comprehensive ranking using local profiles directory"""
    
    print("\n" + "=" * 80)
    print("TESTING COMPREHENSIVE RANKING WITH LOCAL PROFILES")
    print("=" * 80)
    
    job_requirements = {
        "required_skills": ["python"],
        "preferred_skills": [],
        "min_experience": 4,
        "max_experience": 7,
        "domain": "Computer Science Engineering",
        "education_required": "Computer Science"
    }
    
    payload = {
        "job_requirements": job_requirements,
        "profiles_directory": LOCAL_PROFILES_DIR,
        "top_k": 10
    }
    
    print(f"URL: {BASE_URL}/api/comprehensive-profile-ranking")
    print(f"Profiles Directory: {LOCAL_PROFILES_DIR}")
    print(f"Job Requirements: {len(job_requirements['required_skills'])} required skills")
    print()
    
    try:
        print("Sending request...")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/api/comprehensive-profile-ranking",
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
            print(f"Profiles Directory: {data.get('profiles_directory')}")
            print(f"Total Candidates Evaluated: {data.get('total_candidates_evaluated')}")
            print(f"Top Candidates Returned: {data.get('top_candidates_returned')}")
            print(f"Processing Time: {data.get('processing_time_ms')} ms")
            print()
            
            # Display job requirements
            job_reqs = data.get('job_requirements', {})
            print("📋 JOB REQUIREMENTS:")
            print(f"Required Skills: {job_reqs.get('required_skills', [])}")
            print(f"Preferred Skills: {job_reqs.get('preferred_skills', [])}")
            print(f"Experience Range: {job_reqs.get('min_experience')}-{job_reqs.get('max_experience')} years")
            print(f"Domain: {job_reqs.get('domain')}")
            print(f"Education Required: {job_reqs.get('education_required')}")
            print()
            
            # Display top candidates
            ranked_profiles = data.get('ranked_profiles', [])
            if ranked_profiles:
                print("🏆 TOP CANDIDATES:")
                for profile in ranked_profiles:
                    print(f"Rank {profile.get('rank')}: {profile.get('name')} (ID: {profile.get('candidate_id')})")
                    print(f"   Total Score: {profile.get('total_score')}%")
                    print(f"   Skills Score: {profile.get('skills_score')}%")
                    print(f"   Experience Score: {profile.get('experience_score')}%")
                    print(f"   Domain Score: {profile.get('domain_score')}%")
                    print(f"   Education Score: {profile.get('education_score')}%")
                    print(f"   Matched Skills: {', '.join(profile.get('matched_skills', []))}")
                    print(f"   Missing Skills: {', '.join(profile.get('missing_skills', []))}")
                    print(f"   Experience Match: {profile.get('experience_match')}")
                    print(f"   Domain Match: {profile.get('domain_match')}")
                    print(f"   Education Match: {profile.get('education_match')}")
                    print()
            else:
                print("No candidates found")
            
            # Display ranking criteria
            ranking_criteria = data.get('ranking_criteria', {})
            print("⚖️ RANKING CRITERIA:")
            print(f"Weights: {ranking_criteria.get('weights', {})}")
            print(f"Semantic Similarity: {ranking_criteria.get('semantic_similarity')}")
            print(f"Analysis Depth: {ranking_criteria.get('analysis_depth')}")
            
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

def test_with_different_job_requirements():
    """Test with different job requirements to see how ranking changes"""
    
    print("\n" + "=" * 80)
    print("TESTING WITH DIFFERENT JOB REQUIREMENTS")
    print("=" * 80)
    
    # Test with more comprehensive Python developer requirements
    job_requirements = {
        "required_skills": ["python", "django", "sql", "git"],
        "preferred_skills": ["aws", "docker", "agile"],
        "min_experience": 3,
        "max_experience": 8,
        "domain": "Software Development",
        "education_required": "Computer Science"
    }
    
    payload = {
        "job_requirements": job_requirements,
        "profiles_directory": LOCAL_PROFILES_DIR,
        "top_k": 5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/comprehensive-profile-ranking",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ SUCCESS!")
            print(f"Total Candidates Evaluated: {data.get('total_candidates_evaluated')}")
            print()
            
            ranked_profiles = data.get('ranked_profiles', [])
            if ranked_profiles:
                print("🏆 RANKING RESULTS:")
                for profile in ranked_profiles:
                    print(f"Rank {profile.get('rank')}: {profile.get('name')} - Score: {profile.get('total_score')}%")
                    print(f"   Skills: {', '.join(profile.get('matched_skills', []))}")
                    print(f"   Missing: {', '.join(profile.get('missing_skills', []))}")
                    print()
            else:
                print("No candidates found")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Starting comprehensive profile ranking test with local PDF files...")
    print(f"Testing against: {BASE_URL}")
    print()
    
    # Step 1: Check local profiles directory
    if not test_local_profiles_directory():
        print("\n❌ No PDF files found in local profiles directory")
        exit(1)
    
    # Step 2: Test comprehensive ranking
    test_comprehensive_ranking_with_local_profiles()
    
    # Step 3: Test with different requirements
    test_with_different_job_requirements()
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TESTING COMPLETED")
    print("=" * 80)
