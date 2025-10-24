#!/usr/bin/env python3
"""
Test script for the comprehensive profile ranking endpoint on Railway
"""

import requests
import json
import time

# Configuration
BASE_URL = "https://atsai-production.up.railway.app"

def test_comprehensive_ranking_railway():
    """Test with the provided job requirements on Railway"""
    
    # Your provided job requirements
    job_requirements = {
        "required_skills": [
            "git",
            "nosql", 
            "flask",
            "gcp",
            "azure",
            "agile",
            "django",
            "sql",
            "aws",
            "python"
        ],
        "preferred_skills": [],
        "min_experience": 4,
        "max_experience": 7,
        "domain": "Computer Science Engineering",
        "education_required": "Computer Science"
    }
    
    payload = {
        "job_requirements": job_requirements,
        "profiles_directory": "D:\\profiles",
        "top_k": 10
    }
    
    print("=" * 80)
    print("TESTING COMPREHENSIVE PROFILE RANKING ON RAILWAY")
    print("=" * 80)
    print(f"URL: {BASE_URL}/api/comprehensive-profile-ranking")
    print(f"Profiles Directory: {payload['profiles_directory']}")
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

def test_with_sample_data():
    """Test with sample data using the existing profileRankingByJD endpoint"""
    
    payload = {
        "job_id": "COMPREHENSIVE_TEST_001",
        "job_description": "We are looking for a Senior Python Developer with 4-7 years experience in Django, Flask, SQL, Git, AWS, Azure, GCP, NoSQL, and Agile methodologies. Computer Science degree required.",
        "required_skills": "python, django, flask, sql, git, aws, azure, gcp, nosql, agile",
        "preferred_skills": "docker, kubernetes",
        "min_experience": 4,
        "max_experience": 7,
        "domain": "Computer Science Engineering",
        "education_required": "Computer Science"
    }
    
    print("\n" + "=" * 80)
    print("TESTING WITH EXISTING PROFILE RANKING ENDPOINT")
    print("=" * 80)
    print(f"URL: {BASE_URL}/api/profileRankingByJD")
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
            
            ranked_profiles = data.get('ranked_profiles', [])
            print(f"Total Candidates: {data.get('total_candidates_evaluated')}")
            print()
            
            for profile in ranked_profiles[:3]:  # Show top 3
                print(f"Rank {profile.get('rank')}: {profile.get('name')}")
                print(f"   Score: {profile.get('total_score')}%")
                print(f"   Skills: {', '.join(profile.get('matched_skills', []))}")
                print()
            
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
    print("Starting tests for comprehensive profile ranking on Railway...")
    print(f"Testing against: {BASE_URL}")
    print()
    
    # Test with existing endpoint first
    test_with_sample_data()
    
    # Test with new comprehensive endpoint
    test_comprehensive_ranking_railway()
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETED")
    print("=" * 80)
