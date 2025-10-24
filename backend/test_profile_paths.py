#!/usr/bin/env python3
"""
Test script to find the correct path for profiles on Railway
"""

import requests
import json

# Configuration
BASE_URL = "https://atsai-production.up.railway.app"

def test_different_paths():
    """Test different possible paths for profiles"""
    
    # Test different possible paths
    test_paths = [
        "./profiles",
        "/app/profiles", 
        "/tmp/profiles",
        "profiles",
        "/home/railway/profiles",
        "D:\\profiles"  # Original path
    ]
    
    job_requirements = {
        "required_skills": ["python"],
        "preferred_skills": [],
        "min_experience": 4,
        "max_experience": 7,
        "domain": "Computer Science Engineering",
        "education_required": "Computer Science"
    }
    
    print("=" * 80)
    print("TESTING DIFFERENT PROFILE DIRECTORY PATHS")
    print("=" * 80)
    
    for path in test_paths:
        print(f"\n🔍 Testing path: {path}")
        
        payload = {
            "job_requirements": job_requirements,
            "profiles_directory": path,
            "top_k": 10
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/comprehensive-profile-ranking",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                total_candidates = data.get('total_candidates_evaluated', 0)
                
                if total_candidates > 0:
                    print(f"✅ SUCCESS! Found {total_candidates} candidates")
                    print(f"   Message: {data.get('message')}")
                    print(f"   Processing Time: {data.get('processing_time_ms')} ms")
                    
                    # Show first candidate
                    ranked_profiles = data.get('ranked_profiles', [])
                    if ranked_profiles:
                        first_candidate = ranked_profiles[0]
                        print(f"   Top Candidate: {first_candidate.get('name')} (ID: {first_candidate.get('candidate_id')})")
                        print(f"   Score: {first_candidate.get('total_score')}%")
                else:
                    print(f"❌ No candidates found")
                    print(f"   Message: {data.get('message')}")
            else:
                print(f"❌ Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text[:100]}...")
                    
        except Exception as e:
            print(f"❌ Exception: {e}")

def test_with_uploaded_profiles():
    """Test using profiles that might be uploaded to the server"""
    
    print("\n" + "=" * 80)
    print("TESTING WITH UPLOADED PROFILES")
    print("=" * 80)
    
    # Test if there are any profiles in the uploads directory
    job_requirements = {
        "required_skills": ["python"],
        "preferred_skills": [],
        "min_experience": 4,
        "max_experience": 7,
        "domain": "Computer Science Engineering",
        "education_required": "Computer Science"
    }
    
    # Test uploads directory
    payload = {
        "job_requirements": job_requirements,
        "profiles_directory": "./uploads",
        "top_k": 10
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/comprehensive-profile-ranking",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            total_candidates = data.get('total_candidates_evaluated', 0)
            
            if total_candidates > 0:
                print(f"✅ Found {total_candidates} candidates in uploads directory!")
            else:
                print(f"❌ No candidates found in uploads directory")
                print(f"   Message: {data.get('message')}")
        else:
            print(f"❌ Error testing uploads: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception testing uploads: {e}")

if __name__ == "__main__":
    print("Testing different profile directory paths on Railway...")
    print(f"Testing against: {BASE_URL}")
    
    # Test different paths
    test_different_paths()
    
    # Test uploads directory
    test_with_uploaded_profiles()
    
    print("\n" + "=" * 80)
    print("PATH TESTING COMPLETED")
    print("=" * 80)

