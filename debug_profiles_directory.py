#!/usr/bin/env python3
"""
Debug script for profile directory access
"""

import requests
import json

# Configuration
BASE_URL = "https://atsai-production.up.railway.app"

def debug_directory_access():
    """Debug directory access on Railway"""
    
    payload = {
        "profiles_directory": "D:\\profiles"
    }
    
    print("=" * 80)
    print("DEBUGGING PROFILE DIRECTORY ACCESS ON RAILWAY")
    print("=" * 80)
    print(f"URL: {BASE_URL}/api/debug-profiles-directory")
    print(f"Profiles Directory: {payload['profiles_directory']}")
    print()
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/debug-profiles-directory",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"Response Status: {response.status_code}")
        print()
        
        if response.status_code == 200:
            data = response.json()
            
            print("🔍 DEBUG INFORMATION:")
            print(f"Profiles Directory: {data.get('profiles_directory')}")
            print(f"Directory Exists: {data.get('directory_exists')}")
            print(f"Is Directory: {data.get('is_directory')}")
            print(f"Current Working Directory: {data.get('current_working_directory')}")
            print(f"Total Files: {data.get('total_files', 0)}")
            print()
            
            if data.get('files_in_directory'):
                print("📁 FILES IN DIRECTORY:")
                for file in data.get('files_in_directory', []):
                    print(f"  - {file}")
                print()
                
                print("📄 FILE TYPES:")
                print(f"PDF Files: {data.get('pdf_files', [])}")
                print(f"TXT Files: {data.get('txt_files', [])}")
                print(f"DOCX Files: {data.get('docx_files', [])}")
                print()
                
                if data.get('test_file'):
                    print("🧪 TEST FILE INFO:")
                    print(f"Test File: {data.get('test_file')}")
                    print(f"Test File Path: {data.get('test_file_path')}")
                    print(f"Test File Exists: {data.get('test_file_exists')}")
                    print(f"Test File Size: {data.get('test_file_size')} bytes")
                    print()
            
            if data.get('error'):
                print(f"❌ ERROR: {data.get('error')}")
            
        else:
            print("❌ ERROR!")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Response Text: {response.text}")
                
    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_comprehensive_ranking_with_debug():
    """Test comprehensive ranking with debug info"""
    
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
        "profiles_directory": "D:\\profiles",
        "top_k": 10
    }
    
    print("\n" + "=" * 80)
    print("TESTING COMPREHENSIVE RANKING WITH DEBUG")
    print("=" * 80)
    print(f"URL: {BASE_URL}/api/comprehensive-profile-ranking")
    print()
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/comprehensive-profile-ranking",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        print(f"Response Status: {response.status_code}")
        print()
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ SUCCESS!")
            print(f"Status: {data.get('status')}")
            print(f"Message: {data.get('message')}")
            print(f"Total Candidates Evaluated: {data.get('total_candidates_evaluated')}")
            print(f"Processing Time: {data.get('processing_time_ms')} ms")
            
            if data.get('ranked_profiles'):
                print(f"\n🏆 TOP CANDIDATES ({len(data.get('ranked_profiles'))}):")
                for profile in data.get('ranked_profiles', []):
                    print(f"Rank {profile.get('rank')}: {profile.get('name')} (ID: {profile.get('candidate_id')})")
                    print(f"   Score: {profile.get('total_score')}%")
                    print(f"   Skills: {', '.join(profile.get('matched_skills', []))}")
                    print()
            else:
                print("\nNo candidates found")
            
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
    print("Starting debug tests for profile directory access...")
    print(f"Testing against: {BASE_URL}")
    print()
    
    # Debug directory access
    debug_directory_access()
    
    # Test comprehensive ranking
    test_comprehensive_ranking_with_debug()
    
    print("\n" + "=" * 80)
    print("DEBUG TESTS COMPLETED")
    print("=" * 80)

