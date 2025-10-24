#!/usr/bin/env python3
"""
Quick test for the fixed comprehensive profile ranking endpoint
"""

import requests
import json

def test_fixed_endpoint():
    """Test the fixed comprehensive profile ranking endpoint"""
    
    BASE_URL = "https://atsai-production.up.railway.app"
    
    # Correct JSON payload format
    payload = {
        "job_requirements": {
            "required_skills": "python",
            "preferred_skills": "",
            "min_experience": 4,
            "max_experience": 7,
            "domain": "Computer Science Engineering",
            "education_required": "Computer Science"
        },
        "profiles_directory": "./profiles",
        "top_k": 10
    }
    
    print("Testing fixed comprehensive profile ranking endpoint...")
    print(f"URL: {BASE_URL}/api/comprehensive-profile-ranking")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/comprehensive-profile-ranking",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("SUCCESS! Response:")
            print(f"- Status: {data.get('status')}")
            print(f"- Message: {data.get('message')}")
            print(f"- Total Candidates: {data.get('total_candidates_evaluated')}")
            print(f"- Top Candidates Returned: {data.get('top_candidates_returned')}")
            
            ranked_profiles = data.get('ranked_profiles', [])
            if ranked_profiles:
                print(f"\nTop 3 Candidates:")
                for i, profile in enumerate(ranked_profiles[:3]):
                    print(f"{i+1}. {profile.get('name')} (ID: {profile.get('candidate_id')}) - Score: {profile.get('total_score')}%")
            else:
                print("\nNo candidates found")
                
        else:
            print("ERROR!")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error')}")
            except:
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_fixed_endpoint()
