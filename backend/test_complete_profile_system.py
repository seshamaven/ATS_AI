#!/usr/bin/env python3
"""
Complete test script for profile upload and ranking on Railway
"""

import requests
import json
import os

# Configuration
BASE_URL = "https://atsai-production.up.railway.app"

def upload_test_profiles():
    """Upload test profiles to Railway"""
    
    print("=" * 80)
    print("UPLOADING TEST PROFILES TO RAILWAY")
    print("=" * 80)
    
    # Test profile content
    test_profiles = {
        "1.txt": """John Doe - Senior Python Developer

Email: john.doe@example.com
Phone: +1-555-0123

EXPERIENCE:
5+ years of professional experience in Python development
Senior Software Engineer at TechCorp (2020-Present)
Software Developer at StartupXYZ (2018-2020)

SKILLS:
- Python (Expert)
- Django (Advanced)
- Flask (Advanced)
- SQL (Advanced)
- MySQL (Advanced)
- PostgreSQL (Intermediate)
- AWS (Advanced)
- Git (Expert)
- Docker (Intermediate)
- REST API (Advanced)
- Agile/Scrum (Advanced)

EDUCATION:
Bachelor of Computer Science - University of Technology (2018)

DOMAIN:
Software Development, Web Applications, Backend Systems

PROJECTS:
- Built scalable Django applications serving 100K+ users
- Implemented microservices architecture using Flask
- Designed and optimized database schemas
- Led agile development teams""",

        "2.txt": """Jane Smith - Full Stack Developer

Email: jane.smith@example.com
Phone: +1-555-0124

EXPERIENCE:
3 years of experience in full-stack development
Full Stack Developer at WebSolutions Inc (2021-Present)
Junior Developer at Digital Agency (2020-2021)

SKILLS:
- Python (Intermediate)
- JavaScript (Advanced)
- React (Advanced)
- Node.js (Intermediate)
- MySQL (Intermediate)
- MongoDB (Intermediate)
- Docker (Intermediate)
- Git (Advanced)
- HTML/CSS (Expert)
- REST API (Intermediate)

EDUCATION:
Bachelor of Engineering - Computer Science - State University (2020)

DOMAIN:
Web Development, Frontend Development, User Interface Design

PROJECTS:
- Developed responsive web applications using React
- Built REST APIs with Python Flask
- Implemented database solutions with MySQL and MongoDB""",

        "3.txt": """Mike Johnson - Data Scientist

Email: mike.johnson@example.com
Phone: +1-555-0125

EXPERIENCE:
4 years of experience in data science and machine learning
Senior Data Scientist at DataCorp (2021-Present)
Data Analyst at Analytics Inc (2020-2021)

SKILLS:
- Python (Expert)
- Machine Learning (Advanced)
- SQL (Advanced)
- Azure (Advanced)
- Git (Advanced)
- Pandas (Expert)
- NumPy (Advanced)
- Scikit-learn (Advanced)
- TensorFlow (Intermediate)
- Jupyter Notebooks (Expert)

EDUCATION:
Master of Computer Science - Data Science Specialization - Tech University (2020)

DOMAIN:
Data Science, Machine Learning, Analytics, Cloud Computing

PROJECTS:
- Built ML models for predictive analytics
- Implemented data pipelines using Python
- Developed cloud-based analytics solutions on Azure
- Created automated reporting systems"""
    }
    
    uploaded_files = []
    
    for filename, content in test_profiles.items():
        print(f"📤 Uploading {filename}...")
        
        # Create a temporary file
        temp_file_path = f"temp_{filename}"
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        try:
            # Upload file
            with open(temp_file_path, 'rb') as f:
                files = {'file': (filename, f, 'text/plain')}
                response = requests.post(
                    f"{BASE_URL}/api/upload-profile",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Successfully uploaded {filename}")
                print(f"   File path: {data.get('file_path')}")
                uploaded_files.append(filename)
            else:
                print(f"❌ Failed to upload {filename}: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error')}")
                except:
                    print(f"   Response: {response.text}")
            
            # Clean up temp file
            os.remove(temp_file_path)
            
        except Exception as e:
            print(f"❌ Exception uploading {filename}: {e}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    print(f"\n📊 Upload Summary: {len(uploaded_files)}/{len(test_profiles)} files uploaded successfully")
    return uploaded_files

def list_uploaded_profiles():
    """List all uploaded profiles"""
    
    print("\n" + "=" * 80)
    print("LISTING UPLOADED PROFILES")
    print("=" * 80)
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/list-uploaded-profiles",
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Success!")
            print(f"Profiles Directory: {data.get('profiles_directory')}")
            print(f"Total Profiles: {data.get('total_profiles')}")
            print()
            
            profiles = data.get('profiles', [])
            if profiles:
                print("📁 UPLOADED PROFILES:")
                for profile in profiles:
                    print(f"  - {profile.get('filename')} (ID: {profile.get('candidate_id')})")
                    print(f"    Size: {profile.get('file_size')} bytes")
                    print(f"    Path: {profile.get('file_path')}")
                    print()
            else:
                print("No profiles found")
                
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error')}")
            except:
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_comprehensive_ranking():
    """Test comprehensive ranking with uploaded profiles"""
    
    print("\n" + "=" * 80)
    print("TESTING COMPREHENSIVE PROFILE RANKING")
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
        "top_k": 10
    }
    
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
            print(f"Profiles Directory: {data.get('profiles_directory')}")
            print(f"Total Candidates Evaluated: {data.get('total_candidates_evaluated')}")
            print(f"Top Candidates Returned: {data.get('top_candidates_returned')}")
            print(f"Processing Time: {data.get('processing_time_ms')} ms")
            print()
            
            # Display job requirements
            job_reqs = data.get('job_requirements', {})
            print("📋 JOB REQUIREMENTS:")
            print(f"Required Skills: {job_reqs.get('required_skills', [])}")
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
            
        else:
            print("❌ ERROR!")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Response Text: {response.text}")
                
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    print("Starting complete profile upload and ranking test...")
    print(f"Testing against: {BASE_URL}")
    print()
    
    # Step 1: Upload test profiles
    uploaded_files = upload_test_profiles()
    
    # Step 2: List uploaded profiles
    list_uploaded_profiles()
    
    # Step 3: Test comprehensive ranking
    if uploaded_files:
        test_comprehensive_ranking()
    else:
        print("\n❌ No profiles uploaded, skipping ranking test")
    
    print("\n" + "=" * 80)
    print("COMPLETE TEST COMPLETED")
    print("=" * 80)

