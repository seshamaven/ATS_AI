#!/usr/bin/env python3
"""
Test script for JDProcessor API functionality.
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "https://atsai-production.up.railway.app"
# BASE_URL = "http://localhost:5001"  # For local testing

def test_health_check():
    """Test chatbot API health."""
    print("\n" + "="*60)
    print("Testing Chatbot API Health")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✓ Chatbot API is healthy")
            return True
        else:
            print("✗ Chatbot API health check failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_jd_processor_basic():
    """Test basic JD processor functionality."""
    print("\n" + "="*60)
    print("Testing Basic JD Processor")
    print("="*60)
    
    # Sample job descriptions
    job_descriptions = [
        {
            "job_title": "Senior Python Developer",
            "company_name": "Tech Corp",
            "location": "Bangalore",
            "job_description": """
            We are looking for a Senior Python Developer with 5+ years of experience in Python development.
            
            Required Skills:
            - Python programming (5+ years)
            - Django or Flask framework
            - PostgreSQL or MySQL
            - REST API development
            - Git version control
            
            Preferred Skills:
            - AWS cloud services
            - Docker containerization
            - Machine learning experience
            - React.js frontend
            
            Responsibilities:
            - Develop and maintain Python applications
            - Design and implement REST APIs
            - Collaborate with frontend developers
            - Write unit tests and documentation
            - Deploy applications to cloud platforms
            
            Education: Bachelor's degree in Computer Science or related field
            Experience: 5-8 years of relevant experience
            """
        },
        {
            "job_title": "Data Scientist",
            "company_name": "Data Analytics Inc",
            "location": "Mumbai",
            "job_description": """
            Join our data science team to build machine learning models and analytics solutions.
            
            Required Skills:
            - Python programming
            - Machine learning algorithms
            - Pandas, NumPy, Scikit-learn
            - SQL database queries
            - Statistical analysis
            
            Preferred Skills:
            - Deep learning with TensorFlow/PyTorch
            - Big data technologies (Spark, Hadoop)
            - Cloud platforms (AWS, Azure)
            - Data visualization (Tableau, Power BI)
            
            Responsibilities:
            - Build predictive models
            - Analyze large datasets
            - Create data visualizations
            - Collaborate with engineering teams
            - Present findings to stakeholders
            
            Education: Master's degree in Data Science, Statistics, or related field
            Experience: 3-6 years of data science experience
            """
        },
        {
            "job_title": "Full Stack Developer",
            "company_name": "StartupXYZ",
            "location": "Hyderabad",
            "job_description": """
            We need a Full Stack Developer to work on our web application platform.
            
            Required Skills:
            - JavaScript/TypeScript
            - React.js frontend development
            - Node.js backend development
            - MongoDB or PostgreSQL
            - HTML5, CSS3
            
            Preferred Skills:
            - AWS cloud services
            - Docker and Kubernetes
            - GraphQL API development
            - Microservices architecture
            
            Responsibilities:
            - Develop responsive web applications
            - Build RESTful APIs
            - Implement user authentication
            - Optimize application performance
            - Write clean, maintainable code
            
            Education: Bachelor's degree in Computer Science
            Experience: 2-4 years of full-stack development
            """
        }
    ]
    
    for i, jd in enumerate(job_descriptions, 1):
        print(f"\n{i}. Testing Job: {jd['job_title']} at {jd['company_name']}")
        try:
            payload = {
                "job_description": jd["job_description"],
                "job_title": jd["job_title"],
                "company_name": jd["company_name"],
                "location": jd["location"],
                "top_k": 5,
                "min_match_score": 0.3,
                "include_full_details": False
            }
            
            response = requests.post(
                f"{BASE_URL}/jd-processor", 
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Status Code: {response.status_code}")
            result = response.json()
            
            if response.status_code == 200:
                candidates = result.get('candidates', [])
                jd_analysis = result.get('jd_analysis', {})
                
                print(f"  ✓ Found {len(candidates)} matching candidates")
                print(f"  Required Skills: {jd_analysis.get('required_skills', [])}")
                print(f"  Min Experience: {jd_analysis.get('min_experience', 0)} years")
                
                for j, candidate in enumerate(candidates[:3]):  # Show top 3
                    print(f"    {j+1}. {candidate.get('name', 'Unknown')} "
                          f"(ID: {candidate.get('candidate_id')}, Match: {candidate.get('job_match_score', 0):.3f})")
                    print(f"       Skills: {candidate.get('primary_skills', 'N/A')}")
                    print(f"       Experience: {candidate.get('total_experience', 0)} years")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")

def test_jd_processor_advanced():
    """Test advanced JD processor with detailed analysis."""
    print("\n" + "="*60)
    print("Testing Advanced JD Processor with Detailed Analysis")
    print("="*60)
    
    job_description = """
    Senior Software Engineer - Backend Development
    
    Company: TechInnovate Solutions
    Location: Bangalore, India
    Experience: 6-10 years
    Work Mode: Hybrid (3 days office, 2 days remote)
    
    Job Description:
    We are seeking an experienced Senior Software Engineer to join our backend development team.
    The ideal candidate will have strong experience in building scalable web applications and APIs.
    
    Required Qualifications:
    - Bachelor's degree in Computer Science or Engineering
    - 6+ years of experience in software development
    - Strong proficiency in Python programming
    - Experience with Django or Flask frameworks
    - Solid understanding of database design (PostgreSQL, MySQL)
    - Experience with REST API development
    - Knowledge of version control systems (Git)
    - Experience with cloud platforms (AWS preferred)
    
    Preferred Qualifications:
    - Master's degree in Computer Science
    - Experience with microservices architecture
    - Knowledge of containerization (Docker, Kubernetes)
    - Experience with CI/CD pipelines
    - Background in machine learning or data science
    - Experience with message queues (RabbitMQ, Kafka)
    
    Key Responsibilities:
    - Design and develop scalable backend systems
    - Build and maintain RESTful APIs
    - Optimize database queries and application performance
    - Collaborate with frontend developers and product managers
    - Write comprehensive unit tests and documentation
    - Mentor junior developers
    - Participate in code reviews and technical discussions
    
    What We Offer:
    - Competitive salary package
    - Health insurance and wellness programs
    - Flexible working hours
    - Professional development opportunities
    - Collaborative and innovative work environment
    """
    
    try:
        payload = {
            "job_description": job_description,
            "job_title": "Senior Software Engineer - Backend",
            "company_name": "TechInnovate Solutions",
            "location": "Bangalore",
            "top_k": 10,
            "min_match_score": 0.4,
            "include_full_details": True
        }
        
        response = requests.post(
            f"{BASE_URL}/jd-processor", 
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            candidates = result.get('candidates', [])
            jd_analysis = result.get('jd_analysis', {})
            
            print(f"✓ Found {len(candidates)} matching candidates")
            print(f"\nJob Description Analysis:")
            print(f"  Required Skills: {jd_analysis.get('required_skills', [])}")
            print(f"  Preferred Skills: {jd_analysis.get('preferred_skills', [])}")
            print(f"  Experience Range: {jd_analysis.get('min_experience', 0)}-{jd_analysis.get('max_experience', 0)} years")
            print(f"  Education Requirements: {jd_analysis.get('education_requirements', [])}")
            print(f"  Work Mode: {jd_analysis.get('work_mode', [])}")
            
            print(f"\nTop Matching Candidates:")
            for i, candidate in enumerate(candidates[:5]):  # Show top 5
                print(f"\n  {i+1}. {candidate.get('name', 'Unknown')} (ID: {candidate.get('candidate_id')})")
                print(f"     Overall Match Score: {candidate.get('job_match_score', 0):.3f}")
                print(f"     Pinecone Similarity: {candidate.get('pinecone_similarity', 0):.3f}")
                
                match_details = candidate.get('match_details', {})
                print(f"     Skills Match: {match_details.get('skills_match', 0):.3f}")
                print(f"     Experience Match: {match_details.get('experience_match', 0):.3f}")
                print(f"     Education Match: {match_details.get('education_match', 0):.3f}")
                
                print(f"     Skills: {candidate.get('primary_skills', 'N/A')}")
                print(f"     Experience: {candidate.get('total_experience', 0)} years")
                print(f"     Education: {candidate.get('education', 'N/A')}")
                print(f"     Location: {candidate.get('current_location', 'N/A')}")
                print(f"     Company: {candidate.get('current_company', 'N/A')}")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Error: {e}")

def test_jd_processor_performance():
    """Test JD processor performance."""
    print("\n" + "="*60)
    print("Testing JD Processor Performance")
    print("="*60)
    
    job_descriptions = [
        "Python developer with 5+ years experience",
        "Data scientist with machine learning skills",
        "Full stack developer with React and Node.js",
        "DevOps engineer with AWS and Docker",
        "Backend developer with Django and PostgreSQL"
    ]
    
    total_time = 0
    successful_requests = 0
    
    for i, jd in enumerate(job_descriptions, 1):
        try:
            start_time = time.time()
            
            payload = {
                "job_description": jd,
                "job_title": f"Test Position {i}",
                "top_k": 5,
                "min_match_score": 0.2,
                "include_full_details": False
            }
            
            response = requests.post(
                f"{BASE_URL}/jd-processor", 
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            end_time = time.time()
            request_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200:
                successful_requests += 1
                total_time += request_time
                result = response.json()
                processing_time = result.get('processing_time_ms', 0)
                candidates_count = len(result.get('candidates', []))
                
                print(f"  {i}. '{jd}' - {request_time:.0f}ms total, {processing_time:.0f}ms processing, {candidates_count} candidates")
            else:
                print(f"  {i}. '{jd}' - Failed (Status: {response.status_code})")
                
        except Exception as e:
            print(f"  {i}. '{jd}' - Error: {e}")
    
    if successful_requests > 0:
        avg_time = total_time / successful_requests
        print(f"\n  Performance Summary:")
        print(f"    Successful requests: {successful_requests}/{len(job_descriptions)}")
        print(f"    Average response time: {avg_time:.0f}ms")
        print(f"    Total time: {total_time:.0f}ms")

def test_jd_processor_edge_cases():
    """Test JD processor with edge cases."""
    print("\n" + "="*60)
    print("Testing JD Processor Edge Cases")
    print("="*60)
    
    # Test case 1: Very short job description
    print("\n1. Very Short Job Description:")
    try:
        payload = {
            "job_description": "Python developer needed",
            "job_title": "Developer",
            "top_k": 3
        }
        
        response = requests.post(f"{BASE_URL}/jd-processor", json=payload)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Candidates found: {len(result.get('candidates', []))}")
        else:
            print(f"   Error: {response.json().get('error', 'Unknown')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test case 2: Very long job description
    print("\n2. Very Long Job Description:")
    try:
        long_jd = "We need a Python developer. " * 100  # Repeat 100 times
        payload = {
            "job_description": long_jd,
            "job_title": "Python Developer",
            "top_k": 3
        }
        
        response = requests.post(f"{BASE_URL}/jd-processor", json=payload)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Candidates found: {len(result.get('candidates', []))}")
        else:
            print(f"   Error: {response.json().get('error', 'Unknown')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test case 3: No matching candidates (very specific requirements)
    print("\n3. Very Specific Requirements:")
    try:
        specific_jd = """
        We need a developer with 20+ years of experience in COBOL programming,
        experience with mainframe systems, and knowledge of assembly language.
        Must have worked on IBM System/360 and have experience with JCL.
        """
        
        payload = {
            "job_description": specific_jd,
            "job_title": "Mainframe Developer",
            "top_k": 5,
            "min_match_score": 0.8
        }
        
        response = requests.post(f"{BASE_URL}/jd-processor", json=payload)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Candidates found: {len(result.get('candidates', []))}")
        else:
            print(f"   Error: {response.json().get('error', 'Unknown')}")
    except Exception as e:
        print(f"   Error: {e}")

def main():
    """Run all tests."""
    print("JDProcessor API Test Suite")
    print(f"Testing against: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Chatbot API is not available. Exiting.")
        return
    
    # Test 2: Basic JD processor
    test_jd_processor_basic()
    
    # Test 3: Advanced JD processor
    test_jd_processor_advanced()
    
    # Test 4: Performance testing
    test_jd_processor_performance()
    
    # Test 5: Edge cases
    test_jd_processor_edge_cases()
    
    print("\n" + "="*60)
    print("Test Suite Completed")
    print("="*60)
    print("\nJDProcessor API Features:")
    print("1. Job description analysis and requirement extraction")
    print("2. Pinecone vector search for semantic matching")
    print("3. Comprehensive candidate scoring and ranking")
    print("4. Detailed match analysis (skills, experience, education, location)")
    print("5. Configurable filtering and scoring thresholds")
    print("\nExample Usage:")
    print("curl -X POST https://atsai-production.up.railway.app/jd-processor \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"job_description\": \"Python developer with 5+ years experience\", \"top_k\": 10}'")

if __name__ == "__main__":
    main()
