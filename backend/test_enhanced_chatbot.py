#!/usr/bin/env python3
"""
Test script for enhanced chatbot API with resume search functionality.
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

def test_basic_resume_search():
    """Test basic resume search functionality."""
    print("\n" + "="*60)
    print("Testing Basic Resume Search")
    print("="*60)
    
    test_queries = [
        "Python developer with machine learning experience",
        "Software engineer with 5+ years experience",
        "Data scientist with Python and SQL skills",
        "Full stack developer with React and Node.js",
        "DevOps engineer with AWS and Docker experience"
    ]
    
    for query in test_queries:
        print(f"\nSearch Query: '{query}'")
        try:
            payload = {
                "query": query,
                "top_k": 5,
                "include_full_details": False
            }
            
            response = requests.post(
                f"{BASE_URL}/resume-search", 
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Status Code: {response.status_code}")
            result = response.json()
            
            if response.status_code == 200:
                candidates = result.get('candidates', [])
                print(f"  Found {len(candidates)} candidates")
                
                for i, candidate in enumerate(candidates[:3]):  # Show top 3
                    print(f"    {i+1}. {candidate.get('name', 'Unknown')} "
                          f"(ID: {candidate.get('candidate_id')}, Score: {candidate.get('similarity_score', 0):.3f})")
                    print(f"       Skills: {candidate.get('primary_skills', 'N/A')}")
                    print(f"       Experience: {candidate.get('total_experience', 0)} years")
                    print(f"       Domain: {candidate.get('domain', 'N/A')}")
            else:
                print(f"  ✗ Search failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")

def test_advanced_resume_search():
    """Test advanced resume search with filters."""
    print("\n" + "="*60)
    print("Testing Advanced Resume Search with Filters")
    print("="*60)
    
    # Test case 1: Search with experience filter
    print("\n1. Search with Experience Filter:")
    try:
        payload = {
            "query": "Python developer",
            "top_k": 5,
            "filters": {
                "min_experience": 3,
                "max_experience": 10
            },
            "include_full_details": False
        }
        
        response = requests.post(
            f"{BASE_URL}/resume-search-advanced", 
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            candidates = result.get('candidates', [])
            print(f"  Found {len(candidates)} candidates with 3-10 years experience")
            
            for i, candidate in enumerate(candidates[:3]):
                print(f"    {i+1}. {candidate.get('name', 'Unknown')} "
                      f"(Experience: {candidate.get('total_experience', 0)} years)")
        else:
            print(f"  ✗ Search failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test case 2: Search with skills filter
    print("\n2. Search with Skills Filter:")
    try:
        payload = {
            "query": "Software development",
            "top_k": 5,
            "filters": {
                "required_skills": ["Python", "JavaScript", "SQL"]
            },
            "include_full_details": False
        }
        
        response = requests.post(
            f"{BASE_URL}/resume-search-advanced", 
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            candidates = result.get('candidates', [])
            print(f"  Found {len(candidates)} candidates with required skills")
            
            for i, candidate in enumerate(candidates[:3]):
                print(f"    {i+1}. {candidate.get('name', 'Unknown')} "
                      f"(Skills: {candidate.get('primary_skills', 'N/A')})")
        else:
            print(f"  ✗ Search failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test case 3: Search with domain filter
    print("\n3. Search with Domain Filter:")
    try:
        payload = {
            "query": "Technology professional",
            "top_k": 5,
            "filters": {
                "domain": "technology"
            },
            "include_full_details": False
        }
        
        response = requests.post(
            f"{BASE_URL}/resume-search-advanced", 
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            candidates = result.get('candidates', [])
            print(f"  Found {len(candidates)} candidates in technology domain")
            
            for i, candidate in enumerate(candidates[:3]):
                print(f"    {i+1}. {candidate.get('name', 'Unknown')} "
                      f"(Domain: {candidate.get('domain', 'N/A')})")
        else:
            print(f"  ✗ Search failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

def test_full_details_search():
    """Test search with full resume details."""
    print("\n" + "="*60)
    print("Testing Search with Full Resume Details")
    print("="*60)
    
    try:
        payload = {
            "query": "Senior Python developer",
            "top_k": 2,
            "include_full_details": True
        }
        
        response = requests.post(
            f"{BASE_URL}/resume-search", 
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            candidates = result.get('candidates', [])
            print(f"  Found {len(candidates)} candidates with full details")
            
            for i, candidate in enumerate(candidates):
                print(f"\n  Candidate {i+1}:")
                print(f"    Name: {candidate.get('name', 'Unknown')}")
                print(f"    Email: {candidate.get('email', 'N/A')}")
                print(f"    Phone: {candidate.get('phone', 'N/A')}")
                print(f"    Experience: {candidate.get('total_experience', 0)} years")
                print(f"    Skills: {candidate.get('primary_skills', 'N/A')}")
                print(f"    Education: {candidate.get('education', 'N/A')}")
                print(f"    Location: {candidate.get('current_location', 'N/A')}")
                print(f"    Company: {candidate.get('current_company', 'N/A')}")
                print(f"    Similarity Score: {candidate.get('similarity_score', 0):.3f}")
                
                # Show resume text preview (first 200 characters)
                resume_text = candidate.get('resume_text', '')
                if resume_text:
                    print(f"    Resume Preview: {resume_text[:200]}...")
        else:
            print(f"  ✗ Search failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

def test_hybrid_scoring():
    """Test hybrid scoring functionality."""
    print("\n" + "="*60)
    print("Testing Hybrid Scoring")
    print("="*60)
    
    try:
        payload = {
            "query": "Python developer with machine learning",
            "top_k": 5,
            "filters": {
                "required_skills": ["Python", "Machine Learning"],
                "target_experience": 5
            },
            "use_hybrid_search": True,
            "min_similarity_score": 0.1
        }
        
        response = requests.post(
            f"{BASE_URL}/resume-search-advanced", 
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            candidates = result.get('candidates', [])
            print(f"  Found {len(candidates)} candidates with hybrid scoring")
            
            for i, candidate in enumerate(candidates):
                print(f"    {i+1}. {candidate.get('name', 'Unknown')}")
                print(f"       Similarity Score: {candidate.get('similarity_score', 0):.3f}")
                print(f"       Hybrid Score: {candidate.get('hybrid_score', 0):.3f}")
                print(f"       Skills: {candidate.get('primary_skills', 'N/A')}")
        else:
            print(f"  ✗ Search failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

def test_performance():
    """Test API performance."""
    print("\n" + "="*60)
    print("Testing API Performance")
    print("="*60)
    
    queries = [
        "Python developer",
        "Data scientist",
        "Software engineer",
        "DevOps engineer",
        "Full stack developer"
    ]
    
    total_time = 0
    successful_requests = 0
    
    for query in queries:
        try:
            start_time = time.time()
            
            payload = {
                "query": query,
                "top_k": 5,
                "include_full_details": False
            }
            
            response = requests.post(
                f"{BASE_URL}/resume-search", 
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
                
                print(f"  Query: '{query}' - {request_time:.0f}ms total, {processing_time:.0f}ms processing")
            else:
                print(f"  Query: '{query}' - Failed (Status: {response.status_code})")
                
        except Exception as e:
            print(f"  Query: '{query}' - Error: {e}")
    
    if successful_requests > 0:
        avg_time = total_time / successful_requests
        print(f"\n  Performance Summary:")
        print(f"    Successful requests: {successful_requests}/{len(queries)}")
        print(f"    Average response time: {avg_time:.0f}ms")
        print(f"    Total time: {total_time:.0f}ms")

def main():
    """Run all tests."""
    print("Enhanced Chatbot API Test Suite")
    print(f"Testing against: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Chatbot API is not available. Exiting.")
        return
    
    # Test 2: Basic resume search
    test_basic_resume_search()
    
    # Test 3: Advanced resume search with filters
    test_advanced_resume_search()
    
    # Test 4: Full details search
    test_full_details_search()
    
    # Test 5: Hybrid scoring
    test_hybrid_scoring()
    
    # Test 6: Performance testing
    test_performance()
    
    print("\n" + "="*60)
    print("Test Suite Completed")
    print("="*60)
    print("\nAPI Endpoints Available:")
    print("1. POST /resume-search - Basic resume search")
    print("2. POST /resume-search-advanced - Advanced search with filters")
    print("3. POST /chat - Original chatbot functionality")
    print("4. GET /health - Health check")
    print("\nExample Usage:")
    print("curl -X POST https://atsai-production.up.railway.app/resume-search \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"query\": \"Python developer\", \"top_k\": 5}'")

if __name__ == "__main__":
    main()
