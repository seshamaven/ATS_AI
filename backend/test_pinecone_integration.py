#!/usr/bin/env python3
"""
Test script for Pinecone integration with resume metadata.
"""

import requests
import json
import os
import sys
from datetime import datetime

# Configuration
BASE_URL = "https://atsai-production.up.railway.app"
# BASE_URL = "http://localhost:5002"  # For local testing

def test_health_check():
    """Test API health."""
    print("\n" + "="*60)
    print("Testing API Health")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✓ API is healthy")
            return True
        else:
            print("✗ API health check failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_resume_upload():
    """Test resume upload with Pinecone indexing."""
    print("\n" + "="*60)
    print("Testing Resume Upload with Pinecone Indexing")
    print("="*60)
    
    # You would need to provide a test resume file
    test_file_path = "test_resume.pdf"  # Update this path
    
    if not os.path.exists(test_file_path):
        print(f"⚠️  Test file not found: {test_file_path}")
        print("Please provide a test resume file to test upload functionality")
        return None
    
    try:
        with open(test_file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/api/processResume", files=files)
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200:
            candidate_id = result.get('candidate_id')
            pinecone_indexed = result.get('pinecone_indexed', False)
            print(f"\n✓ Resume uploaded successfully!")
            print(f"  Candidate ID: {candidate_id}")
            print(f"  Pinecone Indexed: {pinecone_indexed}")
            return candidate_id
        else:
            print(f"\n✗ Resume upload failed")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def test_batch_indexing():
    """Test batch indexing of existing resumes."""
    print("\n" + "="*60)
    print("Testing Batch Indexing of Existing Resumes")
    print("="*60)
    
    try:
        response = requests.post(f"{BASE_URL}/api/indexExistingResumes")
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200:
            print(f"\n✓ Batch indexing completed!")
            print(f"  Total resumes: {result.get('total_resumes', 0)}")
            print(f"  Indexed: {result.get('indexed_count', 0)}")
            print(f"  Failed: {result.get('failed_count', 0)}")
            return True
        else:
            print(f"\n✗ Batch indexing failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_resume_search():
    """Test resume search using Pinecone."""
    print("\n" + "="*60)
    print("Testing Resume Search with Pinecone")
    print("="*60)
    
    # Test queries
    test_queries = [
        "Python developer with machine learning experience",
        "Software engineer with 5+ years experience",
        "Data scientist with Python and SQL skills",
        "Full stack developer with React and Node.js"
    ]
    
    for query in test_queries:
        print(f"\nSearch Query: '{query}'")
        try:
            payload = {
                "query": query,
                "top_k": 5,
                "filters": {}  # Optional filters
            }
            
            response = requests.post(
                f"{BASE_URL}/api/searchResumes", 
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Status Code: {response.status_code}")
            result = response.json()
            
            if response.status_code == 200:
                matches = result.get('matches', [])
                print(f"  Found {len(matches)} matches")
                
                for i, match in enumerate(matches[:3]):  # Show top 3
                    print(f"    {i+1}. {match.get('name', 'Unknown')} "
                          f"(Score: {match.get('similarity_score', 0):.3f})")
                    print(f"       Skills: {match.get('primary_skills', 'N/A')}")
                    print(f"       Experience: {match.get('total_experience', 0)} years")
            else:
                print(f"  ✗ Search failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")

def test_filtered_search():
    """Test resume search with filters."""
    print("\n" + "="*60)
    print("Testing Filtered Resume Search")
    print("="*60)
    
    try:
        payload = {
            "query": "Python developer",
            "top_k": 10,
            "filters": {
                "domain": "technology",  # Filter by domain
                "total_experience": {"$gte": 2}  # Minimum 2 years experience
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/searchResumes", 
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            matches = result.get('matches', [])
            print(f"✓ Found {len(matches)} filtered matches")
            
            for i, match in enumerate(matches[:5]):  # Show top 5
                print(f"  {i+1}. {match.get('name', 'Unknown')} "
                      f"(Score: {match.get('similarity_score', 0):.3f})")
                print(f"     Domain: {match.get('domain', 'N/A')}")
                print(f"     Experience: {match.get('total_experience', 0)} years")
        else:
            print(f"✗ Filtered search failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    """Run all tests."""
    print("Pinecone Integration Test Suite")
    print(f"Testing against: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ API is not available. Exiting.")
        return
    
    # Test 2: Batch indexing (if resumes exist)
    test_batch_indexing()
    
    # Test 3: Resume search
    test_resume_search()
    
    # Test 4: Filtered search
    test_filtered_search()
    
    # Test 5: Resume upload (optional - requires test file)
    # candidate_id = test_resume_upload()
    
    print("\n" + "="*60)
    print("Test Suite Completed")
    print("="*60)
    print("\nTo enable Pinecone indexing:")
    print("1. Set USE_PINECONE=True in environment variables")
    print("2. Set ATS_PINECONE_API_KEY with your Pinecone API key")
    print("3. Set ATS_PINECONE_INDEX_NAME (default: ats-resumes)")
    print("\nExample environment variables:")
    print("USE_PINECONE=True")
    print("ATS_PINECONE_API_KEY=your_pinecone_api_key")
    print("ATS_PINECONE_INDEX_NAME=ats-resumes")

if __name__ == "__main__":
    main()
