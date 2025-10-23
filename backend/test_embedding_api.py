#!/usr/bin/env python3
"""
Test script for Embedding API
Tests the embedding API endpoint and provides detailed output.
"""

import requests
import json
import time
import sys
from datetime import datetime

def test_embedding_api():
    """Test the embedding API endpoint."""
    
    print("=" * 50)
    print("Regulatory RAG - Embedding API Test")
    print("=" * 50)
    print()
    
    # Configuration
    api_url = "http://localhost:5000/embed"
    timeout = 300  # 5 minutes timeout
    
    print(f"Configuration:")
    print(f"- API Endpoint: {api_url}")
    print(f"- Timeout: {timeout} seconds")
    print(f"- Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        print("Making API call to embedding endpoint...")
        print()
        
        # Start timing
        start_time = time.time()
        
        # Make the API call
        response = requests.post(
            api_url,
            json={},  # Empty JSON payload
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            timeout=timeout
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        print("=" * 50)
        print("API Response")
        print("=" * 50)
        print(f"Status Code: {response.status_code}")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"Response Headers: {dict(response.headers)}")
        print()
        
        if response.status_code == 200:
            print("SUCCESS: Embedding API call completed successfully!")
            print()
            
            try:
                response_data = response.json()
                print("Response Data:")
                print(json.dumps(response_data, indent=2))
                
                # Extract key information
                if isinstance(response_data, dict):
                    message = response_data.get('message', 'No message')
                    processed_regulations = response_data.get('processed_regulations', 0)
                    total_vectors = response_data.get('total_vectors_created', 0)
                    token_usage = response_data.get('token_usage', {})
                    
                    print()
                    print("=" * 50)
                    print("Summary")
                    print("=" * 50)
                    print(f"Message: {message}")
                    print(f"Processed Regulations: {processed_regulations}")
                    print(f"Total Vectors Created: {total_vectors}")
                    
                    if token_usage:
                        print(f"Token Usage:")
                        print(f"  - Total Tokens: {token_usage.get('total_tokens', 'N/A')}")
                        print(f"  - Total Cost: ${token_usage.get('total_cost_usd', 'N/A')}")
                        print(f"  - Model Used: {token_usage.get('model_used', 'N/A')}")
                        print(f"  - Avg Tokens per Chunk: {token_usage.get('avg_tokens_per_chunk', 'N/A')}")
                
            except json.JSONDecodeError:
                print("Response is not valid JSON:")
                print(response.text)
                
        else:
            print(f"ERROR: API call failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the embedding API")
        print("Please ensure the embedding API is running on port 5000")
        print("You can start it with: python embed_api.py")
        
    except requests.exceptions.Timeout:
        print(f"ERROR: Request timed out after {timeout} seconds")
        print("The embedding process may be taking longer than expected")
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
        
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
    
    print()
    print("=" * 50)
    print("Test completed")
    print("=" * 50)

def check_api_status():
    """Check if the API is running."""
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        return True
    except:
        return False

if __name__ == "__main__":
    print("Checking if embedding API is running...")
    
    if check_api_status():
        print("✓ Embedding API is running")
        print()
        test_embedding_api()
    else:
        print("✗ Embedding API is not running")
        print()
        print("To start the embedding API:")
        print("1. Open a terminal/command prompt")
        print("2. Navigate to the regaiagent directory")
        print("3. Run: python embed_api.py")
        print()
        print("Then run this test script again.")
    
    print()
    input("Press Enter to exit...")
