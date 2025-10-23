#!/usr/bin/env python3
"""
Test script to actually call the chatbot API and reproduce the error.
"""

import requests
import json
import time

def test_chatbot_api_call():
    """Test the actual chatbot API call that's causing the error."""
    
    print("Testing actual chatbot API call...")
    
    try:
        # Test the exact query that's causing the error
        user_query = "show RBI regulations that you have"
        
        print(f"Testing query: '{user_query}'")
        
        # Make the API call
        response = requests.post(
            "http://localhost:5001/chat",
            json={"query": user_query},
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("SUCCESS: API call worked!")
            print(f"Response keys: {list(response_data.keys())}")
            return True
        else:
            print(f"ERROR: API call failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to chatbot API")
        print("Please ensure the chatbot API is running on port 5001")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Actual Chatbot API Call")
    print("=" * 60)
    
    success = test_chatbot_api_call()
    
    print("=" * 60)
    if success:
        print("SUCCESS: API call worked!")
    else:
        print("ERROR: API call failed!")
    print("=" * 60)
