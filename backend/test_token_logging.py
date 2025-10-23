#!/usr/bin/env python3
"""
Test script to check the token logging functionality that might be causing the error.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_token_logging():
    """Test the token logging functionality."""
    
    print("Testing token logging functionality...")
    
    try:
        from token_tracker import log_query_embedding_tokens
        from production_prompts import classify_regulatory_query
        
        # Test the exact scenario from the error
        user_query = "show RBI regulations that you have"
        
        # Classify the query
        query_relevance, domains, analysis = classify_regulatory_query(user_query)
        
        print(f"Query relevance: {query_relevance}")
        print(f"Domains: {domains}")
        print(f"Analysis: {analysis}")
        
        # Test the token logging call that might be causing the error
        print("\nTesting token logging call...")
        
        # This is the exact call that might be causing the error
        result = log_query_embedding_tokens(
            model_name="text-embedding-ada-002",
            input_tokens=100,
            user_query=user_query,
            query_relevance=query_relevance.value,
            regulatory_domains=[d.value for d in domains],
            metadata={
                "analysis": analysis,
                "processing_time_ms": 1000,
                "vectors_found": 5
            }
        )
        
        print(f"Token logging result: {result}")
        print("SUCCESS: Token logging worked!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_regulatory_domains_list():
    """Test the regulatory domains list specifically."""
    
    print("\nTesting regulatory domains list...")
    
    try:
        from production_prompts import classify_regulatory_query
        
        user_query = "show RBI regulations that you have"
        query_relevance, domains, analysis = classify_regulatory_query(user_query)
        
        print(f"Domains: {domains}")
        print(f"Domain types: {[type(d) for d in domains]}")
        
        # Test domain value extraction
        domain_values = [d.value for d in domains]
        print(f"Domain values: {domain_values}")
        print(f"Domain value types: {[type(v) for v in domain_values]}")
        
        # Test joining domain values
        domain_string = ', '.join(domain_values)
        print(f"Domain string: {domain_string}")
        
        print("SUCCESS: Regulatory domains list test passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Token Logging Functionality")
    print("=" * 60)
    
    # Test regulatory domains list
    test1_passed = test_regulatory_domains_list()
    
    # Test token logging
    test2_passed = test_token_logging()
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("SUCCESS: No errors found in token logging.")
    else:
        print("ERROR: Found the source of the error!")
    
    print("=" * 60)
