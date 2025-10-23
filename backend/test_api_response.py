#!/usr/bin/env python3
"""
Test script to simulate the exact API call that's causing the error.
"""

import sys
import os
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_api_response_building():
    """Test the API response building that might be causing the error."""
    
    print("Testing API response building...")
    
    try:
        from production_prompts import classify_regulatory_query, ResponseValidator
        from datetime import datetime
        
        # Simulate the exact scenario from the error
        user_query = "show RBI regulations that you have"
        
        # Classify the query
        query_relevance, domains, analysis = classify_regulatory_query(user_query)
        
        # Simulate context_regulations with potential float values
        context_regulations = [
            {
                'vector_id': 'test_vector_1',
                'regulation_title': 'RBI Master Circular on Prudential Norms',
                'summary': 'Guidelines for capital adequacy',
                'regulator': 'Reserve Bank of India',
                'industry': 'Banking',
                'due_date': '2024-12-31',
                'reg_category': 'Circular',
                'reg_subject': 'Capital Adequacy',
                'risk_category': 'High',
                'department': 'Risk Management',
                'chunk_text': 'Banks shall maintain minimum capital adequacy ratio...',
                'chunk_index': 0,
                'total_chunks': 1,
                'relevance_score': 0.85,  # This is a float
                'rerank_score': 0.92     # This is a float
            }
        ]
        
        # Simulate response validation
        response_validator = ResponseValidator()
        validation_result = response_validator.validate_response(
            "Test response", query_relevance, context_regulations
        )
        
        # Build the exact response data structure that might be causing the error
        response_data = {
            'message': 'Chat response generated using production-grade regulatory prompts',
            'user_query': user_query,
            'llm_response': 'Test response',
            'context_used': context_regulations,
            'total_vectors_found': 1,
            'context_regulations_used': 1,
            'query_classification': {
                'relevance': query_relevance.value,
                'domains': [d.value for d in domains],
                'analysis': analysis
            },
            'response_validation': validation_result,
            'processing_time_ms': 1000.0,  # This is a float
            'token_usage': {
                'query_embedding_tokens': 100,
                'rag_input_tokens': 200,
                'rag_output_tokens': 150,
                'total_tokens': 450,
                'models_used': ['text-embedding-ada-002', 'gpt-3.5-turbo']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print("Response data built successfully!")
        
        # Test JSON serialization (this might be where the error occurs)
        print("Testing JSON serialization...")
        json_string = json.dumps(response_data, indent=2)
        print(f"JSON serialization successful! Length: {len(json_string)}")
        
        # Test specific parts that might be causing issues
        print("\nTesting specific parts...")
        
        # Test domains list
        domains_list = [d.value for d in domains]
        print(f"Domains list: {domains_list}")
        print(f"Domains types: {[type(d) for d in domains_list]}")
        
        # Test analysis dictionary
        print(f"Analysis: {analysis}")
        for key, value in analysis.items():
            print(f"  {key}: {value} (type: {type(value)})")
        
        # Test context_regulations
        print(f"Context regulations count: {len(context_regulations)}")
        for i, reg in enumerate(context_regulations):
            print(f"  Regulation {i}: {reg.get('regulation_title', 'No title')}")
            for key, value in reg.items():
                print(f"    {key}: {value} (type: {type(value)})")
        
        print("SUCCESS: API response building worked!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing API Response Building")
    print("=" * 60)
    
    success = test_api_response_building()
    
    print("=" * 60)
    if success:
        print("SUCCESS: No errors found in API response building.")
    else:
        print("ERROR: Found the source of the error!")
    print("=" * 60)
