#!/usr/bin/env python3
"""
Test script to reproduce the exact error and identify the root cause.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_exact_error():
    """Test the exact scenario that's causing the error."""
    
    print("Testing the exact error scenario...")
    
    try:
        from production_prompts import classify_regulatory_query
        
        # Test the exact query that's causing the error
        user_query = "show RBI regulations that you have"
        
        print(f"Testing query: '{user_query}'")
        
        # This is where the error might be occurring
        query_relevance, domains, analysis = classify_regulatory_query(user_query)
        
        print(f"Query relevance: {query_relevance}")
        print(f"Domains: {domains}")
        print(f"Analysis: {analysis}")
        
        # Test the specific operations that might be causing the error
        print("\nTesting domain value extraction...")
        domain_values = [d.value for d in domains]
        print(f"Domain values: {domain_values}")
        
        # Test joining domain values
        print("\nTesting domain value joining...")
        domain_string = ', '.join(domain_values)
        print(f"Domain string: {domain_string}")
        
        # Test analysis dictionary operations
        print("\nTesting analysis dictionary...")
        for key, value in analysis.items():
            print(f"  {key}: {value} (type: {type(value)})")
        
        print("\nSUCCESS: No error occurred!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_building():
    """Test the context building that might be causing the error."""
    
    print("\nTesting context building...")
    
    try:
        from production_prompts import ProductionPromptBuilder
        
        # Create test context data with potential float values
        test_context = [
            {
                'Regulation': 'RBI Master Circular on Prudential Norms',
                'Summary': 'Guidelines for capital adequacy',
                'Reg_Number': 12345,  # This could be a float
                'Reg_Date': '2024-01-15',
                'Reg_Category': 'Circular',
                'Industry': 'Banking',
                'relevance_score': 0.85
            }
        ]
        
        builder = ProductionPromptBuilder()
        
        # Test building context string
        context_string = builder._build_context_string(test_context)
        print(f"Context string length: {len(context_string)}")
        
        print("SUCCESS: Context building worked!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Exact Error Scenario")
    print("=" * 60)
    
    # Test the exact error
    test1_passed = test_exact_error()
    
    # Test context building
    test2_passed = test_context_building()
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("SUCCESS: No errors found in the test scenarios.")
        print("The error might be occurring elsewhere in the code.")
    else:
        print("ERROR: Found the source of the error!")
    
    print("=" * 60)
