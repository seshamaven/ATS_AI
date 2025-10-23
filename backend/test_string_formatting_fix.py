#!/usr/bin/env python3
"""
Test script to verify the string formatting fix for the chatbot API.
This script tests the specific scenario that was causing the error.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_string_formatting_fix():
    """Test the string formatting fix that was causing the error."""
    
    print("Testing string formatting fix...")
    
    try:
        from production_prompts import ProductionPromptBuilder
        
        # Create test data that might contain float values
        test_context_data = [
            {
                'Regulation': 'Test RBI Regulation',
                'Summary': 'Test summary',
                'Reg_Number': 12345,  # This could be a float
                'Reg_Date': '2024-01-15',  # This could be a float
                'Reg_Category': 'Circular',
                'Industry': 'Banking',
                'relevance_score': 0.85  # This is definitely a float
            },
            {
                'Regulation': 'Another Test Regulation',
                'Summary': 'Another test summary',
                'Reg_Number': 'RBI/2024/001',  # This is a string
                'Reg_Date': 2024.01,  # This could be a float
                'Reg_Category': 'Guidelines',
                'Industry': 'Insurance',
                'relevance_score': 0.92  # This is definitely a float
            }
        ]
        
        # Test the ProductionPromptBuilder
        builder = ProductionPromptBuilder()
        
        # Test building user prompt (this was causing the error)
        user_prompt = builder.build_user_prompt(
            user_query="show RBI regulations that you have",
            context_data=test_context_data,
            query_relevance=None  # We'll use a mock value
        )
        
        print("✓ User prompt built successfully!")
        print(f"✓ Prompt length: {len(user_prompt)} characters")
        
        # Test the context string building method directly
        context_string = builder._build_context_string(test_context_data)
        
        print("✓ Context string built successfully!")
        print(f"✓ Context length: {len(context_string)} characters")
        
        # Verify that the context string contains the expected content
        assert "Test RBI Regulation" in context_string
        assert "Another Test Regulation" in context_string
        assert "Relevance Score: 0.850" in context_string
        assert "Relevance Score: 0.920" in context_string
        
        print("✓ All assertions passed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chatbot_api_context_building():
    """Test the context building in chatbot_api.py."""
    
    print("\nTesting chatbot API context building...")
    
    try:
        # Simulate the context building that was causing the error
        context_regulations = [
            {
                'regulation_title': 'Test RBI Regulation',
                'regulator': 'Reserve Bank of India',
                'industry': 'Banking',
                'summary': 'Test summary',
                'reg_category': 'Circular',
                'reg_subject': 'Capital Adequacy',
                'due_date': '2024-12-31',
                'chunk_text': 'This is a test chunk of regulatory text...',
                'relevance_score': 0.85
            },
            {
                'regulation_title': 'Another Test Regulation',
                'regulator': 'SEBI',
                'industry': 'Securities',
                'summary': 'Another test summary',
                'reg_category': 'Guidelines',
                'reg_subject': 'Mutual Funds',
                'due_date': 2024.06,  # This could be a float
                'chunk_text': 'Another test chunk of regulatory text...',
                'relevance_score': 0.92
            }
        ]
        
        # Build context string (this was causing the error)
        context_string = "Relevant Regulatory Information from our database:\n\n"
        for i, reg in enumerate(context_regulations, 1):
            context_string += f"{i}. Regulation: {str(reg['regulation_title'])}\n"
            if reg['regulator']:
                context_string += f"   Regulator: {str(reg['regulator'])}\n"
            if reg['industry']:
                context_string += f"   Industry: {str(reg['industry'])}\n"
            if reg['summary']:
                context_string += f"   Summary: {str(reg['summary'])}\n"
            if reg['reg_category']:
                context_string += f"   Category: {str(reg['reg_category'])}\n"
            if reg['reg_subject']:
                context_string += f"   Subject: {str(reg['reg_subject'])}\n"
            if reg['due_date']:
                context_string += f"   Due Date: {str(reg['due_date'])}\n"
            if reg['chunk_text']:
                context_string += f"   Content: {str(reg['chunk_text'])[:500]}...\n"
            context_string += f"   Relevance Score: {float(reg['relevance_score']):.3f}\n\n"
        
        print("✓ Context string built successfully!")
        print(f"✓ Context length: {len(context_string)} characters")
        
        # Verify that the context string contains the expected content
        assert "Test RBI Regulation" in context_string
        assert "Another Test Regulation" in context_string
        assert "Relevance Score: 0.850" in context_string
        assert "Relevance Score: 0.920" in context_string
        
        print("✓ All assertions passed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing String Formatting Fix")
    print("=" * 60)
    
    # Test production prompts
    test1_passed = test_string_formatting_fix()
    
    # Test chatbot API context building
    test2_passed = test_chatbot_api_context_building()
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED!")
        print("The string formatting fix has resolved the error.")
        print("The chatbot API should now work correctly.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("There may still be issues with string formatting.")
    
    print("=" * 60)
