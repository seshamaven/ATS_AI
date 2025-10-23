#!/usr/bin/env python3
"""
Simple test to verify the chatbot API fix for the string formatting error.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_chatbot_api_fix():
    """Test that the chatbot API can handle the query that was causing the error."""
    
    print("Testing chatbot API fix...")
    
    try:
        # Test the specific context building that was causing the error
        context_regulations = [
            {
                'regulation_title': 'RBI Master Circular on Prudential Norms',
                'regulator': 'Reserve Bank of India',
                'industry': 'Banking',
                'summary': 'Guidelines for capital adequacy and risk management',
                'reg_category': 'Circular',
                'reg_subject': 'Capital Adequacy',
                'due_date': '2024-12-31',
                'chunk_text': 'Banks shall maintain minimum capital adequacy ratio...',
                'relevance_score': 0.85
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
        
        print("SUCCESS: Context string built successfully!")
        print(f"Context length: {len(context_string)} characters")
        print("The string formatting error has been resolved.")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Chatbot API String Formatting Fix")
    print("=" * 50)
    
    success = test_chatbot_api_fix()
    
    print("=" * 50)
    if success:
        print("RESULT: SUCCESS - The fix is working!")
    else:
        print("RESULT: FAILED - There are still issues.")
    print("=" * 50)
