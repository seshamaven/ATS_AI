#!/usr/bin/env python3
"""
Test script to verify the .lower() error fix.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_lower_error_fix():
    """Test that the .lower() error is fixed."""
    
    print("Testing .lower() error fix...")
    
    try:
        from chatbot_api import EnhancedSearchManager
        
        # Create test metadata with float values that might cause the error
        test_metadata = {
            'reg_number': 12345.0,  # This is a float
            'regulator': 123.45,    # This is a float
            'regulation_title': 'Test Regulation',
            'summary': 'Test summary',
            'industry': 'Banking'
        }
        
        # Create EnhancedSearchManager instance
        search_manager = EnhancedSearchManager()
        
        # Test the calculate_keyword_match_score method that was causing the error
        print("Testing calculate_keyword_match_score...")
        score = search_manager.calculate_keyword_match_score("RBI regulation 12345", test_metadata)
        print(f"Keyword match score: {score}")
        
        # Test the calculate_authority_weight method that was also causing the error
        print("Testing calculate_authority_weight...")
        weight = search_manager.calculate_authority_weight(test_metadata)
        print(f"Authority weight: {weight}")
        
        print("SUCCESS: .lower() error fix is working!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing .lower() Error Fix")
    print("=" * 60)
    
    success = test_lower_error_fix()
    
    print("=" * 60)
    if success:
        print("SUCCESS: The .lower() error has been fixed!")
    else:
        print("ERROR: The .lower() error still exists!")
    print("=" * 60)
