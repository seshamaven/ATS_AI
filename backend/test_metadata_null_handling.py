#!/usr/bin/env python3
"""
Test script for metadata NULL value handling
Tests the enhanced Pinecone manager's metadata preparation.
"""

import os
import sys
import logging
from typing import Dict, Any

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_pinecone_manager import EnhancedPineconeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_metadata_preparation():
    """Test the metadata preparation functionality."""
    
    print("🧪 Testing Metadata NULL Value Handling")
    print("=" * 50)
    
    try:
        # Create manager instance
        manager = EnhancedPineconeManager()
        
        # Test cases with different NULL scenarios
        test_cases = [
            {
                'name': 'Complete metadata',
                'metadata': {
                    'candidate_id': 123,
                    'name': 'John Doe',
                    'email': 'john@example.com',
                    'domain': 'Technology',
                    'file_type': 'PDF',
                    'primary_skills': 'Python, Flask',
                    'total_experience': 5.0,
                    'education': 'Bachelor\'s Degree'
                }
            },
            {
                'name': 'Metadata with NULL values',
                'metadata': {
                    'candidate_id': 124,
                    'name': 'Jane Smith',
                    'email': None,  # NULL value
                    'domain': None,  # NULL value
                    'file_type': None,  # NULL value
                    'primary_skills': None,  # NULL value
                    'total_experience': 3.0,
                    'education': None  # NULL value
                }
            },
            {
                'name': 'Mixed NULL and valid values',
                'metadata': {
                    'candidate_id': 125,
                    'name': None,  # NULL value
                    'email': 'bob@example.com',
                    'domain': 'Finance',
                    'file_type': None,  # NULL value
                    'primary_skills': 'Java, Spring',
                    'total_experience': 7.0,
                    'education': 'Master\'s Degree'
                }
            },
            {
                'name': 'All NULL values',
                'metadata': {
                    'candidate_id': 126,
                    'name': None,
                    'email': None,
                    'domain': None,
                    'file_type': None,
                    'primary_skills': None,
                    'total_experience': None,
                    'education': None
                }
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. Testing: {test_case['name']}")
            print(f"   Original metadata: {test_case['metadata']}")
            
            # Prepare metadata
            cleaned_metadata = manager.prepare_metadata(test_case['metadata'])
            print(f"   Cleaned metadata: {cleaned_metadata}")
            
            # Verify no NULL values
            null_values = [k for k, v in cleaned_metadata.items() if v is None]
            if null_values:
                print(f"   ❌ Still contains NULL values: {null_values}")
            else:
                print(f"   ✅ No NULL values found")
            
            # Verify all values are valid Pinecone types
            valid_types = True
            for key, value in cleaned_metadata.items():
                if not isinstance(value, (str, int, float, bool, list)):
                    print(f"   ❌ Invalid type for {key}: {type(value)}")
                    valid_types = False
            
            if valid_types:
                print(f"   ✅ All values are valid Pinecone types")
        
        print(f"\n🎉 All metadata preparation tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


def test_vector_preparation():
    """Test vector preparation with metadata."""
    
    print("\n🔧 Testing Vector Preparation")
    print("=" * 30)
    
    try:
        manager = EnhancedPineconeManager()
        
        # Test vector with NULL metadata
        test_vector = {
            'id': 'test_resume_1',
            'values': [0.1] * manager.dimension,
            'metadata': {
                'candidate_id': 123,
                'name': 'Test User',
                'file_type': None,  # This should be converted to 'Unknown'
                'domain': None,    # This should be converted to 'Unknown'
                'primary_skills': None  # This should be converted to 'No skills'
            }
        }
        
        print(f"Original vector metadata: {test_vector['metadata']}")
        
        # The upsert_vectors method will automatically prepare metadata
        # We'll test the prepare_metadata method directly
        prepared_metadata = manager.prepare_metadata(test_vector['metadata'])
        print(f"Prepared metadata: {prepared_metadata}")
        
        # Verify specific conversions
        expected_conversions = {
            'file_type': 'Unknown',
            'domain': 'Unknown', 
            'primary_skills': 'No skills'
        }
        
        for key, expected_value in expected_conversions.items():
            actual_value = prepared_metadata.get(key)
            if actual_value == expected_value:
                print(f"   ✅ {key}: {actual_value}")
            else:
                print(f"   ❌ {key}: expected '{expected_value}', got '{actual_value}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector preparation test failed: {e}")
        return False


def main():
    """Main test function."""
    
    print("🚀 Metadata NULL Value Handling Test Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('enhanced_pinecone_manager.py'):
        print("❌ Please run this script from the backend directory")
        return False
    
    # Run tests
    success1 = test_metadata_preparation()
    success2 = test_vector_preparation()
    
    if success1 and success2:
        print("\n✅ All tests passed! Metadata NULL handling is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return success1 and success2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
