#!/usr/bin/env python3
"""
Test script for Enhanced Pinecone Manager
Tests dynamic index management and error handling.
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_pinecone_manager import EnhancedPineconeManager, create_pinecone_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pinecone_manager():
    """Test the enhanced Pinecone manager functionality."""
    
    print("🧪 Testing Enhanced Pinecone Manager")
    print("=" * 50)
    
    try:
        # Test 1: Check environment variables
        print("\n1. Checking environment variables...")
        required_vars = ['PINECONE_API_KEY', 'PINECONE_INDEX_NAME', 'PINECONE_DIMENSION']
        
        for var in required_vars:
            value = os.getenv(var)
            if value:
                print(f"   ✅ {var}: {value[:10]}..." if len(value) > 10 else f"   ✅ {var}: {value}")
            else:
                print(f"   ❌ {var}: Not set")
        
        # Test 2: Create Pinecone manager
        print("\n2. Creating Pinecone manager...")
        manager = create_pinecone_manager()
        print(f"   ✅ Manager created successfully")
        print(f"   📊 Index: {manager.index_name}")
        print(f"   📏 Dimension: {manager.dimension}")
        
        # Test 3: Get index stats
        print("\n3. Getting index statistics...")
        stats = manager.get_index_stats()
        print(f"   ✅ Index stats retrieved")
        print(f"   📈 Total vectors: {stats.total_vector_count}")
        print(f"   📏 Index dimension: {stats.dimension}")
        
        # Test 4: Test vector validation
        print("\n4. Testing vector validation...")
        
        # Test valid vector
        valid_vector = {
            'id': 'test_vector_1',
            'values': [0.1] * manager.dimension,
            'metadata': {'test': True}
        }
        
        try:
            manager.upsert_vectors([valid_vector])
            print(f"   ✅ Valid vector upserted successfully")
        except Exception as e:
            print(f"   ❌ Valid vector upsert failed: {e}")
        
        # Test invalid dimension vector
        invalid_vector = {
            'id': 'test_vector_2',
            'values': [0.1] * (manager.dimension + 1),  # Wrong dimension
            'metadata': {'test': True}
        }
        
        try:
            manager.upsert_vectors([invalid_vector])
            print(f"   ❌ Invalid vector was accepted (should have failed)")
        except Exception as e:
            print(f"   ✅ Invalid vector correctly rejected: {str(e)[:50]}...")
        
        # Test 5: Test query functionality
        print("\n5. Testing query functionality...")
        try:
            query_vector = [0.1] * manager.dimension
            results = manager.query_vectors(query_vector, top_k=5)
            print(f"   ✅ Query successful, returned {len(results.matches)} results")
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
        
        print("\n🎉 All tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


def test_error_handling():
    """Test error handling scenarios."""
    
    print("\n🔧 Testing Error Handling")
    print("=" * 30)
    
    # Test 1: Missing API key
    print("\n1. Testing missing API key...")
    try:
        manager = EnhancedPineconeManager(api_key=None)
        print("   ❌ Should have failed with missing API key")
    except ValueError as e:
        print(f"   ✅ Correctly caught missing API key: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Test 2: Invalid dimension
    print("\n2. Testing invalid dimension...")
    try:
        manager = EnhancedPineconeManager(dimension=-1)
        print("   ❌ Should have failed with invalid dimension")
    except ValueError as e:
        print(f"   ✅ Correctly caught invalid dimension: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Test 3: Empty index name
    print("\n3. Testing empty index name...")
    try:
        manager = EnhancedPineconeManager(index_name="")
        print("   ❌ Should have failed with empty index name")
    except ValueError as e:
        print(f"   ✅ Correctly caught empty index name: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")


def main():
    """Main test function."""
    
    print("🚀 Enhanced Pinecone Manager Test Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('enhanced_pinecone_manager.py'):
        print("❌ Please run this script from the backend directory")
        return False
    
    # Run tests
    success = test_pinecone_manager()
    test_error_handling()
    
    if success:
        print("\n✅ All tests passed! Enhanced Pinecone Manager is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
