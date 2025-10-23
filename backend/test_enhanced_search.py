"""
Test script for Enhanced Search Integration
Tests the new EnhancedSearchManager with reranking and intelligent query analysis.
"""

import requests
import json
import time
from typing import Dict, List, Any

# API endpoints
CHAT_API_URL = "http://localhost:5001"

def test_enhanced_search_integration():
    """Test the enhanced search integration in all endpoints."""
    print("Testing Enhanced Search Integration")
    print("=" * 50)
    
    # Test queries that should benefit from enhanced search
    test_queries = [
        {
            "query": "RBI co-lending arrangements",
            "description": "Should extract RBI regulator and banking industry",
            "expected_features": ["regulator_extraction", "industry_extraction", "reranking"]
        },
        {
            "query": "SEBI cybersecurity guidelines for stock brokers",
            "description": "Should extract SEBI regulator, capital markets industry, and guidelines type",
            "expected_features": ["regulator_extraction", "industry_extraction", "regulation_type_extraction"]
        },
        {
            "query": "What are the latest RBI circulars on NBFC compliance?",
            "description": "Should prioritize recent RBI circulars and NBFC-related content",
            "expected_features": ["recency_scoring", "authority_weighting", "keyword_matching"]
        },
        {
            "query": "IRDAI notification RBI/2023-24/123",
            "description": "Should extract regulatory number and IRDAI regulator",
            "expected_features": ["regulatory_number_extraction", "regulator_extraction"]
        }
    ]
    
    # Test each endpoint
    endpoints = [
        {
            "name": "Chat Endpoint",
            "url": f"{CHAT_API_URL}/chat",
            "description": "LLM-powered chatbot with enhanced search"
        },
        {
            "name": "Compare Endpoint", 
            "url": f"{CHAT_API_URL}/compare",
            "description": "Comprehensive data comparison with enhanced search"
        },
        {
            "name": "Search Endpoint",
            "url": f"{CHAT_API_URL}/search", 
            "description": "Metadata-aware search with enhanced search"
        }
    ]
    
    for endpoint in endpoints:
        print(f"\n{'='*60}")
        print(f"Testing {endpoint['name']}")
        print(f"Description: {endpoint['description']}")
        print(f"{'='*60}")
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\nTest {i}: {test_case['query']}")
            print(f"Description: {test_case['description']}")
            print(f"Expected Features: {', '.join(test_case['expected_features'])}")
            
            try:
                # Prepare request payload
                if endpoint['name'] == "Search Endpoint":
                    payload = {
                        "query": test_case['query'],
                        "filters": {}  # Let enhanced search handle intelligent filtering
                    }
                else:
                    payload = {"query": test_case['query']}
                
                # Make API request
                response = requests.post(
                    endpoint['url'],
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ SUCCESS: {endpoint['name']} completed")
                    
                    # Analyze enhanced search features
                    analyze_enhanced_features(result, test_case['expected_features'])
                    
                else:
                    print(f"❌ FAILED: {endpoint['name']} failed with status {response.status_code}")
                    print(f"Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                print(f"❌ CONNECTION ERROR: Cannot connect to {endpoint['name']}")
                print("Make sure chatbot_api.py is running on port 5001")
            except Exception as e:
                print(f"❌ ERROR: {e}")
            
            # Small delay between requests
            time.sleep(1)
    
    print(f"\n{'='*60}")
    print("Enhanced Search Integration Test Complete")
    print(f"{'='*60}")

def analyze_enhanced_features(result: Dict[str, Any], expected_features: List[str]):
    """Analyze the response for enhanced search features."""
    
    # Check for reranking scores
    if 'rerank_scores' in str(result):
        print("  ✅ Reranking: Found rerank scores in results")
    elif 'rerank_score' in str(result):
        print("  ✅ Reranking: Found rerank scores in results")
    else:
        print("  ⚠️  Reranking: No rerank scores found")
    
    # Check for query analysis
    if 'query_analysis' in str(result):
        print("  ✅ Query Analysis: Found intelligent query analysis")
    else:
        print("  ⚠️  Query Analysis: No query analysis found")
    
    # Check for enhanced search manager usage
    if 'enhanced search' in str(result).lower():
        print("  ✅ Enhanced Search: Enhanced search manager used")
    else:
        print("  ⚠️  Enhanced Search: Enhanced search manager usage unclear")
    
    # Check for intelligent filtering
    if 'suggested_filters' in str(result):
        print("  ✅ Intelligent Filtering: Found suggested filters")
    else:
        print("  ⚠️  Intelligent Filtering: No suggested filters found")
    
    # Check for authority weighting
    if 'authority' in str(result).lower():
        print("  ✅ Authority Weighting: Found authority-related scoring")
    else:
        print("  ⚠️  Authority Weighting: No authority weighting found")
    
    # Check for recency scoring
    if 'recency' in str(result).lower():
        print("  ✅ Recency Scoring: Found recency-related scoring")
    else:
        print("  ⚠️  Recency Scoring: No recency scoring found")
    
    # Check for keyword matching
    if 'keyword' in str(result).lower():
        print("  ✅ Keyword Matching: Found keyword-related scoring")
    else:
        print("  ⚠️  Keyword Matching: No keyword matching found")

def test_search_performance():
    """Test search performance with enhanced features."""
    print(f"\n{'='*60}")
    print("Testing Enhanced Search Performance")
    print(f"{'='*60}")
    
    performance_queries = [
        "RBI regulations",
        "SEBI compliance requirements", 
        "IRDAI insurance guidelines",
        "banking sector regulations",
        "capital markets compliance"
    ]
    
    for query in performance_queries:
        print(f"\nTesting: {query}")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{CHAT_API_URL}/search",
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                search_results = result.get('search_results', [])
                
                print(f"  ✅ Response Time: {response_time:.2f} seconds")
                print(f"  ✅ Results Found: {len(search_results)}")
                
                if search_results:
                    top_result = search_results[0]
                    print(f"  ✅ Top Result: {top_result.get('regulation_title', 'N/A')[:50]}...")
                    
                    # Check for enhanced features
                    if 'rerank_score' in top_result:
                        print(f"  ✅ Rerank Score: {top_result['rerank_score']:.3f}")
                    
                    if 'pinecone_score' in top_result:
                        print(f"  ✅ Pinecone Score: {top_result['pinecone_score']:.3f}")
                
            else:
                print(f"  ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_fallback_behavior():
    """Test fallback search behavior."""
    print(f"\n{'='*60}")
    print("Testing Fallback Search Behavior")
    print(f"{'='*60}")
    
    # Test queries that might not have exact matches
    fallback_queries = [
        "obscure financial regulation that probably doesn't exist",
        "very specific compliance requirement with unique terms",
        "hypothetical regulatory scenario"
    ]
    
    for query in fallback_queries:
        print(f"\nTesting Fallback: {query}")
        
        try:
            response = requests.post(
                f"{CHAT_API_URL}/chat",
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check if fallback was used
                if 'fallback' in str(result).lower() or 'no relevant regulations' in str(result).lower():
                    print("  ✅ Fallback: Fallback behavior triggered")
                else:
                    print("  ✅ Fallback: Found relevant results (no fallback needed)")
                
                # Check LLM response
                llm_response = result.get('llm_response', '')
                if llm_response:
                    print(f"  ✅ LLM Response: {llm_response[:100]}...")
                
            else:
                print(f"  ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def main():
    """Run all enhanced search tests."""
    print("Enhanced Search Integration Test Suite")
    print("=" * 70)
    print("Testing the integration of EnhancedPineconeSearchManager")
    print("with reranking, intelligent query analysis, and fallback search")
    print("=" * 70)
    
    # Test enhanced search integration
    test_enhanced_search_integration()
    
    # Test search performance
    test_search_performance()
    
    # Test fallback behavior
    test_fallback_behavior()
    
    print(f"\n{'='*70}")
    print("Enhanced Search Integration Test Suite Complete")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Review the test results above")
    print("2. Check for any missing enhanced features")
    print("3. Optimize reranking weights if needed")
    print("4. Monitor search performance in production")
    print("5. Collect user feedback on search quality")

if __name__ == "__main__":
    main()
