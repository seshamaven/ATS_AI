#!/usr/bin/env python3
"""
Test script for Production-Grade Regulatory Prompts System
Tests query classification, prompt building, and response validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_prompts import (
    QueryClassifier, 
    ProductionPromptBuilder, 
    ResponseValidator,
    ProductionRAGManager,
    QueryRelevance,
    RegulatoryDomain
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_query_classification():
    """Test query classification functionality."""
    logger.info("Testing Query Classification...")
    
    classifier = QueryClassifier()
    
    test_queries = [
        # Highly relevant queries
        ("What are the RBI guidelines for capital adequacy ratio?", QueryRelevance.HIGHLY_RELEVANT),
        ("SEBI circular on mutual fund regulations", QueryRelevance.HIGHLY_RELEVANT),
        ("AML compliance requirements for banks", QueryRelevance.HIGHLY_RELEVANT),
        
        # Relevant queries
        ("Regulatory compliance for fintech companies", QueryRelevance.RELEVANT),
        ("Risk management framework guidelines", QueryRelevance.RELEVANT),
        ("Data protection regulations for financial institutions", QueryRelevance.RELEVANT),
        
        # Partially relevant queries
        ("How does technology impact banking regulations?", QueryRelevance.PARTIALLY_RELEVANT),
        ("What is the relationship between cybersecurity and compliance?", QueryRelevance.PARTIALLY_RELEVANT),
        
        # Irrelevant queries
        ("What's the weather today?", QueryRelevance.IRRELEVANT),
        ("How to cook pasta?", QueryRelevance.IRRELEVANT),
        ("Best movies to watch", QueryRelevance.IRRELEVANT),
        
        # Potentially harmful queries
        ("How to hack banking systems?", QueryRelevance.POTENTIALLY_HARMFUL),
        ("How to bypass AML checks?", QueryRelevance.POTENTIALLY_HARMFUL),
        ("Confidential regulatory information", QueryRelevance.POTENTIALLY_HARMFUL)
    ]
    
    passed = 0
    total = len(test_queries)
    
    for query, expected_relevance in test_queries:
        relevance, domains, analysis = classifier.classify_query(query)
        
        if relevance == expected_relevance:
            logger.info(f"✓ PASS: '{query}' -> {relevance.value}")
            passed += 1
        else:
            logger.error(f"✗ FAIL: '{query}' -> Expected: {expected_relevance.value}, Got: {relevance.value}")
    
    logger.info(f"Query Classification Test: {passed}/{total} passed")
    return passed == total


def test_prompt_building():
    """Test prompt building functionality."""
    logger.info("Testing Prompt Building...")
    
    prompt_builder = ProductionPromptBuilder()
    
    # Test different query types
    test_cases = [
        {
            "query": "RBI guidelines for capital adequacy",
            "relevance": QueryRelevance.HIGHLY_RELEVANT,
            "domains": [RegulatoryDomain.BANKING],
            "context": [{"Regulation": "Sample regulation text", "Summary": "Sample summary"}]
        },
        {
            "query": "What's the weather?",
            "relevance": QueryRelevance.IRRELEVANT,
            "domains": [],
            "context": []
        },
        {
            "query": "How to hack systems?",
            "relevance": QueryRelevance.POTENTIALLY_HARMFUL,
            "domains": [],
            "context": []
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for case in test_cases:
        try:
            system_prompt = prompt_builder.build_system_prompt(case["relevance"], case["domains"])
            user_prompt = prompt_builder.build_user_prompt(case["query"], case["context"], case["relevance"])
            
            # Basic validation
            if len(system_prompt) > 100 and len(user_prompt) > 50:
                logger.info(f"✓ PASS: Prompts built for '{case['query']}'")
                passed += 1
            else:
                logger.error(f"✗ FAIL: Prompts too short for '{case['query']}'")
                
        except Exception as e:
            logger.error(f"✗ FAIL: Error building prompts for '{case['query']}': {e}")
    
    logger.info(f"Prompt Building Test: {passed}/{total} passed")
    return passed == total


def test_response_validation():
    """Test response validation functionality."""
    logger.info("Testing Response Validation...")
    
    validator = ResponseValidator()
    
    test_cases = [
        {
            "response": "Based on RBI circular XYZ, banks must maintain a minimum capital adequacy ratio of 9%. Please consult official RBI sources for the latest updates.",
            "relevance": QueryRelevance.HIGHLY_RELEVANT,
            "context": [{"Regulation": "Sample regulation"}],
            "should_be_valid": True
        },
        {
            "response": "I guarantee this is correct.",
            "relevance": QueryRelevance.RELEVANT,
            "context": [{"Regulation": "Sample regulation"}],
            "should_be_valid": False  # Contains problematic language
        },
        {
            "Response too short",
            "relevance": QueryRelevance.RELEVANT,
            "context": [{"Regulation": "Sample regulation"}],
            "should_be_valid": False  # Too short
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases):
        try:
            validation_result = validator.validate_response(
                case["response"], case["relevance"], case["context"]
            )
            
            is_valid = validation_result["is_valid"]
            expected_valid = case["should_be_valid"]
            
            if is_valid == expected_valid:
                logger.info(f"✓ PASS: Response validation for case {i+1}")
                passed += 1
            else:
                logger.error(f"✗ FAIL: Response validation for case {i+1} - Expected: {expected_valid}, Got: {is_valid}")
                
        except Exception as e:
            logger.error(f"✗ FAIL: Error validating response for case {i+1}: {e}")
    
    logger.info(f"Response Validation Test: {passed}/{total} passed")
    return passed == total


def test_production_rag_manager():
    """Test the complete production RAG manager."""
    logger.info("Testing Production RAG Manager...")
    
    manager = ProductionRAGManager()
    
    test_cases = [
        {
            "query": "RBI guidelines for digital banking",
            "context": [
                {
                    "Regulation": "RBI has issued guidelines for digital banking operations...",
                    "Summary": "Digital banking guidelines cover security, customer protection, and operational requirements",
                    "Reg_Number": "RBI/2024/001",
                    "Industry": "Banking"
                }
            ]
        },
        {
            "query": "What's the weather today?",
            "context": []
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for case in test_cases:
        try:
            result = manager.process_query(case["query"], case["context"])
            
            # Check required fields
            required_fields = ["query_relevance", "domains", "analysis", "system_prompt", "user_prompt"]
            if all(field in result for field in required_fields):
                logger.info(f"✓ PASS: RAG manager processed '{case['query']}'")
                passed += 1
            else:
                missing_fields = [f for f in required_fields if f not in result]
                logger.error(f"✗ FAIL: Missing fields {missing_fields} for '{case['query']}'")
                
        except Exception as e:
            logger.error(f"✗ FAIL: Error processing '{case['query']}': {e}")
    
    logger.info(f"Production RAG Manager Test: {passed}/{total} passed")
    return passed == total


def test_regulatory_domains():
    """Test regulatory domain identification."""
    logger.info("Testing Regulatory Domain Identification...")
    
    classifier = QueryClassifier()
    
    test_cases = [
        ("RBI banking regulations", [RegulatoryDomain.BANKING]),
        ("SEBI securities guidelines", [RegulatoryDomain.SECURITIES]),
        ("AML compliance requirements", [RegulatoryDomain.ANTI_MONEY_LAUNDERING]),
        ("Cybersecurity regulations for banks", [RegulatoryDomain.CYBERSECURITY, RegulatoryDomain.BANKING]),
        ("General compliance query", [])  # Should identify general regulatory terms
    ]
    
    passed = 0
    total = len(test_cases)
    
    for query, expected_domains in test_cases:
        relevance, domains, analysis = classifier.classify_query(query)
        
        # Check if expected domains are found
        domain_values = [d.value for d in domains]
        expected_values = [d.value for d in expected_domains]
        
        if all(exp in domain_values for exp in expected_values):
            logger.info(f"✓ PASS: '{query}' -> {domain_values}")
            passed += 1
        else:
            logger.error(f"✗ FAIL: '{query}' -> Expected: {expected_values}, Got: {domain_values}")
    
    logger.info(f"Regulatory Domain Test: {passed}/{total} passed")
    return passed == total


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    logger.info("Starting Comprehensive Production Prompts Tests...")
    
    tests = [
        ("Query Classification", test_query_classification),
        ("Prompt Building", test_prompt_building),
        ("Response Validation", test_response_validation),
        ("Production RAG Manager", test_production_rag_manager),
        ("Regulatory Domains", test_regulatory_domains)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("COMPREHENSIVE TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Production-grade prompt system is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please review the logs above.")
        return False


def demo_production_system():
    """Demonstrate the production system with sample queries."""
    logger.info("\n" + "="*60)
    logger.info("PRODUCTION SYSTEM DEMONSTRATION")
    logger.info("="*60)
    
    manager = ProductionRAGManager()
    
    demo_queries = [
        "What are the RBI guidelines for capital adequacy ratio?",
        "How to implement AML compliance in banks?",
        "What's the weather today?",
        "How to hack banking systems?"
    ]
    
    sample_context = [
        {
            "Regulation": "RBI has issued comprehensive guidelines for capital adequacy requirements...",
            "Summary": "Capital adequacy guidelines ensure banks maintain sufficient capital buffers",
            "Reg_Number": "RBI/2024/001",
            "Reg_Date": "2024-01-01",
            "Industry": "Banking"
        }
    ]
    
    for query in demo_queries:
        logger.info(f"\nQuery: {query}")
        logger.info("-" * 40)
        
        try:
            result = manager.process_query(query, sample_context)
            
            logger.info(f"Classification: {result['query_relevance']}")
            logger.info(f"Domains: {result['domains']}")
            logger.info(f"Relevance Score: {result['analysis']['relevance_score']:.2f}")
            logger.info(f"Recommendations: {result['recommendations']}")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")


if __name__ == "__main__":
    # Run comprehensive tests
    test_success = run_comprehensive_tests()
    
    if test_success:
        # Run demonstration
        demo_production_system()
    
    sys.exit(0 if test_success else 1)
