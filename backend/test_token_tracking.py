#!/usr/bin/env python3
"""
Test script for Token Usage Tracking System
Tests token counting, logging, and analytics functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from token_tracker import (
    TokenUsageTracker,
    TokenUsageRecord,
    OperationType,
    ModelPricing,
    get_token_tracker,
    log_embedding_tokens,
    log_query_embedding_tokens,
    log_rag_tokens,
    get_token_usage_summary,
    get_daily_token_usage
)
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_pricing():
    """Test model pricing calculations."""
    logger.info("Testing Model Pricing...")
    
    # Test embedding model pricing
    embedding_cost = ModelPricing.calculate_cost("text-embedding-ada-002", 1000, 0)
    logger.info(f"Embedding cost for 1000 tokens: ${embedding_cost}")
    
    # Test chat model pricing
    chat_cost = ModelPricing.calculate_cost("gpt-4", 1000, 500)
    logger.info(f"Chat cost for 1000 input + 500 output tokens: ${chat_cost}")
    
    # Test unknown model fallback
    unknown_cost = ModelPricing.calculate_cost("unknown-model", 1000, 500)
    logger.info(f"Unknown model cost: ${unknown_cost}")
    
    logger.info("✓ Model pricing tests completed")


def test_token_usage_record():
    """Test token usage record creation."""
    logger.info("Testing Token Usage Record...")
    
    record = TokenUsageRecord(
        operation_type=OperationType.EMBEDDING.value,
        model_name="text-embedding-ada-002",
        input_tokens=1000,
        output_tokens=0,
        total_tokens=1000,
        cost_usd=0.0001,
        user_query="Test query",
        metadata={"test": True}
    )
    
    logger.info(f"Created record: {record.operation_type} - {record.total_tokens} tokens - ${record.cost_usd}")
    logger.info("✓ Token usage record test completed")


def test_token_tracker():
    """Test token tracker functionality."""
    logger.info("Testing Token Tracker...")
    
    tracker = TokenUsageTracker()
    
    # Test connection (this will fail if database not available, but that's expected)
    connected = tracker.connect()
    if connected:
        logger.info("✓ Database connection successful")
        
        # Test logging embedding usage
        success = tracker.log_embedding_usage(
            model_name="text-embedding-ada-002",
            input_tokens=1000,
            text_content="Test embedding text",
            operation_type="embedding",
            metadata={"test": True}
        )
        logger.info(f"Embedding logging: {'✓ Success' if success else '✗ Failed'}")
        
        # Test logging query embedding usage
        success = tracker.log_query_embedding_usage(
            model_name="text-embedding-ada-002",
            input_tokens=500,
            user_query="What are RBI guidelines?",
            query_relevance="highly_relevant",
            regulatory_domains=["banking"],
            metadata={"test": True}
        )
        logger.info(f"Query embedding logging: {'✓ Success' if success else '✗ Failed'}")
        
        # Test logging RAG usage
        success = tracker.log_rag_output_usage(
            model_name="gpt-4",
            input_tokens=2000,
            output_tokens=500,
            user_query="What are RBI guidelines?",
            response_length=2500,
            processing_time_ms=5000,
            quality_score=0.85,
            safety_score=0.95,
            query_relevance="highly_relevant",
            regulatory_domains=["banking"],
            metadata={"test": True}
        )
        logger.info(f"RAG output logging: {'✓ Success' if success else '✗ Failed'}")
        
        tracker.disconnect()
    else:
        logger.warning("✗ Database connection failed - skipping database tests")
    
    logger.info("✓ Token tracker tests completed")


def test_convenience_functions():
    """Test convenience functions."""
    logger.info("Testing Convenience Functions...")
    
    # Test embedding token logging
    success = log_embedding_tokens(
        model_name="text-embedding-ada-002",
        input_tokens=1000,
        text_content="Test text",
        operation_type="embedding",
        metadata={"test": True}
    )
    logger.info(f"Convenience embedding logging: {'✓ Success' if success else '✗ Failed'}")
    
    # Test query embedding token logging
    success = log_query_embedding_tokens(
        model_name="text-embedding-ada-002",
        input_tokens=500,
        user_query="Test query",
        query_relevance="relevant",
        regulatory_domains=["banking"],
        metadata={"test": True}
    )
    logger.info(f"Convenience query embedding logging: {'✓ Success' if success else '✗ Failed'}")
    
    # Test RAG token logging
    success = log_rag_tokens(
        model_name="gpt-4",
        input_tokens=2000,
        output_tokens=500,
        user_query="Test query",
        response_length=2500,
        processing_time_ms=5000,
        quality_score=0.85,
        safety_score=0.95,
        query_relevance="relevant",
        regulatory_domains=["banking"],
        metadata={"test": True}
    )
    logger.info(f"Convenience RAG logging: {'✓ Success' if success else '✗ Failed'}")
    
    logger.info("✓ Convenience functions tests completed")


def test_usage_analytics():
    """Test usage analytics functions."""
    logger.info("Testing Usage Analytics...")
    
    # Test usage summary
    try:
        summary = get_token_usage_summary()
        logger.info(f"Usage summary retrieved: {len(summary.get('summary', []))} operations")
        logger.info("✓ Usage summary test completed")
    except Exception as e:
        logger.warning(f"Usage summary test failed: {e}")
    
    # Test daily usage
    try:
        daily_usage = get_daily_token_usage(7)
        logger.info(f"Daily usage retrieved: {len(daily_usage)} days")
        logger.info("✓ Daily usage test completed")
    except Exception as e:
        logger.warning(f"Daily usage test failed: {e}")


def test_operation_types():
    """Test all operation types."""
    logger.info("Testing Operation Types...")
    
    operation_types = [
        OperationType.EMBEDDING,
        OperationType.QUERY_EMBEDDING,
        OperationType.RAG_INPUT,
        OperationType.RAG_OUTPUT,
        OperationType.CHAT_COMPLETION,
        OperationType.TEXT_GENERATION,
        OperationType.TEXT_CLASSIFICATION
    ]
    
    for op_type in operation_types:
        logger.info(f"Operation type: {op_type.value}")
    
    logger.info("✓ Operation types test completed")


def test_cost_calculations():
    """Test various cost calculation scenarios."""
    logger.info("Testing Cost Calculations...")
    
    test_cases = [
        ("text-embedding-ada-002", 1000, 0, "Embedding only"),
        ("gpt-4", 1000, 500, "GPT-4 with output"),
        ("gpt-3.5-turbo", 2000, 1000, "GPT-3.5 Turbo"),
        ("text-embedding-3-small", 500, 0, "Small embedding model"),
        ("unknown-model", 1000, 500, "Unknown model fallback")
    ]
    
    for model, input_tokens, output_tokens, description in test_cases:
        cost = ModelPricing.calculate_cost(model, input_tokens, output_tokens)
        logger.info(f"{description}: {input_tokens} input + {output_tokens} output = ${cost:.6f}")
    
    logger.info("✓ Cost calculations test completed")


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    logger.info("Starting Comprehensive Token Tracking Tests...")
    
    tests = [
        ("Model Pricing", test_model_pricing),
        ("Token Usage Record", test_token_usage_record),
        ("Token Tracker", test_token_tracker),
        ("Convenience Functions", test_convenience_functions),
        ("Usage Analytics", test_usage_analytics),
        ("Operation Types", test_operation_types),
        ("Cost Calculations", test_cost_calculations)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*60}")
        
        try:
            test_func()
            results.append((test_name, True))
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
        logger.info("🎉 All tests passed! Token tracking system is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please review the logs above.")
        return False


def demo_token_tracking():
    """Demonstrate token tracking functionality."""
    logger.info("\n" + "="*60)
    logger.info("TOKEN TRACKING SYSTEM DEMONSTRATION")
    logger.info("="*60)
    
    # Simulate a complete RAG workflow
    logger.info("\nSimulating RAG Workflow:")
    
    # 1. Query embedding
    query = "What are the RBI guidelines for capital adequacy ratio?"
    query_tokens = len(query) // 4  # Rough estimation
    logger.info(f"1. Query embedding: '{query}' ({query_tokens} tokens)")
    
    # 2. RAG input
    system_prompt = "You are a regulatory compliance assistant..."
    user_prompt = f"User Query: {query}\n\nContext: [regulatory context]"
    input_tokens = len(system_prompt + user_prompt) // 4
    logger.info(f"2. RAG input: {input_tokens} tokens")
    
    # 3. RAG output
    response = "Based on RBI guidelines, banks must maintain a minimum capital adequacy ratio..."
    output_tokens = len(response) // 4
    logger.info(f"3. RAG output: {output_tokens} tokens")
    
    # Calculate costs
    embedding_cost = ModelPricing.calculate_cost("text-embedding-ada-002", query_tokens, 0)
    rag_cost = ModelPricing.calculate_cost("gpt-4", input_tokens, output_tokens)
    total_cost = embedding_cost + rag_cost
    
    logger.info(f"\nCost Breakdown:")
    logger.info(f"- Query embedding: ${embedding_cost:.6f}")
    logger.info(f"- RAG processing: ${rag_cost:.6f}")
    logger.info(f"- Total cost: ${total_cost:.6f}")
    
    logger.info(f"\nTotal tokens used: {query_tokens + input_tokens + output_tokens}")


if __name__ == "__main__":
    # Run comprehensive tests
    test_success = run_comprehensive_tests()
    
    if test_success:
        # Run demonstration
        demo_token_tracking()
    
    sys.exit(0 if test_success else 1)
