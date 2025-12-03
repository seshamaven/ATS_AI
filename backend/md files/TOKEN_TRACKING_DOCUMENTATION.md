# Token Usage Tracking System Documentation

## Overview

This document describes the comprehensive token usage tracking system implemented for the Regulatory RAG system. The system tracks OpenAI API token usage across all operations to monitor costs, optimize performance, and provide detailed analytics.

## System Components

### 1. Database Schema (`token_usage_schema.sql`)

The `token_usage` table tracks all API token usage with the following key fields:

- **Operation Details**: `operation_type`, `model_name`, `input_tokens`, `output_tokens`, `total_tokens`
- **Cost Tracking**: `cost_usd` with automatic calculation based on current OpenAI pricing
- **Query Context**: `user_query`, `query_relevance`, `regulatory_domains`, `context_count`
- **Performance Metrics**: `processing_time_ms`, `quality_score`, `safety_score`
- **User Tracking**: `session_id`, `user_id`, `request_id`
- **Metadata**: `metadata` JSON field for additional context

### 2. Token Tracker (`token_tracker.py`)

Core tracking system with the following classes:

#### `ModelPricing`
- Maintains current OpenAI pricing for all models
- Automatically calculates costs based on token usage
- Supports embedding models, GPT models, and fallback pricing

#### `TokenUsageTracker`
- Database operations for logging token usage
- Batch processing capabilities
- Comprehensive analytics and reporting functions

#### `TokenUsageRecord`
- Structured data class for token usage records
- Type-safe data handling
- Easy serialization for database storage

### 3. Operation Types

The system tracks the following operation types:

- **`embedding`**: Document embedding operations
- **`query_embedding`**: User query embedding
- **`rag_input`**: RAG system input processing
- **`rag_output`**: RAG system output generation
- **`chat_completion`**: Direct chat completions
- **`text_generation`**: General text generation
- **`text_classification`**: Text classification operations

## Integration Points

### 1. Embedding Operations (`embed_api.py`)

Token logging integrated into the document embedding process:

```python
# Count tokens for each chunk
chunk_tokens = text_processor.count_tokens(chunk)

# Log token usage
log_embedding_tokens(
    model_name="text-embedding-ada-002",
    input_tokens=chunk_tokens,
    text_content=chunk[:100],
    operation_type="embedding",
    metadata={
        "row_id": row_id,
        "chunk_index": i,
        "total_chunks": len(chunks)
    }
)
```

### 2. Query Processing (`chatbot_api.py`)

Comprehensive token logging throughout the RAG pipeline:

#### Query Embedding
```python
# Log query embedding token usage
query_tokens = token_counter.count_embedding_tokens(user_query)
log_query_embedding_tokens(
    model_name="text-embedding-ada-002",
    input_tokens=query_tokens,
    user_query=user_query,
    query_relevance=query_relevance.value,
    regulatory_domains=[d.value for d in domains]
)
```

#### RAG Input/Output
```python
# Log RAG input token usage
input_tokens = token_counter.count_chat_tokens(system_prompt + user_prompt)
log_rag_tokens(
    model_name=Config.OPENAI_MODEL,
    input_tokens=input_tokens,
    output_tokens=0,  # Updated after response
    user_query=user_query,
    response_length=0,  # Updated after response
    processing_time_ms=(time.time() - start_time) * 1000
)

# After LLM response
output_tokens = token_counter.count_chat_tokens(llm_response)
log_rag_tokens(
    model_name=Config.OPENAI_MODEL,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    user_query=user_query,
    response_length=len(llm_response),
    processing_time_ms=(time.time() - start_time) * 1000
)
```

## API Endpoints

### 1. Token Usage Statistics
```
GET /token-usage?days=30&start_date=2024-01-01&end_date=2024-01-31
```

**Response:**
```json
{
  "message": "Token usage statistics retrieved successfully",
  "summary": {
    "summary": [
      {
        "operation_type": "embedding",
        "model_name": "text-embedding-ada-002",
        "operation_count": 150,
        "total_input_tokens": 150000,
        "total_output_tokens": 0,
        "total_tokens": 150000,
        "total_cost_usd": 0.015
      }
    ],
    "totals": {
      "total_operations": 200,
      "total_input_tokens": 200000,
      "total_output_tokens": 50000,
      "total_tokens": 250000,
      "total_cost_usd": 0.025
    }
  },
  "daily_usage": [
    {
      "usage_date": "2024-01-15",
      "operation_count": 10,
      "total_input_tokens": 10000,
      "total_output_tokens": 2500,
      "total_tokens": 12500,
      "total_cost_usd": 0.00125
    }
  ]
}
```

### 2. User-Specific Usage
```
GET /token-usage/user/{user_id}?days=30
```

**Response:**
```json
{
  "message": "Token usage statistics for user user123",
  "user_usage": {
    "user_id": "user123",
    "summary": [
      {
        "operation_type": "rag_output",
        "operation_count": 25,
        "total_cost_usd": 0.0125,
        "avg_quality_score": 0.85,
        "avg_safety_score": 0.95
      }
    ],
    "totals": {
      "total_operations": 25,
      "total_cost_usd": 0.0125
    }
  }
}
```

### 3. Cost Analysis
```
GET /token-usage/cost-analysis?days=30
```

**Response:**
```json
{
  "message": "Cost analysis completed successfully",
  "cost_analysis": {
    "total_cost_usd": 0.025,
    "total_tokens": 250000,
    "cost_per_token": 0.0000001,
    "operation_costs": {
      "embedding": {
        "total_cost": 0.015,
        "total_tokens": 150000,
        "operation_count": 150,
        "cost_per_operation": 0.0001,
        "cost_per_token": 0.0000001
      },
      "rag_output": {
        "total_cost": 0.01,
        "total_tokens": 100000,
        "operation_count": 50,
        "cost_per_operation": 0.0002,
        "cost_per_token": 0.0000001
      }
    },
    "recommendations": [
      "Consider implementing caching for frequent queries",
      "High embedding costs - consider optimizing chunk sizes"
    ]
  }
}
```

## Token Counting

### Token Counter Class

The system includes a `TokenCounter` class that accurately counts tokens for different models:

```python
class TokenCounter:
    def __init__(self):
        # Initialize token encoders for different models
        self.embedding_encoder = tiktoken.encoding_for_model("text-embedding-ada-002")
        self.chat_encoder = tiktoken.encoding_for_model(Config.OPENAI_MODEL)
    
    def count_embedding_tokens(self, text: str) -> int:
        """Count tokens for embedding model."""
        return len(self.embedding_encoder.encode(text))
    
    def count_chat_tokens(self, text: str) -> int:
        """Count tokens for chat model."""
        return len(self.chat_encoder.encode(text))
```

### Supported Models

- **Embedding Models**: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
- **Chat Models**: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`
- **Fallback**: Uses `cl100k_base` encoding for unknown models

## Cost Calculation

### Pricing Structure

Current OpenAI pricing (as of 2024):

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|------------------------|
| text-embedding-ada-002 | $0.0001 | $0.0 |
| text-embedding-3-small | $0.00002 | $0.0 |
| text-embedding-3-large | $0.00013 | $0.0 |
| gpt-4 | $0.03 | $0.06 |
| gpt-4-turbo | $0.01 | $0.03 |
| gpt-4o | $0.005 | $0.015 |
| gpt-3.5-turbo | $0.0015 | $0.002 |

### Automatic Cost Calculation

```python
def calculate_cost(model_name: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Calculate cost for token usage."""
    pricing = ModelPricing.get_pricing(model_name)
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return round(input_cost + output_cost, 6)
```

## Analytics and Reporting

### Usage Summary

The system provides comprehensive usage summaries including:

- **Operation Counts**: Number of operations by type and model
- **Token Usage**: Total input, output, and combined tokens
- **Cost Analysis**: Total costs and cost per operation
- **Performance Metrics**: Average processing times, quality scores, safety scores
- **Time-based Analysis**: Daily, weekly, monthly usage patterns

### Cost Optimization Recommendations

The system automatically generates optimization recommendations:

- **High Embedding Costs**: Suggests optimizing chunk sizes
- **High RAG Output Costs**: Recommends more efficient models
- **Frequent Queries**: Suggests implementing caching
- **Cost Thresholds**: Alerts when costs exceed predefined limits

### User Analytics

- **Per-user Usage**: Track individual user consumption
- **Session Analysis**: Monitor usage patterns per session
- **Query Type Analysis**: Understand which query types consume most tokens
- **Domain-specific Usage**: Track usage by regulatory domain

## Database Indexes

Optimized indexes for efficient querying:

```sql
-- Primary indexes
KEY `idx_operation_type` (`operation_type`),
KEY `idx_model_name` (`model_name`),
KEY `idx_timestamp` (`timestamp`),
KEY `idx_user_id` (`user_id`),
KEY `idx_cost` (`cost_usd`),

-- Composite indexes for common queries
KEY `idx_daily_usage` (`timestamp`, `operation_type`),
KEY `idx_cost_analysis` (`timestamp`, `cost_usd`, `model_name`),
KEY `idx_user_usage` (`user_id`, `timestamp`),
KEY `idx_operation_cost` (`operation_type`, `cost_usd`)
```

## Testing

### Test Coverage

The system includes comprehensive tests (`test_token_tracking.py`):

- **Model Pricing**: Cost calculation accuracy
- **Token Counting**: Token counting for different models
- **Database Operations**: CRUD operations for token records
- **Analytics Functions**: Usage summary and reporting
- **Integration Tests**: End-to-end workflow testing

### Test Scenarios

- **Embedding Operations**: Document chunking and embedding
- **Query Processing**: User query embedding and processing
- **RAG Workflows**: Complete RAG input/output cycles
- **Cost Calculations**: Various pricing scenarios
- **Error Handling**: Database connection failures, invalid data

## Monitoring and Alerting

### Real-time Monitoring

- **Token Usage Tracking**: Real-time token consumption monitoring
- **Cost Alerts**: Automatic alerts when costs exceed thresholds
- **Performance Monitoring**: Processing time and quality score tracking
- **Error Tracking**: Failed operations and error rates

### Alert Thresholds

Configurable thresholds for automated alerts:

```python
alert_thresholds = {
    'max_daily_cost_usd': 50.0,
    'max_operation_cost_usd': 1.0,
    'max_tokens_per_query': 10000,
    'min_quality_score': 0.7,
    'max_processing_time_ms': 10000
}
```

## Best Practices

### Token Optimization

1. **Chunk Size Optimization**: Balance between context and token usage
2. **Model Selection**: Choose appropriate models for different tasks
3. **Caching**: Implement caching for frequent queries
4. **Batch Processing**: Group operations to reduce overhead

### Cost Management

1. **Regular Monitoring**: Check usage patterns and costs daily
2. **User Limits**: Implement per-user token limits
3. **Budget Alerts**: Set up automated budget alerts
4. **Optimization Reviews**: Regular reviews of cost optimization opportunities

### Data Management

1. **Retention Policies**: Implement data retention policies for token logs
2. **Privacy Compliance**: Ensure user data privacy in token logs
3. **Backup Strategies**: Regular backups of token usage data
4. **Performance Optimization**: Monitor database performance for large datasets

## Future Enhancements

### Planned Features

1. **Predictive Analytics**: ML-based cost prediction
2. **Auto-scaling**: Automatic model selection based on cost/performance
3. **Advanced Caching**: Intelligent caching strategies
4. **Cost Budgeting**: Per-user and per-project budget management
5. **Integration APIs**: Enhanced integration with external monitoring tools

### Performance Optimizations

1. **Batch Processing**: Enhanced batch processing capabilities
2. **Async Logging**: Asynchronous token logging to reduce latency
3. **Compression**: Data compression for large token logs
4. **Partitioning**: Database partitioning for large datasets

---

## Conclusion

The token usage tracking system provides comprehensive monitoring and analytics for OpenAI API usage across the Regulatory RAG system. With detailed cost tracking, performance monitoring, and optimization recommendations, it enables efficient management of API costs while maintaining high-quality service delivery.

The system is designed to scale with usage growth and provides the insights needed to optimize costs and performance in production environments.
