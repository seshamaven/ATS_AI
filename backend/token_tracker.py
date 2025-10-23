"""
Token Usage Tracking System for Regulatory RAG
Comprehensive tracking of OpenAI API token usage across all operations.
"""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import mysql.connector
from mysql.connector import Error
from config import Config

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that consume tokens."""
    EMBEDDING = "embedding"
    QUERY_EMBEDDING = "query_embedding"
    RAG_INPUT = "rag_input"
    RAG_OUTPUT = "rag_output"
    CHAT_COMPLETION = "chat_completion"
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"


class ModelPricing:
    """OpenAI model pricing information (per 1K tokens)."""
    
    # Pricing as of 2024 (update as needed)
    PRICING = {
        # Embedding models
        "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
        "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
        "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
        
        # GPT models
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        
        # Default fallback
        "default": {"input": 0.002, "output": 0.002}
    }
    
    @classmethod
    def get_pricing(cls, model_name: str) -> Dict[str, float]:
        """Get pricing for a specific model."""
        return cls.PRICING.get(model_name, cls.PRICING["default"])
    
    @classmethod
    def calculate_cost(cls, model_name: str, input_tokens: int, output_tokens: int = 0) -> float:
        """Calculate cost for token usage."""
        pricing = cls.get_pricing(model_name)
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return round(input_cost + output_cost, 6)


@dataclass
class TokenUsageRecord:
    """Token usage record for database storage."""
    operation_type: str
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    user_query: Optional[str] = None
    query_relevance: Optional[str] = None
    regulatory_domains: Optional[List[str]] = None
    context_count: Optional[int] = None
    response_length: Optional[int] = None
    processing_time_ms: Optional[int] = None
    quality_score: Optional[float] = None
    safety_score: Optional[float] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class TokenUsageTracker:
    """Comprehensive token usage tracking system."""
    
    def __init__(self, db_config: Dict[str, Any] = None):
        self.db_config = db_config or Config.get_mysql_config()
        self.connection = None
        self.batch_records = []
        self.batch_size = 100
        
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            if self.connection.is_connected():
                logger.info("Token usage tracker connected to database")
                return True
        except Error as e:
            logger.error(f"Error connecting to database for token tracking: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Token usage tracker disconnected")
    
    def log_token_usage(self, record: TokenUsageRecord) -> bool:
        """Log token usage to database."""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                logger.error("Failed to connect to database for token logging")
                return False
        
        try:
            cursor = self.connection.cursor()
            
            # Prepare data for insertion
            record_dict = asdict(record)
            record_dict['timestamp'] = record_dict['timestamp'] or datetime.now()
            
            # Convert regulatory_domains to JSON string
            if record_dict['regulatory_domains']:
                record_dict['regulatory_domains'] = json.dumps(record_dict['regulatory_domains'])
            
            # Convert metadata to JSON string
            if record_dict['metadata']:
                record_dict['metadata'] = json.dumps(record_dict['metadata'])
            
            # Insert query
            insert_query = """
            INSERT INTO token_usage (
                operation_type, model_name, input_tokens, output_tokens, total_tokens,
                cost_usd, user_query, query_relevance, regulatory_domains, context_count,
                response_length, processing_time_ms, quality_score, safety_score,
                session_id, user_id, request_id, metadata, timestamp
            ) VALUES (
                %(operation_type)s, %(model_name)s, %(input_tokens)s, %(output_tokens)s, %(total_tokens)s,
                %(cost_usd)s, %(user_query)s, %(query_relevance)s, %(regulatory_domains)s, %(context_count)s,
                %(response_length)s, %(processing_time_ms)s, %(quality_score)s, %(safety_score)s,
                %(session_id)s, %(user_id)s, %(request_id)s, %(metadata)s, %(timestamp)s
            )
            """
            
            cursor.execute(insert_query, record_dict)
            self.connection.commit()
            cursor.close()
            
            logger.debug(f"Token usage logged: {record.operation_type} - {record.total_tokens} tokens - ${record.cost_usd}")
            return True
            
        except Error as e:
            logger.error(f"Error logging token usage: {e}")
            return False
    
    def log_embedding_usage(self, model_name: str, input_tokens: int, 
                           text_content: str, operation_type: str = "embedding",
                           metadata: Dict[str, Any] = None) -> bool:
        """Log embedding operation token usage."""
        cost = ModelPricing.calculate_cost(model_name, input_tokens, 0)
        
        record = TokenUsageRecord(
            operation_type=operation_type,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=0,
            total_tokens=input_tokens,
            cost_usd=cost,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        return self.log_token_usage(record)
    
    def log_query_embedding_usage(self, model_name: str, input_tokens: int,
                                 user_query: str, query_relevance: str = None,
                                 regulatory_domains: List[str] = None,
                                 metadata: Dict[str, Any] = None) -> bool:
        """Log query embedding operation token usage."""
        cost = ModelPricing.calculate_cost(model_name, input_tokens, 0)
        
        record = TokenUsageRecord(
            operation_type=OperationType.QUERY_EMBEDDING.value,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=0,
            total_tokens=input_tokens,
            cost_usd=cost,
            user_query=user_query,
            query_relevance=query_relevance,
            regulatory_domains=regulatory_domains,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        return self.log_token_usage(record)
    
    def log_rag_input_usage(self, model_name: str, input_tokens: int,
                           user_query: str, context_count: int,
                           query_relevance: str = None,
                           regulatory_domains: List[str] = None,
                           session_id: str = None, user_id: str = None,
                           metadata: Dict[str, Any] = None) -> bool:
        """Log RAG input operation token usage."""
        cost = ModelPricing.calculate_cost(model_name, input_tokens, 0)
        
        record = TokenUsageRecord(
            operation_type=OperationType.RAG_INPUT.value,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=0,
            total_tokens=input_tokens,
            cost_usd=cost,
            user_query=user_query,
            query_relevance=query_relevance,
            regulatory_domains=regulatory_domains,
            context_count=context_count,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        return self.log_token_usage(record)
    
    def log_rag_output_usage(self, model_name: str, input_tokens: int, output_tokens: int,
                            user_query: str, response_length: int,
                            processing_time_ms: int = None,
                            quality_score: float = None, safety_score: float = None,
                            query_relevance: str = None,
                            regulatory_domains: List[str] = None,
                            session_id: str = None, user_id: str = None,
                            request_id: str = None,
                            metadata: Dict[str, Any] = None) -> bool:
        """Log RAG output operation token usage."""
        cost = ModelPricing.calculate_cost(model_name, input_tokens, output_tokens)
        
        record = TokenUsageRecord(
            operation_type=OperationType.RAG_OUTPUT.value,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            user_query=user_query,
            query_relevance=query_relevance,
            regulatory_domains=regulatory_domains,
            response_length=response_length,
            processing_time_ms=processing_time_ms,
            quality_score=quality_score,
            safety_score=safety_score,
            session_id=session_id,
            user_id=user_id,
            request_id=request_id or str(uuid.uuid4()),
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        return self.log_token_usage(record)
    
    def log_chat_completion_usage(self, model_name: str, input_tokens: int, output_tokens: int,
                                 user_query: str, response_length: int,
                                 processing_time_ms: int = None,
                                 quality_score: float = None, safety_score: float = None,
                                 query_relevance: str = None,
                                 regulatory_domains: List[str] = None,
                                 session_id: str = None, user_id: str = None,
                                 request_id: str = None,
                                 metadata: Dict[str, Any] = None) -> bool:
        """Log chat completion operation token usage."""
        cost = ModelPricing.calculate_cost(model_name, input_tokens, output_tokens)
        
        record = TokenUsageRecord(
            operation_type=OperationType.CHAT_COMPLETION.value,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            user_query=user_query,
            query_relevance=query_relevance,
            regulatory_domains=regulatory_domains,
            response_length=response_length,
            processing_time_ms=processing_time_ms,
            quality_score=quality_score,
            safety_score=safety_score,
            session_id=session_id,
            user_id=user_id,
            request_id=request_id or str(uuid.uuid4()),
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        return self.log_token_usage(record)
    
    def get_usage_summary(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Get token usage summary for a date range."""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return {}
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Default to last 30 days if no dates provided
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # Summary query
            summary_query = """
            SELECT 
                operation_type,
                model_name,
                COUNT(*) as operation_count,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as total_cost_usd,
                AVG(processing_time_ms) as avg_processing_time_ms,
                AVG(quality_score) as avg_quality_score,
                AVG(safety_score) as avg_safety_score
            FROM token_usage 
            WHERE timestamp BETWEEN %s AND %s
            GROUP BY operation_type, model_name
            ORDER BY total_cost_usd DESC
            """
            
            cursor.execute(summary_query, (start_date, end_date))
            results = cursor.fetchall()
            
            # Overall totals
            totals_query = """
            SELECT 
                COUNT(*) as total_operations,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as total_cost_usd
            FROM token_usage 
            WHERE timestamp BETWEEN %s AND %s
            """
            
            cursor.execute(totals_query, (start_date, end_date))
            totals = cursor.fetchone()
            
            cursor.close()
            
            return {
                'summary': results,
                'totals': totals,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                }
            }
            
        except Error as e:
            logger.error(f"Error getting usage summary: {e}")
            return {}
    
    def get_daily_usage(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily usage statistics."""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            daily_query = """
            SELECT 
                DATE(timestamp) as usage_date,
                COUNT(*) as operation_count,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as total_cost_usd
            FROM token_usage 
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY DATE(timestamp)
            ORDER BY usage_date DESC
            """
            
            cursor.execute(daily_query, (days,))
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Error as e:
            logger.error(f"Error getting daily usage: {e}")
            return []
    
    def get_user_usage(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for a specific user."""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return {}
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            user_query = """
            SELECT 
                operation_type,
                COUNT(*) as operation_count,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as total_cost_usd,
                AVG(processing_time_ms) as avg_processing_time_ms,
                AVG(quality_score) as avg_quality_score,
                AVG(safety_score) as avg_safety_score
            FROM token_usage 
            WHERE user_id = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY operation_type
            ORDER BY total_cost_usd DESC
            """
            
            cursor.execute(user_query, (user_id, days))
            results = cursor.fetchall()
            
            # Total for user
            totals_query = """
            SELECT 
                COUNT(*) as total_operations,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as total_cost_usd
            FROM token_usage 
            WHERE user_id = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            """
            
            cursor.execute(totals_query, (user_id, days))
            totals = cursor.fetchone()
            
            cursor.close()
            
            return {
                'user_id': user_id,
                'summary': results,
                'totals': totals,
                'days': days
            }
            
        except Error as e:
            logger.error(f"Error getting user usage: {e}")
            return {}


# Global token tracker instance
_token_tracker = None


def get_token_tracker() -> TokenUsageTracker:
    """Get the global token tracker instance."""
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenUsageTracker()
    return _token_tracker


# Convenience functions for easy integration
def log_embedding_tokens(model_name: str, input_tokens: int, text_content: str,
                        operation_type: str = "embedding", metadata: Dict[str, Any] = None) -> bool:
    """Log embedding token usage."""
    tracker = get_token_tracker()
    return tracker.log_embedding_usage(model_name, input_tokens, text_content, operation_type, metadata)


def log_query_embedding_tokens(model_name: str, input_tokens: int, user_query: str,
                              query_relevance: str = None, regulatory_domains: List[str] = None,
                              metadata: Dict[str, Any] = None) -> bool:
    """Log query embedding token usage."""
    tracker = get_token_tracker()
    return tracker.log_query_embedding_usage(model_name, input_tokens, user_query, 
                                           query_relevance, regulatory_domains, metadata)


def log_rag_tokens(model_name: str, input_tokens: int, output_tokens: int,
                  user_query: str, response_length: int,
                  processing_time_ms: int = None,
                  quality_score: float = None, safety_score: float = None,
                  query_relevance: str = None, regulatory_domains: List[str] = None,
                  session_id: str = None, user_id: str = None,
                  request_id: str = None, metadata: Dict[str, Any] = None) -> bool:
    """Log RAG operation token usage."""
    tracker = get_token_tracker()
    return tracker.log_rag_output_usage(model_name, input_tokens, output_tokens,
                                      user_query, response_length, processing_time_ms,
                                      quality_score, safety_score, query_relevance,
                                      regulatory_domains, session_id, user_id,
                                      request_id, metadata)


def log_chat_completion_usage(model_name: str, input_tokens: int, output_tokens: int,
                             user_query: str, response_length: int,
                             processing_time_ms: int = None,
                             quality_score: float = None, safety_score: float = None,
                             query_relevance: str = None, regulatory_domains: List[str] = None,
                             session_id: str = None, user_id: str = None,
                             request_id: str = None, metadata: Dict[str, Any] = None) -> bool:
    """Log chat completion operation token usage."""
    tracker = get_token_tracker()
    return tracker.log_chat_completion_usage(model_name, input_tokens, output_tokens,
                                           user_query, response_length, processing_time_ms,
                                           quality_score, safety_score, query_relevance,
                                           regulatory_domains, session_id, user_id,
                                           request_id, metadata)


def get_token_usage_summary(start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
    """Get token usage summary."""
    tracker = get_token_tracker()
    return tracker.get_usage_summary(start_date, end_date)


def get_daily_token_usage(days: int = 30) -> List[Dict[str, Any]]:
    """Get daily token usage statistics."""
    tracker = get_token_tracker()
    return tracker.get_daily_usage(days)
