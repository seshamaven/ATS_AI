"""
Production Monitoring and Logging System for Regulatory RAG
Comprehensive logging, metrics collection, and alerting for production deployment.
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from enum import Enum


class LogLevel(Enum):
    """Log levels for different types of events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"


class EventType(Enum):
    """Types of events to monitor."""
    QUERY_RECEIVED = "query_received"
    QUERY_CLASSIFIED = "query_classified"
    CONTEXT_RETRIEVED = "context_retrieved"
    RESPONSE_GENERATED = "response_generated"
    RESPONSE_VALIDATED = "response_validated"
    SECURITY_ALERT = "security_alert"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class LogEntry:
    """Structured log entry for production monitoring."""
    timestamp: str
    event_type: str
    log_level: str
    message: str
    user_query: Optional[str] = None
    query_relevance: Optional[str] = None
    regulatory_domains: Optional[List[str]] = None
    context_count: Optional[int] = None
    response_length: Optional[int] = None
    processing_time_ms: Optional[float] = None
    quality_score: Optional[float] = None
    safety_score: Optional[float] = None
    error_details: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProductionMetrics:
    """Collects and manages production metrics."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.lock = threading.Lock()
        
    def increment_counter(self, metric_name: str, value: int = 1):
        """Increment a counter metric."""
        with self.lock:
            self.counters[metric_name] += value
    
    def record_timing(self, metric_name: str, duration_ms: float):
        """Record a timing metric."""
        with self.lock:
            self.timers[metric_name].append(duration_ms)
            # Keep only recent timings
            if len(self.timers[metric_name]) > self.max_history:
                self.timers[metric_name] = self.timers[metric_name][-self.max_history:]
    
    def record_value(self, metric_name: str, value: float):
        """Record a value metric."""
        with self.lock:
            self.metrics[metric_name].append({
                'timestamp': datetime.now().isoformat(),
                'value': value
            })
    
    def get_counter(self, metric_name: str) -> int:
        """Get counter value."""
        with self.lock:
            return self.counters.get(metric_name, 0)
    
    def get_average_timing(self, metric_name: str, window_minutes: int = 60) -> float:
        """Get average timing for a window."""
        with self.lock:
            if metric_name not in self.timers:
                return 0.0
            
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_timings = [
                t for t in self.timers[metric_name]
                if datetime.fromisoformat(t.get('timestamp', '1970-01-01')) > cutoff_time
            ]
            
            return sum(recent_timings) / len(recent_timings) if recent_timings else 0.0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self.lock:
            summary = {
                'counters': dict(self.counters),
                'average_timings': {},
                'recent_values': {}
            }
            
            # Calculate average timings
            for metric_name, timings in self.timers.items():
                if timings:
                    summary['average_timings'][metric_name] = sum(timings) / len(timings)
            
            # Get recent values
            for metric_name, values in self.metrics.items():
                if values:
                    recent_values = list(values)[-10:]  # Last 10 values
                    summary['recent_values'][metric_name] = recent_values
            
            return summary


class ProductionLogger:
    """Production-grade logger with structured logging and monitoring."""
    
    def __init__(self, log_file: str = "production_regulatory_rag.log"):
        self.log_file = log_file
        self.metrics = ProductionMetrics()
        self.logger = self._setup_logger()
        self.security_alerts = deque(maxlen=1000)
        self.error_alerts = deque(maxlen=1000)
        
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger."""
        logger = logging.getLogger("production_regulatory_rag")
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_event(self, log_entry: LogEntry):
        """Log a structured event."""
        # Update metrics
        self.metrics.increment_counter(f"events_{log_entry.event_type}")
        self.metrics.increment_counter(f"logs_{log_entry.log_level}")
        
        if log_entry.processing_time_ms:
            self.metrics.record_timing("processing_time", log_entry.processing_time_ms)
        
        if log_entry.quality_score:
            self.metrics.record_value("quality_score", log_entry.quality_score)
        
        if log_entry.safety_score:
            self.metrics.record_value("safety_score", log_entry.safety_score)
        
        # Log the event
        log_message = json.dumps(asdict(log_entry), indent=2)
        
        if log_entry.log_level == LogLevel.CRITICAL.value:
            self.logger.critical(log_message)
        elif log_entry.log_level == LogLevel.ERROR.value:
            self.logger.error(log_message)
        elif log_entry.log_level == LogLevel.WARNING.value:
            self.logger.warning(log_message)
        elif log_entry.log_level == LogLevel.SECURITY.value:
            self.logger.critical(f"SECURITY ALERT: {log_message}")
            self.security_alerts.append(log_entry)
        else:
            self.logger.info(log_message)
        
        # Handle error alerts
        if log_entry.log_level == LogLevel.ERROR.value:
            self.error_alerts.append(log_entry)
    
    def log_query_received(self, user_query: str, session_id: str = None, user_id: str = None):
        """Log query received event."""
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.QUERY_RECEIVED.value,
            log_level=LogLevel.INFO.value,
            message="User query received",
            user_query=user_query,
            session_id=session_id,
            user_id=user_id
        )
        self.log_event(log_entry)
    
    def log_query_classified(self, user_query: str, relevance: str, domains: List[str], 
                           analysis: Dict[str, Any], processing_time_ms: float):
        """Log query classification event."""
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.QUERY_CLASSIFIED.value,
            log_level=LogLevel.INFO.value,
            message=f"Query classified as {relevance}",
            user_query=user_query,
            query_relevance=relevance,
            regulatory_domains=domains,
            processing_time_ms=processing_time_ms,
            metadata=analysis
        )
        self.log_event(log_entry)
    
    def log_context_retrieved(self, user_query: str, context_count: int, 
                             processing_time_ms: float):
        """Log context retrieval event."""
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.CONTEXT_RETRIEVED.value,
            log_level=LogLevel.INFO.value,
            message=f"Retrieved {context_count} context items",
            user_query=user_query,
            context_count=context_count,
            processing_time_ms=processing_time_ms
        )
        self.log_event(log_entry)
    
    def log_response_generated(self, user_query: str, response_length: int, 
                              processing_time_ms: float):
        """Log response generation event."""
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.RESPONSE_GENERATED.value,
            log_level=LogLevel.INFO.value,
            message=f"Response generated ({response_length} chars)",
            user_query=user_query,
            response_length=response_length,
            processing_time_ms=processing_time_ms
        )
        self.log_event(log_entry)
    
    def log_response_validated(self, user_query: str, validation_result: Dict[str, Any]):
        """Log response validation event."""
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.RESPONSE_VALIDATED.value,
            log_level=LogLevel.INFO.value,
            message=f"Response validated: valid={validation_result.get('is_valid', False)}",
            user_query=user_query,
            quality_score=validation_result.get('quality_score', 0),
            safety_score=validation_result.get('safety_score', 0),
            metadata=validation_result
        )
        self.log_event(log_entry)
    
    def log_security_alert(self, user_query: str, alert_type: str, details: str):
        """Log security alert."""
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.SECURITY_ALERT.value,
            log_level=LogLevel.SECURITY.value,
            message=f"Security alert: {alert_type}",
            user_query=user_query,
            error_details=details
        )
        self.log_event(log_entry)
    
    def log_error(self, user_query: str, error_type: str, error_details: str):
        """Log error event."""
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.ERROR_OCCURRED.value,
            log_level=LogLevel.ERROR.value,
            message=f"Error occurred: {error_type}",
            user_query=user_query,
            error_details=error_details
        )
        self.log_event(log_entry)
    
    def get_security_alerts(self, hours: int = 24) -> List[LogEntry]:
        """Get recent security alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.security_alerts
            if datetime.fromisoformat(alert.timestamp) > cutoff_time
        ]
    
    def get_error_alerts(self, hours: int = 24) -> List[LogEntry]:
        """Get recent error alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            error for error in self.error_alerts
            if datetime.fromisoformat(error.timestamp) > cutoff_time
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'metrics': self.metrics.get_metrics_summary(),
            'security_alerts_24h': len(self.get_security_alerts(24)),
            'error_alerts_24h': len(self.get_error_alerts(24)),
            'total_queries': self.metrics.get_counter('events_query_received'),
            'avg_processing_time': self.metrics.get_average_timing('processing_time', 60),
            'avg_quality_score': self.metrics.get_average_timing('quality_score', 60),
            'avg_safety_score': self.metrics.get_average_timing('safety_score', 60)
        }


class ProductionMonitor:
    """Production monitoring system with alerting capabilities."""
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.alert_thresholds = {
            'max_processing_time_ms': 10000,  # 10 seconds
            'min_quality_score': 0.5,
            'min_safety_score': 0.7,
            'max_error_rate_percent': 5.0,
            'max_security_alerts_per_hour': 10
        }
        self.last_alert_times = {}
        
    def check_performance_thresholds(self, processing_time_ms: float, 
                                   quality_score: float, safety_score: float):
        """Check if performance metrics exceed thresholds."""
        alerts = []
        
        if processing_time_ms > self.alert_thresholds['max_processing_time_ms']:
            alerts.append({
                'type': 'performance',
                'message': f'Processing time exceeded threshold: {processing_time_ms}ms',
                'severity': 'warning'
            })
        
        if quality_score < self.alert_thresholds['min_quality_score']:
            alerts.append({
                'type': 'quality',
                'message': f'Quality score below threshold: {quality_score}',
                'severity': 'warning'
            })
        
        if safety_score < self.alert_thresholds['min_safety_score']:
            alerts.append({
                'type': 'safety',
                'message': f'Safety score below threshold: {safety_score}',
                'severity': 'critical'
            })
        
        return alerts
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        performance_summary = self.logger.get_performance_summary()
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'metrics': performance_summary,
            'alerts': []
        }
        
        # Check error rate
        total_queries = performance_summary['total_queries']
        error_count = performance_summary['error_alerts_24h']
        if total_queries > 0:
            error_rate = (error_count / total_queries) * 100
            if error_rate > self.alert_thresholds['max_error_rate_percent']:
                health_status['alerts'].append({
                    'type': 'error_rate',
                    'message': f'Error rate too high: {error_rate:.1f}%',
                    'severity': 'critical'
                })
                health_status['status'] = 'unhealthy'
        
        # Check security alerts
        security_alerts = performance_summary['security_alerts_24h']
        if security_alerts > self.alert_thresholds['max_security_alerts_per_hour']:
            health_status['alerts'].append({
                'type': 'security',
                'message': f'Too many security alerts: {security_alerts}',
                'severity': 'critical'
            })
            health_status['status'] = 'unhealthy'
        
        return health_status


# Global production logger instance
production_logger = ProductionLogger()
production_monitor = ProductionMonitor(production_logger)


def get_production_logger() -> ProductionLogger:
    """Get the global production logger instance."""
    return production_logger


def get_production_monitor() -> ProductionMonitor:
    """Get the global production monitor instance."""
    return production_monitor


# Convenience functions for easy integration
def log_query_processing(user_query: str, start_time: float, end_time: float,
                        relevance: str, domains: List[str], analysis: Dict[str, Any],
                        context_count: int, response_length: int, 
                        validation_result: Dict[str, Any]):
    """Log complete query processing pipeline."""
    total_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Log each step
    production_logger.log_query_received(user_query)
    production_logger.log_query_classified(user_query, relevance, domains, analysis, total_time * 0.1)
    production_logger.log_context_retrieved(user_query, context_count, total_time * 0.2)
    production_logger.log_response_generated(user_query, response_length, total_time * 0.5)
    production_logger.log_response_validated(user_query, validation_result)
    
    # Check performance thresholds
    alerts = production_monitor.check_performance_thresholds(
        total_time,
        validation_result.get('quality_score', 0),
        validation_result.get('safety_score', 0)
    )
    
    for alert in alerts:
        if alert['severity'] == 'critical':
            production_logger.log_security_alert(user_query, alert['type'], alert['message'])
        else:
            production_logger.log_error(user_query, alert['type'], alert['message'])


def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    return production_monitor.check_system_health()


def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics."""
    return production_logger.get_performance_summary()
