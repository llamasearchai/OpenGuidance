"""
Advanced monitoring and metrics collection system.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """Individual metric data point."""
    
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    
    session_id: str
    execution_time: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_type: Optional[str] = None
    model_name: Optional[str] = None
    token_usage: Dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """
    Comprehensive metrics collection and monitoring system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize metrics collector with optional configuration."""
        self.config = config or {}
        self.max_metrics_history = self.config.get('max_metrics_history', 10000)
        self.aggregation_window = self.config.get('aggregation_window', 300)  # 5 minutes
        
        # Metrics storage
        self._metrics_history: deque = deque(maxlen=self.max_metrics_history)
        self._execution_metrics: List[ExecutionMetrics] = []
        self._aggregated_metrics: Dict[str, Any] = defaultdict(list)
        
        # Performance counters
        self._counters = defaultdict(int)
        self._timers = defaultdict(float)
        self._gauges = defaultdict(float)
        
        # System health metrics
        self._health_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'error_rate': 0.0,
            'uptime_start': datetime.utcnow()
        }
        
        logger.info("MetricsCollector initialized")
    
    async def record_execution(
        self,
        session_id: str,
        execution_time: float,
        success: bool,
        error_type: Optional[str] = None,
        model_name: Optional[str] = None,
        token_usage: Optional[Dict[str, int]] = None
    ) -> None:
        """Record execution metrics."""
        metrics = ExecutionMetrics(
            session_id=session_id,
            execution_time=execution_time,
            success=success,
            error_type=error_type,
            model_name=model_name,
            token_usage=token_usage or {}
        )
        
        self._execution_metrics.append(metrics)
        
        # Update health metrics
        self._health_metrics['total_executions'] += 1
        if success:
            self._health_metrics['successful_executions'] += 1
        else:
            self._health_metrics['failed_executions'] += 1
        
        # Update average execution time
        total_time = sum(m.execution_time for m in self._execution_metrics[-100:])  # Last 100 executions
        count = min(len(self._execution_metrics), 100)
        self._health_metrics['average_execution_time'] = total_time / count if count > 0 else 0.0
        
        # Update error rate
        recent_executions = self._execution_metrics[-100:]
        failed_count = sum(1 for m in recent_executions if not m.success)
        self._health_metrics['error_rate'] = failed_count / len(recent_executions) if recent_executions else 0.0
        
        logger.debug(f"Recorded execution metrics for session: {session_id}")
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a custom metric."""
        metric = MetricData(
            name=name,
            value=value,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self._metrics_history.append(metric)
        
        # Update aggregated metrics
        self._aggregated_metrics[name].append({
            'value': value,
            'timestamp': metric.timestamp,
            'tags': tags or {}
        })
        
        logger.debug(f"Recorded metric: {name} = {value}")
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self._counters[name] += value
        self.record_metric(f"counter.{name}", self._counters[name])
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric value."""
        self._gauges[name] = value
        self.record_metric(f"gauge.{name}", value)
    
    def start_timer(self, name: str) -> str:
        """Start a timer and return timer ID."""
        timer_id = f"{name}_{int(time.time() * 1000000)}"
        self._timers[timer_id] = time.time()
        return timer_id
    
    def stop_timer(self, timer_id: str) -> float:
        """Stop a timer and record the duration."""
        if timer_id in self._timers:
            duration = time.time() - self._timers[timer_id]
            del self._timers[timer_id]
            
            # Extract metric name from timer ID
            metric_name = timer_id.rsplit('_', 1)[0]
            self.record_metric(f"timer.{metric_name}", duration)
            
            return duration
        return 0.0
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        uptime = datetime.utcnow() - self._health_metrics['uptime_start']
        
        return {
            **self._health_metrics,
            'uptime_seconds': uptime.total_seconds(),
            'metrics_collected': len(self._metrics_history),
            'active_timers': len(self._timers)
        }
    
    def get_execution_stats(self, limit: int = 100) -> Dict[str, Any]:
        """Get execution statistics."""
        recent_metrics = self._execution_metrics[-limit:]
        
        if not recent_metrics:
            return {
                'total_executions': 0,
                'success_rate': 0.0,
                'average_execution_time': 0.0,
                'error_types': {}
            }
        
        successful = sum(1 for m in recent_metrics if m.success)
        failed = sum(1 for m in recent_metrics if not m.success)
        
        # Error type distribution
        error_types = defaultdict(int)
        for m in recent_metrics:
            if not m.success and m.error_type:
                error_types[m.error_type] += 1
        
        return {
            'total_executions': len(recent_metrics),
            'successful_executions': successful,
            'failed_executions': failed,
            'success_rate': successful / len(recent_metrics),
            'average_execution_time': sum(m.execution_time for m in recent_metrics) / len(recent_metrics),
            'error_types': dict(error_types),
            'token_usage_total': sum(
                sum(m.token_usage.values()) for m in recent_metrics if m.token_usage
            )
        }
    
    def get_metric_history(self, metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get history for a specific metric."""
        matching_metrics = [
            {
                'value': m.value,
                'timestamp': m.timestamp.isoformat(),
                'tags': m.tags,
                'metadata': m.metadata
            }
            for m in list(self._metrics_history)[-limit:]
            if m.name == metric_name
        ]
        
        return matching_metrics
    
    def get_aggregated_metrics(self, time_window: int = 300) -> Dict[str, Any]:
        """Get aggregated metrics for the specified time window (seconds)."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
        
        aggregated = {}
        for metric_name, values in self._aggregated_metrics.items():
            recent_values = [
                v for v in values
                if v['timestamp'] >= cutoff_time
            ]
            
            if recent_values:
                values_only = [v['value'] for v in recent_values]
                aggregated[metric_name] = {
                    'count': len(values_only),
                    'sum': sum(values_only),
                    'avg': sum(values_only) / len(values_only),
                    'min': min(values_only),
                    'max': max(values_only),
                    'latest': recent_values[-1]['value'],
                    'time_window': time_window
                }
        
        return aggregated
    
    def export_metrics(self, format: str = 'json') -> Dict[str, Any]:
        """Export all metrics in specified format."""
        export_data = {
            'health_metrics': self.get_health_metrics(),
            'execution_stats': self.get_execution_stats(),
            'aggregated_metrics': self.get_aggregated_metrics(),
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'export_timestamp': datetime.utcnow().isoformat()
        }
        
        return export_data
    
    def reset_metrics(self) -> None:
        """Reset all metrics (use with caution)."""
        self._metrics_history.clear()
        self._execution_metrics.clear()
        self._aggregated_metrics.clear()
        self._counters.clear()
        self._timers.clear()
        self._gauges.clear()
        
        # Reset health metrics but keep uptime start
        uptime_start = self._health_metrics['uptime_start']
        self._health_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'error_rate': 0.0,
            'uptime_start': uptime_start
        }
        
        logger.info("All metrics have been reset")


def create_metrics_collector(config: Optional[Dict[str, Any]] = None) -> MetricsCollector:
    """Factory function to create a metrics collector with default configuration."""
    default_config = {
        'max_metrics_history': 10000,
        'aggregation_window': 300,
        'enable_detailed_logging': False
    }
    
    if config:
        default_config.update(config)
    
    return MetricsCollector(default_config) 