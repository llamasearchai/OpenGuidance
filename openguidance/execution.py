"""
Advanced execution engine with performance monitoring and optimization.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ExecutionMetrics:
    """Comprehensive execution metrics tracking."""
    
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    token_usage: Dict[str, int] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    api_calls: int = 0
    retry_count: int = 0
    error_details: Optional[str] = None
    
    def mark_completed(self) -> None:
        """Mark execution as completed and calculate duration."""
        self.end_time = datetime.utcnow()
        self.status = ExecutionStatus.COMPLETED
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
    
    def mark_failed(self, error: str) -> None:
        """Mark execution as failed with error details."""
        self.end_time = datetime.utcnow()
        self.status = ExecutionStatus.FAILED
        self.error_details = error
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()


@dataclass
class ExecutionResult:
    """Comprehensive execution result container."""
    
    execution_id: str
    content: Any
    metrics: ExecutionMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: List[Any] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.metrics.status == ExecutionStatus.COMPLETED
    
    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        return self.metrics.duration


class ExecutionEngine:
    """
    Advanced execution engine with sophisticated monitoring,
    optimization, and resource management.
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        prompt_manager: Optional[Any] = None,
        max_concurrent: int = 10,
        default_timeout: int = 30,
        enable_monitoring: bool = True
    ):
        self.config = config
        self.memory_manager = memory_manager
        self.prompt_manager = prompt_manager
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.enable_monitoring = enable_monitoring
        
        self._active_executions: Dict[str, asyncio.Task] = {}
        self._execution_queue: asyncio.Queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._metrics_history: List[ExecutionMetrics] = []
        self._performance_stats: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0.0,
            "peak_concurrent": 0
        }
        
        # Start background tasks
        self._monitor_task = None
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the execution engine."""
        if self.is_initialized:
            logger.warning("ExecutionEngine already initialized")
            return
        
        # Start monitoring if enabled
        if self.enable_monitoring:
            self._start_monitoring()
        
        self.is_initialized = True
        logger.info("ExecutionEngine initialization completed")
    
    async def cleanup(self) -> None:
        """Cleanup execution engine resources."""
        if not self.is_initialized:
            return
        
        # Stop monitoring task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.is_initialized = False
        logger.info("ExecutionEngine cleanup completed")
    
    def _start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        async def monitor_loop():
            while True:
                await self._update_performance_stats()
                await self._cleanup_completed_executions()
                await asyncio.sleep(5)  # Monitor every 5 seconds
        
        self._monitor_task = asyncio.create_task(monitor_loop())
    
    async def execute_async(
        self,
        execution_func: Callable,
        execution_id: str = None,
        timeout: int = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute function asynchronously with comprehensive monitoring."""
        if execution_id is None:
            execution_id = f"exec_{int(time.time() * 1000)}"
        
        metrics = ExecutionMetrics(
            execution_id=execution_id,
            start_time=datetime.utcnow()
        )
        
        async with self._semaphore:
            try:
                metrics.status = ExecutionStatus.RUNNING
                
                # Execute with timeout
                execution_timeout = timeout or self.default_timeout
                result_content = await asyncio.wait_for(
                    execution_func(**kwargs),
                    timeout=execution_timeout
                )
                
                metrics.mark_completed()
                self._performance_stats["successful_executions"] += 1
                
                result = ExecutionResult(
                    execution_id=execution_id,
                    content=result_content,
                    metrics=metrics,
                    metadata=metadata or {}
                )
                
                if self.enable_monitoring:
                    self._metrics_history.append(metrics)
                
                logger.info(f"Execution completed: {execution_id} in {metrics.duration:.2f}s")
                return result
                
            except asyncio.TimeoutError:
                metrics.status = ExecutionStatus.TIMEOUT
                metrics.mark_failed("Execution timeout")
                self._performance_stats["failed_executions"] += 1
                
                return ExecutionResult(
                    execution_id=execution_id,
                    content=None,
                    metrics=metrics,
                    metadata=metadata or {}
                )
                
            except Exception as e:
                metrics.mark_failed(str(e))
                self._performance_stats["failed_executions"] += 1
                logger.error(f"Execution failed: {execution_id} - {str(e)}")
                
                return ExecutionResult(
                    execution_id=execution_id,
                    content=None,
                    metrics=metrics,
                    metadata=metadata or {}
                )
            finally:
                self._performance_stats["total_executions"] += 1
    
    async def execute_batch(
        self,
        execution_funcs: List[Callable],
        timeout: int = None,
        max_retries: int = 2
    ) -> List[ExecutionResult]:
        """Execute multiple functions in parallel with retry logic."""
        tasks = []
        
        for i, func in enumerate(execution_funcs):
            execution_id = f"batch_{int(time.time() * 1000)}_{i}"
            task = asyncio.create_task(
                self._execute_with_retry(func, execution_id, timeout, max_retries)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_metrics = ExecutionMetrics(
                    execution_id=f"batch_{int(time.time() * 1000)}_{i}",
                    start_time=datetime.utcnow()
                )
                error_metrics.mark_failed(str(result))
                
                processed_results.append(ExecutionResult(
                    execution_id=error_metrics.execution_id,
                    content=None,
                    metrics=error_metrics
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_with_retry(
        self,
        execution_func: Callable,
        execution_id: str,
        timeout: int,
        max_retries: int
    ) -> ExecutionResult:
        """Execute function with retry logic."""
        last_result = None
        
        for attempt in range(max_retries + 1):
            result = await self.execute_async(
                execution_func,
                f"{execution_id}_attempt_{attempt}",
                timeout
            )
            
            if result.success:
                return result
            
            last_result = result
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return last_result
    
    async def execute_request(
        self,
        request: str,
        session_id: str,
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute a request with the given context."""
        async def request_handler():
            # Simple request processing - in production, this would involve
            # more sophisticated AI processing
            response = f"Processed request: {request}"
            return response
        
        return await self.execute_async(
            request_handler,
            execution_id=f"request_{session_id}_{int(time.time() * 1000)}",
            metadata={"session_id": session_id, "context": context}
        )
    
    async def execute_streaming_request(
        self,
        request: str,
        session_id: str,
        context: Dict[str, Any]
    ):
        """Execute a streaming request."""
        # Simple streaming implementation - yield response chunks
        response = f"Streaming response for: {request}"
        for i, chunk in enumerate(response.split()):
            yield f"{chunk} "
            await asyncio.sleep(0.1)  # Simulate streaming delay
    
    async def _update_performance_stats(self) -> None:
        """Update performance statistics."""
        if not self._metrics_history:
            return
        
        successful = sum(1 for m in self._metrics_history if m.status == ExecutionStatus.COMPLETED)
        failed = sum(1 for m in self._metrics_history if m.status == ExecutionStatus.FAILED)
        total_duration = sum(m.duration or 0 for m in self._metrics_history if m.duration)
        
        self._performance_stats.update({
            "successful_executions": successful,
            "failed_executions": failed,
            "total_executions": len(self._metrics_history),
            "average_duration": total_duration / len(self._metrics_history) if self._metrics_history else 0,
            "peak_concurrent": len(self._active_executions)
        })
    
    async def _cleanup_completed_executions(self) -> None:
        """Clean up completed executions to prevent memory leaks."""
        completed_ids = []
        for exec_id, task in self._active_executions.items():
            if task.done():
                completed_ids.append(exec_id)
        
        for exec_id in completed_ids:
            del self._active_executions[exec_id]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self._performance_stats.copy()
    
    def get_execution_history(self, limit: int = 100) -> List[ExecutionMetrics]:
        """Get execution history with optional limit."""
        return self._metrics_history[-limit:] if limit else self._metrics_history
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the execution engine."""
        # Cancel monitoring task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active executions to complete
        if self._active_executions:
            await asyncio.gather(*self._active_executions.values(), return_exceptions=True)
        
        logger.info("ExecutionEngine shutdown completed")