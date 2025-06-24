"""
Main OpenGuidance system orchestrator.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, AsyncIterator
from dataclasses import dataclass, field

from ..memory import MemoryManager
from ..prompts import PromptManager
from ..execution import ExecutionEngine
from ..models import ExecutionResult, ExecutionStatus, ExecutionError
from .config import Config


logger = logging.getLogger(__name__)


@dataclass
class SystemStats:
    """System statistics container."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    total_memory_items: int = 0
    active_sessions: int = 0
    uptime_seconds: float = 0.0
    last_request_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "average_response_time": self.average_response_time,
            "total_memory_items": self.total_memory_items,
            "active_sessions": self.active_sessions,
            "uptime_seconds": self.uptime_seconds,
            "last_request_time": self.last_request_time
        }


class OpenGuidance:
    """
    Main OpenGuidance system class that orchestrates all components.
    
    This class serves as the primary interface for the OpenGuidance system,
    coordinating between memory management, prompt processing, and execution.
    """
    
    def __init__(self, config: Config):
        """
        Initialize OpenGuidance system.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.stats = SystemStats()
        self.start_time = time.time()
        self.is_initialized = False
        
        # Component instances
        self.memory_manager: Optional[MemoryManager] = None
        self.prompt_manager: Optional[PromptManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        
        # Response time tracking
        self._response_times: List[float] = []
        
        logger.info(f"OpenGuidance system created with config: {config.model_name}")
    
    async def initialize(self) -> None:
        """Initialize all system components."""
        if self.is_initialized:
            logger.warning("System already initialized")
            return
        
        logger.info("Initializing OpenGuidance system...")
        
        try:
            # Initialize memory manager
            if self.config.enable_memory:
                self.memory_manager = MemoryManager(self.config.memory_config)
                await self.memory_manager.initialize()
                logger.info("Memory manager initialized")
            
            # Initialize prompt manager
            self.prompt_manager = PromptManager(self.config.prompt_config)
            await self.prompt_manager.initialize()
            logger.info("Prompt manager initialized")
            
            # Initialize execution engine
            self.execution_engine = ExecutionEngine(
                self.config,
                memory_manager=self.memory_manager,
                prompt_manager=self.prompt_manager
            )
            await self.execution_engine.initialize()
            logger.info("Execution engine initialized")
            
            self.is_initialized = True
            logger.info("OpenGuidance system initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenGuidance system: {e}")
            await self.cleanup()
            raise
    
    async def process_request(
        self,
        request: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> ExecutionResult:
        """
        Process a user request through the complete OpenGuidance pipeline.
        
        Args:
            request: User request text
            session_id: Session identifier
            context: Additional context information
            timeout: Request timeout in seconds
            
        Returns:
            ExecutionResult containing the response
            
        Raises:
            ExecutionError: If processing fails
        """
        if not self.is_initialized:
            raise ExecutionError("System not initialized")
        
        start_time = time.time()
        self.stats.total_requests += 1
        self.stats.last_request_time = start_time
        
        try:
            logger.debug(f"Processing request for session {session_id}: {request[:100]}...")
            
            # Apply timeout if specified
            if timeout:
                result = await asyncio.wait_for(
                    self._process_request_internal(request, session_id, context),
                    timeout=timeout
                )
            else:
                result = await self._process_request_internal(request, session_id, context)
            
            # Update statistics
            execution_time = time.time() - start_time
            self._update_response_stats(execution_time, success=True)
            
            logger.info(
                f"Request processed successfully in {execution_time:.3f}s "
                f"for session {session_id}"
            )
            
            return result
            
        except asyncio.TimeoutError:
            self.stats.failed_requests += 1
            error_msg = f"Request timeout after {timeout}s"
            logger.error(error_msg)
            raise ExecutionError(error_msg)
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_response_stats(execution_time, success=False)
            
            logger.error(f"Request processing failed: {e}", exc_info=True)
            raise ExecutionError(f"Request processing failed: {e}")
    
    async def _process_request_internal(
        self,
        request: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Internal request processing logic."""
        # Retrieve relevant memories
        relevant_memories = []
        if self.memory_manager:
            relevant_memories = await self.memory_manager.retrieve_relevant_memories(
                request, session_id, limit=10
            )
        
        # Build context with memories
        enriched_context = {
            "request": request,
            "session_id": session_id,
            "memories": relevant_memories,
            "user_context": context or {}
        }
        
        # Process through execution engine
        result = await self.execution_engine.execute_request(
            request, session_id, enriched_context
        )
        
        # Store important information in memory
        if self.memory_manager and result.metrics.status == ExecutionStatus.COMPLETED:
            await self._store_interaction_memory(request, result, session_id)
        
        return result
    
    async def process_streaming_request(
        self,
        request: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """
        Process a request with streaming response.
        
        Args:
            request: User request text
            session_id: Session identifier
            context: Additional context information
            
        Yields:
            Response chunks as they become available
        """
        if not self.is_initialized:
            raise ExecutionError("System not initialized")
        
        logger.debug(f"Processing streaming request for session {session_id}")
        
        try:
            # Retrieve relevant memories
            relevant_memories = []
            if self.memory_manager:
                relevant_memories = await self.memory_manager.retrieve_relevant_memories(
                    request, session_id, limit=10
                )
            
            # Build context
            enriched_context = {
                "request": request,
                "session_id": session_id,
                "memories": relevant_memories,
                "user_context": context or {}
            }
            
            # Stream response from execution engine
            full_response = ""
            async for chunk in self.execution_engine.execute_streaming_request(
                request, session_id, enriched_context
            ):
                full_response += chunk
                yield chunk
            
            # Store interaction in memory after completion
            if self.memory_manager and full_response:
                # Create a simple result object for memory storage
                from ..models.execution import ExecutionResult as ModelExecutionResult
                memory_result = ModelExecutionResult(
                    content=full_response,
                    status=ExecutionStatus.COMPLETED,
                    session_id=session_id,
                    execution_time=0.0
                )
                await self._store_interaction_memory(request, memory_result, session_id)
                
        except Exception as e:
            logger.error(f"Streaming request processing failed: {e}", exc_info=True)
            yield f'{{"error": "{str(e)}"}}'
    
    async def _store_interaction_memory(
        self,
        request: str,
        result: ExecutionResult,
        session_id: str
    ) -> None:
        """Store interaction in memory for future reference."""
        try:
            # Store user request
            await self.memory_manager.store_memory(
                {"request": request, "session_id": session_id, "timestamp": time.time()},
                "user_request",
                importance="medium"
            )
            
            # Store assistant response
            await self.memory_manager.store_memory(
                {
                    "response": result.content,
                    "session_id": session_id,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "timestamp": time.time()
                },
                "assistant_response",
                importance="medium"
            )
            
            # Extract and store any important facts or insights
            if result.metadata and "extracted_facts" in result.metadata:
                for fact in result.metadata["extracted_facts"]:
                    await self.memory_manager.store_memory(
                        {"fact": fact, "session_id": session_id},
                        "fact",
                        importance="high"
                    )
                    
        except Exception as e:
            logger.warning(f"Failed to store interaction memory: {e}")
    
    def _update_response_stats(self, execution_time: float, success: bool) -> None:
        """Update response time statistics."""
        if success:
            self.stats.successful_requests += 1
        else:
            self.stats.failed_requests += 1
        
        # Track response times for average calculation
        self._response_times.append(execution_time)
        if len(self._response_times) > 1000:  # Keep last 1000 times
            self._response_times = self._response_times[-1000:]
        
        # Calculate average response time
        if self._response_times:
            self.stats.average_response_time = sum(self._response_times) / len(self._response_times)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        current_time = time.time()
        self.stats.uptime_seconds = current_time - self.start_time
        
        # Get memory statistics
        if self.memory_manager:
            memory_stats = self.memory_manager.get_stats()
            self.stats.total_memory_items = memory_stats.get("total_items", 0)
            self.stats.active_sessions = memory_stats.get("active_sessions", 0)
        
        return self.stats.to_dict()
    
    async def export_system_state(self) -> Dict[str, Any]:
        """Export complete system state for backup/migration."""
        logger.info("Exporting system state...")
        
        export_data = {
            "version": "0.1.0",
            "timestamp": time.time(),
            "config": self.config.to_dict(),
            "stats": self.get_system_stats()
        }
        
        # Export memory data
        if self.memory_manager:
            export_data["memory"] = await self.memory_manager.export_memories()
        
        # Export prompt templates
        if self.prompt_manager:
            export_data["prompts"] = self.prompt_manager.get_export_data()
        
        logger.info("System state export complete")
        return export_data
    
    async def import_system_state(self, import_data: Dict[str, Any]) -> Dict[str, int]:
        """Import system state from backup/migration data."""
        logger.info("Importing system state...")
        
        imported_items = {"memories": 0, "templates": 0}
        
        try:
            # Import memory data
            if "memory" in import_data and self.memory_manager:
                count = await self.memory_manager.import_memories(import_data["memory"])
                imported_items["memories"] = count
                logger.info(f"Imported {count} memory items")
            
            # Import prompt templates
            if "prompts" in import_data and self.prompt_manager:
                # For now, just return 0 since import_templates writes to file
                imported_items["templates"] = 0
                logger.info("Imported 0 prompt templates")
            
            logger.info("System state import complete")
            return imported_items
            
        except Exception as e:
            logger.error(f"Failed to import system state: {e}")
            raise ExecutionError(f"Import failed: {e}")
    
    async def cleanup(self) -> None:
        """Clean up system resources."""
        logger.info("Cleaning up OpenGuidance system...")
        
        try:
            if self.execution_engine:
                await self.execution_engine.cleanup()
                logger.debug("Execution engine cleaned up")
            
            if self.memory_manager:
                await self.memory_manager.cleanup()
                logger.debug("Memory manager cleaned up")
            
            if self.prompt_manager:
                await self.prompt_manager.cleanup()
                logger.debug("Prompt manager cleaned up")
            
            self.is_initialized = False
            logger.info("OpenGuidance system cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()