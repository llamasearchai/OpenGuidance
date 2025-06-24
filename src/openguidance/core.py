"""
Core guidance engine implementation with advanced context management.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import with explicit typing support
if TYPE_CHECKING:
    from .memory import MemoryManager
    from .validation import ValidationEngine
    from .monitoring import MetricsCollector
else:
    from .memory import MemoryManager
    from .validation import ValidationEngine
    from .monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class GuidanceMode(Enum):
    """Guidance execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    STREAMING = "streaming"


@dataclass
class GuidanceContext:
    """Enhanced context container for guidance operations."""
    
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    system_context: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_context(self, **kwargs) -> None:
        """Update context with new data."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()
    
    def add_conversation_turn(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a conversation turn to history."""
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        self.conversation_history.append(turn)
        self.updated_at = datetime.utcnow()


@dataclass
class GuidanceResult:
    """Comprehensive result container for guidance operations."""
    
    content: str
    confidence: float
    execution_time: float
    token_usage: Dict[str, int]
    validation_results: Any  # Can be ValidationReport or Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    context_id: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if guidance execution was successful."""
        return self.error is None and self.confidence > 0.5


class GuidanceEngine:
    """
    Advanced AI guidance engine with sophisticated context management,
    validation, and optimization capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        memory_manager: Optional["MemoryManager"] = None,
        validation_engine: Optional["ValidationEngine"] = None,
        metrics_collector: Optional["MetricsCollector"] = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.model_name = model_name
        
        # Initialize components with explicit typing and error handling
        if memory_manager is not None:
            self.memory_manager: "MemoryManager" = memory_manager
        else:
            # Create default MemoryManager with optional parameters
            self.memory_manager = MemoryManager()
        
        if validation_engine is not None:
            self.validation_engine: "ValidationEngine" = validation_engine
        else:
            self.validation_engine = ValidationEngine()
            
        if metrics_collector is not None:
            self.metrics_collector: "MetricsCollector" = metrics_collector
        else:
            # Create MetricsCollector with optional config
            try:
                self.metrics_collector = MetricsCollector()
            except TypeError:
                # Handle case where config is required
                self.metrics_collector = MetricsCollector(config={})
        
        self.max_retries = max_retries
        self.timeout = timeout
        
        self._active_contexts: Dict[str, GuidanceContext] = {}
        self._execution_hooks: Dict[str, List[Callable]] = {
            "pre_execution": [],
            "post_execution": [],
            "error_handling": []
        }
        
        logger.info(f"GuidanceEngine initialized with model: {model_name}")
    
    def register_hook(self, event: str, callback: Callable) -> None:
        """Register execution hooks for extensibility."""
        if event in self._execution_hooks:
            self._execution_hooks[event].append(callback)
        else:
            raise ValueError(f"Unknown hook event: {event}")
    
    async def create_context(self, **kwargs) -> GuidanceContext:
        """Create and register a new guidance context."""
        context = GuidanceContext(**kwargs)
        self._active_contexts[context.session_id] = context
        
        # Initialize memory for this context with method existence check
        if hasattr(self.memory_manager, 'initialize_session') and callable(getattr(self.memory_manager, 'initialize_session')):
            try:
                await self.memory_manager.initialize_session(context.session_id)
            except Exception as e:
                logger.warning(f"Failed to initialize memory session: {e}")
        else:
            logger.debug("Memory manager does not support session initialization")
        
        logger.debug(f"Created guidance context: {context.session_id}")
        return context
    
    async def execute_guidance(
        self,
        prompt: str,
        context: GuidanceContext,
        mode: GuidanceMode = GuidanceMode.SEQUENTIAL,
        **kwargs
    ) -> GuidanceResult:
        """
        Execute guidance with comprehensive error handling and validation.
        """
        start_time = datetime.utcnow()
        
        try:
            # Execute pre-execution hooks
            for hook in self._execution_hooks["pre_execution"]:
                await self._safe_hook_execution(hook, context, prompt)
            
            # Main execution logic
            result = await self._execute_with_retry(prompt, context, mode, **kwargs)
            
            # Validate result
            context_dict = {
                "session_id": context.session_id,
                "user_id": context.user_id,
                "conversation_history": context.conversation_history,
                "system_context": context.system_context,
                "execution_context": context.execution_context,
                "metadata": context.metadata
            }
            validation_results = await self.validation_engine.validate_response(
                result.content, context_dict
            )
            result.validation_results = validation_results
            
            # Execute post-execution hooks
            for hook in self._execution_hooks["post_execution"]:
                await self._safe_hook_execution(hook, context, result)
            
            # Update context with result
            context.add_conversation_turn("assistant", result.content, {
                "confidence": result.confidence,
                "validation": validation_results
            })
            
            # Collect metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            await self.metrics_collector.record_execution(
                context.session_id, execution_time, result.success
            )
            
            result.execution_time = execution_time
            result.context_id = context.session_id
            
            logger.info(f"Guidance executed successfully for context: {context.session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Guidance execution failed: {str(e)}")
            
            # Execute error hooks
            for hook in self._execution_hooks["error_handling"]:
                await self._safe_hook_execution(hook, context, e)
            
            return GuidanceResult(
                content="",
                confidence=0.0,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                token_usage={},
                validation_results={},
                error=str(e),
                context_id=context.session_id
            )
    
    async def stream_guidance(
        self,
        prompt: str,
        context: GuidanceContext,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream guidance response in real-time."""
        try:
            # Simulate streaming response (in real implementation, this would connect to actual model)
            response_chunks = self._simulate_streaming_response(prompt, context)
            
            full_response = ""
            async for chunk in response_chunks:
                full_response += chunk
                yield chunk
            
            # Update context with complete response
            context.add_conversation_turn("assistant", full_response)
            
        except Exception as e:
            logger.error(f"Streaming guidance failed: {str(e)}")
            yield f"Error: {str(e)}"
    
    async def _execute_with_retry(
        self,
        prompt: str,
        context: GuidanceContext,
        mode: GuidanceMode,
        **kwargs
    ) -> GuidanceResult:
        """Execute guidance with retry logic."""
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            try:
                return await self._core_execution(prompt, context, mode, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Guidance execution attempt {attempt + 1} failed: {str(e)}")
                
                # Exponential backoff
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        # Ensure we always raise a valid exception
        if last_error is not None:
            raise last_error
        else:
            raise Exception("Maximum retry attempts exceeded")
    
    async def _core_execution(
        self,
        prompt: str,
        context: GuidanceContext,
        mode: GuidanceMode,
        **kwargs
    ) -> GuidanceResult:
        """Core guidance execution logic."""
        # Retrieve relevant memory
        memory_context = await self.memory_manager.retrieve_context(
            context.session_id, prompt
        )
        
        # Enhance prompt with memory and context
        enhanced_prompt = self._enhance_prompt(prompt, context, memory_context)
        
        # Simulate model execution (in real implementation, this would call actual model)
        response_content = await self._simulate_model_call(enhanced_prompt, context)
        
        # Calculate confidence based on response quality
        confidence = self._calculate_confidence(response_content, context)
        
        # Store in memory
        await self.memory_manager.store_interaction(
            context.session_id, prompt, response_content
        )
        
        return GuidanceResult(
            content=response_content,
            confidence=confidence,
            execution_time=0.0,  # Will be set by caller
            token_usage={"prompt_tokens": len(enhanced_prompt.split()), "completion_tokens": len(response_content.split())},
            validation_results={}
        )
    
    def _enhance_prompt(self, prompt: str, context: GuidanceContext, memory_context: str) -> str:
        """Enhance prompt with context and memory."""
        enhanced = f"""Context: {context.system_context}
Memory: {memory_context}
Conversation History: {context.conversation_history[-5:] if context.conversation_history else 'None'}

User Query: {prompt}

Please provide a comprehensive, accurate response based on the context provided."""
        
        return enhanced
    
    async def _simulate_model_call(self, prompt: str, context: GuidanceContext) -> str:
        """Simulate model API call (replace with actual implementation)."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate response based on prompt content
        if "code" in prompt.lower():
            return "Here's a Python implementation that addresses your requirements:\n\n\ndef solution():\n    return 'Implementation complete'\n"
        elif "explain" in prompt.lower():
            return "Based on the context provided, here's a detailed explanation of the concept with relevant examples and best practices."
        else:
            return "I understand your request. Here's a comprehensive response that addresses your specific needs with actionable insights."
    
    async def _simulate_streaming_response(self, prompt: str, context: GuidanceContext) -> AsyncGenerator[str, None]:
        """Simulate streaming response chunks."""
        response = await self._simulate_model_call(prompt, context)
        words = response.split()
        
        for i, word in enumerate(words):
            await asyncio.sleep(0.05)  # Simulate streaming delay
            yield word + (" " if i < len(words) - 1 else "")
    
    def _calculate_confidence(self, response: str, context: GuidanceContext) -> float:
        """Calculate confidence score for response."""
        base_confidence = 0.8
        
        # Adjust based on response length
        if len(response) < 10:
            base_confidence -= 0.3
        elif len(response) > 100:
            base_confidence += 0.1
        
        # Adjust based on context history
        if len(context.conversation_history) > 5:
            base_confidence += 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    async def _safe_hook_execution(self, hook: Callable, *args) -> None:
        """Safely execute hooks without affecting main execution."""
        try:
            if asyncio.iscoroutinefunction(hook):
                await hook(*args)
            else:
                hook(*args)
        except Exception as e:
            logger.warning(f"Hook execution failed: {str(e)}")
    
    async def cleanup_context(self, session_id: str) -> None:
        """Clean up resources for a context."""
        if session_id in self._active_contexts:
            del self._active_contexts[session_id]
            
            # Clean up memory session with method existence check
            if hasattr(self.memory_manager, 'cleanup_session') and callable(getattr(self.memory_manager, 'cleanup_session')):
                try:
                    await self.memory_manager.cleanup_session(session_id)
                except Exception as e:
                    logger.warning(f"Failed to cleanup memory session: {e}")
            else:
                logger.debug("Memory manager does not support session cleanup")
                
            logger.debug(f"Cleaned up context: {session_id}")
    
    def get_active_contexts(self) -> List[str]:
        """Get list of active context IDs."""
        return list(self._active_contexts.keys())