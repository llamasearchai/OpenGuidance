"""
Execution-related data models.
"""

import time
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


class ExecutionStatus(Enum):
    """Execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of request execution."""
    content: str
    status: ExecutionStatus
    session_id: str
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "status": self.status.value,
            "session_id": self.session_id,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "error_message": self.error_message,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """Create from dictionary representation."""
        data = data.copy()
        data['status'] = ExecutionStatus(data['status'])
        return cls(**data)


class ExecutionError(Exception):
    """Custom exception for execution errors."""
    
    def __init__(
        self,
        message: str,
        error_type: str = "execution_error",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "message": self.message,
            "error_type": self.error_type,
            "details": self.details,
            "timestamp": self.timestamp
        }


@dataclass
class CodeExecutionResult:
    """Result of code execution."""
    output: str
    error: Optional[str] = None
    return_value: Any = None
    execution_time: float = 0.0
    memory_usage: int = 0  # bytes
    exit_code: int = 0
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.exit_code == 0 and self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "output": self.output,
            "error": self.error,
            "return_value": str(self.return_value) if self.return_value is not None else None,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "exit_code": self.exit_code,
            "success": self.success
        }


@dataclass
class RequestContext:
    """Context information for request processing."""
    session_id: str
    user_message: str
    relevant_memories: List[Dict[str, Any]] = field(default_factory=list)
    user_context: Dict[str, Any] = field(default_factory=dict)
    system_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "user_message": self.user_message,
            "relevant_memories": self.relevant_memories,
            "user_context": self.user_context,
            "system_context": self.system_context,
            "timestamp": self.timestamp
        }