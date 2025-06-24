"""
Memory-related data models.
"""

import time
import json
import hashlib
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


class MemoryType(Enum):
    """Types of memories that can be stored."""
    CONVERSATION = "conversation"
    FACT = "fact"
    PREFERENCE = "preference"
    SKILL = "skill"
    CONTEXT = "context"
    USER_REQUEST = "user_request"
    ASSISTANT_RESPONSE = "assistant_response"
    CODE_EXECUTION = "code_execution"
    ERROR = "error"
    SYSTEM_EVENT = "system_event"


class MemoryImportance(Enum):
    """Memory importance levels."""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Memory:
    """Memory item data structure."""
    content: Dict[str, Any]
    memory_type: MemoryType
    importance: MemoryImportance
    session_id: Optional[str] = None
    memory_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.memory_id is None:
            self.memory_id = self._generate_id()
        
        if isinstance(self.memory_type, str):
            self.memory_type = MemoryType(self.memory_type)
        
        if isinstance(self.importance, str):
            self.importance = MemoryImportance(self.importance)
    
    def _generate_id(self) -> str:
        """Generate unique memory ID."""
        content_str = json.dumps(
            self.content, 
            sort_keys=True, 
            default=str
        )
        
        hash_input = f"{content_str}{self.session_id}{self.timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def update_access(self) -> None:
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def get_searchable_text(self) -> str:
        """Get text representation for search indexing."""
        text_parts = []
        
        # Extract text from content
        def extract_text(obj, prefix=""):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, (dict, list)):
                        extract_text(value, f"{prefix}{key}.")
            elif isinstance(obj, list):
                for item in obj:
                    extract_text(item, prefix)
        
        extract_text(self.content)
        
        # Add tags
        if self.tags:
            text_parts.extend(self.tags)
        
        # Add metadata text
        if self.metadata:
            extract_text(self.metadata)
        
        return " ".join(text_parts)
    
    def calculate_relevance_score(
        self, 
        query: str, 
        current_time: Optional[float] = None
    ) -> float:
        """
        Calculate relevance score for a query.
        
        Factors:
        - Text similarity (placeholder - would use actual embeddings)
        - Recency
        - Importance
        - Access frequency
        """
        if current_time is None:
            current_time = time.time()
        
        # Text similarity (simplified - would use embeddings in real implementation)
        searchable_text = self.get_searchable_text().lower()
        query_lower = query.lower()
        
        query_words = set(query_lower.split())
        content_words = set(searchable_text.split())
        
        if not query_words or not content_words:
            text_score = 0.0
        else:
            intersection = query_words.intersection(content_words)
            text_score = len(intersection) / len(query_words.union(content_words))
        
        # Recency score (decay over 30 days)
        age_hours = (current_time - self.timestamp) / 3600
        recency_score = max(0, 1 - (age_hours / (30 * 24)))
        
        # Importance score
        importance_scores = {
            MemoryImportance.LOW: 0.25,
            MemoryImportance.MEDIUM: 0.5,
            MemoryImportance.HIGH: 0.75,
            MemoryImportance.CRITICAL: 1.0
        }
        importance_score = importance_scores[self.importance]
        
        # Access frequency score
        access_score = min(1.0, self.access_count / 10.0)
        
        # Combine scores with weights
        total_score = (
            text_score * 0.4 +
            recency_score * 0.3 +
            importance_score * 0.2 +
            access_score * 0.1
        )
        
        return total_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary representation."""
        data = data.copy()
        
        # Convert enum strings back to enums
        if 'memory_type' in data:
            data['memory_type'] = MemoryType(data['memory_type'])
        if 'importance' in data:
            data['importance'] = MemoryImportance(data['importance'])
        
        return cls(**data)


@dataclass
class MemoryQuery:
    """Query structure for memory retrieval."""
    query_text: str
    session_id: Optional[str] = None
    memory_types: Optional[List[MemoryType]] = None
    importance_threshold: MemoryImportance = MemoryImportance.LOW
    limit: int = 10
    time_range: Optional[tuple] = None  # (start_time, end_time)
    tags: Optional[List[str]] = None
    
    def matches_memory(self, memory: Memory) -> bool:
        """Check if memory matches query criteria."""
        # Session filter
        if self.session_id and memory.session_id != self.session_id:
            return False
        
        # Memory type filter
        if self.memory_types and memory.memory_type not in self.memory_types:
            return False
        
        # Importance threshold
        importance_order = [
            MemoryImportance.LOW,
            MemoryImportance.MEDIUM,
            MemoryImportance.HIGH,
            MemoryImportance.CRITICAL
        ]
        
        if importance_order.index(memory.importance) < importance_order.index(self.importance_threshold):
            return False
        
        # Time range filter
        if self.time_range:
            start_time, end_time = self.time_range
            if not (start_time <= memory.timestamp <= end_time):
                return False
        
        # Tags filter
        if self.tags:
            if not any(tag in memory.tags for tag in self.tags):
                return False
        
        return True


@dataclass
class MemoryStats:
    """Memory system statistics."""
    total_memories: int = 0
    memories_by_type: Dict[str, int] = field(default_factory=dict)
    memories_by_importance: Dict[str, int] = field(default_factory=dict)
    active_sessions: int = 0
    average_memory_age: float = 0.0
    most_accessed_memory: Optional[str] = None
    storage_size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_memories": self.total_memories,
            "memories_by_type": self.memories_by_type,
            "memories_by_importance": self.memories_by_importance,
            "active_sessions": self.active_sessions,
            "average_memory_age": self.average_memory_age,
            "most_accessed_memory": self.most_accessed_memory,
            "storage_size_bytes": self.storage_size_bytes
        }