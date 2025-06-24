"""
Advanced memory management system with intelligent context retrieval.
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import logging
from pathlib import Path
import pickle
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory items."""
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    CONTEXT = "context"
    SYSTEM = "system"
    USER_PREFERENCE = "user_preference"
    CODE_EXECUTION = "code_execution"
    ERROR = "error"


@dataclass
class MemoryItem:
    """Individual memory item with metadata."""
    
    id: str
    content: str
    content_type: str
    session_id: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    importance_score: float = 1.0
    tags: List[str] = field(default_factory=list)
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ConversationMemory:
    """Conversation-specific memory container."""
    
    session_id: str
    turns: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    topics: List[str] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add conversation turn."""
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        self.turns.append(turn)
        self.updated_at = datetime.utcnow()


@dataclass
class ContextMemory:
    """Long-term contextual memory storage."""
    
    domain: str
    knowledge_items: List[MemoryItem] = field(default_factory=list)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    concepts: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class MemoryStorage(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    async def store(self, memory: MemoryItem) -> str:
        """Store a memory item and return its ID."""
        pass
    
    @abstractmethod
    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search for memories matching the query."""
        pass
    
    @abstractmethod
    async def get_session_memories(self, session_id: str, limit: int = 50) -> List[MemoryItem]:
        """Get memories for a specific session."""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete all memories for a session."""
        pass
    
    @abstractmethod
    async def cleanup_old_memories(self, max_age_days: int = 30) -> int:
        """Clean up old memories and return count deleted."""
        pass


class FileMemoryStorage(MemoryStorage):
    """File-based memory storage implementation."""
    
    def __init__(self, storage_path: str = "./data/memories"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memories: Dict[str, MemoryItem] = {}
        self.session_index: Dict[str, List[str]] = {}
        self._load_memories()
    
    def _generate_id(self, memory: MemoryItem) -> str:
        """Generate unique ID for memory item."""
        content_hash = hashlib.md5(memory.content.encode()).hexdigest()
        timestamp_str = str(memory.created_at.timestamp())
        return f"{memory.session_id}_{timestamp_str}_{content_hash[:8]}"
    
    def _load_memories(self):
        """Load existing memories from disk."""
        try:
            memories_file = self.storage_path / "memories.json"
            if memories_file.exists():
                with open(memories_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for memory_id, memory_data in data.items():
                        memory = MemoryItem.from_dict(memory_data)
                        self.memories[memory_id] = memory
                        
                        # Update session index
                        if memory.session_id not in self.session_index:
                            self.session_index[memory.session_id] = []
                        self.session_index[memory.session_id].append(memory_id)
                
                logger.info(f"Loaded {len(self.memories)} memories from disk")
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
    
    def _save_memories(self):
        """Save memories to disk."""
        try:
            memories_file = self.storage_path / "memories.json"
            data = {memory_id: memory.to_dict() for memory_id, memory in self.memories.items()}
            
            with open(memories_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    async def store(self, memory: MemoryItem) -> str:
        """Store a memory item."""
        memory_id = self._generate_id(memory)
        self.memories[memory_id] = memory
        
        # Update session index
        if memory.session_id not in self.session_index:
            self.session_index[memory.session_id] = []
        self.session_index[memory.session_id].append(memory_id)
        
        # Save to disk
        self._save_memories()
        
        logger.debug(f"Stored memory {memory_id} for session {memory.session_id}")
        return memory_id
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        return self.memories.get(memory_id)
    
    async def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search for memories matching the query."""
        results = []
        query_lower = query.lower()
        
        for memory in self.memories.values():
            if query_lower in memory.content.lower():
                results.append(memory)
        
        # Sort by timestamp (most recent first) and limit
        results.sort(key=lambda m: m.created_at, reverse=True)
        return results[:limit]
    
    async def get_session_memories(self, session_id: str, limit: int = 50) -> List[MemoryItem]:
        """Get memories for a specific session."""
        if session_id not in self.session_index:
            return []
        
        memories = []
        for memory_id in self.session_index[session_id]:
            if memory_id in self.memories:
                memories.append(self.memories[memory_id])
        
        # Sort by timestamp and limit
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]
    
    async def delete_session(self, session_id: str) -> None:
        """Delete all memories for a session."""
        if session_id not in self.session_index:
            return
        
        # Remove memories
        for memory_id in self.session_index[session_id]:
            if memory_id in self.memories:
                del self.memories[memory_id]
        
        # Remove from session index
        del self.session_index[session_id]
        
        # Save changes
        self._save_memories()
        
        logger.info(f"Deleted all memories for session {session_id}")
    
    async def cleanup_old_memories(self, max_age_days: int = 30) -> int:
        """Clean up old memories."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        deleted_count = 0
        
        # Find old memories
        old_memory_ids = []
        for memory_id, memory in self.memories.items():
            if memory.created_at.timestamp() < cutoff_time:
                old_memory_ids.append(memory_id)
        
        # Delete old memories
        for memory_id in old_memory_ids:
            memory = self.memories[memory_id]
            
            # Remove from memories
            del self.memories[memory_id]
            
            # Remove from session index
            if memory.session_id in self.session_index:
                if memory_id in self.session_index[memory.session_id]:
                    self.session_index[memory.session_id].remove(memory_id)
                
                # Remove empty session entries
                if not self.session_index[memory.session_id]:
                    del self.session_index[memory.session_id]
            
            deleted_count += 1
        
        if deleted_count > 0:
            self._save_memories()
            logger.info(f"Cleaned up {deleted_count} old memories")
        
        return deleted_count


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_items: int = 10000
    cleanup_interval: int = 3600  # seconds
    storage_backend: str = "file"
    similarity_threshold: float = 0.7
    file_storage_path: str = "./data/memories"
    enable_embeddings: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class MemoryManager:
    """Advanced memory management system with intelligent retrieval."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.storage = self._create_storage()
        self.is_initialized = False
        self._cleanup_task = None
    
    def _create_storage(self) -> MemoryStorage:
        """Create storage backend based on configuration."""
        if self.config.storage_backend == "file":
            return FileMemoryStorage(self.config.file_storage_path)
        else:
            raise ValueError(f"Unsupported storage backend: {self.config.storage_backend}")
    
    async def initialize(self) -> None:
        """Initialize the memory manager."""
        if self.is_initialized:
            logger.warning("MemoryManager already initialized")
            return
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.is_initialized = True
        logger.info("MemoryManager initialization completed")
    
    async def cleanup(self) -> None:
        """Cleanup memory manager resources."""
        if not self.is_initialized:
            return
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.is_initialized = False
        logger.info("MemoryManager cleanup completed")
    
    async def store_memory(self, content: str, memory_type: str, session_id: str, 
                          importance: float = 1.0, metadata: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None) -> str:
        """Store a new memory item."""
        memory = MemoryItem(
            id=f"{session_id}_{int(time.time())}_{hashlib.md5(content.encode()).hexdigest()[:8]}",
            content=content,
            content_type=memory_type,
            session_id=session_id,
            importance_score=importance,
            metadata=metadata or {},
            tags=tags or []
        )
        
        return await self.storage.store(memory)
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a specific memory item."""
        return await self.storage.retrieve(memory_id)
    
    async def search_memories(self, query: str, session_id: Optional[str] = None, 
                            limit: int = 10) -> List[MemoryItem]:
        """Search for relevant memories."""
        if session_id:
            # Search within session
            session_memories = await self.storage.get_session_memories(session_id, limit * 2)
            results = []
            query_lower = query.lower()
            
            for memory in session_memories:
                if query_lower in memory.content.lower():
                    results.append(memory)
            
            return results[:limit]
        else:
            # Global search
            return await self.storage.search(query, limit)
    
    async def get_session_memories(self, session_id: str, limit: int = 50) -> List[MemoryItem]:
        """Get all memories for a session."""
        return await self.storage.get_session_memories(session_id, limit)
    
    async def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        memories = await self.get_session_memories(session_id, limit)
        
        history = []
        for memory in memories:
            history.append({
                "timestamp": memory.created_at.timestamp(),
                "content": memory.content,
                "type": memory.content_type,
                "metadata": memory.metadata,
                "tags": memory.tags
            })
        
        return history
    
    async def delete_session(self, session_id: str) -> None:
        """Delete all memories for a session."""
        await self.storage.delete_session(session_id)
    
    async def clear_session(self, session_id: str) -> None:
        """Clear session data (alias for delete_session)."""
        await self.delete_session(session_id)
    
    async def store_memory_dict(self, memory_data: Dict[str, Any]) -> str:
        """Store memory from dictionary data."""
        return await self.store_memory(
            content=memory_data.get("content", ""),
            memory_type=memory_data.get("type", "general"),
            session_id=memory_data.get("session_id", "default"),
            importance=memory_data.get("importance", 1.0),
            metadata=memory_data.get("metadata", {}),
            tags=memory_data.get("tags", [])
        )
    
    async def retrieve_relevant_memories(
        self,
        query: str,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to the query and session."""
        relevant = []
        
        # Search session memories
        session_memories = await self.search_memories(query, session_id, limit)
        for memory in session_memories:
            relevant.append({
                "type": memory.content_type,
                "content": memory.content,
                "session_id": memory.session_id,
                "importance": memory.importance_score
            })
        
        # If we need more, search globally
        if len(relevant) < limit:
            global_memories = await self.search_memories(query, None, limit - len(relevant))
            for memory in global_memories:
                if memory.session_id != session_id:  # Avoid duplicates
                    relevant.append({
                        "type": memory.content_type,
                        "content": memory.content,
                        "session_id": memory.session_id,
                        "importance": memory.importance_score
                    })
        
        return relevant[:limit]
    
    async def store_interaction(
        self,
        session_id: str,
        user_input: str,
        assistant_response: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Store a complete interaction in memory."""
        # Store user input
        await self.store_memory(
            content=user_input,
            memory_type="user_input",
            session_id=session_id,
            metadata=metadata or {}
        )
        
        # Store assistant response
        await self.store_memory(
            content=assistant_response,
            memory_type="assistant_response",
            session_id=session_id,
            metadata=metadata or {}
        )
    
    async def retrieve_context(
        self,
        session_id: str,
        query: str,
        max_items: int = 10
    ) -> str:
        """Retrieve contextual information for a query."""
        memories = await self.retrieve_relevant_memories(query, session_id, max_items)
        
        if not memories:
            return ""
        
        context_parts = []
        for memory in memories:
            context_parts.append(f"[{memory['type']}] {memory['content']}")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if isinstance(self.storage, FileMemoryStorage):
            return {
                "total_memories": len(self.storage.memories),
                "total_sessions": len(self.storage.session_index),
                "storage_backend": self.config.storage_backend,
                "cleanup_interval": self.config.cleanup_interval
            }
        else:
            return {
                "storage_backend": self.config.storage_backend,
                "cleanup_interval": self.config.cleanup_interval
            }
    
    async def export_memories(self) -> Dict[str, Any]:
        """Export all memory data."""
        export_data = {
            "config": {
                "storage_backend": self.config.storage_backend,
                "max_memory_items": self.config.max_memory_items
            },
            "memories": [],
            "sessions": []
        }
        
        if isinstance(self.storage, FileMemoryStorage):
            # Export memories
            for memory_id, memory in self.storage.memories.items():
                export_data["memories"].append({
                    "id": memory_id,
                    "content": memory.content,
                    "content_type": memory.content_type,
                    "session_id": memory.session_id,
                    "importance_score": memory.importance_score,
                    "created_at": memory.created_at.isoformat(),
                    "tags": memory.tags,
                    "metadata": memory.metadata
                })
            
            # Export session index
            export_data["sessions"] = dict(self.storage.session_index)
        
        return export_data
    
    async def import_memories(self, import_data: Dict[str, Any]) -> int:
        """Import memory data."""
        count = 0
        
        if isinstance(self.storage, FileMemoryStorage):
            # Import memories
            for memory_data in import_data.get("memories", []):
                memory = MemoryItem(
                    id=memory_data["id"],
                    content=memory_data["content"],
                    content_type=memory_data["content_type"],
                    session_id=memory_data["session_id"],
                    importance_score=memory_data.get("importance_score", 1.0),
                    created_at=datetime.fromisoformat(memory_data["created_at"]),
                    tags=memory_data.get("tags", []),
                    metadata=memory_data.get("metadata", {})
                )
                
                self.storage.memories[memory.id] = memory
                
                # Update session index
                if memory.session_id not in self.storage.session_index:
                    self.storage.session_index[memory.session_id] = []
                if memory.id not in self.storage.session_index[memory.session_id]:
                    self.storage.session_index[memory.session_id].append(memory.id)
                
                count += 1
            
            # Save changes
            self.storage._save_memories()
        
        logger.info(f"Imported {count} memories")
        return count
    
    async def _cleanup_loop(self):
        """Background cleanup task."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                deleted_count = await self.storage.cleanup_old_memories()
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old memories")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory cleanup error: {e}")