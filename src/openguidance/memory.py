"""
Advanced memory management system with intelligent context retrieval.
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Individual memory item with metadata."""
    
    id: str
    content: str
    content_type: str
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


class MemoryManager:
    """
    Comprehensive memory management system with intelligent retrieval,
    summarization, and optimization capabilities.
    """
    
    def __init__(
        self,
        max_conversation_turns: int = 50,
        max_memory_items: int = 1000,
        embedding_dimension: int = 384
    ):
        self.max_conversation_turns = max_conversation_turns
        self.max_memory_items = max_memory_items
        self.embedding_dimension = embedding_dimension
        
        self._conversation_memories: Dict[str, ConversationMemory] = {}
        self._context_memories: Dict[str, ContextMemory] = {}
        self._global_memory: List[MemoryItem] = []
        self._memory_index: Dict[str, List[str]] = defaultdict(list)
        
        # Performance tracking
        self._retrieval_stats = {
            "total_retrievals": 0,
            "cache_hits": 0,
            "average_retrieval_time": 0.0
        }
        
        logger.info("MemoryManager initialized")
    
    async def initialize_session(self, session_id: str) -> None:
        """Initialize memory for a new session."""
        if session_id not in self._conversation_memories:
            self._conversation_memories[session_id] = ConversationMemory(session_id=session_id)
            logger.debug(f"Initialized memory for session: {session_id}")
    
    async def store_interaction(
        self,
        session_id: str,
        user_input: str,
        assistant_response: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Store a complete interaction in memory."""
        if session_id not in self._conversation_memories:
            await self.initialize_session(session_id)
        
        memory = self._conversation_memories[session_id]
        
        # Add user turn
        memory.add_turn("user", user_input, metadata)
        
        # Add assistant turn
        memory.add_turn("assistant", assistant_response, metadata)
        
        # Trim conversation if too long
        if len(memory.turns) > self.max_conversation_turns:
            # Keep recent turns and summarize old ones
            await self._summarize_old_turns(memory)
        
        # Extract and store important information
        await self._extract_knowledge(session_id, user_input, assistant_response)
        
        logger.debug(f"Stored interaction for session: {session_id}")
    
    async def retrieve_context(
        self,
        session_id: str,
        query: str,
        max_items: int = 10
    ) -> str:
        """Retrieve relevant context for a query."""
        start_time = datetime.utcnow()
        
        try:
            # Get conversation context
            conversation_context = await self._get_conversation_context(session_id)
            
            # Get relevant knowledge items
            knowledge_context = await self._retrieve_relevant_knowledge(query, max_items)
            
            # Combine contexts
            combined_context = self._combine_contexts(conversation_context, knowledge_context)
            
            # Update retrieval stats
            retrieval_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_retrieval_stats(retrieval_time)
            
            return combined_context
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return ""
    
    async def _get_conversation_context(self, session_id: str) -> str:
        """Get recent conversation context."""
        if session_id not in self._conversation_memories:
            return ""
        
        memory = self._conversation_memories[session_id]
        
        # Get recent turns (last 10)
        recent_turns = memory.turns[-10:] if memory.turns else []
        
        # Format conversation context
        context_parts = []
        if memory.summary:
            context_parts.append(f"Previous conversation summary: {memory.summary}")
        
        if recent_turns:
            context_parts.append("Recent conversation:")
            for turn in recent_turns:
                context_parts.append(f"{turn['role']}: {turn['content']}")
        
        return "\n".join(context_parts)
    
    async def _retrieve_relevant_knowledge(self, query: str, max_items: int) -> str:
        """Retrieve relevant knowledge items for query."""
        # Simple keyword-based retrieval (in production, use semantic search)
        query_words = set(query.lower().split())
        
        relevant_items = []
        for item in self._global_memory:
            item_words = set(item.content.lower().split())
            overlap = len(query_words.intersection(item_words))
            
            if overlap > 0:
                relevance_score = overlap / len(query_words)
                relevant_items.append((item, relevance_score))
        
        # Sort by relevance and recency
        relevant_items.sort(key=lambda x: (x[1], x[0].last_accessed), reverse=True)
        
        # Format knowledge context
        context_parts = []
        for item, score in relevant_items[:max_items]:
            item.update_access()  # Update access statistics
            context_parts.append(f"Knowledge: {item.content}")
        
        return "\n".join(context_parts)
    
    def _combine_contexts(self, conversation_context: str, knowledge_context: str) -> str:
        """Combine different context sources."""
        contexts = []
        
        if conversation_context:
            contexts.append(conversation_context)
        
        if knowledge_context:
            contexts.append(knowledge_context)
        
        return "\n\n".join(contexts)
    
    async def _summarize_old_turns(self, memory: ConversationMemory) -> None:
        """Summarize old conversation turns to save space."""
        if len(memory.turns) <= self.max_conversation_turns:
            return
        
        # Keep recent turns, summarize old ones
        turns_to_summarize = memory.turns[:-self.max_conversation_turns//2]
        memory.turns = memory.turns[-self.max_conversation_turns//2:]
        
        # Create summary of old turns
        summary_content = []
        for turn in turns_to_summarize:
            summary_content.append(f"{turn['role']}: {turn['content'][:100]}...")
        
        new_summary = f"Previous conversation ({len(turns_to_summarize)} turns): " + "; ".join(summary_content)
        
        if memory.summary:
            memory.summary = f"{memory.summary}\n\n{new_summary}"
        else:
            memory.summary = new_summary
        
        logger.debug(f"Summarized {len(turns_to_summarize)} old turns for session {memory.session_id}")
    
    async def _extract_knowledge(self, session_id: str, user_input: str, assistant_response: str) -> None:
        """Extract and store important knowledge from interaction."""
        # Simple knowledge extraction (in production, use NLP techniques)
        knowledge_indicators = ['fact:', 'remember:', 'important:', 'key point:', 'note:']
        
        combined_text = f"{user_input} {assistant_response}".lower()
        
        for indicator in knowledge_indicators:
            if indicator in combined_text:
                # Extract knowledge item
                start_idx = combined_text.find(indicator)
                end_idx = min(start_idx + 200, len(combined_text))
                knowledge_content = combined_text[start_idx:end_idx].strip()
                
                # Create memory item
                memory_item = MemoryItem(
                    id=hashlib.md5(f"{session_id}_{knowledge_content}".encode()).hexdigest(),
                    content=knowledge_content,
                    content_type="knowledge",
                    importance_score=0.8,
                    tags=[session_id, "extracted_knowledge"]
                )
                
                self._global_memory.append(memory_item)
                self._memory_index[indicator].append(memory_item.id)
                
                # Limit global memory size
                if len(self._global_memory) > self.max_memory_items:
                    # Remove least important items
                    self._global_memory.sort(key=lambda x: (x.importance_score, x.last_accessed))
                    removed_items = self._global_memory[:len(self._global_memory) - self.max_memory_items]
                    self._global_memory = self._global_memory[len(self._global_memory) - self.max_memory_items:]
                    
                    # Update index
                    for item in removed_items:
                        for key, item_list in self._memory_index.items():
                            if item.id in item_list:
                                item_list.remove(item.id)
                
                break
    
    def _update_retrieval_stats(self, retrieval_time: float) -> None:
        """Update retrieval performance statistics."""
        stats = self._retrieval_stats
        stats["total_retrievals"] += 1
        
        # Update average retrieval time
        current_avg = stats["average_retrieval_time"]
        total_retrievals = stats["total_retrievals"]
        new_avg = ((current_avg * (total_retrievals - 1)) + retrieval_time) / total_retrievals
        stats["average_retrieval_time"] = new_avg
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up memory for a specific session."""
        if session_id in self._conversation_memories:
            del self._conversation_memories[session_id]
            
            # Remove session-specific items from global memory
            session_items = [item for item in self._global_memory if session_id in item.tags]
            for item in session_items:
                self._global_memory.remove(item)
                
                # Update index
                for key, item_list in self._memory_index.items():
                    if item.id in item_list:
                        item_list.remove(item.id)
            
            logger.debug(f"Cleaned up memory for session: {session_id}")
    
    async def store_memory(self, content: Dict[str, Any], memory_type: str, importance: str = "medium") -> str:
        """Store a general memory item."""
        memory_item = MemoryItem(
            id=hashlib.md5(f"{time.time()}_{str(content)}".encode()).hexdigest(),
            content=str(content),
            content_type=memory_type,
            importance_score={"low": 0.3, "medium": 0.6, "high": 0.9}.get(importance, 0.6),
            tags=[memory_type]
        )
        
        self._global_memory.append(memory_item)
        self._memory_index[memory_type].append(memory_item.id)
        
        return memory_item.id
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "total_conversations": len(self._conversation_memories),
            "total_memory_items": len(self._global_memory),
            "memory_by_type": {
                memory_type: len(items) for memory_type, items in self._memory_index.items()
            },
            "retrieval_stats": self._retrieval_stats,
            "average_conversation_length": sum(
                len(conv.turns) for conv in self._conversation_memories.values()
            ) / len(self._conversation_memories) if self._conversation_memories else 0
        }
    
    async def search_memories(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search through stored memories."""
        query_words = set(query.lower().split())
        
        scored_items = []
        for item in self._global_memory:
            item_words = set(item.content.lower().split())
            overlap = len(query_words.intersection(item_words))
            
            if overlap > 0:
                # Calculate relevance score
                relevance = overlap / len(query_words)
                recency = 1.0 / (1.0 + (time.time() - item.created_at.timestamp()) / 86400)  # Decay over days
                
                total_score = (relevance * 0.7) + (recency * 0.2) + (item.importance_score * 0.1)
                scored_items.append((item, total_score))
        
        # Sort by score and return top results
        scored_items.sort(key=lambda x: x[1], reverse=True)
        results = [item for item, score in scored_items[:limit]]
        
        # Update access statistics
        for item in results:
            item.update_access()
        
        return results
    
    async def export_memory(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Export memory data for backup or analysis."""
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "memory_stats": self.get_memory_stats(),
            "conversations": {},
            "global_memory": []
        }
        
        # Export conversations
        conversations_to_export = (
            {session_id: self._conversation_memories[session_id]} 
            if session_id and session_id in self._conversation_memories
            else self._conversation_memories
        )
        
        for sid, conv in conversations_to_export.items():
            export_data["conversations"][sid] = {
                "session_id": conv.session_id,
                "turns": conv.turns,
                "summary": conv.summary,
                "topics": conv.topics,
                "entities": conv.entities,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat()
            }
        
        # Export global memory
        for item in self._global_memory:
            export_data["global_memory"].append({
                "id": item.id,
                "content": item.content,
                "content_type": item.content_type,
                "importance_score": item.importance_score,
                "created_at": item.created_at.isoformat(),
                "last_accessed": item.last_accessed.isoformat(),
                "access_count": item.access_count,
                "tags": item.tags
            })
        
        return export_data
    
    async def import_memory(self, import_data: Dict[str, Any]) -> None:
        """Import memory data from backup."""
        # Import conversations
        for session_id, conv_data in import_data.get("conversations", {}).items():
            memory = ConversationMemory(
                session_id=conv_data["session_id"],
                turns=conv_data["turns"],
                summary=conv_data["summary"],
                topics=conv_data["topics"],
                entities=conv_data["entities"],
                created_at=datetime.fromisoformat(conv_data["created_at"]),
                updated_at=datetime.fromisoformat(conv_data["updated_at"])
            )
            self._conversation_memories[session_id] = memory
        
        # Import global memory
        for item_data in import_data.get("global_memory", []):
            item = MemoryItem(
                id=item_data["id"],
                content=item_data["content"],
                content_type=item_data["content_type"],
                importance_score=item_data["importance_score"],
                created_at=datetime.fromisoformat(item_data["created_at"]),
                last_accessed=datetime.fromisoformat(item_data["last_accessed"]),
                access_count=item_data["access_count"],
                tags=item_data["tags"]
            )
            self._global_memory.append(item)
            
            # Update index
            for tag in item.tags:
                self._memory_index[tag].append(item.id)
        
        logger.info(f"Imported memory data with {len(import_data.get('conversations', {}))} conversations and {len(import_data.get('global_memory', []))} memory items")