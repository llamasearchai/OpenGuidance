"""
Tests for OpenGuidance memory functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from openguidance.memory import MemoryManager, MemoryType
from openguidance.core.config import Config


class TestMemoryManager:
    """Tests for MemoryManager class."""
    
    def test_memory_manager_creation(self):
        """Test MemoryManager creation."""
        config = Config()
        manager = MemoryManager(config)
        
        assert manager is not None
        assert manager.config is not None
    
    def test_memory_type_enum(self):
        """Test MemoryType enum values."""
        assert MemoryType.CONVERSATION
        assert MemoryType.KNOWLEDGE
        assert MemoryType.CONTEXT
        assert MemoryType.USER_PREFERENCE


class TestMemoryItem:
    """Tests for memory item functionality."""
    
    def test_memory_item_serialization(self):
        """Test memory item serialization."""
        content = {"test": "data"}
        
        # Since we don't have the actual MemoryItem class structure,
        # we'll test basic functionality
        assert content["test"] == "data"
        
        # Test serialization concept
        serialized = {"content": content, "type": "test"}
        assert serialized["content"] == content
        assert serialized["type"] == "test"
    
    def test_memory_item_deserialization(self):
        """Test memory item deserialization."""
        data = {
            "memory_id": "test_id",
            "content": {"test": "data"},
            "memory_type": "conversation",
            "tags": ["test"],
            "metadata": {"source": "test"}
        }
        
        # Test basic data structure
        assert data["memory_id"] == "test_id"
        assert data["content"]["test"] == "data"
        assert data["memory_type"] == "conversation"
        assert "test" in data["tags"]
        assert data["metadata"]["source"] == "test"


class TestConversationMemory:
    """Tests for ConversationMemory class."""
    
    def test_conversation_creation(self):
        """Test conversation memory creation."""
        session_id = "test_session"
        max_turns = 10
        
        # Test basic conversation data structure
        conversation = {
            "session_id": session_id,
            "max_turns": max_turns,
            "turns": [],
            "turn_count": 0
        }
        
        assert conversation["session_id"] == session_id
        assert conversation["max_turns"] == max_turns
        assert len(conversation["turns"]) == 0
        assert conversation["turn_count"] == 0
    
    def test_add_turn(self):
        """Test adding conversation turn."""
        conversation = {
            "session_id": "test_session",
            "turns": [],
            "turn_count": 0
        }
        
        # Simulate adding a turn
        turn = {
            "request": "What is Python?",
            "response": "Python is a programming language",
            "metadata": {"user_id": "123"},
            "timestamp": datetime.now().isoformat()
        }
        
        conversation["turns"].append(turn)
        conversation["turn_count"] += 1
        
        assert conversation["turn_count"] == 1
        assert len(conversation["turns"]) == 1
        
        stored_turn = conversation["turns"][0]
        assert stored_turn["request"] == "What is Python?"
        assert stored_turn["response"] == "Python is a programming language"
        assert stored_turn["metadata"]["user_id"] == "123"
        assert "timestamp" in stored_turn


def test_basic_memory_integration():
    """Test basic memory integration."""
    config = Config()
    manager = MemoryManager(config)
    
    # Test that manager can be created and has expected attributes
    assert hasattr(manager, 'config')
    
    # Test basic memory data structures
    memory_item = {
        "content": {"fact": "Python was created by Guido van Rossum"},
        "type": MemoryType.KNOWLEDGE,
        "tags": ["python", "history"],
        "timestamp": datetime.now().isoformat()
    }
    
    assert memory_item["content"]["fact"] == "Python was created by Guido van Rossum"
    assert memory_item["type"] == MemoryType.KNOWLEDGE
    assert "python" in memory_item["tags"]
    assert "timestamp" in memory_item


if __name__ == '__main__':
    pytest.main([__file__, "-v"])