"""
Basic functionality test for OpenGuidance system.
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import pytest
from openguidance.core.system import OpenGuidance
from openguidance.core.config import Config
from openguidance.memory import MemoryManager
from openguidance.api.server import app
from fastapi.testclient import TestClient


def test_config_creation():
    """Test that configuration can be created successfully."""
    config = Config()
    assert config.model_name == "gpt-4"
    assert config.enable_memory is True
    assert config.enable_code_execution is True
    print("[SUCCESS] Configuration created successfully")


def test_memory_manager_creation():
    """Test that MemoryManager can be created successfully."""
    memory_manager = MemoryManager()
    assert memory_manager.max_conversation_turns == 50
    assert memory_manager.max_memory_items == 1000
    print("[SUCCESS] MemoryManager created successfully")


def test_openguidance_system_creation():
    """Test that OpenGuidance system can be created successfully."""
    config = Config()
    system = OpenGuidance(config)
    assert system.config == config
    assert system.is_initialized is False
    print("[SUCCESS] OpenGuidance system created successfully")


def test_fastapi_app_creation():
    """Test that FastAPI app can be created and basic routes work."""
    client = TestClient(app)
    
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "OpenGuidance API"
    assert data["version"] == "1.0.0"
    print("[SUCCESS] FastAPI app working correctly")


@pytest.mark.asyncio
async def test_memory_manager_session_initialization():
    """Test that MemoryManager can initialize sessions."""
    memory_manager = MemoryManager()
    session_id = "test_session_123"
    
    await memory_manager.initialize_session(session_id)
    assert session_id in memory_manager._conversation_memories
    print("[SUCCESS] MemoryManager session initialization working")


def test_api_health_endpoint():
    """Test that health endpoint works correctly."""
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "uptime" in data
    assert "components" in data
    print("[SUCCESS] Health endpoint working correctly")


def test_api_monitoring_endpoints():
    """Test that monitoring endpoints work correctly."""
    client = TestClient(app)
    
    # Test basic health endpoint
    response = client.get("/monitoring/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("[SUCCESS] Monitoring endpoints working correctly")


if __name__ == "__main__":
    print("Running basic functionality tests...")
    
    # Run synchronous tests
    test_config_creation()
    test_memory_manager_creation()
    test_openguidance_system_creation()
    test_fastapi_app_creation()
    test_api_health_endpoint()
    test_api_monitoring_endpoints()
    
    # Run async tests
    asyncio.run(test_memory_manager_session_initialization())
    
    print("\n[SUCCESS] All basic functionality tests passed!")
    print("OpenGuidance system is working correctly.") 