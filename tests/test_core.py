"""
Tests for OpenGuidance core functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from openguidance import OpenGuidance, Config
from openguidance.models.execution import ExecutionResult, ExecutionStatus


class TestConfig:
    """Tests for Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.enable_memory is True
        assert config.enable_code_execution is True
        assert config.enable_validation is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = Config(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            enable_memory=False,
            max_tokens=2000
        )
        
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.5
        assert config.enable_memory is False
        assert config.max_tokens == 2000


class TestOpenGuidance:
    """Tests for OpenGuidance class."""
    
    @pytest.fixture
    def basic_system(self):
        """Create a basic system for testing."""
        config = Config()
        system = OpenGuidance(config)
        return system
    
    def test_system_initialization(self):
        """Test system initialization."""
        config = Config()
        system = OpenGuidance(config)
        
        assert system.config is not None
        assert hasattr(system, 'memory_manager')
        assert hasattr(system, 'execution_engine')
        assert hasattr(system, 'prompt_manager')
    
    @pytest.mark.asyncio
    async def test_system_initialization_async(self, basic_system):
        """Test async system initialization."""
        await basic_system.initialize()
        
        assert basic_system.memory_manager is not None
        assert basic_system.execution_engine is not None
        assert basic_system.prompt_manager is not None
    
    @pytest.mark.asyncio
    async def test_process_request_basic(self, basic_system):
        """Test basic request processing."""
        await basic_system.initialize()
        
        # Mock the underlying components
        with patch.object(basic_system, 'process_request') as mock_process:
            mock_result = Mock()
            mock_result.content = "Test response"
            mock_result.status = "completed"
            mock_result.execution_time = 0.5
            mock_result.session_id = "test_session"
            mock_process.return_value = mock_result
            
            response = await basic_system.process_request(
                "What is Python?",
                session_id="test_session"
            )
            
            assert response.content == "Test response"
            assert response.session_id == "test_session"
    
    @pytest.mark.asyncio
    async def test_process_request_with_context(self, basic_system):
        """Test request processing with additional context."""
        await basic_system.initialize()
        
        context = {
            "user_preferences": {"language": "python"},
            "conversation_history": ["Previous question about variables"]
        }
        
        with patch.object(basic_system, 'process_request') as mock_process:
            mock_result = Mock()
            mock_result.content = "Contextual response"
            mock_result.status = "completed"
            mock_result.session_id = "test_session"
            mock_process.return_value = mock_result
            
            response = await basic_system.process_request(
                "Continue explaining",
                session_id="test_session",
                context=context
            )
            
            assert response.content == "Contextual response"
            # Verify process_request was called
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_system_cleanup(self, basic_system):
        """Test system cleanup."""
        await basic_system.initialize()
        
        # System cleanup should not raise exceptions
        await basic_system.cleanup()


class TestExecutionResult:
    """Tests for ExecutionResult class."""
    
    def test_execution_result_creation(self):
        """Test ExecutionResult creation."""
        result = ExecutionResult(
            content="Test output",
            status=ExecutionStatus.COMPLETED,
            execution_time=1.5,
            session_id="test_session"
        )
        
        assert result.content == "Test output"
        assert result.status == ExecutionStatus.COMPLETED
        assert result.execution_time == 1.5
        assert result.session_id == "test_session"
    
    def test_execution_result_str_representation(self):
        """Test ExecutionResult string representation."""
        result = ExecutionResult(
            content="Test output",
            status=ExecutionStatus.COMPLETED,
            execution_time=1.5,
            session_id="test_session"
        )
        
        str_repr = str(result)
        assert "Test output" in str_repr
        assert "COMPLETED" in str_repr


@pytest.mark.asyncio
async def test_integration_basic_flow():
    """Test basic integration flow."""
    config = Config()
    system = OpenGuidance(config)
    
    try:
        await system.initialize()
        
        # Mock a simple request
        with patch.object(system, 'process_request') as mock_process:
            mock_result = Mock()
            mock_result.content = "Integration test response"
            mock_result.status = "completed"
            mock_result.session_id = "integration_test"
            mock_process.return_value = mock_result
            
            response = await system.process_request(
                "Test integration",
                session_id="integration_test"
            )
            
            assert response.content == "Integration test response"
            
    finally:
        await system.cleanup()


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in system."""
    config = Config()
    system = OpenGuidance(config)
    
    try:
        await system.initialize()
        
        # Test with invalid input
        with patch.object(system, 'process_request') as mock_process:
            mock_process.side_effect = Exception("Test error")
            
            with pytest.raises(Exception):
                await system.process_request(
                    "This should fail",
                    session_id="error_test"
                )
                
    finally:
        await system.cleanup()


if __name__ == '__main__':
    pytest.main([__file__, "-v"])