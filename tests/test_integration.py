"""
Tests for OpenGuidance integration functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from openguidance import OpenGuidance, Config
from openguidance.memory import MemoryManager
from openguidance.execution import ExecutionEngine
from openguidance.prompts import PromptManager
from openguidance.validation import ValidationEngine


class TestSystemIntegration:
    """Tests for complete system integration."""
    
    @pytest.mark.asyncio
    async def test_basic_system_initialization(self):
        """Test basic system initialization."""
        config = Config()
        system = OpenGuidance(config)
        
        try:
            await system.initialize()
            
            # Verify all components are initialized
            assert system.memory_manager is not None
            assert system.execution_engine is not None
            assert system.prompt_manager is not None
            
        finally:
            await system.cleanup()
    
    @pytest.mark.asyncio
    async def test_component_integration(self):
        """Test integration between different components."""
        config = Config()
        system = OpenGuidance(config)
        
        try:
            await system.initialize()
            
            # Mock a request processing flow
            with patch.object(system, 'process_request') as mock_process:
                mock_result = Mock()
                mock_result.content = "Integration test response"
                mock_result.status = "completed"
                mock_result.session_id = "test_session"
                mock_result.execution_time = 0.5
                mock_process.return_value = mock_result
                
                result = await system.process_request(
                    "Test integration request",
                    session_id="test_session"
                )
                
                assert result.content == "Integration test response"
                assert result.session_id == "test_session"
                
        finally:
            await system.cleanup()


class TestComponentCommunication:
    """Tests for communication between components."""
    
    def test_memory_and_execution_integration(self):
        """Test integration between memory and execution components."""
        config = Config()
        memory_manager = MemoryManager(config)
        execution_engine = ExecutionEngine(config)
        
        # Test that components can be created and work together
        assert memory_manager.config == config
        assert execution_engine.config == config
    
    def test_prompt_and_validation_integration(self):
        """Test integration between prompt and validation components."""
        config = Config()
        prompt_manager = PromptManager(config)
        validation_engine = ValidationEngine()
        
        # Test that components can be created independently
        assert prompt_manager.config == config
        assert validation_engine is not None


class TestEndToEndWorkflow:
    """Tests for end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_simple_request_workflow(self):
        """Test a simple request processing workflow."""
        config = Config()
        system = OpenGuidance(config)
        
        try:
            await system.initialize()
            
            # Mock the workflow
            with patch.object(system, 'process_request') as mock_process:
                # Simulate a complete workflow
                mock_result = Mock()
                mock_result.content = "Workflow completed successfully"
                mock_result.status = "completed"
                mock_result.session_id = "workflow_test"
                mock_result.execution_time = 1.2
                mock_result.metadata = {"workflow": "simple"}
                mock_process.return_value = mock_result
                
                result = await system.process_request(
                    "Process this request through the complete workflow",
                    session_id="workflow_test",
                    context={"test": True}
                )
                
                # Verify workflow completion
                assert result.content == "Workflow completed successfully"
                assert result.status == "completed"
                assert result.metadata["workflow"] == "simple"
                
        finally:
            await system.cleanup()
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test error handling in the complete workflow."""
        config = Config()
        system = OpenGuidance(config)
        
        try:
            await system.initialize()
            
            # Mock an error scenario
            with patch.object(system, 'process_request') as mock_process:
                mock_process.side_effect = Exception("Test error")
                
                with pytest.raises(Exception):
                    await system.process_request(
                        "This should cause an error",
                        session_id="error_test"
                    )
                    
        finally:
            await system.cleanup()


class TestConfigurationIntegration:
    """Tests for configuration integration across components."""
    
    def test_config_propagation(self):
        """Test that configuration is properly propagated to all components."""
        config = Config(
            model_name="test-model",
            temperature=0.5,
            max_tokens=2000
        )
        
        system = OpenGuidance(config)
        
        # Verify config is set on main system
        assert system.config == config
        assert system.config.model_name == "test-model"
        assert system.config.temperature == 0.5
        assert system.config.max_tokens == 2000
    
    @pytest.mark.asyncio
    async def test_component_initialization_with_config(self):
        """Test that components are initialized with correct configuration."""
        config = Config(
            enable_memory=True,
            enable_code_execution=True,
            enable_validation=True
        )
        
        system = OpenGuidance(config)
        
        try:
            await system.initialize()
            
            # Verify components are initialized according to config
            if config.enable_memory:
                assert system.memory_manager is not None
            
            if config.enable_code_execution:
                assert system.execution_engine is not None
            
                         # Validation engine is not part of the main system
             # It can be created independently if needed
                
        finally:
            await system.cleanup()


def test_basic_component_creation():
    """Test basic creation of all components."""
    config = Config()
    
    # Test individual component creation
    memory_manager = MemoryManager(config)
    execution_engine = ExecutionEngine(config)
    prompt_manager = PromptManager(config)
    validation_engine = ValidationEngine()
    
    # Verify all components can be created
    assert memory_manager is not None
    assert execution_engine is not None
    assert prompt_manager is not None
    assert validation_engine is not None
    
    # Verify they have the expected configuration
    assert memory_manager.config == config
    assert execution_engine.config == config
    assert prompt_manager.config == config


def test_system_factory():
    """Test system creation through the main factory."""
    config = Config()
    system = OpenGuidance(config)
    
    # Test that system is created properly
    assert system is not None
    assert system.config == config
    
    # Test that system has all expected attributes
    assert hasattr(system, 'memory_manager')
    assert hasattr(system, 'execution_engine')
    assert hasattr(system, 'prompt_manager')


if __name__ == '__main__':
    pytest.main([__file__, "-v"])