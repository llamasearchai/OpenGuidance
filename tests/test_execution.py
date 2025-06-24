"""
Tests for OpenGuidance execution functionality.
"""

import pytest
from unittest.mock import Mock

from openguidance.execution import ExecutionEngine
from openguidance.models.execution import ExecutionResult, ExecutionStatus, ExecutionError
from openguidance.core.config import Config


class TestExecutionEngine:
    """Tests for ExecutionEngine class."""
    
    def test_execution_engine_creation(self):
        """Test ExecutionEngine creation."""
        config = Config()
        engine = ExecutionEngine(config)
        
        assert engine is not None
        assert engine.config is not None
        assert engine.max_concurrent == 10
        assert engine.default_timeout == 30


class TestExecutionResult:
    """Tests for ExecutionResult class."""
    
    def test_successful_result_creation(self):
        """Test creating a successful ExecutionResult."""
        result = ExecutionResult(
            content="Success output",
            status=ExecutionStatus.COMPLETED,
            execution_time=1.5,
            session_id="test_session"
        )
        
        assert result.content == "Success output"
        assert result.status == ExecutionStatus.COMPLETED
        assert result.execution_time == 1.5
        assert result.session_id == "test_session"
        assert result.error_message is None
    
    def test_failed_result_creation(self):
        """Test creating a failed ExecutionResult."""
        result = ExecutionResult(
            content="",
            status=ExecutionStatus.FAILED,
            execution_time=0.5,
            session_id="test_session",
            error_message="Invalid input"
        )
        
        assert result.content == ""
        assert result.status == ExecutionStatus.FAILED
        assert result.execution_time == 0.5
        assert result.session_id == "test_session"
        assert result.error_message == "Invalid input"
    
    def test_result_to_dict(self):
        """Test ExecutionResult to_dict method."""
        result = ExecutionResult(
            content="Test content",
            status=ExecutionStatus.COMPLETED,
            execution_time=1.0,
            session_id="test_session",
            metadata={"key": "value"}
        )
        
        data = result.to_dict()
        
        assert data["content"] == "Test content"
        assert data["status"] == "completed"
        assert data["execution_time"] == 1.0
        assert data["session_id"] == "test_session"
        assert data["metadata"]["key"] == "value"


class TestExecutionError:
    """Tests for ExecutionError class."""
    
    def test_execution_error_creation(self):
        """Test ExecutionError creation."""
        error = ExecutionError(
            message="Something went wrong",
            error_type="RuntimeError",
            details={"line": 42, "function": "test_func"}
        )
        
        assert error.message == "Something went wrong"
        assert error.error_type == "RuntimeError"
        assert error.details["line"] == 42
        assert error.details["function"] == "test_func"
    
    def test_execution_error_str_representation(self):
        """Test ExecutionError string representation."""
        error = ExecutionError(
            message="Test error message",
            error_type="TestError"
        )
        
        str_repr = str(error)
        assert "Test error message" in str_repr
    
    def test_execution_error_to_dict(self):
        """Test ExecutionError to_dict method."""
        error = ExecutionError(
            message="Error message",
            error_type="ValueError",
            details={"context": "test"}
        )
        
        data = error.to_dict()
        
        assert data["message"] == "Error message"
        assert data["error_type"] == "ValueError"
        assert data["details"]["context"] == "test"
        assert "timestamp" in data


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""
    
    def test_execution_status_values(self):
        """Test ExecutionStatus enum values."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.TIMEOUT.value == "timeout"
        assert ExecutionStatus.CANCELLED.value == "cancelled"


def test_basic_integration():
    """Test basic integration between components."""
    config = Config()
    engine = ExecutionEngine(config)
    
    # Test that engine can be created and has expected attributes
    assert hasattr(engine, 'config')
    assert hasattr(engine, 'max_concurrent')
    assert hasattr(engine, 'default_timeout')
    
    # Test that we can create execution results
    result = ExecutionResult(
        content="Integration test",
        status=ExecutionStatus.COMPLETED,
        execution_time=0.1,
        session_id="integration"
    )
    
    assert result.content == "Integration test"
    assert result.status == ExecutionStatus.COMPLETED


if __name__ == '__main__':
    pytest.main([__file__, "-v"])