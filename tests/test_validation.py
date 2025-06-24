"""
Tests for OpenGuidance validation functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock

from openguidance.validation import (
    ValidationEngine,
    ValidationResult,
    ValidationReport,
    ValidationLevel,
    ResponseValidator,
    SafetyValidator
)


class TestValidationResult:
    """Tests for ValidationResult class."""
    
    def test_validation_result_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(
            validator_name="test_validator",
            level=ValidationLevel.INFO,
            message="Test validation passed",
            score=0.9
        )
        
        assert result.validator_name == "test_validator"
        assert result.level == ValidationLevel.INFO
        assert result.message == "Test validation passed"
        assert result.score == 0.9
        assert result.details == {}


class TestValidationReport:
    """Tests for ValidationReport class."""
    
    def test_validation_report_creation(self):
        """Test creating a ValidationReport."""
        report = ValidationReport(
            overall_score=0.8,
            passed=True
        )
        
        assert report.overall_score == 0.8
        assert report.passed is True
        assert len(report.results) == 0
    
    def test_add_result(self):
        """Test adding results to report."""
        report = ValidationReport(overall_score=0.0, passed=False)
        
        result = ValidationResult(
            validator_name="test",
            level=ValidationLevel.INFO,
            message="Test passed",
            score=0.9
        )
        
        report.add_result(result)
        
        assert len(report.results) == 1
        assert report.results[0] == result
        assert report.overall_score > 0  # Should be updated


class TestResponseValidator:
    """Tests for ResponseValidator class."""
    
    @pytest.mark.asyncio
    async def test_response_validator_creation(self):
        """Test ResponseValidator creation."""
        validator = ResponseValidator()
        
        assert validator.name == "response_quality"
        assert validator.enabled is True
    
    @pytest.mark.asyncio
    async def test_validate_good_response(self):
        """Test validating a good response."""
        validator = ResponseValidator()
        
        content = "This is a well-structured response that provides helpful information about the topic."
        context = {}
        
        result = await validator.validate(content, context)
        
        assert result.validator_name == "response_quality"
        assert result.score > 0.5
        assert result.level in [ValidationLevel.INFO, ValidationLevel.WARNING]
    
    @pytest.mark.asyncio
    async def test_validate_short_response(self):
        """Test validating a too-short response."""
        validator = ResponseValidator()
        
        content = "Short"  # Too short
        context = {}
        
        result = await validator.validate(content, context)
        
        assert result.score < 1.0
        assert "short" in result.message.lower()


class TestSafetyValidator:
    """Tests for SafetyValidator class."""
    
    @pytest.mark.asyncio
    async def test_safety_validator_creation(self):
        """Test SafetyValidator creation."""
        validator = SafetyValidator()
        
        assert validator.name == "safety"
        assert validator.enabled is True
    
    @pytest.mark.asyncio
    async def test_validate_safe_content(self):
        """Test validating safe content."""
        validator = SafetyValidator()
        
        content = "This is safe content about programming and learning."
        context = {}
        
        result = await validator.validate(content, context)
        
        assert result.validator_name == "safety"
        assert result.score > 0.5
        assert result.level in [ValidationLevel.INFO, ValidationLevel.WARNING]
    
    @pytest.mark.asyncio
    async def test_validate_potentially_harmful_content(self):
        """Test validating potentially harmful content."""
        validator = SafetyValidator()
        
        content = "Here's how to hack into a system and exploit vulnerabilities."
        context = {}
        
        result = await validator.validate(content, context)
        
        assert result.score < 1.0
        assert result.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]


class TestValidationEngine:
    """Tests for ValidationEngine class."""
    
    def test_validation_engine_creation(self):
        """Test ValidationEngine creation."""
        engine = ValidationEngine()
        
        assert engine is not None
        assert isinstance(engine.validators, dict)
    
    def test_register_validator(self):
        """Test registering a validator."""
        engine = ValidationEngine()
        validator = ResponseValidator()
        
        engine.register_validator(validator)
        
        assert validator.name in engine.validators
        assert engine.validators[validator.name] == validator
    
    @pytest.mark.asyncio
    async def test_validate_response(self):
        """Test validating a response."""
        engine = ValidationEngine()
        validator = ResponseValidator()
        engine.register_validator(validator)
        
        content = "This is a test response for validation."
        context = {}
        
        report = await engine.validate_response(content, context)
        
        assert isinstance(report, ValidationReport)
        assert len(report.results) > 0
        assert report.overall_score >= 0.0
        assert isinstance(report.passed, bool)
    
    def test_enable_disable_validator(self):
        """Test enabling and disabling validators."""
        engine = ValidationEngine()
        validator = ResponseValidator()
        engine.register_validator(validator)
        
        # Test disable
        engine.disable_validator(validator.name)
        assert not engine.validators[validator.name].enabled
        
        # Test enable
        engine.enable_validator(validator.name)
        assert engine.validators[validator.name].enabled
    
    def test_unregister_validator(self):
        """Test unregistering a validator."""
        engine = ValidationEngine()
        validator = ResponseValidator()
        engine.register_validator(validator)
        
        assert validator.name in engine.validators
        
        engine.unregister_validator(validator.name)
        
        assert validator.name not in engine.validators


class TestValidationLevel:
    """Tests for ValidationLevel enum."""
    
    def test_validation_level_values(self):
        """Test ValidationLevel enum values."""
        assert ValidationLevel.INFO.value == "info"
        assert ValidationLevel.WARNING.value == "warning"
        assert ValidationLevel.ERROR.value == "error"
        assert ValidationLevel.CRITICAL.value == "critical"


def test_basic_validation_integration():
    """Test basic validation integration."""
    # Create engine and validators
    engine = ValidationEngine()
    response_validator = ResponseValidator()
    safety_validator = SafetyValidator()
    
    # Register validators
    engine.register_validator(response_validator)
    engine.register_validator(safety_validator)
    
    # Test that validators are registered
    assert "response_quality" in engine.validators
    assert "safety" in engine.validators
    # ValidationEngine comes with default validators, so check for our specific ones


if __name__ == '__main__':
    pytest.main([__file__, "-v"])