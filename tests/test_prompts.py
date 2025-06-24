"""
Tests for OpenGuidance prompts functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from openguidance.prompts import PromptTemplate, PromptManager
from openguidance.core.config import Config


class TestPromptTemplate:
    """Tests for PromptTemplate class."""
    
    def test_template_creation(self):
        """Test prompt template creation."""
        template = PromptTemplate(
            name="test_template",
            template="Hello {{name}}, welcome to {{place}}!",
            metadata={"author": "test"}
        )
        
        assert template.name == "test_template"
        assert template.template == "Hello {{name}}, welcome to {{place}}!"
        assert template.metadata["author"] == "test"
        assert "name" in template.variables
        assert "place" in template.variables
    
    def test_template_rendering(self):
        """Test template rendering with variables."""
        template = PromptTemplate(
            name="greeting",
            template="Hello {{name}}!"
        )
        
        rendered = template.render(name="Alice")
        assert rendered == "Hello Alice!"
    
    def test_template_rendering_missing_variable(self):
        """Test template rendering with missing variable."""
        template = PromptTemplate(
            name="greeting",
            template="Hello {{name}}!"
        )
        
        with pytest.raises(ValueError):
            template.render()  # Missing 'name' variable
    
    def test_variable_extraction(self):
        """Test automatic variable extraction."""
        template = PromptTemplate(
            name="complex",
            template="Hello {{name}}, you have {{count}} messages from {{sender}}."
        )
        
        expected_vars = {"name", "count", "sender"}
        assert set(template.variables) == expected_vars
    
    def test_validate_variables(self):
        """Test variable validation."""
        template = PromptTemplate(
            name="test",
            template="Hello {{name}}!"
        )
        
        # Valid variables
        result = template.validate_variables(name="Alice")
        assert result["valid"] is True
        assert len(result["missing"]) == 0
        
        # Missing variables
        result = template.validate_variables()
        assert result["valid"] is False
        assert "name" in result["missing"]


class TestPromptManager:
    """Tests for PromptManager class."""
    
    def test_manager_creation(self):
        """Test PromptManager creation."""
        config = Config()
        manager = PromptManager(config)
        
        assert manager is not None
        assert manager.config is not None
        assert isinstance(manager.templates, dict)
    
    def test_register_template(self):
        """Test registering a template."""
        config = Config()
        manager = PromptManager(config)
        
        template = PromptTemplate(
            name="test_template",
            template="Hello {{name}}!",
            version="1.0.0"
        )
        
        manager.register_template(template)
        
        assert "test_template" in manager.templates
        assert "1.0.0" in manager.templates["test_template"]
    
    def test_get_template(self):
        """Test retrieving a template."""
        config = Config()
        manager = PromptManager(config)
        
        template = PromptTemplate(
            name="retrieve_test",
            template="Test template",
            version="1.0.0"
        )
        
        manager.register_template(template)
        
        retrieved = manager.get_template("retrieve_test", "1.0.0")
        assert retrieved.name == "retrieve_test"
        assert retrieved.template == "Test template"
    
    def test_get_nonexistent_template(self):
        """Test retrieving non-existent template."""
        config = Config()
        manager = PromptManager(config)
        
        with pytest.raises(ValueError):
            manager.get_template("nonexistent")
    
    def test_list_templates(self):
        """Test listing all templates."""
        config = Config()
        manager = PromptManager(config)
        
        template1 = PromptTemplate(name="template1", template="Test 1", version="1.0.0")
        template2 = PromptTemplate(name="template2", template="Test 2", version="1.0.0")
        
        manager.register_template(template1)
        manager.register_template(template2)
        
        templates = manager.list_templates()
        
        assert "template1" in templates
        assert "template2" in templates
        assert "1.0.0" in templates["template1"]
        assert "1.0.0" in templates["template2"]
    
    def test_record_usage(self):
        """Test recording template usage."""
        config = Config()
        manager = PromptManager(config)
        
        template = PromptTemplate(name="usage_test", template="Test", version="1.0.0")
        manager.register_template(template)
        
        # Record successful usage
        manager.record_usage("usage_test", "1.0.0", success=True, response_time=0.5)
        
        key = "usage_test:1.0.0"
        assert key in manager.usage_stats
        assert manager.usage_stats[key]["usage_count"] == 1
        assert manager.usage_stats[key]["success_rate"] == 1.0


def test_basic_prompt_integration():
    """Test basic prompt integration."""
    config = Config()
    manager = PromptManager(config)
    
    # Create and register a template
    template = PromptTemplate(
        name="integration_test",
        template="Hello {{user}}, your score is {{score}}!",
        version="1.0.0"
    )
    
    manager.register_template(template)
    
    # Retrieve and use the template
    retrieved = manager.get_template("integration_test", "1.0.0")
    rendered = retrieved.render(user="Alice", score=95)
    
    assert rendered == "Hello Alice, your score is 95!"
    
    # Record usage
    manager.record_usage("integration_test", "1.0.0", success=True, response_time=0.3)
    
    # Check stats
    key = "integration_test:1.0.0"
    assert manager.usage_stats[key]["usage_count"] == 1


if __name__ == '__main__':
    pytest.main([__file__, "-v"])