"""
Prompt-related data models.
"""

import time
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field


class PromptType(Enum):
    """Types of prompt templates."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    COMPLETION = "completion"


@dataclass
class PromptVariable:
    """Variable definition for prompt templates."""
    name: str
    description: str
    var_type: str = "string"  # string, integer, float, boolean, list, dict
    required: bool = True
    default_value: Any = None
    validation_pattern: Optional[str] = None
    choices: Optional[List[Any]] = None
    
    def validate_value(self, value: Any) -> bool:
        """Validate a value against this variable definition."""
        if value is None:
            return not self.required
        
        # Type validation
        type_mapping = {
            "string": str,
            "integer": int,
            "float": (int, float),
            "boolean": bool,
            "list": list,
            "dict": dict
        }
        
        expected_type = type_mapping.get(self.var_type)
        if expected_type and not isinstance(value, expected_type):
            return False
        
        # Choice validation
        if self.choices and value not in self.choices:
            return False
        
        # Pattern validation for strings
        if self.var_type == "string" and self.validation_pattern:
            import re
            return bool(re.match(self.validation_pattern, str(value)))
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "var_type": self.var_type,
            "required": self.required,
            "default_value": self.default_value,
            "validation_pattern": self.validation_pattern,
            "choices": self.choices
        }


@dataclass
class PromptTemplate:
    """Prompt template with variables and metadata."""
    template_id: str
    name: str
    template: str
    prompt_type: PromptType
    description: str = ""
    variables: List[PromptVariable] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    author: str = "system"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.prompt_type, str):
            self.prompt_type = PromptType(self.prompt_type)
    
    def render(self, **kwargs) -> str:
        """
        Render the template with provided variables.
        
        Args:
            **kwargs: Variable values to substitute
            
        Returns:
            Rendered template string
            
        Raises:
            ValueError: If required variables are missing or invalid
        """
        # Validate variables
        for variable in self.variables:
            if variable.name in kwargs:
                value = kwargs[variable.name]
                if not variable.validate_value(value):
                    raise ValueError(
                        f"Invalid value for variable '{variable.name}': {value}"
                    )
            elif variable.required:
                if variable.default_value is not None:
                    kwargs[variable.name] = variable.default_value
                else:
                    raise ValueError(f"Required variable '{variable.name}' not provided")
        
        # Render template
        try:
            rendered = self.template.format(**kwargs)
            self.usage_count += 1
            self.updated_at = time.time()
            return rendered
        except KeyError as e:
            raise ValueError(f"Template variable not found: {e}")
        except Exception as e:
            raise ValueError(f"Template rendering failed: {e}")
    
    def get_variable_names(self) -> List[str]:
        """Get list of all variable names in template."""
        import re
        # Find all {variable} patterns
        pattern = r'\{(\w+)\}'
        return list(set(re.findall(pattern, self.template)))
    
    def validate_template(self) -> List[str]:
        """
        Validate template consistency.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check if all template variables have definitions
        template_vars = set(self.get_variable_names())
        defined_vars = set(var.name for var in self.variables)
        
        undefined_vars = template_vars - defined_vars
        if undefined_vars:
            errors.append(f"Undefined variables in template: {undefined_vars}")
        
        unused_vars = defined_vars - template_vars
        if unused_vars:
            errors.append(f"Unused variable definitions: {unused_vars}")
        
        # Check for duplicate variable definitions
        var_names = [var.name for var in self.variables]
        duplicates = set([name for name in var_names if var_names.count(name) > 1])
        if duplicates:
            errors.append(f"Duplicate variable definitions: {duplicates}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "template": self.template,
            "prompt_type": self.prompt_type.value,
            "description": self.description,
            "variables": [var.to_dict() for var in self.variables],
            "tags": self.tags,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage_count": self.usage_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create from dictionary representation."""
        data = data.copy()
        
        # Convert prompt_type string to enum
        if 'prompt_type' in data:
            data['prompt_type'] = PromptType(data['prompt_type'])
        
        # Convert variable dictionaries back to PromptVariable objects
        if 'variables' in data:
            variables = []
            for var_data in data['variables']:
                variables.append(PromptVariable(**var_data))
            data['variables'] = variables
        
        return cls(**data)