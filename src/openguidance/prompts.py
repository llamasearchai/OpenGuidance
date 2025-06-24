"""
Advanced prompt management system with dynamic templating and optimization.
"""

import re
import json
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Advanced prompt template with variable substitution and validation."""
    
    name: str
    template: str
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Extract variables from template after initialization."""
        self.variables = self._extract_variables()
    
    def _extract_variables(self) -> List[str]:
        """Extract variable names from template using regex."""
        pattern = r'\{\{(\w+)\}\}'
        return list(set(re.findall(pattern, self.template)))
    
    def render(self, **kwargs) -> str:
        """Render template with provided variables."""
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        rendered = self.template
        for var, value in kwargs.items():
            rendered = rendered.replace(f"{{{{{var}}}}}", str(value))
        
        return rendered
    
    def validate_variables(self, **kwargs) -> Dict[str, Any]:
        """Validate provided variables against template requirements."""
        result = {
            "valid": True,
            "missing": [],
            "extra": [],
            "errors": []
        }
        
        provided_vars = set(kwargs.keys())
        required_vars = set(self.variables)
        
        result["missing"] = list(required_vars - provided_vars)
        result["extra"] = list(provided_vars - required_vars)
        
        if result["missing"]:
            result["valid"] = False
            result["errors"].append(f"Missing variables: {result['missing']}")
        
        return result


class DynamicPrompt:
    """Dynamic prompt that adapts based on context and conditions."""
    
    def __init__(self, base_template: PromptTemplate):
        self.base_template = base_template
        self.conditions: List[Callable] = []
        self.transformations: List[Callable] = []
        self.adaptations: Dict[str, PromptTemplate] = {}
    
    def add_condition(self, condition: Callable[[Dict], bool]) -> 'DynamicPrompt':
        """Add condition for dynamic adaptation."""
        self.conditions.append(condition)
        return self
    
    def add_transformation(self, transform: Callable[[str], str]) -> 'DynamicPrompt':
        """Add transformation function."""
        self.transformations.append(transform)
        return self
    
    def add_adaptation(self, name: str, template: PromptTemplate, condition: Callable) -> 'DynamicPrompt':
        """Add conditional template adaptation."""
        self.adaptations[name] = {
            "template": template,
            "condition": condition
        }
        return self
    
    def render(self, context: Dict[str, Any], **kwargs) -> str:
        """Render dynamic prompt based on context."""
        # Select appropriate template
        selected_template = self.base_template
        
        for name, adaptation in self.adaptations.items():
            if adaptation["condition"](context):
                selected_template = adaptation["template"]
                logger.debug(f"Using adaptation: {name}")
                break
        
        # Render base content
        content = selected_template.render(**kwargs)
        
        # Apply transformations
        for transform in self.transformations:
            content = transform(content)
        
        return content


class PromptManager:
    """Comprehensive prompt management system with versioning and optimization."""
    
    def __init__(self):
        self.templates: Dict[str, Dict[str, PromptTemplate]] = {}
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
    
    def register_template(self, template: PromptTemplate) -> None:
        """Register a new prompt template."""
        if template.name not in self.templates:
            self.templates[template.name] = {}
        
        self.templates[template.name][template.version] = template
        self.usage_stats[f"{template.name}:{template.version}"] = {
            "usage_count": 0,
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "last_used": None
        }
        
        logger.info(f"Registered template: {template.name} v{template.version}")
    
    def get_template(self, name: str, version: str = None) -> PromptTemplate:
        """Retrieve prompt template by name and version."""
        if name not in self.templates:
            raise ValueError(f"Template not found: {name}")
        
        if version is None:
            # Get latest version
            version = max(self.templates[name].keys())
        
        if version not in self.templates[name]:
            raise ValueError(f"Template version not found: {name} v{version}")
        
        return self.templates[name][version]
    
    def list_templates(self) -> Dict[str, List[str]]:
        """List all templates with their versions."""
        return {name: list(versions.keys()) for name, versions in self.templates.items()}
    
    def record_usage(self, name: str, version: str, success: bool, response_time: float) -> None:
        """Record template usage for analytics."""
        key = f"{name}:{version}"
        if key not in self.usage_stats:
            return
        
        stats = self.usage_stats[key]
        stats["usage_count"] += 1
        stats["last_used"] = datetime.utcnow().isoformat()
        
        # Update success rate
        current_successes = stats["success_rate"] * (stats["usage_count"] - 1)
        new_successes = current_successes + (1 if success else 0)
        stats["success_rate"] = new_successes / stats["usage_count"]
        
        # Update average response time
        current_total_time = stats["avg_response_time"] * (stats["usage_count"] - 1)
        new_total_time = current_total_time + response_time
        stats["avg_response_time"] = new_total_time / stats["usage_count"]
    
    def get_best_template(self, name: str, metric: str = "success_rate") -> PromptTemplate:
        """Get best performing version of a template."""
        if name not in self.templates:
            raise ValueError(f"Template not found: {name}")
        
        best_version = None
        best_score = -1
        
        for version in self.templates[name].keys():
            key = f"{name}:{version}"
            if key in self.usage_stats and self.usage_stats[key]["usage_count"] > 0:
                score = self.usage_stats[key].get(metric, 0)
                if score > best_score:
                    best_score = score
                    best_version = version
        
        if best_version is None:
            best_version = max(self.templates[name].keys())
        
        return self.templates[name][best_version]
    
    def export_templates(self, filename: str) -> None:
        """Export templates to JSON file."""
        export_data = {}
        for name, versions in self.templates.items():
            export_data[name] = {}
            for version, template in versions.items():
                export_data[name][version] = {
                    "template": template.template,
                    "variables": template.variables,
                    "metadata": template.metadata,
                    "version": template.version,
                    "created_at": template.created_at.isoformat(),
                    "tags": template.tags
                }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Templates exported to: {filename}")
    
    def import_templates(self, filename: str) -> None:
        """Import templates from JSON file."""
        with open(filename, 'r') as f:
            import_data = json.load(f)
        
        for name, versions in import_data.items():
            for version, data in versions.items():
                template = PromptTemplate(
                    name=name,
                    template=data["template"],
                    metadata=data.get("metadata", {}),
                    version=data.get("version", version),
                    created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
                    tags=data.get("tags", [])
                )
                self.register_template(template)
        
        logger.info(f"Templates imported from: {filename}")


# Predefined professional templates
SYSTEM_TEMPLATES = {
    "code_review": PromptTemplate(
        name="code_review",
        template="""You are an expert code reviewer. Please analyze the following code:

{{code}}

Provide feedback on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Security implications
5. Suggestions for improvement

Language: {{language}}
Context: {{context}}""",
        tags=["code", "review", "analysis"]
    ),
    
    "technical_documentation": PromptTemplate(
        name="technical_documentation",
        template="""Create comprehensive technical documentation for:

Topic: {{topic}}
Audience: {{audience}}
Complexity Level: {{complexity}}

Requirements:
- Clear explanations with examples
- Proper structure and formatting
- Code samples where applicable
- Best practices and guidelines

Additional Context: {{context}}""",
        tags=["documentation", "technical", "writing"]
    ),
    
    "problem_solving": PromptTemplate(
        name="problem_solving",
        template="""Analyze and solve the following problem:

Problem: {{problem}}
Domain: {{domain}}
Constraints: {{constraints}}

Please provide:
1. Problem analysis and breakdown
2. Proposed solution approach
3. Implementation details
4. Potential challenges and mitigations
5. Alternative approaches

Context: {{context}}""",
        tags=["problem-solving", "analysis", "solution"]
    ),
    
    "system_design": PromptTemplate(
        name="system_design",
        template="""Design a system for the following requirements:

System: {{system_name}}
Requirements: {{requirements}}
Scale: {{scale}}
Constraints: {{constraints}}

Provide:
1. High-level architecture
2. Component breakdown
3. Data flow and storage
4. Scalability considerations
5. Technology recommendations
6. Trade-offs and alternatives

Context: {{context}}""",
        tags=["architecture", "design", "system"]
    )
}