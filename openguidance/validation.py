"""
Advanced validation system for response quality and safety.
"""

import re
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Individual validation result."""
    
    validator_name: str
    level: ValidationLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    score: float = 1.0  # 0.0 to 1.0, higher is better
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    
    overall_score: float
    passed: bool
    results: List[ValidationResult] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: ValidationResult) -> None:
        """Add validation result to report."""
        self.results.append(result)
        self._update_overall_score()
    
    def _update_overall_score(self) -> None:
        """Update overall score based on all results."""
        if not self.results:
            self.overall_score = 1.0
            self.passed = True
            return
        
        # Calculate weighted average
        total_weight = 0
        weighted_score = 0
        
        for result in self.results:
            weight = self._get_level_weight(result.level)
            weighted_score += result.score * weight
            total_weight += weight
        
        self.overall_score = weighted_score / total_weight if total_weight > 0 else 1.0
        
        # Check if validation passed (no critical errors, overall score > 0.6)
        has_critical = any(r.level == ValidationLevel.CRITICAL for r in self.results)
        self.passed = not has_critical and self.overall_score > 0.6
    
    def _get_level_weight(self, level: ValidationLevel) -> float:
        """Get weight for validation level."""
        weights = {
            ValidationLevel.INFO: 0.25,
            ValidationLevel.WARNING: 0.5,
            ValidationLevel.ERROR: 1.0,
            ValidationLevel.CRITICAL: 2.0
        }
        return weights.get(level, 1.0)


class BaseValidator:
    """Base class for all validators."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
    
    async def validate(self, content: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate content and return result."""
        raise NotImplementedError("Subclasses must implement validate method")


class ResponseValidator(BaseValidator):
    """Validates response quality and completeness."""
    
    def __init__(self):
        super().__init__("response_quality")
        self.min_length = 10
        self.max_length = 10000
    
    async def validate(self, content: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate response quality."""
        issues = []
        score = 1.0
        
        # Length validation
        if len(content) < self.min_length:
            issues.append("Response too short")
            score -= 0.3
        elif len(content) > self.max_length:
            issues.append("Response too long")
            score -= 0.2
        
        # Completeness validation
        if content.strip().endswith(("...", "etc.", "and so on")):
            issues.append("Response appears incomplete")
            score -= 0.2
        
        # Structure validation
        if len(content.split('\n')) == 1 and len(content) > 200:
            issues.append("Response lacks proper structure")
            score -= 0.1
        
        # Code block validation
        code_blocks = re.findall(r'```.*?```', content, re.DOTALL)
        for block in code_blocks:
            if 'placeholder' in block.lower() or 'todo' in block.lower():
                issues.append("Code contains placeholders")
                score -= 0.2
                break
        
        level = ValidationLevel.ERROR if score < 0.5 else ValidationLevel.WARNING if issues else ValidationLevel.INFO
        message = "; ".join(issues) if issues else "Response quality validated"
        
        return ValidationResult(
            validator_name=self.name,
            level=level,
            message=message,
            score=max(0.0, score),
            details={"issues": issues, "length": len(content), "code_blocks": len(code_blocks)}
        )


class SafetyValidator(BaseValidator):
    """Validates content for safety and appropriateness."""
    
    def __init__(self):
        super().__init__("safety")
        self.harmful_patterns = [
            r'\b(hack|exploit|vulnerability)\b.*\b(system|server|database)\b',
            r'\b(bypass|circumvent)\b.*\b(security|authentication)\b',
            r'\b(malware|virus|trojan)\b',
            r'\b(illegal|unlawful)\b.*\b(activity|action)\b'
        ]
        self.sensitive_topics = [
            'personal information', 'private data', 'credentials', 'passwords',
            'financial information', 'medical records'
        ]
    
    async def validate(self, content: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate content safety."""
        issues = []
        score = 1.0
        content_lower = content.lower()
        
        # Check for harmful patterns
        for pattern in self.harmful_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                issues.append(f"Potentially harmful content detected: {pattern}")
                score -= 0.5
        
        # Check for sensitive topics
        for topic in self.sensitive_topics:
            if topic in content_lower:
                issues.append(f"Sensitive topic mentioned: {topic}")
                score -= 0.2
        
        # Check for excessive technical detail that could be misused
        if len(re.findall(r'```.*?```', content, re.DOTALL)) > 5:
            security_keywords = ['admin', 'root', 'sudo', 'privilege', 'escalation']
            if any(keyword in content_lower for keyword in security_keywords):
                issues.append("Excessive technical detail with security implications")
                score -= 0.3
        
        level = (ValidationLevel.CRITICAL if score < 0.3 else 
                ValidationLevel.ERROR if score < 0.6 else 
                ValidationLevel.WARNING if issues else ValidationLevel.INFO)
        
        message = "; ".join(issues) if issues else "Safety validation passed"
        
        return ValidationResult(
            validator_name=self.name,
            level=level,
            message=message,
            score=max(0.0, score),
            details={"detected_issues": len(issues), "sensitive_topics_found": len([t for t in self.sensitive_topics if t in content_lower])}
        )


class AccuracyValidator(BaseValidator):
    """Validates technical accuracy and consistency."""
    
    def __init__(self):
        super().__init__("accuracy")
        self.known_facts = {
            'python': ['indentation', 'interpreted', 'dynamic typing'],
            'javascript': ['prototype-based', 'event-driven', 'interpreted'],
            'java': ['compiled', 'object-oriented', 'static typing']
        }
    
    async def validate(self, content: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate technical accuracy."""
        issues = []
        score = 1.0
        content_lower = content.lower()
        
        # Check for common technical inaccuracies
        inaccuracies = [
            (r'python.*compiled.*bytecode', 'Python compilation detail may be misleading'),
            (r'javascript.*single.?threaded.*always', 'JavaScript threading model oversimplified'),
            (r'sql.*always.*faster', 'SQL performance claims too absolute'),
            (r'nosql.*never.*consistent', 'NoSQL consistency claims inaccurate')
        ]
        
        for pattern, issue in inaccuracies:
            if re.search(pattern, content_lower):
                issues.append(issue)
                score -= 0.2
        
        # Check for contradictions within the response
        if 'however' in content_lower and 'but' in content_lower:
            sentences = content.split('.')
            contradictory_pairs = [
                ('always', 'never'), ('all', 'none'), ('every', 'no'),
                ('best', 'worst'), ('fastest', 'slowest')
            ]
            
            for word1, word2 in contradictory_pairs:
                if any(word1 in s.lower() for s in sentences) and any(word2 in s.lower() for s in sentences):
                    issues.append(f"Potential contradiction detected: {word1} vs {word2}")
                    score -= 0.1
        
        # Validate code syntax if present
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)
        for lang, code in code_blocks:
            if lang and lang.lower() in ['python', 'javascript', 'java']:
                syntax_issues = self._validate_code_syntax(code, lang.lower())
                if syntax_issues:
                    issues.extend(syntax_issues)
                    score -= 0.15 * len(syntax_issues)
        
        level = (ValidationLevel.ERROR if score < 0.5 else 
                ValidationLevel.WARNING if issues else ValidationLevel.INFO)
        
        message = "; ".join(issues) if issues else "Accuracy validation passed"
        
        return ValidationResult(
            validator_name=self.name,
            level=level,
            message=message,
            score=max(0.0, score),
            details={"code_blocks_validated": len(code_blocks), "accuracy_issues": len(issues)}
        )
    
    def _validate_code_syntax(self, code: str, language: str) -> List[str]:
        """Basic syntax validation for code blocks."""
        issues = []
        
        if language == 'python':
            # Basic Python syntax checks
            lines = code.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.endswith(':') and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith((' ', '\t')):
                        issues.append(f"Python indentation issue at line {i + 2}")
                
                if 'print(' in stripped and not stripped.count('(') == stripped.count(')'):
                    issues.append(f"Unmatched parentheses in print statement at line {i + 1}")
        
        elif language == 'javascript':
            # Basic JavaScript syntax checks
            if code.count('{') != code.count('}'):
                issues.append("Unmatched braces in JavaScript code")
            
            if 'function' in code and 'return' not in code and 'console.log' not in code:
                issues.append("Function without return statement or output")
        
        return issues


class RelevanceValidator(BaseValidator):
    """Validates response relevance to the query."""
    
    def __init__(self):
        super().__init__("relevance")
    
    async def validate(self, content: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate response relevance."""
        issues = []
        score = 1.0
        
        # Get the original query from context
        query = context.get('query', '')
        if not query:
            return ValidationResult(
                validator_name=self.name,
                level=ValidationLevel.INFO,
                message="No query context available for relevance check",
                score=1.0
            )
        
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Calculate word overlap
        overlap = len(query_words.intersection(content_words))
        overlap_ratio = overlap / len(query_words) if query_words else 0
        
        if overlap_ratio < 0.2:
            issues.append("Low relevance to original query")
            score -= 0.4
        elif overlap_ratio < 0.4:
            issues.append("Moderate relevance concerns")
            score -= 0.2
        
        # Check for topic drift
        if len(content) > 500:
            content_parts = [content[i:i+500] for i in range(0, len(content), 500)]
            relevance_scores = []
            
            for part in content_parts:
                part_words = set(part.lower().split())
                part_overlap = len(query_words.intersection(part_words))
                part_score = part_overlap / len(query_words) if query_words else 0
                relevance_scores.append(part_score)
            
            if len(relevance_scores) > 1:
                score_variance = max(relevance_scores) - min(relevance_scores)
                if score_variance > 0.5:
                    issues.append("Significant topic drift detected")
                    score -= 0.2
        
        # Check for generic responses
        generic_phrases = [
            'it depends', 'there are many ways', 'it varies', 'generally speaking',
            'in most cases', 'typically', 'usually', 'often'
        ]
        
        generic_count = sum(1 for phrase in generic_phrases if phrase in content.lower())
        if generic_count > 3:
            issues.append("Response appears overly generic")
            score -= 0.15
        
        level = (ValidationLevel.ERROR if score < 0.5 else 
                ValidationLevel.WARNING if issues else ValidationLevel.INFO)
        
        message = "; ".join(issues) if issues else "Relevance validation passed"
        
        return ValidationResult(
            validator_name=self.name,
            level=level,
            message=message,
            score=max(0.0, score),
            details={
                "query_overlap_ratio": overlap_ratio,
                "generic_phrase_count": generic_count,
                "content_length": len(content)
            }
        )


class ValidationEngine:
    """
    Comprehensive validation engine that orchestrates multiple validators
    and provides detailed validation reports.
    """
    
    def __init__(self):
        self.validators: Dict[str, BaseValidator] = {}
        self.validation_history: List[ValidationReport] = []
        self.performance_stats = {
            "total_validations": 0,
            "average_validation_time": 0.0,
            "validation_pass_rate": 0.0
        }
        
        # Register default validators
        self.register_validator(ResponseValidator())
        self.register_validator(SafetyValidator())
        self.register_validator(AccuracyValidator())
        self.register_validator(RelevanceValidator())
        
        logger.info("ValidationEngine initialized with default validators")
    
    def register_validator(self, validator: BaseValidator) -> None:
        """Register a new validator."""
        self.validators[validator.name] = validator
        logger.debug(f"Registered validator: {validator.name}")
    
    def unregister_validator(self, name: str) -> None:
        """Unregister a validator."""
        if name in self.validators:
            del self.validators[name]
            logger.debug(f"Unregistered validator: {name}")
    
    def enable_validator(self, name: str) -> None:
        """Enable a specific validator."""
        if name in self.validators:
            self.validators[name].enabled = True
    
    def disable_validator(self, name: str) -> None:
        """Disable a specific validator."""
        if name in self.validators:
            self.validators[name].enabled = False
    
    async def validate_response(
        self,
        content: str,
        context: Dict[str, Any],
        validators: Optional[List[str]] = None
    ) -> ValidationReport:
        """Validate response using specified or all enabled validators."""
        start_time = datetime.utcnow()
        report = ValidationReport(overall_score=1.0, passed=True)
        
        # Determine which validators to use
        validators_to_use = validators or list(self.validators.keys())
        active_validators = [
            self.validators[name] for name in validators_to_use
            if name in self.validators and self.validators[name].enabled
        ]
        
        # Run validations
        validation_tasks = []
        for validator in active_validators:
            task = validator.validate(content, context)
            validation_tasks.append(task)
        
        try:
            # Execute all validations concurrently
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Validator {active_validators[i].name} failed: {str(result)}")
                    # Create error result
                    error_result = ValidationResult(
                        validator_name=active_validators[i].name,
                        level=ValidationLevel.ERROR,
                        message=f"Validation failed: {str(result)}",
                        score=0.0
                    )
                    report.add_result(error_result)
                elif isinstance(result, ValidationResult):
                    report.add_result(result)
            
            # Calculate execution time
            end_time = datetime.utcnow()
            report.execution_time = (end_time - start_time).total_seconds()
            
            # Update statistics
            self._update_performance_stats(report)
            
            # Store in history
            self.validation_history.append(report)
            
            # Limit history size
            if len(self.validation_history) > 1000:
                self.validation_history = self.validation_history[-1000:]
            
            logger.debug(f"Validation completed: score={report.overall_score:.2f}, passed={report.passed}")
            return report
            
        except Exception as e:
            logger.error(f"Validation engine error: {str(e)}")
            report.add_result(ValidationResult(
                validator_name="validation_engine",
                level=ValidationLevel.CRITICAL,
                message=f"Validation engine error: {str(e)}",
                score=0.0
            ))
            return report
    
    def _update_performance_stats(self, report: ValidationReport) -> None:
        """Update performance statistics."""
        stats = self.performance_stats
        stats["total_validations"] += 1
        
        # Update average validation time
        current_avg = stats["average_validation_time"]
        total_validations = stats["total_validations"]
        new_avg = ((current_avg * (total_validations - 1)) + report.execution_time) / total_validations
        stats["average_validation_time"] = new_avg
        
        # Update pass rate
        total_passed = sum(1 for r in self.validation_history if r.passed)
        stats["validation_pass_rate"] = total_passed / len(self.validation_history) if self.validation_history else 0.0
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics."""
        return {
            **self.performance_stats,
            "active_validators": [name for name, validator in self.validators.items() if validator.enabled],
            "total_validators": len(self.validators),
            "recent_validations": len(self.validation_history)
        }
    
    def get_validation_history(self, limit: int = 50) -> List[ValidationReport]:
        """Get recent validation history."""
        return self.validation_history[-limit:] if limit else self.validation_history
    
    async def create_custom_validator(
        self,
        name: str,
        validation_func: Callable[[str, Dict[str, Any]], ValidationResult]
    ) -> None:
        """Create and register a custom validator."""
        class CustomValidator(BaseValidator):
            def __init__(self, name: str, func: Callable):
                super().__init__(name)
                self.validation_func = func
            
            async def validate(self, content: str, context: Dict[str, Any]) -> ValidationResult:
                if asyncio.iscoroutinefunction(self.validation_func):
                    return await self.validation_func(content, context)
                else:
                    return self.validation_func(content, context)
        
        custom_validator = CustomValidator(name, validation_func)
        self.register_validator(custom_validator)
    
    def export_validation_report(self, report: ValidationReport) -> Dict[str, Any]:
        """Export validation report to dictionary format."""
        return {
            "overall_score": report.overall_score,
            "passed": report.passed,
            "execution_time": report.execution_time,
            "results": [
                {
                    "validator_name": result.validator_name,
                    "level": result.level.value,
                    "message": result.message,
                    "score": result.score,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in report.results
            ],
            "metadata": report.metadata
        }


# Factory functions for common validation configurations
def create_validation_engine_with_defaults() -> ValidationEngine:
    """Create validation engine with all default validators enabled."""
    return ValidationEngine()


def create_strict_validation_engine() -> ValidationEngine:
    """Create validation engine with strict validation settings."""
    engine = ValidationEngine()
    
    # Configure stricter thresholds
    response_validator = engine.validators["response_quality"]
    if isinstance(response_validator, ResponseValidator):
        response_validator.min_length = 50
        response_validator.max_length = 5000
    
    return engine


def create_permissive_validation_engine() -> ValidationEngine:
    """Create validation engine with more permissive settings."""
    engine = ValidationEngine()
    
    # Disable some validators for more permissive validation
    engine.disable_validator("accuracy")
    
    # Configure more lenient thresholds
    response_validator = engine.validators["response_quality"]
    if isinstance(response_validator, ResponseValidator):
        response_validator.min_length = 5
        response_validator.max_length = 20000
    
    return engine