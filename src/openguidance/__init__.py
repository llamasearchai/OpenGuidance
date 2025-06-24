"""
OpenGuidance - Advanced AI Guidance System
==========================================

A comprehensive framework for building sophisticated AI assistants with
advanced prompt management, execution control, memory systems, and validation.

Key Features:
- Dynamic prompt templating and management
- Sophisticated execution engine with monitoring
- Intelligent memory and context management  
- Comprehensive validation and safety systems
- Performance optimization and analytics
- Extensible architecture for custom implementations

Example Usage:
    from openguidance import OpenGuidanceSystem
    
    # Initialize the system
    system = OpenGuidanceSystem()
    
    # Process a request
    response = await system.process_request(
        "Explain how to implement a binary search algorithm",
        session_id="user_123"
    )
    
    print(response.content)
    print(f"Validation Score: {response.validation_report.overall_score}")
"""

from .core import GuidanceEngine, GuidanceContext, GuidanceResult, GuidanceMode
from .prompts import PromptManager, PromptTemplate, DynamicPrompt, SYSTEM_TEMPLATES
from .execution import ExecutionEngine, ExecutionResult, ExecutionStatus
from .memory import MemoryManager, MemoryItem, ConversationMemory
from .validation import ValidationEngine, ValidationReport, ValidationLevel
from .validation import create_validation_engine_with_defaults, create_strict_validation_engine

__version__ = "1.0.0"
__author__ = "OpenGuidance Team"
__email__ = "contact@openguidance.ai"

__all__ = [
    # Core system
    "GuidanceEngine",
    "GuidanceContext", 
    "GuidanceResult",
    "GuidanceMode",
    
    # Prompt management
    "PromptManager",
    "PromptTemplate",
    "DynamicPrompt",
    "SYSTEM_TEMPLATES",
    
    # Execution engine
    "ExecutionEngine",
    "ExecutionResult", 
    "ExecutionStatus",
    
    # Memory management
    "MemoryManager",
    "MemoryItem",
    "ConversationMemory",
    
    # Validation system
    "ValidationEngine",
    "ValidationReport",
    "ValidationLevel",
    "create_validation_engine_with_defaults",
    "create_strict_validation_engine",
]


# Quick setup functions for common use cases
def create_basic_system(**kwargs) -> GuidanceEngine:
    """Create a basic OpenGuidance system with default configuration."""
    return GuidanceEngine(
        model_name=kwargs.get('model_name', 'gpt-4'),
        max_retries=kwargs.get('max_retries', 3),
        timeout=kwargs.get('timeout', 30),
        **kwargs
    )


def create_advanced_system(**kwargs) -> GuidanceEngine:
    """Create an advanced OpenGuidance system with full features."""
    return GuidanceEngine(
        model_name=kwargs.get('model_name', 'gpt-4'),
        max_retries=kwargs.get('max_retries', 5),
        timeout=kwargs.get('timeout', 60),
        **kwargs
    )


def create_development_system(**kwargs) -> GuidanceEngine:
    """Create a development-focused system with enhanced code validation."""
    return GuidanceEngine(
        model_name=kwargs.get('model_name', 'gpt-4'),
        max_retries=kwargs.get('max_retries', 3),
        timeout=kwargs.get('timeout', 30),
        **kwargs
    )


# Version and compatibility information
SUPPORTED_PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]
MINIMUM_PYTHON_VERSION = "3.8"

# Feature flags for optional dependencies
FEATURES = {
    "async_support": True,
    "memory_persistence": True,
    "advanced_validation": True,
    "performance_monitoring": True,
    "code_analysis": True,
    "export_capabilities": True
}


def get_version_info():
    """Get detailed version and feature information."""
    import sys
    import platform
    
    return {
        "openguidance_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "features": FEATURES,
        "supported_python_versions": SUPPORTED_PYTHON_VERSIONS
    }


# Configuration helpers
class QuickConfig:
    """Quick configuration presets for common scenarios."""
    
    @staticmethod
    def for_chatbot():
        """Configuration optimized for chatbot applications."""
        return {
            'model_name': 'gpt-4',
            'max_retries': 3,
            'timeout': 30
        }
    
    @staticmethod
    def for_code_assistant():
        """Configuration optimized for code assistance."""
        return {
            'model_name': 'gpt-4',
            'max_retries': 5,
            'timeout': 60
        }
    
    @staticmethod
    def for_content_generation():
        """Configuration optimized for content generation."""
        return {
            'model_name': 'gpt-4',
            'max_retries': 3,
            'timeout': 45
        }
    
    @staticmethod
    def for_research_assistant():
        """Configuration optimized for research assistance."""
        return {
            'model_name': 'gpt-4',
            'max_retries': 5,
            'timeout': 90
        }


# Logging configuration
def configure_logging(level="INFO", format_string=None):
    """Configure logging for OpenGuidance components."""
    import logging
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set specific loggers
    loggers = [
        "openguidance.core",
        "openguidance.prompts", 
        "openguidance.execution",
        "openguidance.memory",
        "openguidance.validation"
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))


# Health check and diagnostics
async def health_check() -> dict:
    """Perform a health check of the OpenGuidance system."""
    import asyncio
    import time
    
    results = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {},
        "performance": {}
    }
    
    try:
        # Test basic system initialization
        start_time = time.time()
        system = create_basic_system()
        init_time = time.time() - start_time
        results["components"]["core"] = {"status": "ok", "init_time": init_time}
        
        # Test prompt management
        start_time = time.time()
        prompt_manager = PromptManager()
        prompt_time = time.time() - start_time
        results["components"]["prompts"] = {"status": "ok", "init_time": prompt_time}
        
        # Test execution engine
        start_time = time.time()
        execution_engine = ExecutionEngine()
        exec_time = time.time() - start_time
        results["components"]["execution"] = {"status": "ok", "init_time": exec_time}
        
        # Test memory management
        start_time = time.time()
        memory_manager = MemoryManager()
        memory_time = time.time() - start_time
        results["components"]["memory"] = {"status": "ok", "init_time": memory_time}
        
        # Test validation engine
        start_time = time.time()
        validation_engine = create_validation_engine_with_defaults()
        validation_time = time.time() - start_time
        results["components"]["validation"] = {"status": "ok", "init_time": validation_time}
        
        # Performance metrics
        results["performance"] = {
            "total_init_time": sum(comp.get("init_time", 0) for comp in results["components"].values()),
            "components_tested": len(results["components"]),
            "all_components_healthy": all(comp["status"] == "ok" for comp in results["components"].values())
        }
        
    except Exception as e:
        results["status"] = "unhealthy"
        results["error"] = str(e)
        results["components"]["error"] = {"status": "error", "message": str(e)}
    
    return results


# Compatibility and migration helpers
def check_compatibility():
    """Check system compatibility and requirements."""
    import sys
    import warnings
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_version not in SUPPORTED_PYTHON_VERSIONS:
        warnings.warn(
            f"Python {python_version} is not officially supported. "
            f"Supported versions: {', '.join(SUPPORTED_PYTHON_VERSIONS)}",
            UserWarning
        )
    
    # Check for optional dependencies
    optional_deps = {
        'numpy': 'Enhanced performance for numerical operations',
        'scipy': 'Advanced scientific computing features', 
        'nltk': 'Natural language processing capabilities',
        'transformers': 'Advanced language model integration',
        'redis': 'Distributed memory and caching support'
    }
    
    missing_deps = []
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append((dep, description))
    
    if missing_deps:
        print("Optional dependencies not found:")
        for dep, desc in missing_deps:
            print(f"  - {dep}: {desc}")
        print("\nInstall with: pip install openguidance[full] for all features")
    
    return {
        "python_compatible": python_version in SUPPORTED_PYTHON_VERSIONS,
        "missing_optional_deps": missing_deps,
        "core_features_available": True
    }


# Initialize compatibility check on import
try:
    check_compatibility()
except Exception:
    pass  # Don't fail on import if compatibility check fails