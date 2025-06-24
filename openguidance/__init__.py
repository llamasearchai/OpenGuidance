"""
OpenGuidance AI Assistant Framework

A production-ready AI assistant framework with advanced memory management,
code execution capabilities, intelligent prompting, and comprehensive API endpoints.

Author: Nik Jois <nikjois@llamasearch.ai>
Version: 1.0.0
"""

from .core.system import OpenGuidance
from .core.config import Config, load_config
from .memory import MemoryManager, MemoryType
from .prompts import PromptManager
from .execution import ExecutionEngine
from .validation import ValidationEngine, ValidationResult, ValidationLevel
from .models.execution import ExecutionResult, ExecutionStatus, ExecutionError

# API components
from .api.server import app
from .api.routes import guidance_router, admin_router, monitoring_router

# CLI
from .cli import cli

__version__ = "1.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

__all__ = [
    # Core components
    "OpenGuidance",
    "Config",
    "load_config",
    
    # Managers
    "MemoryManager", 
    "MemoryType",
    "PromptManager",
    "ExecutionEngine",
    "ValidationEngine",
    
    # Models
    "ExecutionResult",
    "ExecutionStatus", 
    "ExecutionError",
    "ValidationResult",
    "ValidationLevel",
    
    # API
    "app",
    "guidance_router",
    "admin_router", 
    "monitoring_router",
    
    # CLI
    "cli",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__"
] 