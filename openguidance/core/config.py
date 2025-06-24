"""
Configuration management for OpenGuidance system.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    max_memory_items: int = 10000
    cleanup_interval: int = 3600  # seconds
    storage_backend: str = "file"  # file, redis, postgres
    similarity_threshold: float = 0.7
    max_context_memories: int = 10
    enable_compression: bool = True
    
    # Backend-specific settings
    redis_url: Optional[str] = None
    postgres_url: Optional[str] = None
    file_storage_path: str = "./data/memories"


@dataclass
class ExecutionConfig:
    """Execution engine configuration."""
    max_execution_time: int = 30  # seconds
    enable_validation: bool = True
    sandbox_timeout: int = 10
    allowed_modules: List[str] = field(default_factory=lambda: [
        "math", "json", "re", "datetime", "random", "string", "itertools"
    ])
    max_output_size: int = 10000  # characters
    enable_file_access: bool = False


@dataclass
class PromptConfig:
    """Prompt management configuration."""
    templates_path: str = "./data/prompts"
    default_model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    system_prompt: str = "You are OpenGuidance, an AI assistant focused on providing helpful, accurate, and safe responses."
    max_history_tokens: int = 2000


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    enable_auth: bool = False
    secret_key: Optional[str] = None


@dataclass
class Config:
    """Main configuration class for OpenGuidance system."""
    
    # Model configuration
    model_name: str = "gpt-4"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000
    
    # Feature toggles
    enable_memory: bool = True
    enable_code_execution: bool = True
    enable_validation: bool = True
    
    # Component configurations
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    api_config: APIConfig = field(default_factory=APIConfig)
    
    # Environment settings
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "./data"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    def __post_init__(self):
        """Post-initialization configuration setup."""
        # Ensure data directory exists
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Update nested config paths based on data_dir
        if not os.path.isabs(self.memory_config.file_storage_path):
            self.memory_config.file_storage_path = os.path.join(
                self.data_dir, "memories"
            )
        
        if not os.path.isabs(self.prompt_config.templates_path):
            self.prompt_config.templates_path = os.path.join(
                self.data_dir, "prompts"
            )
        
        # Environment-specific overrides
        if self.environment == "production":
            self.debug = False
            self.api_config.enable_cors = False
            self.api_config.cors_origins = []
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration settings."""
        if self.enable_code_execution and not self.enable_validation:
            logger.warning(
                "Code execution is enabled without validation. This may be unsafe."
            )
        
        if not self.api_key and self.model_name.startswith(("gpt-", "text-")):
            logger.warning(
                "OpenAI API key not configured. Some features may not work."
            )
        
        if self.memory_config.max_memory_items <= 0:
            raise ValueError("max_memory_items must be positive")
        
        if self.execution_config.max_execution_time <= 0:
            raise ValueError("max_execution_time must be positive")
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                logger.warning("Empty config file, using defaults")
                return cls()
            
            # Create nested config objects
            memory_config = MemoryConfig(**config_data.get("memory", {}))
            execution_config = ExecutionConfig(**config_data.get("execution", {}))
            prompt_config = PromptConfig(**config_data.get("prompt", {}))
            api_config = APIConfig(**config_data.get("api", {}))
            
            # Remove nested configs from main config
            main_config = {
                k: v for k, v in config_data.items()
                if k not in ["memory", "execution", "prompt", "api"]
            }
            
            return cls(
                memory_config=memory_config,
                execution_config=execution_config,
                prompt_config=prompt_config,
                api_config=api_config,
                **main_config
            )
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_file(self, config_path: Union[str, Path]):
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            raise
    
    def update_from_env(self):
        """Update configuration from environment variables."""
        env_mappings = {
            "OPENGUIDANCE_MODEL": "model_name",
            "OPENGUIDANCE_TEMPERATURE": ("temperature", float),
            "OPENGUIDANCE_MAX_TOKENS": ("max_tokens", int),
            "OPENGUIDANCE_ENABLE_MEMORY": ("enable_memory", lambda x: x.lower() == "true"),
            "OPENGUIDANCE_ENABLE_EXECUTION": ("enable_code_execution", lambda x: x.lower() == "true"),
            "OPENGUIDANCE_API_HOST": ("api_config.host", str),
            "OPENGUIDANCE_API_PORT": ("api_config.port", int),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if isinstance(config_path, tuple):
                        attr_path, converter = config_path
                        value = converter(value)
                    else:
                        attr_path = config_path
                    
                    # Handle nested attributes
                    if "." in attr_path:
                        obj_name, attr_name = attr_path.split(".", 1)
                        obj = getattr(self, obj_name)
                        setattr(obj, attr_name, value)
                    else:
                        setattr(self, attr_path, value)
                    
                    logger.info(f"Updated {attr_path} from environment variable {env_var}")
                    
                except Exception as e:
                    logger.error(f"Failed to set {config_path} from {env_var}: {e}")


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file or environment.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Loaded configuration object
    """
    if config_path:
        config = Config.from_file(config_path)
    else:
        # Look for config file in standard locations
        possible_paths = [
            "config.yaml",
            "config/config.yaml",
            os.path.expanduser("~/.openguidance/config.yaml"),
            "/etc/openguidance/config.yaml"
        ]
        
        config = None
        for path in possible_paths:
            if os.path.exists(path):
                config = Config.from_file(path)
                break
        
        if config is None:
            config = Config()
    
    # Update from environment variables
    config.update_from_env()
    
    return config