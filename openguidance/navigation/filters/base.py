"""
Base classes for state estimation filters.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from openguidance.core.types import State


@dataclass
class FilterConfig:
    """Base configuration for filters."""
    state_dim: int
    process_noise_std: float = 0.1
    initial_state_uncertainty: float = 1.0
    enable_adaptive_tuning: bool = False
    max_iterations: int = 100


class Filter(ABC):
    """Abstract base class for state estimation filters."""
    
    def __init__(self, config: FilterConfig):
        """Initialize filter with configuration."""
        self.config = config
        self.is_initialized = False
        self.iteration_count = 0
        
    @abstractmethod
    def predict(self, control: Optional[np.ndarray], dt: float) -> None:
        """Prediction step."""
        pass
    
    @abstractmethod
    def update(self, sensor_type: str, measurement: np.ndarray) -> None:
        """Measurement update step."""
        pass
    
    @abstractmethod
    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate and covariance."""
        pass
    
    @abstractmethod
    def reset(self, state: np.ndarray, covariance: np.ndarray) -> None:
        """Reset filter state."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filter performance statistics."""
        return {
            "iterations": self.iteration_count,
            "initialized": self.is_initialized,
            "state_dim": self.config.state_dim
        } 