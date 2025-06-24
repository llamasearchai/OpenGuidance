"""
Advanced AI integration for OpenGuidance framework.

This package provides:
- Reinforcement learning for optimal control
- Neural network-based state estimation
- AI-powered trajectory planning
- Machine learning for system identification
- Deep learning for sensor fusion

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .reinforcement_learning import RLController, RLConfig
from .neural_networks import NeuralStateEstimator, NeuralNetworkConfig
from .ml_trajectory_planner import MLTrajectoryPlanner, MLPlannerConfig
from .system_identification import MLSystemID, SystemIDConfig

__all__ = [
    "RLController",
    "RLConfig", 
    "NeuralStateEstimator",
    "NeuralNetworkConfig",
    "MLTrajectoryPlanner",
    "MLPlannerConfig",
    "MLSystemID",
    "SystemIDConfig",
] 