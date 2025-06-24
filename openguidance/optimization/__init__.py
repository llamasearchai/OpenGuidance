"""
Advanced optimization algorithms for OpenGuidance.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .trajectory_optimization import TrajectoryOptimizer, TrajectoryOptimizerConfig

# Import optional modules with error handling
try:
    from .model_predictive_control import ModelPredictiveController, MPCConfig
except ImportError:
    ModelPredictiveController = None
    MPCConfig = None

try:
    from .genetic_algorithm import GeneticAlgorithm, GAConfig
except ImportError:
    GeneticAlgorithm = None
    GAConfig = None

try:
    from .particle_swarm import ParticleSwarmOptimizer, PSOConfig
except ImportError:
    ParticleSwarmOptimizer = None
    PSOConfig = None

__all__ = [
    "TrajectoryOptimizer",
    "TrajectoryOptimizerConfig",
    "ModelPredictiveController", 
    "MPCConfig",
    "GeneticAlgorithm",
    "GAConfig",
    "ParticleSwarmOptimizer",
    "PSOConfig",
] 