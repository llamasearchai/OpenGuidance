"""
Navigation and state estimation algorithms for OpenGuidance.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .filters.extended_kalman_filter import ExtendedKalmanFilter

# Import optional modules with error handling
try:
    from .filters.unscented_kalman_filter import UnscentedKalmanFilter
except ImportError:
    UnscentedKalmanFilter = None

try:
    from .filters.particle_filter import ParticleFilter
except ImportError:
    ParticleFilter = None

try:
    from .sensor_fusion import SensorFusionSystem
except ImportError:
    SensorFusionSystem = None

try:
    from .inertial_navigation import InertialNavigationSystem
except ImportError:
    InertialNavigationSystem = None

__all__ = [
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter", 
    "ParticleFilter",
    "SensorFusionSystem",
    "InertialNavigationSystem",
] 