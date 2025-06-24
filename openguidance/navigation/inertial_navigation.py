"""Inertial Navigation System for OpenGuidance.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from openguidance.core.types import State

logger = logging.getLogger(__name__)


@dataclass
class INSConfig:
    """Configuration for Inertial Navigation System."""
    # Initial alignment parameters
    alignment_time: float = 300.0  # seconds
    stationary_threshold: float = 0.01  # m/s for stationary detection
    
    # Gravity model
    gravity_magnitude: float = 9.81  # m/s^2
    
    # Earth parameters
    earth_radius: float = 6378137.0  # m (WGS84)
    earth_rotation_rate: float = 7.2921159e-5  # rad/s
    
    # Coordinate frame
    reference_frame: str = "NED"  # "NED" or "ENU"


class InertialNavigationSystem:
    """Strapdown Inertial Navigation System.
    
    Provides dead-reckoning navigation using IMU measurements.
    """
    
    def __init__(self, config: INSConfig):
        self.config = config
        
        # Navigation state
        self.position = np.zeros(3)  # m
        self.velocity = np.zeros(3)  # m/s
        self.attitude = np.eye(3)    # Direction Cosine Matrix (DCM)
        
        # IMU biases
        self.accel_bias = np.zeros(3)  # m/s^2
        self.gyro_bias = np.zeros(3)   # rad/s
        
        # Alignment state
        self.is_aligned = False
        self.alignment_samples = []
        self.alignment_start_time = None
        
        # Navigation parameters
        self.gravity_vector = np.array([0, 0, self.config.gravity_magnitude])
        if self.config.reference_frame == "ENU":
            self.gravity_vector = np.array([0, 0, -self.config.gravity_magnitude])
        
        # Earth rotation vector (simplified - assumes stationary on Earth surface)
        self.earth_rotation = np.array([0, 0, self.config.earth_rotation_rate])
        
        # Navigation statistics
        self.update_count = 0
        self.last_update_time = None
        
        logger.info("Inertial Navigation System initialized")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get INS diagnostic information."""
        return {
            "is_aligned": self.is_aligned,
            "update_count": self.update_count,
            "alignment_samples": len(self.alignment_samples),
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "accel_bias": self.accel_bias.tolist(),
            "gyro_bias": self.gyro_bias.tolist(),
            "attitude_determinant": float(np.linalg.det(self.attitude)),
            "velocity_magnitude": float(np.linalg.norm(self.velocity))
        } 