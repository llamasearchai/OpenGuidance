"""Sensor fusion module for OpenGuidance navigation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

from openguidance.core.types import State

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Types of sensors supported by the fusion system."""
    IMU = auto()
    GPS = auto()
    MAGNETOMETER = auto()
    BAROMETER = auto()
    VISION = auto()
    RADAR = auto()
    LIDAR = auto()


@dataclass
class SensorMeasurement:
    """Container for sensor measurements."""
    sensor_type: SensorType
    timestamp: float
    data: np.ndarray
    covariance: np.ndarray
    valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorFusionConfig:
    """Configuration for sensor fusion system."""
    # Sensor configurations
    enabled_sensors: List[SensorType] = field(default_factory=list)
    
    # Timing parameters
    max_time_offset: float = 0.1  # Maximum allowed time offset for measurements
    
    # Fault detection
    enable_fault_detection: bool = True
    innovation_threshold: float = 5.0  # Chi-squared threshold for innovation gating
    
    # Adaptive weighting
    enable_adaptive_weighting: bool = True
    weight_adaptation_rate: float = 0.1


class SensorFusionSystem:
    """Multi-sensor fusion system for navigation state estimation."""
    
    def __init__(self, config: SensorFusionConfig):
        self.config = config
        
        # Sensor buffers
        self.measurement_buffer: Dict[SensorType, List[SensorMeasurement]] = {}
        for sensor_type in config.enabled_sensors:
            self.measurement_buffer[sensor_type] = []
        
        # Sensor weights and health status
        self.sensor_weights: Dict[SensorType, float] = {}
        self.sensor_health: Dict[SensorType, bool] = {}
        
        # Initialize sensor weights
        for sensor_type in config.enabled_sensors:
            self.sensor_weights[sensor_type] = 1.0
            self.sensor_health[sensor_type] = True
        
        # Fusion statistics
        self.fusion_count = 0
        self.rejected_measurements = 0
        
        logger.info(f"Sensor fusion system initialized with sensors: {config.enabled_sensors}")
    
    def add_measurement(self, measurement: SensorMeasurement) -> None:
        """Add a new sensor measurement to the fusion system."""
        if measurement.sensor_type not in self.config.enabled_sensors:
            logger.warning(f"Sensor type {measurement.sensor_type} not enabled")
            return
        
        # Add to buffer
        self.measurement_buffer[measurement.sensor_type].append(measurement)
        
        # Maintain buffer size (keep only recent measurements)
        max_buffer_size = 100
        if len(self.measurement_buffer[measurement.sensor_type]) > max_buffer_size:
            self.measurement_buffer[measurement.sensor_type].pop(0)
        
        logger.debug(f"Added {measurement.sensor_type} measurement at time {measurement.timestamp}")
    
    def get_synchronized_measurements(self, target_time: float) -> Dict[SensorType, SensorMeasurement]:
        """Get synchronized measurements closest to the target time."""
        synchronized = {}
        
        for sensor_type in self.config.enabled_sensors:
            if not self.measurement_buffer[sensor_type]:
                continue
            
            # Find measurement closest to target time
            best_measurement = None
            min_time_diff = float('inf')
            
            for measurement in self.measurement_buffer[sensor_type]:
                time_diff = abs(measurement.timestamp - target_time)
                if time_diff < min_time_diff and time_diff <= self.config.max_time_offset:
                    min_time_diff = time_diff
                    best_measurement = measurement
            
            if best_measurement is not None:
                synchronized[sensor_type] = best_measurement
        
        return synchronized
    
    def fuse_measurements(self, measurements: Dict[SensorType, SensorMeasurement], 
                         current_state: State) -> Tuple[np.ndarray, np.ndarray]:
        """Fuse multiple sensor measurements into a single estimate."""
        if not measurements:
            return np.zeros(12), np.eye(12) * 1e6  # Return high uncertainty
        
        # Convert current state to vector
        state_vector = self._state_to_vector(current_state)
        
        # Initialize fusion variables
        fused_measurement = np.zeros_like(state_vector)
        fused_covariance = np.zeros((len(state_vector), len(state_vector)))
        total_weight = 0.0
        
        # Process each measurement
        for sensor_type, measurement in measurements.items():
            if not self.sensor_health[sensor_type]:
                continue
            
            # Convert measurement to state vector format
            measurement_vector, measurement_cov = self._convert_measurement_to_state_format(
                measurement, state_vector
            )
            
            # Fault detection
            if self.config.enable_fault_detection:
                if not self._validate_measurement(measurement_vector, measurement_cov, state_vector):
                    logger.warning(f"Rejecting {sensor_type} measurement due to fault detection")
                    self.rejected_measurements += 1
                    continue
            
            # Compute weight
            weight = self._compute_sensor_weight(sensor_type, measurement)
            
            # Weighted fusion
            if total_weight == 0:
                fused_measurement = weight * measurement_vector
                fused_covariance = weight * measurement_cov
            else:
                # Information filter approach
                try:
                    inv_cov = np.linalg.inv(measurement_cov)
                    fused_inv_cov = np.linalg.inv(fused_covariance) if total_weight > 0 else np.zeros_like(fused_covariance)
                    
                    combined_inv_cov = fused_inv_cov + weight * inv_cov
                    fused_covariance = np.linalg.inv(combined_inv_cov)
                    
                    fused_measurement = fused_covariance @ (
                        fused_inv_cov @ fused_measurement + weight * inv_cov @ measurement_vector
                    )
                except np.linalg.LinAlgError:
                    # Fallback to simple weighted average
                    fused_measurement = (total_weight * fused_measurement + weight * measurement_vector) / (total_weight + weight)
            
            total_weight += weight
        
        # Normalize if we have measurements
        if total_weight > 0:
            # Update sensor weights if adaptive weighting is enabled
            if self.config.enable_adaptive_weighting:
                self._update_adaptive_weights(measurements, fused_measurement)
        else:
            # No valid measurements
            fused_covariance = np.eye(len(state_vector)) * 1e6
        
        self.fusion_count += 1
        return fused_measurement, fused_covariance
    
    def _state_to_vector(self, state: State) -> np.ndarray:
        """Convert State object to vector representation."""
        vector = np.zeros(12)  # [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz] -> 13D, but simplified to 12D
        
        if hasattr(state, 'position') and state.position is not None:
            vector[0:3] = state.position
        if hasattr(state, 'velocity') and state.velocity is not None:
            vector[3:6] = state.velocity
        if hasattr(state, 'attitude') and state.attitude is not None:
            # Convert quaternion to Euler angles for simplicity
            euler = state.attitude.yaw_pitch_roll  # [yaw, pitch, roll]
            vector[6:9] = euler
        if hasattr(state, 'angular_velocity') and state.angular_velocity is not None:
            vector[9:12] = state.angular_velocity
        
        return vector
    
    def _convert_measurement_to_state_format(self, measurement: SensorMeasurement, 
                                           state_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert sensor measurement to state vector format."""
        measurement_vector = np.zeros_like(state_vector)
        measurement_cov = np.eye(len(state_vector)) * 1e6  # High uncertainty by default
        
        if measurement.sensor_type == SensorType.GPS:
            # GPS provides position (and possibly velocity)
            if len(measurement.data) >= 3:
                measurement_vector[0:3] = measurement.data[0:3]  # Position
                measurement_cov[0:3, 0:3] = measurement.covariance[0:3, 0:3]
            if len(measurement.data) >= 6:
                measurement_vector[3:6] = measurement.data[3:6]  # Velocity
                measurement_cov[3:6, 3:6] = measurement.covariance[3:6, 3:6]
        
        elif measurement.sensor_type == SensorType.IMU:
            # IMU provides acceleration and angular velocity
            if len(measurement.data) >= 6:
                # Use angular velocity directly
                measurement_vector[9:12] = measurement.data[3:6]
                measurement_cov[9:12, 9:12] = measurement.covariance[3:6, 3:6]
                
                # Integrate acceleration for velocity estimate (simplified)
                # This would normally require proper integration over time
                dt = 0.01  # Assume 100Hz IMU
                measurement_vector[3:6] = state_vector[3:6] + measurement.data[0:3] * dt
                measurement_cov[3:6, 3:6] = measurement.covariance[0:3, 0:3] * dt**2
        
        elif measurement.sensor_type == SensorType.MAGNETOMETER:
            # Magnetometer provides heading information
            if len(measurement.data) >= 3:
                # Convert magnetic field to heading (simplified)
                heading = np.arctan2(measurement.data[1], measurement.data[0])
                measurement_vector[6] = heading  # Yaw angle
                measurement_cov[6, 6] = measurement.covariance[0, 0]  # Simplified
        
        elif measurement.sensor_type == SensorType.BAROMETER:
            # Barometer provides altitude
            if len(measurement.data) >= 1:
                measurement_vector[2] = -measurement.data[0]  # Altitude (NED frame)
                measurement_cov[2, 2] = measurement.covariance[0, 0]
        
        return measurement_vector, measurement_cov
    
    def _validate_measurement(self, measurement: np.ndarray, measurement_cov: np.ndarray, 
                            state_estimate: np.ndarray) -> bool:
        """Validate measurement using innovation gating."""
        # Compute innovation
        innovation = measurement - state_estimate
        
        # Chi-squared test
        try:
            inv_cov = np.linalg.inv(measurement_cov)
            chi_squared = innovation.T @ inv_cov @ innovation
            
            # Compare with threshold
            return bool(chi_squared < self.config.innovation_threshold)
        except np.linalg.LinAlgError:
            # If covariance is singular, accept the measurement
            return True
    
    def _compute_sensor_weight(self, sensor_type: SensorType, measurement: SensorMeasurement) -> float:
        """Compute weight for sensor based on reliability and accuracy."""
        base_weight = self.sensor_weights[sensor_type]
        
        # Adjust weight based on measurement quality
        quality_factor = 1.0
        
        # Check measurement validity
        if not measurement.valid:
            quality_factor *= 0.1
        
        # Check covariance trace (lower trace = higher confidence)
        cov_trace = np.trace(measurement.covariance)
        if cov_trace > 0:
            quality_factor *= 1.0 / (1.0 + cov_trace)
        
        return base_weight * quality_factor
    
    def _update_adaptive_weights(self, measurements: Dict[SensorType, SensorMeasurement], 
                               fused_estimate: np.ndarray) -> None:
        """Update sensor weights based on performance."""
        for sensor_type, measurement in measurements.items():
            if sensor_type not in self.sensor_weights:
                continue
            
            # Convert measurement to compare with fused estimate
            measurement_vector, _ = self._convert_measurement_to_state_format(
                measurement, fused_estimate
            )
            
            # Compute error
            error = np.linalg.norm(measurement_vector - fused_estimate)
            
            # Update weight (lower error = higher weight)
            error_factor = 1.0 / (1.0 + error)
            
            # Exponential moving average
            alpha = self.config.weight_adaptation_rate
            self.sensor_weights[sensor_type] = float(
                (1 - alpha) * self.sensor_weights[sensor_type] + 
                alpha * error_factor
            )
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """Get status of all sensors."""
        status = {
            "fusion_count": self.fusion_count,
            "rejected_measurements": self.rejected_measurements,
            "sensor_weights": dict(self.sensor_weights),
            "sensor_health": dict(self.sensor_health),
            "buffer_sizes": {
                sensor_type.name: len(buffer) 
                for sensor_type, buffer in self.measurement_buffer.items()
            }
        }
        return status
    
    def set_sensor_health(self, sensor_type: SensorType, healthy: bool) -> None:
        """Set health status of a sensor."""
        if sensor_type in self.sensor_health:
            self.sensor_health[sensor_type] = healthy
            logger.info(f"Sensor {sensor_type} health set to {healthy}")
    
    def clear_measurement_buffer(self, sensor_type: Optional[SensorType] = None) -> None:
        """Clear measurement buffer for specified sensor or all sensors."""
        if sensor_type is not None:
            if sensor_type in self.measurement_buffer:
                self.measurement_buffer[sensor_type].clear()
        else:
            for buffer in self.measurement_buffer.values():
                buffer.clear()
        
        logger.info(f"Cleared measurement buffer for {sensor_type or 'all sensors'}") 