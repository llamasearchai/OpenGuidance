"""Proportional Navigation guidance law implementation."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from numba import jit
from dataclasses import dataclass

from openguidance.core.types import State, Control, Vehicle, Environment


@dataclass
class PNConfig:
    """Configuration for Proportional Navigation guidance."""
    # Basic PN parameters
    navigation_constant: float = 3.0  # N (typically 3-5)
    augmented: bool = True  # Use augmented PN
    bias_shaping: bool = False  # Use bias shaping for terminal guidance
    optimal_gain: bool = False  # Use optimal gain scheduling
    
    # Advanced features
    lead_angle_compensation: bool = True
    gravity_compensation: bool = True
    drag_compensation: bool = False
    
    # Filtering parameters
    los_rate_filter_alpha: float = 0.1  # LOS rate filtering coefficient
    target_accel_filter_alpha: float = 0.1  # Target acceleration filter
    
    # Engagement constraints
    min_range: float = 10.0  # m - minimum engagement range
    max_acceleration: float = 100.0  # m/s² - maximum commanded acceleration
    
    # Terminal guidance
    terminal_guidance_range: float = 1000.0  # m - range to switch to terminal mode
    desired_impact_angle: float = 0.0  # rad - desired impact angle
    
    # Safety limits
    max_los_rate: float = 10.0  # rad/s - maximum line-of-sight rate
    min_closing_velocity: float = 10.0  # m/s - minimum closing velocity
    
    # Performance tuning
    enable_saturation_limits: bool = True
    enable_noise_filtering: bool = True
    enable_performance_monitoring: bool = True


class ProportionalNavigation:
    """Proportional Navigation guidance law for missile intercept."""
    
    def __init__(
        self,
        navigation_constant: float = 3.0,
        augmented: bool = True,
        bias_shaping: bool = False,
        optimal_gain: bool = False
    ):
        """Initialize proportional navigation guidance.
        
        Args:
            navigation_constant: Navigation constant (typically 3-5)
            augmented: Use augmented PN (compensates for target acceleration)
            bias_shaping: Use bias-shaped PN for terminal guidance
            optimal_gain: Use optimal gain scheduling
        """
        self.N = navigation_constant
        self.augmented = augmented
        self.bias_shaping = bias_shaping
        self.optimal_gain = optimal_gain
        
        # Internal state
        self.previous_los_rate = np.zeros(3)
        self.previous_time = 0.0
        self.estimated_target_accel = np.zeros(3)
        
    def compute_command(
        self, 
        missile_state: State, 
        target_state: State,
        environment: Optional[Environment] = None
    ) -> np.ndarray:
        """Compute guidance acceleration command.
        
        Args:
            missile_state: Current missile state
            target_state: Current target state
            environment: Environment conditions
            
        Returns:
            Acceleration command in inertial frame [m/s^2]
        """
        # Relative geometry
        relative_position = target_state.position - missile_state.position
        relative_velocity = target_state.velocity - missile_state.velocity
        
        # Range and closing velocity
        range_to_target = np.linalg.norm(relative_position)
        if range_to_target < 1.0:  # Very close
            return np.zeros(3)
        
        # Line-of-sight unit vector
        los_unit = relative_position / range_to_target
        
        # Closing velocity (negative means closing)
        v_closing = -np.dot(relative_velocity, los_unit)
        
        if v_closing <= 0:  # Opening
            return np.zeros(3)
        
        # Line-of-sight rate
        los_rate = self._compute_los_rate(relative_position, relative_velocity)
        
        # Basic proportional navigation command
        accel_cmd = self.N * v_closing * los_rate
        
        # Augmented PN: compensate for target acceleration
        if self.augmented:
            # Estimate target acceleration
            target_accel = self._estimate_target_acceleration(target_state)
            
            # Component perpendicular to LOS
            target_accel_perp = target_accel - np.dot(target_accel, los_unit) * los_unit
            
            # Add to command
            accel_cmd += 0.5 * self.N * target_accel_perp
        
        # Bias shaping for terminal guidance
        if self.bias_shaping:
            bias_term = self._compute_bias_term(
                missile_state, target_state, float(range_to_target), v_closing
            )
            accel_cmd += bias_term
        
        # Optimal gain scheduling
        if self.optimal_gain:
            optimal_N = self._compute_optimal_gain(float(range_to_target), v_closing)
            accel_cmd *= optimal_N / self.N
        
        # Store for next iteration
        self.previous_los_rate = los_rate
        self.previous_time = missile_state.time
        
        return accel_cmd
    
    @staticmethod
    @jit(nopython=True)
    def _compute_los_rate(relative_position: np.ndarray, relative_velocity: np.ndarray) -> np.ndarray:
        """Compute line-of-sight angular rate vector."""
        r = np.linalg.norm(relative_position)
        if r < 1e-6:
            return np.zeros(3)
        
        # LOS rate = (r × v_rel) / r^2
        los_rate = np.cross(relative_position, relative_velocity) / (r * r)
        return los_rate
    
    def _estimate_target_acceleration(self, target_state: State) -> np.ndarray:
        """Estimate target acceleration from velocity history."""
        if target_state.acceleration is not None:
            # Use direct acceleration if available
            return target_state.acceleration
        
        # Simple finite difference estimation
        dt = target_state.time - self.previous_time
        if dt > 0 and hasattr(self, 'previous_target_velocity'):
            accel_estimate = (target_state.velocity - self.previous_target_velocity) / dt
            
            # Low-pass filter to reduce noise
            alpha = 0.1  # Filter coefficient
            self.estimated_target_accel = (
                alpha * accel_estimate + (1 - alpha) * self.estimated_target_accel
            )
        
        # Store for next iteration
        self.previous_target_velocity = target_state.velocity.copy()
        
        return self.estimated_target_accel
    
    def _compute_bias_term(
        self, 
        missile_state: State, 
        target_state: State, 
        range_to_target: float, 
        v_closing: float
    ) -> np.ndarray:
        """Compute bias shaping term for terminal guidance."""
        # Time-to-go estimate
        t_go = range_to_target / v_closing if v_closing > 0 else 1.0
        
        # Bias shaping gain (increases as t_go decreases)
        if t_go > 0.1:
            bias_gain = 1.0 / t_go
        else:
            bias_gain = 10.0  # Saturate for very small t_go
        
        # Desired impact angle (could be parameter)
        desired_impact_angle = 0.0  # Head-on
        
        # Current flight path angle
        missile_velocity_norm = np.linalg.norm(missile_state.velocity)
        if missile_velocity_norm > 0:
            flight_path_angle = np.arcsin(-missile_state.velocity[2] / missile_velocity_norm)
        else:
            flight_path_angle = 0.0
        
        # Angle error
        angle_error = desired_impact_angle - flight_path_angle
        
        # Bias command (simplified)
        bias_cmd = bias_gain * angle_error * np.array([0, 0, 1])  # Vertical correction
        
        return bias_cmd
    
    def _compute_optimal_gain(self, range_to_target: float, v_closing: float) -> float:
        """Compute optimal navigation gain based on engagement geometry."""
        # Time-to-go
        t_go = range_to_target / v_closing if v_closing > 0 else 1.0
        
        # Optimal gain varies with time-to-go
        # This is a simplified model - real optimal gain depends on target maneuvers
        if t_go > 10.0:
            # Long range: use lower gain
            return 3.0
        elif t_go > 2.0:
            # Medium range: standard gain
            return 4.0
        else:
            # Short range: higher gain for precision
            return 5.0
    
    def get_miss_distance_estimate(
        self, 
        missile_state: State, 
        target_state: State
    ) -> float:
        """Estimate miss distance based on current trajectory."""
        # Relative position and velocity
        rel_pos = target_state.position - missile_state.position
        rel_vel = target_state.velocity - missile_state.velocity
        
        # Range and closing velocity
        range_to_target = np.linalg.norm(rel_pos)
        v_closing = -np.dot(rel_vel, rel_pos) / range_to_target if range_to_target > 0 else 0
        
        if v_closing <= 0:
            return float('inf')  # Not closing
        
        # Time to closest approach
        t_ca = range_to_target / v_closing
        
        # Predicted positions at closest approach
        missile_pos_ca = missile_state.position + missile_state.velocity * t_ca
        target_pos_ca = target_state.position + target_state.velocity * t_ca
        
        # Miss distance
        miss_distance = np.linalg.norm(target_pos_ca - missile_pos_ca)
        
        return float(miss_distance)
    
    def is_intercept_feasible(
        self, 
        missile_state: State, 
        target_state: State,
        missile_max_accel: float = 100.0
    ) -> bool:
        """Check if intercept is feasible given missile acceleration limits."""
        # Compute required acceleration
        accel_cmd = self.compute_command(missile_state, target_state)
        required_accel = np.linalg.norm(accel_cmd)
        
        # Check against limits
        return bool(required_accel <= missile_max_accel)
    
    def get_guidance_status(
        self, 
        missile_state: State, 
        target_state: State
    ) -> Dict[str, Any]:
        """Get detailed guidance status information."""
        rel_pos = target_state.position - missile_state.position
        rel_vel = target_state.velocity - missile_state.velocity
        
        range_to_target = np.linalg.norm(rel_pos)
        los_unit = rel_pos / range_to_target if range_to_target > 0 else np.zeros(3)
        v_closing = -np.dot(rel_vel, los_unit)
        
        # Time-to-go
        t_go = range_to_target / v_closing if v_closing > 0 else float('inf')
        
        # Line-of-sight rate
        los_rate = self._compute_los_rate(rel_pos, rel_vel)
        los_rate_magnitude = np.linalg.norm(los_rate)
        
        # Miss distance estimate
        miss_distance = self.get_miss_distance_estimate(missile_state, target_state)
        
        return {
            'range_to_target': range_to_target,
            'closing_velocity': v_closing,
            'time_to_go': t_go,
            'los_rate_magnitude': los_rate_magnitude,
            'estimated_miss_distance': miss_distance,
            'intercept_feasible': self.is_intercept_feasible(missile_state, target_state),
        }


class AdvancedProportionalNavigation(ProportionalNavigation):
    """Advanced PN with additional features for complex scenarios."""
    
    def __init__(
        self,
        navigation_constant: float = 3.0,
        lead_angle_compensation: bool = True,
        gravity_compensation: bool = True,
        drag_compensation: bool = False,
        **kwargs
    ):
        """Initialize advanced proportional navigation."""
        super().__init__(navigation_constant, **kwargs)
        
        self.lead_angle_compensation = lead_angle_compensation
        self.gravity_compensation = gravity_compensation
        self.drag_compensation = drag_compensation
        
    def compute_command(
        self, 
        missile_state: State, 
        target_state: State,
        environment: Optional[Environment] = None
    ) -> np.ndarray:
        """Compute advanced guidance command with compensations."""
        # Base PN command
        accel_cmd = super().compute_command(missile_state, target_state, environment)
        
        # Lead angle compensation
        if self.lead_angle_compensation:
            lead_compensation = self._compute_lead_angle_compensation(
                missile_state, target_state
            )
            accel_cmd += lead_compensation
        
        # Gravity compensation
        if self.gravity_compensation and environment:
            gravity_compensation = self._compute_gravity_compensation(
                missile_state, environment
            )
            accel_cmd += gravity_compensation
        
        # Drag compensation
        if self.drag_compensation and environment:
            drag_compensation = self._compute_drag_compensation(
                missile_state, environment
            )
            accel_cmd += drag_compensation
        
        return accel_cmd
    
    def _compute_lead_angle_compensation(
        self, 
        missile_state: State, 
        target_state: State
    ) -> np.ndarray:
        """Compute lead angle compensation for maneuvering targets."""
        # Estimate target maneuver
        target_accel = self._estimate_target_acceleration(target_state)
        
        # Time delay compensation
        time_delay = 0.1  # Typical autopilot delay
        
        # Lead compensation
        lead_compensation = time_delay * target_accel
        
        return lead_compensation
    
    def _compute_gravity_compensation(
        self, 
        missile_state: State, 
        environment: Environment
    ) -> np.ndarray:
        """Compute gravity compensation."""
        # Simply compensate for gravity
        return -environment.gravity
    
    def _compute_drag_compensation(
        self, 
        missile_state: State, 
        environment: Environment
    ) -> np.ndarray:
        """Compute drag compensation (simplified)."""
        # Simplified drag model
        velocity = missile_state.velocity
        speed = np.linalg.norm(velocity)
        
        if speed < 1.0:
            return np.zeros(3)
        
        # Drag acceleration (opposite to velocity)
        drag_coeff = 0.01  # Simplified
        altitude = missile_state.altitude
        density = environment.get_density_at_altitude(altitude)
        
        drag_accel = -0.5 * density * speed * drag_coeff * velocity
        
        return drag_accel 