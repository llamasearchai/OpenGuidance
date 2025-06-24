"""PID Controller implementation for OpenGuidance."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from openguidance.core.types import State, Control


@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float = 1.0  # Proportional gain
    ki: float = 0.0  # Integral gain
    kd: float = 0.0  # Derivative gain
    
    # Anti-windup
    integral_limit: float = 10.0
    output_limit: Optional[Tuple[float, float]] = None
    
    # Derivative filtering
    derivative_filter_tau: float = 0.01  # Time constant for derivative filter


class PIDController:
    """PID controller with anti-windup and derivative filtering."""
    
    def __init__(
        self,
        gains: PIDGains,
        dt: float = 0.01,
        enable_antiwindup: bool = True,
        enable_derivative_filter: bool = True
    ):
        """Initialize PID controller.
        
        Args:
            gains: PID gains and limits
            dt: Sample time
            enable_antiwindup: Enable integral anti-windup
            enable_derivative_filter: Enable derivative term filtering
        """
        self.gains = gains
        self.dt = dt
        self.enable_antiwindup = enable_antiwindup
        self.enable_derivative_filter = enable_derivative_filter
        
        # Internal state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_derivative = 0.0
        self.previous_time = 0.0
        
        # Reset flag
        self.first_call = True
        
    def update(self, setpoint: float, measurement: float, current_time: float) -> float:
        """Update PID controller.
        
        Args:
            setpoint: Desired value
            measurement: Current measured value
            current_time: Current time
            
        Returns:
            Control output
        """
        # Compute error
        error = setpoint - measurement
        
        # Handle first call
        if self.first_call:
            self.previous_error = error
            self.previous_time = current_time
            self.first_call = False
            return self.gains.kp * error
        
        # Time step
        dt = current_time - self.previous_time
        if dt <= 0:
            dt = self.dt  # Use default if time didn't advance
        
        # Proportional term
        proportional = self.gains.kp * error
        
        # Integral term
        self.integral += error * dt
        
        # Anti-windup
        if self.enable_antiwindup:
            self.integral = np.clip(
                self.integral, 
                -self.gains.integral_limit, 
                self.gains.integral_limit
            )
        
        integral = self.gains.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative_raw = (error - self.previous_error) / dt
        else:
            derivative_raw = 0.0
        
        # Derivative filtering
        if self.enable_derivative_filter and self.gains.derivative_filter_tau > 0:
            alpha = dt / (self.gains.derivative_filter_tau + dt)
            derivative_filtered = alpha * derivative_raw + (1 - alpha) * self.previous_derivative
            self.previous_derivative = derivative_filtered
        else:
            derivative_filtered = derivative_raw
        
        derivative = self.gains.kd * derivative_filtered
        
        # Total output
        output = proportional + integral + derivative
        
        # Output limiting
        if self.gains.output_limit:
            output_limited = np.clip(output, self.gains.output_limit[0], self.gains.output_limit[1])
            
            # Back-calculate integral for anti-windup
            if self.enable_antiwindup and output != output_limited:
                integral_back_calc = output_limited - proportional - derivative
                self.integral = integral_back_calc / self.gains.ki if self.gains.ki != 0 else 0.0
            
            output = output_limited
        
        # Store for next iteration
        self.previous_error = error
        self.previous_time = current_time
        
        return output
    
    def reset(self):
        """Reset controller internal state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_derivative = 0.0
        self.first_call = True
    
    def set_gains(self, gains: PIDGains):
        """Update controller gains."""
        self.gains = gains
    
    def get_components(self) -> Dict[str, float]:
        """Get individual PID components for debugging."""
        error = self.previous_error
        return {
            'proportional': self.gains.kp * error,
            'integral': self.gains.ki * self.integral,
            'derivative': self.gains.kd * self.previous_derivative,
            'error': error,
            'integral_state': self.integral,
        }


class CascadedPIDController:
    """Cascaded PID controller for position/velocity control."""
    
    def __init__(
        self,
        outer_gains: PIDGains,  # Position controller
        inner_gains: PIDGains,  # Velocity controller
        dt: float = 0.01
    ):
        """Initialize cascaded PID controller.
        
        Args:
            outer_gains: Outer loop (position) PID gains
            inner_gains: Inner loop (velocity) PID gains
            dt: Sample time
        """
        self.outer_pid = PIDController(outer_gains, dt)
        self.inner_pid = PIDController(inner_gains, dt)
        
    def update(
        self, 
        position_setpoint: float, 
        position_measurement: float,
        velocity_measurement: float,
        current_time: float
    ) -> float:
        """Update cascaded controller.
        
        Args:
            position_setpoint: Desired position
            position_measurement: Current position
            velocity_measurement: Current velocity
            current_time: Current time
            
        Returns:
            Acceleration command
        """
        # Outer loop: position to velocity
        velocity_command = self.outer_pid.update(
            position_setpoint, position_measurement, current_time
        )
        
        # Inner loop: velocity to acceleration
        acceleration_command = self.inner_pid.update(
            velocity_command, velocity_measurement, current_time
        )
        
        return acceleration_command
    
    def reset(self):
        """Reset both controllers."""
        self.outer_pid.reset()
        self.inner_pid.reset()


class AttitudePIDController:
    """3-axis attitude PID controller."""
    
    def __init__(
        self,
        roll_gains: PIDGains,
        pitch_gains: PIDGains,
        yaw_gains: PIDGains,
        dt: float = 0.01
    ):
        """Initialize attitude controller.
        
        Args:
            roll_gains: Roll axis PID gains
            pitch_gains: Pitch axis PID gains
            yaw_gains: Yaw axis PID gains
            dt: Sample time
        """
        self.roll_pid = PIDController(roll_gains, dt)
        self.pitch_pid = PIDController(pitch_gains, dt)
        self.yaw_pid = PIDController(yaw_gains, dt)
        
    def update(
        self,
        attitude_setpoint: np.ndarray,  # [roll, pitch, yaw] in radians
        attitude_measurement: np.ndarray,  # [roll, pitch, yaw] in radians
        current_time: float
    ) -> np.ndarray:
        """Update attitude controller.
        
        Args:
            attitude_setpoint: Desired attitude [roll, pitch, yaw]
            attitude_measurement: Current attitude [roll, pitch, yaw]
            current_time: Current time
            
        Returns:
            Angular rate commands [p, q, r]
        """
        # Handle angle wrapping for yaw
        yaw_error = attitude_setpoint[2] - attitude_measurement[2]
        yaw_error = self._wrap_angle(yaw_error)
        
        # Adjust setpoint for wrapped yaw
        yaw_setpoint_wrapped = attitude_measurement[2] + yaw_error
        
        roll_cmd = self.roll_pid.update(
            attitude_setpoint[0], attitude_measurement[0], current_time
        )
        pitch_cmd = self.pitch_pid.update(
            attitude_setpoint[1], attitude_measurement[1], current_time
        )
        yaw_cmd = self.yaw_pid.update(
            yaw_setpoint_wrapped, attitude_measurement[2], current_time
        )
        
        return np.array([roll_cmd, pitch_cmd, yaw_cmd])
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def reset(self):
        """Reset all attitude controllers."""
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()


class VectorPIDController:
    """Vector PID controller for 3D position/velocity control."""
    
    def __init__(
        self,
        gains_x: PIDGains,
        gains_y: PIDGains,
        gains_z: PIDGains,
        dt: float = 0.01,
        coordinate_frame: str = "inertial"  # "inertial" or "body"
    ):
        """Initialize vector PID controller.
        
        Args:
            gains_x: X-axis PID gains
            gains_y: Y-axis PID gains
            gains_z: Z-axis PID gains
            dt: Sample time
            coordinate_frame: Coordinate frame for control
        """
        self.pid_x = PIDController(gains_x, dt)
        self.pid_y = PIDController(gains_y, dt)
        self.pid_z = PIDController(gains_z, dt)
        
        self.coordinate_frame = coordinate_frame
        
    def update(
        self,
        setpoint: np.ndarray,  # 3D setpoint
        measurement: np.ndarray,  # 3D measurement
        current_time: float,
        rotation_matrix: Optional[np.ndarray] = None  # Body to inertial
    ) -> np.ndarray:
        """Update vector controller.
        
        Args:
            setpoint: Desired 3D vector
            measurement: Current 3D measurement
            current_time: Current time
            rotation_matrix: Rotation matrix (if frame conversion needed)
            
        Returns:
            3D control command
        """
        # Transform to control frame if needed
        if self.coordinate_frame == "body" and rotation_matrix is not None:
            # Transform setpoint and measurement to body frame
            setpoint_body = rotation_matrix.T @ setpoint
            measurement_body = rotation_matrix.T @ measurement
        else:
            setpoint_body = setpoint
            measurement_body = measurement
        
        # Update each axis
        cmd_x = self.pid_x.update(setpoint_body[0], measurement_body[0], current_time)
        cmd_y = self.pid_y.update(setpoint_body[1], measurement_body[1], current_time)
        cmd_z = self.pid_z.update(setpoint_body[2], measurement_body[2], current_time)
        
        command = np.array([cmd_x, cmd_y, cmd_z])
        
        # Transform back to inertial frame if needed
        if self.coordinate_frame == "body" and rotation_matrix is not None:
            command = rotation_matrix @ command
        
        return command
    
    def reset(self):
        """Reset all axis controllers."""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
    
    def get_all_components(self) -> Dict[str, Dict[str, float]]:
        """Get PID components for all axes."""
        return {
            'x': self.pid_x.get_components(),
            'y': self.pid_y.get_components(),
            'z': self.pid_z.get_components(),
        } 