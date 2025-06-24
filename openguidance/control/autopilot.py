"""Autopilot system for OpenGuidance."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass

from openguidance.core.types import State, Control, Vehicle, VehicleType, ControlMode
from openguidance.control.controllers.pid import PIDController, PIDGains, AttitudePIDController, VectorPIDController


class AutopilotMode(Enum):
    """Autopilot operating modes."""
    MANUAL = auto()
    STABILIZE = auto()
    ALTITUDE_HOLD = auto()
    POSITION_HOLD = auto()
    GUIDED = auto()
    AUTO = auto()
    LAND = auto()
    RTL = auto()


@dataclass
class AutopilotConfig:
    """Configuration for autopilot system."""
    # Default mode
    default_mode: AutopilotMode = AutopilotMode.STABILIZE
    
    # Attitude control gains
    roll_kp: float = 10.0
    roll_ki: float = 0.0
    roll_kd: float = 2.0
    roll_integral_limit: float = 5.0
    
    pitch_kp: float = 10.0
    pitch_ki: float = 0.0
    pitch_kd: float = 2.0
    pitch_integral_limit: float = 5.0
    
    yaw_kp: float = 5.0
    yaw_ki: float = 0.0
    yaw_kd: float = 1.0
    yaw_integral_limit: float = 5.0
    
    # Position control gains
    pos_x_kp: float = 1.0
    pos_x_ki: float = 0.0
    pos_x_kd: float = 0.5
    pos_x_integral_limit: float = 10.0
    
    pos_y_kp: float = 1.0
    pos_y_ki: float = 0.0
    pos_y_kd: float = 0.5
    pos_y_integral_limit: float = 10.0
    
    pos_z_kp: float = 2.0
    pos_z_ki: float = 0.0
    pos_z_kd: float = 1.0
    pos_z_integral_limit: float = 10.0
    
    # Velocity control gains
    vel_x_kp: float = 2.0
    vel_x_ki: float = 0.0
    vel_x_kd: float = 0.1
    vel_x_integral_limit: float = 5.0
    
    vel_y_kp: float = 2.0
    vel_y_ki: float = 0.0
    vel_y_kd: float = 0.1
    vel_y_integral_limit: float = 5.0
    
    vel_z_kp: float = 3.0
    vel_z_ki: float = 0.0
    vel_z_kd: float = 0.2
    vel_z_integral_limit: float = 5.0
    
    # Control limits
    max_attitude_angle: float = 0.785  # rad (45 degrees)
    max_angular_rate: float = 3.14  # rad/s (180 degrees/s)
    max_acceleration: float = 20.0  # m/s²
    
    # Update rates
    attitude_update_rate: float = 100.0  # Hz
    position_update_rate: float = 50.0  # Hz
    
    # Safety features
    enable_geofence: bool = True
    geofence_radius: float = 1000.0  # m
    enable_altitude_limit: bool = True
    max_altitude: float = 500.0  # m
    min_altitude: float = 5.0  # m


class Autopilot:
    """Main autopilot system for aircraft and missiles."""
    
    def __init__(self, vehicle: Vehicle, mode: AutopilotMode = AutopilotMode.STABILIZE):
        """Initialize autopilot system.
        
        Args:
            vehicle: Vehicle configuration
            mode: Initial autopilot mode
        """
        self.vehicle = vehicle
        self.mode = mode
        
        # Control gains (would typically be loaded from config)
        self._setup_default_gains()
        
        # Initialize controllers
        self.attitude_controller = AttitudePIDController(
            roll_gains=self.roll_gains,
            pitch_gains=self.pitch_gains,
            yaw_gains=self.yaw_gains
        )
        
        self.position_controller = VectorPIDController(
            gains_x=self.pos_x_gains,
            gains_y=self.pos_y_gains,
            gains_z=self.pos_z_gains
        )
        
        self.velocity_controller = VectorPIDController(
            gains_x=self.vel_x_gains,
            gains_y=self.vel_y_gains,
            gains_z=self.vel_z_gains
        )
        
        # Command tracking
        self.position_command = np.zeros(3)
        self.velocity_command = np.zeros(3)
        self.acceleration_command = np.zeros(3)
        self.attitude_command = np.zeros(3)
        self.angular_rate_command = np.zeros(3)
        
        # Limits
        self.max_attitude_angle = np.deg2rad(45.0)  # Maximum bank/pitch angle
        self.max_angular_rate = np.deg2rad(180.0)  # Maximum angular rate
        self.max_acceleration = 20.0  # Maximum acceleration command
        
    def _setup_default_gains(self):
        """Setup default control gains based on vehicle type."""
        if self.vehicle.type == VehicleType.AIRCRAFT:
            # Aircraft gains
            self.roll_gains = PIDGains(kp=10.0, ki=0.0, kd=2.0, integral_limit=5.0)
            self.pitch_gains = PIDGains(kp=10.0, ki=0.0, kd=2.0, integral_limit=5.0)
            self.yaw_gains = PIDGains(kp=5.0, ki=0.0, kd=1.0, integral_limit=5.0)
            
            self.pos_x_gains = PIDGains(kp=1.0, ki=0.0, kd=0.5, integral_limit=10.0)
            self.pos_y_gains = PIDGains(kp=1.0, ki=0.0, kd=0.5, integral_limit=10.0)
            self.pos_z_gains = PIDGains(kp=2.0, ki=0.0, kd=1.0, integral_limit=10.0)
            
            self.vel_x_gains = PIDGains(kp=2.0, ki=0.0, kd=0.1, integral_limit=5.0)
            self.vel_y_gains = PIDGains(kp=2.0, ki=0.0, kd=0.1, integral_limit=5.0)
            self.vel_z_gains = PIDGains(kp=3.0, ki=0.0, kd=0.2, integral_limit=5.0)
            
        elif self.vehicle.type == VehicleType.MISSILE:
            # Missile gains (higher bandwidth)
            self.roll_gains = PIDGains(kp=20.0, ki=0.0, kd=1.0, integral_limit=10.0)
            self.pitch_gains = PIDGains(kp=20.0, ki=0.0, kd=1.0, integral_limit=10.0)
            self.yaw_gains = PIDGains(kp=15.0, ki=0.0, kd=0.5, integral_limit=10.0)
            
            self.pos_x_gains = PIDGains(kp=0.5, ki=0.0, kd=0.1, integral_limit=5.0)
            self.pos_y_gains = PIDGains(kp=0.5, ki=0.0, kd=0.1, integral_limit=5.0)
            self.pos_z_gains = PIDGains(kp=1.0, ki=0.0, kd=0.2, integral_limit=5.0)
            
            self.vel_x_gains = PIDGains(kp=1.0, ki=0.0, kd=0.05, integral_limit=2.0)
            self.vel_y_gains = PIDGains(kp=1.0, ki=0.0, kd=0.05, integral_limit=2.0)
            self.vel_z_gains = PIDGains(kp=1.5, ki=0.0, kd=0.1, integral_limit=2.0)
            
        else:
            # Default gains
            self.roll_gains = PIDGains(kp=5.0, ki=0.0, kd=1.0)
            self.pitch_gains = PIDGains(kp=5.0, ki=0.0, kd=1.0)
            self.yaw_gains = PIDGains(kp=3.0, ki=0.0, kd=0.5)
            
            self.pos_x_gains = PIDGains(kp=1.0, ki=0.0, kd=0.2)
            self.pos_y_gains = PIDGains(kp=1.0, ki=0.0, kd=0.2)
            self.pos_z_gains = PIDGains(kp=1.5, ki=0.0, kd=0.3)
            
            self.vel_x_gains = PIDGains(kp=1.0, ki=0.0, kd=0.1)
            self.vel_y_gains = PIDGains(kp=1.0, ki=0.0, kd=0.1)
            self.vel_z_gains = PIDGains(kp=1.5, ki=0.0, kd=0.2)
    
    def update(self, state: State) -> Control:
        """Update autopilot and generate control commands.
        
        Args:
            state: Current vehicle state
            
        Returns:
            Control commands
        """
        if self.mode == AutopilotMode.MANUAL:
            return self._manual_control(state)
        elif self.mode == AutopilotMode.STABILIZE:
            return self._stabilize_control(state)
        elif self.mode == AutopilotMode.ALTITUDE_HOLD:
            return self._altitude_hold_control(state)
        elif self.mode == AutopilotMode.POSITION_HOLD:
            return self._position_hold_control(state)
        elif self.mode == AutopilotMode.GUIDED:
            return self._guided_control(state)
        elif self.mode == AutopilotMode.AUTO:
            return self._auto_control(state)
        elif self.mode == AutopilotMode.LAND:
            return self._landing_control(state)
        elif self.mode == AutopilotMode.RTL:
            return self._return_to_launch_control(state)
        else:
            return Control()  # Default: no control
    
    def _manual_control(self, state: State) -> Control:
        """Manual control mode - pass through pilot commands."""
        # In manual mode, return zero control (pilot has direct control)
        return Control()
    
    def _stabilize_control(self, state: State) -> Control:
        """Stabilize mode - attitude stabilization only."""
        # Target attitude (level flight)
        target_attitude = np.array([0.0, 0.0, state.euler_angles[2]])  # Keep current heading
        
        # Attitude control
        angular_rate_cmd = self.attitude_controller.update(
            target_attitude, state.euler_angles, state.time
        )
        
        # Apply limits
        angular_rate_cmd = np.clip(angular_rate_cmd, -self.max_angular_rate, self.max_angular_rate)
        
        # Convert to control surface commands
        return self._angular_rate_to_control(angular_rate_cmd, state)
    
    def _altitude_hold_control(self, state: State) -> Control:
        """Altitude hold mode - maintain current altitude."""
        # Target position (hold current x,y but maintain altitude)
        target_position = np.array([state.position[0], state.position[1], self.position_command[2]])
        
        # Position control
        velocity_cmd = self.position_controller.update(
            target_position, state.position, state.time
        )
        
        # Velocity control
        acceleration_cmd = self.velocity_controller.update(
            velocity_cmd, state.velocity, state.time
        )
        
        return self.track_acceleration(state, acceleration_cmd)
    
    def _position_hold_control(self, state: State) -> Control:
        """Position hold mode - maintain current position."""
        # Use commanded position or current position
        target_position = self.position_command
        
        # Position control
        velocity_cmd = self.position_controller.update(
            target_position, state.position, state.time
        )
        
        # Velocity control  
        acceleration_cmd = self.velocity_controller.update(
            velocity_cmd, state.velocity, state.time
        )
        
        return self.track_acceleration(state, acceleration_cmd)
    
    def _guided_control(self, state: State) -> Control:
        """Guided mode - track external commands."""
        # Use external acceleration command
        return self.track_acceleration(state, self.acceleration_command)
    
    def _auto_control(self, state: State) -> Control:
        """Auto mode - follow pre-programmed mission."""
        # Simplified auto mode - could integrate with mission planner
        return self._position_hold_control(state)
    
    def _landing_control(self, state: State) -> Control:
        """Landing mode - controlled descent."""
        # Target: descend at controlled rate
        descent_rate = -2.0  # m/s
        target_velocity = np.array([0.0, 0.0, descent_rate])
        
        # Velocity control
        acceleration_cmd = self.velocity_controller.update(
            target_velocity, state.velocity, state.time
        )
        
        return self.track_acceleration(state, acceleration_cmd)
    
    def _return_to_launch_control(self, state: State) -> Control:
        """Return to launch mode."""
        # Simplified RTL - go to origin
        home_position = np.array([0.0, 0.0, -100.0])  # 100m altitude
        
        # Position control
        velocity_cmd = self.position_controller.update(
            home_position, state.position, state.time
        )
        
        # Velocity control
        acceleration_cmd = self.velocity_controller.update(
            velocity_cmd, state.velocity, state.time
        )
        
        return self.track_acceleration(state, acceleration_cmd)
    
    def track_acceleration(self, state: State, acceleration_cmd: np.ndarray) -> Control:
        """Convert acceleration command to control surface commands.
        
        Args:
            state: Current vehicle state
            acceleration_cmd: Desired acceleration in inertial frame
            
        Returns:
            Control surface commands
        """
        # Apply acceleration limits
        accel_magnitude = np.linalg.norm(acceleration_cmd)
        if accel_magnitude > self.max_acceleration:
            acceleration_cmd = acceleration_cmd * (self.max_acceleration / accel_magnitude)
        
        # Transform to body frame
        accel_body = np.array(state.attitude.inverse.rotate(acceleration_cmd))
        
        # For aircraft: convert to attitude commands
        if self.vehicle.type == VehicleType.AIRCRAFT:
            return self._acceleration_to_attitude_control(accel_body, state)
        
        # For missiles: direct acceleration control
        elif self.vehicle.type == VehicleType.MISSILE:
            return self._acceleration_to_missile_control(accel_body, state)
        
        else:
            return Control()
    
    def _acceleration_to_attitude_control(self, accel_body: np.ndarray, state: State) -> Control:
        """Convert acceleration command to attitude control for aircraft."""
        # Simplified aircraft control allocation
        
        # Thrust command (longitudinal acceleration)
        thrust_cmd = accel_body[0] / 9.81  # Normalize by gravity
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)
        
        # Attitude commands from lateral/vertical acceleration
        # Bank angle for lateral acceleration
        bank_cmd = np.arctan2(accel_body[1], 9.81)
        bank_cmd = np.clip(bank_cmd, -self.max_attitude_angle, self.max_attitude_angle)
        
        # Pitch angle for vertical acceleration  
        pitch_cmd = np.arctan2(-accel_body[2], 9.81)
        pitch_cmd = np.clip(pitch_cmd, -self.max_attitude_angle, self.max_attitude_angle)
        
        # Keep current heading
        yaw_cmd = state.euler_angles[2]
        
        # Attitude control
        target_attitude = np.array([bank_cmd, pitch_cmd, yaw_cmd])
        angular_rate_cmd = self.attitude_controller.update(
            target_attitude, state.euler_angles, state.time
        )
        
        # Convert to control surfaces
        control = Control()
        control.throttle = thrust_cmd
        control.aileron = angular_rate_cmd[0] * 0.1  # Scale factor
        control.elevator = angular_rate_cmd[1] * 0.1
        control.rudder = angular_rate_cmd[2] * 0.1
        
        return control
    
    def _acceleration_to_missile_control(self, accel_body: np.ndarray, state: State) -> Control:
        """Convert acceleration command to missile control (fins + TVC)."""
        # Missile control allocation
        
        # Thrust vector control
        if accel_body[0] > 0:  # Forward acceleration
            throttle = min(1.0, accel_body[0] / 50.0)  # Scale factor
        else:
            throttle = 0.0
        
        # TVC for pitch/yaw control
        tvc_pitch = np.arctan2(-accel_body[2], max(1.0, accel_body[0]))
        tvc_yaw = np.arctan2(accel_body[1], max(1.0, accel_body[0]))
        
        # Limit TVC angles
        max_tvc = np.deg2rad(15.0)
        tvc_pitch = np.clip(tvc_pitch, -max_tvc, max_tvc)
        tvc_yaw = np.clip(tvc_yaw, -max_tvc, max_tvc)
        
        # Fin commands for fine control
        fin_cmd_pitch = accel_body[2] * 0.01  # Scale factor
        fin_cmd_yaw = accel_body[1] * 0.01
        
        control = Control()
        control.throttle = throttle
        control.tvc_pitch = tvc_pitch
        control.tvc_yaw = tvc_yaw
        control.elevator = fin_cmd_pitch
        control.rudder = fin_cmd_yaw
        
        return control
    
    def _angular_rate_to_control(self, angular_rate_cmd: np.ndarray, state: State) -> Control:
        """Convert angular rate command to control surface deflections."""
        # Simple proportional control allocation
        control = Control()
        
        if self.vehicle.type == VehicleType.AIRCRAFT:
            # Aircraft control surfaces
            control.aileron = angular_rate_cmd[0] * 0.1  # Roll rate to aileron
            control.elevator = angular_rate_cmd[1] * 0.1  # Pitch rate to elevator  
            control.rudder = angular_rate_cmd[2] * 0.1  # Yaw rate to rudder
            
        elif self.vehicle.type == VehicleType.MISSILE:
            # Missile fin control
            control.elevator = angular_rate_cmd[1] * 0.05  # Pitch rate to pitch fin
            control.rudder = angular_rate_cmd[2] * 0.05  # Yaw rate to yaw fin
            control.aileron = angular_rate_cmd[0] * 0.05  # Roll rate to roll fin
        
        return control
    
    def set_mode(self, mode: AutopilotMode):
        """Set autopilot mode."""
        self.mode = mode
        
        # Reset controllers when changing modes
        self.attitude_controller.reset()
        self.position_controller.reset()
        self.velocity_controller.reset()
    
    def set_position_command(self, position: np.ndarray):
        """Set position command for guided modes."""
        self.position_command = position.copy()
    
    def set_velocity_command(self, velocity: np.ndarray):
        """Set velocity command for guided modes."""
        self.velocity_command = velocity.copy()
    
    def set_acceleration_command(self, acceleration: np.ndarray):
        """Set acceleration command for guided mode."""
        self.acceleration_command = acceleration.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get autopilot status information."""
        return {
            'mode': self.mode.name,
            'position_command': self.position_command.tolist(),
            'velocity_command': self.velocity_command.tolist(),
            'acceleration_command': self.acceleration_command.tolist(),
            'attitude_command': self.attitude_command.tolist(),
        } 