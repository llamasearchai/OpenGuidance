"""Quadrotor dynamics model for OpenGuidance."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from numba import jit
from pyquaternion import Quaternion

from openguidance.core.types import State, Control, Vehicle, VehicleType, Environment


class QuadrotorDynamics:
    """6-DOF quadrotor dynamics model."""
    
    def __init__(self, vehicle: Vehicle, use_gpu: bool = False):
        """Initialize quadrotor dynamics model."""
        assert vehicle.type == VehicleType.QUADROTOR
        self.vehicle = vehicle
        self.use_gpu = use_gpu
        
        # Quadrotor specific parameters
        self.arm_length = vehicle.parameters.get('arm_length', 0.25)  # m
        self.rotor_thrust_coeff = vehicle.parameters.get('rotor_thrust_coeff', 1.0e-5)
        self.rotor_torque_coeff = vehicle.parameters.get('rotor_torque_coeff', 1.0e-7)
        self.drag_coeff = vehicle.parameters.get('drag_coeff', 0.01)
        
        # Motor configuration (+ configuration)
        self.motor_positions = np.array([
            [self.arm_length, 0, 0],      # Front
            [0, self.arm_length, 0],      # Right  
            [-self.arm_length, 0, 0],     # Back
            [0, -self.arm_length, 0]      # Left
        ])
        
        # Motor spin directions (CCW: +1, CW: -1)
        self.motor_directions = np.array([1, -1, 1, -1])
        
    def derivatives(self, state: State, control: Control, environment: Environment) -> Dict[str, np.ndarray]:
        """Compute state derivatives for quadrotor dynamics."""
        # Extract state components
        pos = state.position
        vel = state.velocity
        q = state.attitude
        omega = state.angular_velocity
        
        # Get mass properties
        mass = self.vehicle.get_total_mass()
        inertia = self.vehicle.get_inertia_with_fuel()
        
        # Motor commands (4 motors)
        if control.motor_commands is not None and len(control.motor_commands) == 4:
            motor_thrusts = control.motor_commands
        else:
            # Convert generic control to motor commands
            motor_thrusts = self._control_to_motors(control)
        
        # Compute forces and moments from rotors
        rotor_forces, rotor_moments = self._compute_rotor_forces_moments(motor_thrusts)
        
        # Aerodynamic drag
        drag_force = self._compute_drag(vel, environment)
        
        # Total forces in body frame
        forces_body = rotor_forces + drag_force
        
        # Add gravity (transform to body frame)
        gravity_body = q.inverse.rotate(environment.gravity * mass)
        forces_body += gravity_body
        
        # Total moments in body frame
        moments_body = rotor_moments
        
        # Newton's second law
        accel_body = forces_body / mass
        accel_inertial = q.rotate(accel_body)
        
        # Euler's equation for rotational dynamics
        omega_cross_I_omega = np.cross(omega, inertia @ omega)
        omega_dot = np.linalg.solve(inertia, moments_body - omega_cross_I_omega)
        
        # Quaternion derivative
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_array = np.array([q.w, q.x, q.y, q.z])
        q_dot = 0.5 * self._quaternion_multiply(q_array, omega_quat)
        
        return {
            'position_dot': vel,
            'velocity_dot': accel_inertial,
            'quaternion_dot': q_dot,
            'angular_velocity_dot': omega_dot,
            'forces': forces_body,
            'moments': moments_body,
            'motor_thrusts': motor_thrusts
        }
    
    def step(self, state: State, control: Control, environment: Environment, dt: float) -> State:
        """Integrate dynamics one time step using RK4."""
        # RK4 integration
        k1 = self._get_state_derivative(state, control, environment)
        
        state2 = self._add_derivative(state, k1, dt * 0.5)
        k2 = self._get_state_derivative(state2, control, environment)
        
        state3 = self._add_derivative(state, k2, dt * 0.5)
        k3 = self._get_state_derivative(state3, control, environment)
        
        state4 = self._add_derivative(state, k3, dt)
        k4 = self._get_state_derivative(state4, control, environment)
        
        # Combine RK4 stages
        new_state = state.copy()
        new_state.position += dt / 6.0 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        new_state.velocity += dt / 6.0 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        
        # Update quaternion with normalization
        q_array = np.array([state.attitude.w, state.attitude.x, state.attitude.y, state.attitude.z])
        q_dot_avg = 1/6.0 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        q_array += dt * q_dot_avg
        q_array /= np.linalg.norm(q_array)
        new_state.attitude = Quaternion(q_array)
        
        new_state.angular_velocity += dt / 6.0 * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        new_state.time = state.time + dt
        
        # Store additional info
        derivatives = self.derivatives(state, control, environment)
        new_state.acceleration = derivatives['velocity_dot']
        new_state.angular_acceleration = derivatives['angular_velocity_dot']
        
        return new_state
    
    def _control_to_motors(self, control: Control) -> np.ndarray:
        """Convert generic control inputs to motor thrust commands."""
        # Base thrust from throttle
        max_thrust = self.vehicle.max_thrust or 1000.0
        base_thrust = control.thrust * max_thrust / 4.0
        
        # Control mixing matrix for + configuration
        # [thrust, roll, pitch, yaw] -> [m1, m2, m3, m4]
        mixing_matrix = np.array([
            [1,  0, -1,  1],  # Motor 1 (front)
            [1, -1,  0, -1],  # Motor 2 (right)
            [1,  0,  1,  1],  # Motor 3 (back)
            [1,  1,  0, -1]   # Motor 4 (left)
        ])
        
        # Control vector
        control_vec = np.array([
            base_thrust,
            control.aileron * 0.1,   # Roll moment
            control.elevator * 0.1,  # Pitch moment
            control.rudder * 0.05    # Yaw moment
        ])
        
        # Compute motor thrusts
        motor_thrusts = mixing_matrix @ control_vec
        
        # Ensure non-negative thrusts
        motor_thrusts = np.maximum(motor_thrusts, 0.0)
        
        return motor_thrusts
    
    def _compute_rotor_forces_moments(self, motor_thrusts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forces and moments from rotor thrusts."""
        # Total thrust (upward in body frame)
        total_thrust = np.sum(motor_thrusts)
        thrust_force = np.array([0, 0, -total_thrust])  # Negative Z (up)
        
        # Moments from rotor positions and thrusts
        roll_moment = 0.0
        pitch_moment = 0.0
        yaw_moment = 0.0
        
        for i, (pos, thrust, direction) in enumerate(zip(self.motor_positions, motor_thrusts, self.motor_directions)):
            # Moment arm effects
            roll_moment += pos[1] * thrust   # Y position * thrust
            pitch_moment -= pos[0] * thrust  # -X position * thrust
            
            # Yaw moment from rotor torque
            rotor_torque = direction * self.rotor_torque_coeff * thrust
            yaw_moment += rotor_torque
        
        moments = np.array([roll_moment, pitch_moment, yaw_moment])
        
        return thrust_force, moments
    
    def _compute_drag(self, velocity: np.ndarray, environment: Environment) -> np.ndarray:
        """Compute aerodynamic drag forces."""
        # Simple quadratic drag model
        speed = np.linalg.norm(velocity)
        if speed > 0.1:
            drag_direction = -velocity / speed
            ref_area = self.vehicle.reference_area or 0.1
            drag_magnitude = 0.5 * environment.density * speed**2 * ref_area * self.drag_coeff
            return drag_direction * drag_magnitude
        else:
            return np.zeros(3)
    
    def _get_state_derivative(self, state: State, control: Control, environment: Environment) -> Tuple:
        """Get state derivatives as tuple for RK4."""
        derivs = self.derivatives(state, control, environment)
        return (
            derivs['position_dot'],
            derivs['velocity_dot'],
            derivs['quaternion_dot'],
            derivs['angular_velocity_dot']
        )
    
    def _add_derivative(self, state: State, deriv: Tuple, dt: float) -> State:
        """Add derivative scaled by dt to state."""
        new_state = state.copy()
        new_state.position = state.position + dt * deriv[0]
        new_state.velocity = state.velocity + dt * deriv[1]
        
        # Update quaternion
        q_array = np.array([state.attitude.w, state.attitude.x, state.attitude.y, state.attitude.z])
        q_array += dt * deriv[2]
        q_array /= np.linalg.norm(q_array)
        new_state.attitude = Quaternion(q_array)
        
        new_state.angular_velocity = state.angular_velocity + dt * deriv[3]
        
        return new_state
    
    @staticmethod
    @jit(nopython=True)
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def get_flight_envelope(self) -> Dict[str, Tuple[float, float]]:
        """Get quadrotor flight envelope limits."""
        return {
            'max_speed': (0.0, 20.0),  # m/s
            'max_altitude': (0.0, 500.0),  # m
            'max_acceleration': (0.0, 2.0),  # g
            'max_angular_rate': (0.0, np.radians(180)),  # rad/s
        }
    
    def check_flight_envelope(self, state: State, environment: Environment) -> Dict[str, bool]:
        """Check if current state is within flight envelope."""
        envelope = self.get_flight_envelope()
        
        speed = state.speed
        altitude = state.altitude
        accel = np.linalg.norm(state.acceleration) / 9.81 if state.acceleration is not None else 0.0
        angular_rate = np.linalg.norm(state.angular_velocity)
        
        return {
            'speed_ok': bool(envelope['max_speed'][0] <= speed <= envelope['max_speed'][1]),
            'altitude_ok': bool(envelope['max_altitude'][0] <= altitude <= envelope['max_altitude'][1]),
            'acceleration_ok': bool(accel <= envelope['max_acceleration'][1]),
            'angular_rate_ok': bool(angular_rate <= envelope['max_angular_rate'][1]),
        } 