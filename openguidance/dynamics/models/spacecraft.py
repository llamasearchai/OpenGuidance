"""Spacecraft dynamics model for OpenGuidance."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from numba import jit
from pyquaternion import Quaternion

from openguidance.core.types import State, Control, Vehicle, VehicleType, Environment


class SpacecraftDynamics:
    """6-DOF spacecraft dynamics model."""
    
    def __init__(self, vehicle: Vehicle, use_gpu: bool = False):
        """Initialize spacecraft dynamics model."""
        assert vehicle.type == VehicleType.SPACECRAFT
        self.vehicle = vehicle
        self.use_gpu = use_gpu
        
        # Spacecraft specific parameters
        self.num_thrusters = vehicle.parameters.get('num_thrusters', 8)
        self.thruster_positions = vehicle.parameters.get('thruster_positions', self._default_thruster_layout())
        self.thruster_directions = vehicle.parameters.get('thruster_directions', self._default_thruster_directions())
        
        # RCS parameters
        self.rcs_thrust = vehicle.parameters.get('rcs_thrust', 100.0)  # N per thruster
        
        # Disturbance parameters
        self.solar_pressure_coeff = vehicle.parameters.get('solar_pressure_coeff', 1.3)
        self.atmospheric_drag_coeff = vehicle.parameters.get('atmospheric_drag_coeff', 2.2)
        
    def derivatives(self, state: State, control: Control, environment: Environment) -> Dict[str, np.ndarray]:
        """Compute state derivatives for spacecraft dynamics."""
        # Extract state components
        pos = state.position
        vel = state.velocity
        q = state.attitude
        omega = state.angular_velocity
        
        # Get mass properties
        mass = self.vehicle.get_total_mass()
        inertia = self.vehicle.get_inertia_with_fuel()
        
        # Compute thruster forces and moments
        thruster_forces, thruster_moments = self._compute_thruster_forces_moments(control)
        
        # Environmental disturbances
        disturbance_forces, disturbance_moments = self._compute_disturbances(state, environment)
        
        # Total forces and moments in body frame
        forces_body = thruster_forces + disturbance_forces
        moments_body = thruster_moments + disturbance_moments
        
        # Add gravitational force (if in orbit)
        if np.linalg.norm(pos) > 6.4e6:  # Above Earth surface
            gravity_inertial = self._compute_orbital_gravity(pos)
            gravity_body = q.inverse.rotate(gravity_inertial * mass)
            forces_body += gravity_body
        else:
            # Use standard gravity for low altitude
            gravity_body = q.inverse.rotate(environment.gravity * mass)
            forces_body += gravity_body
        
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
    
    def _compute_thruster_forces_moments(self, control: Control) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forces and moments from RCS thrusters."""
        # Convert control inputs to thruster commands
        thruster_commands = self._control_to_thrusters(control)
        
        # Compute total force and moment
        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        
        for i, (cmd, pos, direction) in enumerate(zip(thruster_commands, 
                                                     self.thruster_positions, 
                                                     self.thruster_directions)):
            thrust_magnitude = cmd * self.rcs_thrust
            thrust_vector = thrust_magnitude * direction
            
            total_force += thrust_vector
            total_moment += np.cross(pos, thrust_vector)
        
        return total_force, total_moment
    
    def _control_to_thrusters(self, control: Control) -> np.ndarray:
        """Convert control inputs to individual thruster commands."""
        # Simple allocation matrix (would be more complex in reality)
        # Assumes 8 thrusters in standard RCS configuration
        
        # Control vector: [Fx, Fy, Fz, Mx, My, Mz]
        control_vec = np.array([
            control.thrust * 1000.0,        # Forward thrust
            control.aileron * 100.0,        # Side force
            control.elevator * 100.0,       # Vertical force
            control.aileron * 10.0,         # Roll moment
            control.elevator * 10.0,        # Pitch moment
            control.rudder * 10.0           # Yaw moment
        ])
        
        # Simplified allocation (8 thrusters)
        # In reality, would solve: B * u = F_desired
        allocation_matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],    # Fx
            [0, 1, 0, -1, 0, 0, 0, 0],   # Fy
            [0, 0, 1, 0, 0, -1, 0, 0],   # Fz
            [0, 0, 0, 0, 1, 0, -1, 0],   # Mx
            [0, 0, 0, 0, 0, 0, 0, 1],    # My
            [0, 1, 0, 1, 0, 0, 0, 0]     # Mz
        ])
        
        # Pseudo-inverse allocation
        try:
            thruster_commands = np.linalg.pinv(allocation_matrix) @ control_vec
        except np.linalg.LinAlgError:
            thruster_commands = np.zeros(8)
        
        # Ensure non-negative commands and apply limits
        thruster_commands = np.clip(thruster_commands, 0.0, 1.0)
        
        return thruster_commands
    
    def _compute_disturbances(self, state: State, environment: Environment) -> Tuple[np.ndarray, np.ndarray]:
        """Compute environmental disturbance forces and moments."""
        pos = state.position
        vel = state.velocity
        
        # Initialize disturbances
        dist_force = np.zeros(3)
        dist_moment = np.zeros(3)
        
        # Atmospheric drag (if in atmosphere)
        altitude = np.linalg.norm(pos) - 6.371e6  # Earth radius
        if altitude < 500e3:  # Below 500 km
            # Simplified atmospheric density model
            density = environment.density * np.exp(-altitude / 8500.0)
            
            if density > 1e-12:  # Significant atmosphere
                vel_magnitude = np.linalg.norm(vel)
                if vel_magnitude > 0:
                    drag_direction = -vel / vel_magnitude
                    ref_area = self.vehicle.reference_area or 1.0
                    drag_magnitude = 0.5 * density * vel_magnitude**2 * ref_area * self.atmospheric_drag_coeff
                    dist_force += drag_direction * drag_magnitude
        
        # Solar radiation pressure (simplified)
        if hasattr(environment, 'solar_flux'):
            solar_flux = getattr(environment, 'solar_flux', 1361.0)  # W/m^2
            c = 299792458.0  # Speed of light
            ref_area = self.vehicle.reference_area or 1.0
            
            # Assume sun direction (simplified)
            sun_direction = np.array([1, 0, 0])  # X direction
            solar_pressure = solar_flux / c * self.solar_pressure_coeff * ref_area
            dist_force += sun_direction * solar_pressure
        
        # Gravity gradient torque
        if np.linalg.norm(pos) > 6.4e6:
            gravity_gradient_moment = self._compute_gravity_gradient_torque(state)
            dist_moment += gravity_gradient_moment
        
        return dist_force, dist_moment
    
    def _compute_orbital_gravity(self, position: np.ndarray) -> np.ndarray:
        """Compute gravitational acceleration for orbital mechanics."""
        mu_earth = 3.986004418e14  # m^3/s^2
        r = np.linalg.norm(position)
        
        if r > 0:
            return -mu_earth / r**3 * position
        else:
            return np.zeros(3)
    
    def _compute_gravity_gradient_torque(self, state: State) -> np.ndarray:
        """Compute gravity gradient torque."""
        pos = state.position
        r = np.linalg.norm(pos)
        
        if r < 6.4e6:  # Too close to Earth
            return np.zeros(3)
        
        # Unit vector to Earth center
        r_hat = pos / r
        
        # Transform to body frame
        r_body = np.array(state.attitude.inverse.rotate(r_hat))
        
        # Gravity gradient coefficient
        mu_earth = 3.986004418e14
        coeff = 3.0 * mu_earth / r**3
        
        # Inertia matrix
        I = self.vehicle.inertia
        
        # Gravity gradient torque
        torque = coeff * np.cross(r_body, np.array(I @ r_body))
        
        return torque
    
    def _default_thruster_layout(self) -> np.ndarray:
        """Default RCS thruster positions."""
        return np.array([
            [1.0, 0.0, 0.0],   # Forward
            [0.0, 1.0, 0.0],   # Right
            [0.0, 0.0, 1.0],   # Up
            [0.0, -1.0, 0.0],  # Left
            [0.0, 0.0, -1.0],  # Down
            [-1.0, 1.0, 0.0],  # Aft-right
            [-1.0, -1.0, 0.0], # Aft-left
            [-1.0, 0.0, 1.0]   # Aft-up
        ])
    
    def _default_thruster_directions(self) -> np.ndarray:
        """Default RCS thruster directions."""
        return np.array([
            [1.0, 0.0, 0.0],   # Forward
            [0.0, -1.0, 0.0],  # Left
            [0.0, 0.0, -1.0],  # Down
            [0.0, 1.0, 0.0],   # Right
            [0.0, 0.0, 1.0],   # Up
            [0.0, 1.0, 0.0],   # Right
            [0.0, -1.0, 0.0],  # Left
            [0.0, 0.0, -1.0]   # Down
        ])
    
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
    
    def get_orbital_elements(self, state: State) -> Dict[str, float]:
        """Compute classical orbital elements from state."""
        pos = state.position
        vel = state.velocity
        
        mu = 3.986004418e14  # Earth gravitational parameter
        
        # Position and velocity magnitudes
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        
        # Specific energy
        energy = v**2 / 2 - mu / r
        
        # Semi-major axis
        a = -mu / (2 * energy) if energy < 0 else float('inf')
        
        # Angular momentum vector
        h_vec = np.cross(pos, vel)
        h = np.linalg.norm(h_vec)
        
        # Eccentricity vector
        e_vec = np.cross(vel, h_vec) / mu - pos / r
        e = np.linalg.norm(e_vec)
        
        # Inclination
        i = np.arccos(h_vec[2] / h) if h > 0 else 0.0
        
        return {
            'semi_major_axis': float(a),
            'eccentricity': float(e),
            'inclination': float(np.degrees(i)),
            'period': float(2 * np.pi * np.sqrt(a**3 / mu)) if a > 0 else 0.0,
            'apogee': float(a * (1 + e) - 6.371e6) if a > 0 else 0.0,
            'perigee': float(a * (1 - e) - 6.371e6) if a > 0 else 0.0,
        } 