"""Missile dynamics model for OpenGuidance."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from numba import jit
from pyquaternion import Quaternion

from openguidance.core.types import State, Control, Vehicle, VehicleType, Environment
from openguidance.dynamics.aerodynamics import AerodynamicsModel
from openguidance.dynamics.propulsion import PropulsionModel


class MissileDynamics:
    """6-DOF missile dynamics model with thrust vector control."""
    
    def __init__(self, vehicle: Vehicle):
        """Initialize missile dynamics model."""
        assert vehicle.type == VehicleType.MISSILE
        self.vehicle = vehicle
        
        # Initialize subsystem models
        self.aero = AerodynamicsModel(vehicle)
        self.prop = PropulsionModel(vehicle)
        
        # Missile-specific parameters
        self.has_tvc = True  # Thrust vector control
        self.max_tvc_angle = np.deg2rad(15.0)  # Maximum TVC deflection
        self.fuel_burn_rate = 0.0  # kg/s (will be computed)
        
    def derivatives(self, state: State, control: Control, environment: Environment) -> Dict[str, np.ndarray]:
        """Compute state derivatives for missile dynamics."""
        # Extract state components
        pos = state.position
        vel = state.velocity
        q = state.attitude
        omega = state.angular_velocity
        
        # Get mass properties (accounting for fuel burn)
        mass = self.vehicle.get_total_mass()
        inertia = self.vehicle.get_inertia_with_fuel()
        
        # Transform velocity to body frame
        vel_body = q.inverse.rotate(vel)
        
        # Compute airspeed and angles
        wind = environment.get_wind_at_position(pos, state.time)
        vel_air = vel - wind
        airspeed = float(np.linalg.norm(vel_air))
        
        if airspeed > 0.1:
            alpha = np.arctan2(vel_body[2], vel_body[0])
            beta = np.arcsin(np.clip(vel_body[1] / airspeed, -1, 1))
        else:
            alpha = beta = 0.0
        
        # Get atmospheric properties
        altitude = -pos[2] if state.frame == "NED" else pos[2]
        density = environment.get_density_at_altitude(altitude)
        
        # Compute aerodynamic forces and moments
        aero_forces, aero_moments = self.aero.compute_forces_moments(
            airspeed=airspeed,
            alpha=alpha,
            beta=beta,
            angular_velocity=omega,
            control_surfaces={
                'fin1': control.aileron,  # Fin deflections
                'fin2': control.elevator,
                'fin3': control.rudder,
                'fin4': 0.0  # Could add fourth fin
            },
            density=density
        )
        
        # Compute thrust forces and moments with TVC
        thrust_force, thrust_moment = self.prop.compute_thrust(
            throttle=control.throttle,
            airspeed=airspeed,
            altitude=altitude,
            tvc_pitch=control.tvc_pitch,
            tvc_yaw=control.tvc_yaw
        )
        
        # Total forces in body frame
        forces_body = aero_forces + thrust_force
        
        # Add gravity (transform to body frame)
        gravity_body = q.inverse.rotate(environment.gravity * mass)
        forces_body += gravity_body
        
        # Total moments in body frame
        moments_body = aero_moments + thrust_moment
        
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
        
        # Fuel consumption
        self.fuel_burn_rate = self.prop.compute_fuel_flow(control.throttle, altitude)
        
        return {
            'position_dot': vel,
            'velocity_dot': accel_inertial,
            'quaternion_dot': q_dot,
            'angular_velocity_dot': omega_dot,
            'forces': forces_body,
            'moments': moments_body,
            'alpha': alpha,
            'beta': beta,
            'airspeed': airspeed,
            'fuel_burn_rate': self.fuel_burn_rate
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
        
        # Update fuel mass
        if self.vehicle.fuel_mass is not None and self.vehicle.fuel_mass > 0:
            fuel_consumed = self.fuel_burn_rate * dt
            self.vehicle.fuel_mass = max(0.0, self.vehicle.fuel_mass - fuel_consumed)
        
        # Store additional info
        derivatives = self.derivatives(state, control, environment)
        new_state.acceleration = derivatives['velocity_dot']
        new_state.angular_acceleration = derivatives['angular_velocity_dot']
        
        return new_state
    
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
    
    def get_performance_envelope(self) -> Dict[str, Tuple[float, float]]:
        """Get missile performance envelope."""
        return {
            'mach_number': (0.5, 4.0),
            'altitude': (0.0, 25000.0),  # m
            'load_factor': (-20.0, 20.0),  # g
            'angle_of_attack': (np.deg2rad(-30), np.deg2rad(30)),  # rad
            'tvc_angle': (-self.max_tvc_angle, self.max_tvc_angle),  # rad
        }
    
    def compute_guidance_acceleration_limits(self, state: State, environment: Environment) -> Dict[str, float]:
        """Compute available acceleration for guidance commands."""
        # Get current flight conditions
        airspeed = state.speed
        altitude = state.altitude
        density = environment.get_density_at_altitude(altitude)
        
        # Dynamic pressure
        q_bar = 0.5 * density * airspeed**2
        
        # Maximum aerodynamic force (simplified)
        reference_area = self.vehicle.reference_area if self.vehicle.reference_area is not None else 1.0
        max_aero_force = q_bar * reference_area * 2.0  # Rough estimate
        
        # Maximum thrust force
        max_thrust = self.vehicle.max_thrust or 1000.0
        
        # Total available force
        total_force = max_aero_force + max_thrust
        
        # Convert to acceleration
        mass = self.vehicle.get_total_mass()
        max_acceleration = total_force / mass
        
        return {
            'max_lateral_accel': max_acceleration,
            'max_normal_accel': max_acceleration,
            'max_axial_accel': max_thrust / mass,
        }
    
    def is_fuel_depleted(self) -> bool:
        """Check if missile fuel is depleted."""
        if self.vehicle.fuel_mass is None:
            return False
        return self.vehicle.fuel_mass <= 0.0
    
    def get_remaining_flight_time(self) -> float:
        """Estimate remaining flight time based on fuel."""
        if self.vehicle.fuel_mass is None or self.fuel_burn_rate <= 0:
            return float('inf')
        
        return self.vehicle.fuel_mass / self.fuel_burn_rate
    
    def compute_impact_point(self, state: State, environment: Environment) -> np.ndarray:
        """Compute ballistic impact point (no thrust)."""
        # Simple ballistic trajectory calculation
        pos = state.position.copy()
        vel = state.velocity.copy()
        
        # Time to impact (assuming flat earth)
        if vel[2] >= 0:  # Going up
            return pos  # Can't compute impact
        
        t_impact = -pos[2] / vel[2]  # Time to reach ground
        
        # Impact position
        impact_pos = pos + vel * t_impact
        
        # Add gravity effect (simplified)
        gravity_effect = 0.5 * environment.gravity * t_impact**2
        impact_pos += gravity_effect
        
        return impact_pos
    
    def get_seeker_field_of_view(self) -> float:
        """Get seeker field of view in radians."""
        return self.vehicle.parameters.get('seeker_fov', np.deg2rad(30.0))
    
    def is_target_in_seeker_fov(self, state: State, target_position: np.ndarray) -> bool:
        """Check if target is within seeker field of view."""
        # Vector from missile to target in body frame
        los_inertial = target_position - state.position
        los_body = np.array(state.attitude.inverse.rotate(los_inertial))
        
        # Angle from missile nose (x-axis)
        los_body_magnitude = np.linalg.norm(los_body)
        los_body_norm = los_body / los_body_magnitude
        angle_off_boresight = np.arccos(np.clip(los_body_norm[0], -1, 1))
        
        fov = self.get_seeker_field_of_view()
        return angle_off_boresight <= fov / 2.0 