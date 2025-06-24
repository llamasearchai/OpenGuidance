"""Aircraft dynamics model for OpenGuidance."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from numba import jit
from pyquaternion import Quaternion
from dataclasses import dataclass

from openguidance.core.types import State, Control, Vehicle, VehicleType, Environment
from openguidance.dynamics.aerodynamics import AerodynamicsModel
from openguidance.dynamics.propulsion import PropulsionModel


@dataclass
class AircraftConfig:
    """Configuration for aircraft models."""
    # Physical properties
    mass: float = 9200.0  # kg
    reference_area: float = 27.87  # m²
    reference_length: float = 4.96  # m
    wingspan: float = 9.96  # m
    max_thrust: float = 129000.0  # N
    
    # Aerodynamic coefficients
    cd0: float = 0.02  # Zero-lift drag coefficient
    cl_alpha: float = 5.5  # Lift curve slope (1/rad)
    cm_alpha: float = -0.8  # Pitching moment curve slope (1/rad)
    
    # Control surface effectiveness
    aileron_effectiveness: float = 0.1  # rad/rad
    elevator_effectiveness: float = 0.2  # rad/rad
    rudder_effectiveness: float = 0.15  # rad/rad
    
    # Limits
    max_alpha: float = 0.35  # rad (20 degrees)
    max_beta: float = 0.17  # rad (10 degrees)
    max_roll_rate: float = 6.28  # rad/s (360 deg/s)
    max_pitch_rate: float = 3.14  # rad/s (180 deg/s)
    max_yaw_rate: float = 1.57  # rad/s (90 deg/s)
    
    # Engine properties
    thrust_response_time: float = 1.0  # s
    fuel_flow_rate: float = 0.5  # kg/s at max thrust
    
    # Flight envelope
    max_altitude: float = 15000.0  # m
    max_speed: float = 600.0  # m/s
    stall_speed: float = 70.0  # m/s
    
    # Inertia tensor (kg⋅m²)
    inertia_xx: float = 12875.0
    inertia_yy: float = 75674.0
    inertia_zz: float = 85552.0
    inertia_xz: float = 1331.0


class Aircraft:
    """High-level aircraft model with configuration."""
    
    def __init__(self, config: AircraftConfig):
        """Initialize aircraft with configuration."""
        self.config = config
        
        # Create vehicle object
        self.vehicle = Vehicle(
            name="Configured Aircraft",
            type=VehicleType.AIRCRAFT,
            mass=config.mass,
            inertia=np.array([
                [config.inertia_xx, 0, config.inertia_xz],
                [0, config.inertia_yy, 0],
                [config.inertia_xz, 0, config.inertia_zz]
            ]),
            reference_area=config.reference_area,
            reference_length=config.reference_length,
            wingspan=config.wingspan,
            max_thrust=config.max_thrust,
            control_limits={
                'aileron': (-25.0, 25.0),  # degrees
                'elevator': (-25.0, 25.0),  # degrees
                'rudder': (-30.0, 30.0),   # degrees
                'throttle': (0.0, 1.0)     # normalized
            }
        )
        
        # Initialize dynamics model
        self.dynamics = AircraftDynamics(self.vehicle)
        
    def get_vehicle(self) -> Vehicle:
        """Get the vehicle configuration."""
        return self.vehicle
    
    def get_dynamics(self) -> 'AircraftDynamics':
        """Get the dynamics model."""
        return self.dynamics
    
    def validate_configuration(self) -> bool:
        """Validate aircraft configuration."""
        checks = [
            self.config.mass > 0,
            self.config.reference_area > 0,
            self.config.reference_length > 0,
            self.config.wingspan > 0,
            self.config.max_thrust > 0,
            self.config.max_alpha > 0,
            self.config.max_beta > 0,
            self.config.stall_speed > 0,
            self.config.max_speed > self.config.stall_speed,
            self.config.max_altitude > 0
        ]
        
        return bool(all(checks))


class AircraftDynamics:
    """6-DOF aircraft dynamics model."""
    
    def __init__(self, vehicle: Vehicle, use_gpu: bool = False):
        """Initialize aircraft dynamics model."""
        assert vehicle.type == VehicleType.AIRCRAFT
        self.vehicle = vehicle
        self.use_gpu = use_gpu
        
        # Initialize subsystem models
        self.aero = AerodynamicsModel(vehicle)
        self.prop = PropulsionModel(vehicle)
        
        # State indices for easier access
        self.idx_pos = slice(0, 3)
        self.idx_vel = slice(3, 6)
        self.idx_quat = slice(6, 10)
        self.idx_omega = slice(10, 13)
        
    def derivatives(self, state: State, control: Control, environment: Environment) -> Dict[str, np.ndarray]:
        """Compute state derivatives for aircraft dynamics."""
        # Extract state components
        pos = state.position
        vel = state.velocity
        q = state.attitude
        omega = state.angular_velocity
        
        # Get mass properties
        mass = self.vehicle.get_total_mass()
        inertia = self.vehicle.get_inertia_with_fuel()
        
        # Transform velocity to body frame
        vel_body = q.inverse.rotate(vel)
        
        # Compute airspeed and angles
        wind = environment.get_wind_at_position(pos, state.time)
        vel_air = vel - wind
        airspeed = np.linalg.norm(vel_air)
        
        if airspeed > 0.1:  # Avoid division by zero
            alpha = np.arctan2(vel_body[2], vel_body[0])  # Angle of attack
            beta = np.arcsin(np.clip(vel_body[1] / airspeed, -1, 1))  # Sideslip angle
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
                'aileron': control.aileron,
                'elevator': control.elevator,
                'rudder': control.rudder,
                'flaps': control.flaps
            },
            density=density
        )
        
        # Compute propulsion forces and moments
        thrust_force, thrust_moment = self.prop.compute_thrust(
            throttle=control.throttle,
            airspeed=airspeed,
            altitude=altitude
        )
        
        # Total forces in body frame
        forces_body = aero_forces + thrust_force
        
        # Add gravity (transform to body frame)
        gravity_body = q.inverse.rotate(environment.gravity * mass)
        forces_body += gravity_body
        
        # Total moments in body frame
        moments_body = aero_moments + thrust_moment
        
        # Newton's second law (F = ma)
        accel_body = forces_body / mass
        
        # Transform acceleration to inertial frame
        accel_inertial = q.rotate(accel_body)
        
        # Euler's equation for rotational dynamics
        # I * omega_dot + omega x (I * omega) = M
        omega_cross_I_omega = np.cross(omega, inertia @ omega)
        omega_dot = np.linalg.solve(inertia, moments_body - omega_cross_I_omega)
        
        # Quaternion derivative
        # q_dot = 0.5 * q * [0, omega]
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
            'alpha': alpha,
            'beta': beta,
            'airspeed': airspeed
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
        q_array /= np.linalg.norm(q_array)  # Normalize
        new_state.attitude = Quaternion(q_array)
        
        new_state.angular_velocity += dt / 6.0 * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        new_state.time = state.time + dt
        
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
    
    def linearize(self, state: State, control: Control, environment: Environment) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize dynamics around operating point for control design."""
        # Numerical linearization using finite differences
        eps = 1e-6
        
        # State vector: [pos(3), vel(3), euler(3), omega(3)]
        x0 = np.concatenate([
            state.position,
            state.velocity,
            state.euler_angles,
            state.angular_velocity
        ])
        
        # Control vector: [thrust, aileron, elevator, rudder]
        u0 = np.array([control.thrust, control.aileron, control.elevator, control.rudder])
        
        n_states = 12
        n_controls = 4
        
        A = np.zeros((n_states, n_controls))
        B = np.zeros((n_states, n_controls))
        
        # Compute A matrix (state Jacobian)
        for i in range(n_states):
            # Perturb state
            x_plus = x0.copy()
            x_plus[i] += eps
            
            x_minus = x0.copy()
            x_minus[i] -= eps
            
            # Create perturbed states
            state_plus = self._vector_to_state(x_plus, state.time)
            state_minus = self._vector_to_state(x_minus, state.time)
            
            # Compute derivatives
            f_plus = self._state_to_derivative_vector(
                self.derivatives(state_plus, control, environment)
            )
            f_minus = self._state_to_derivative_vector(
                self.derivatives(state_minus, control, environment)
            )
            
            # Central difference
            A[:, i] = (f_plus - f_minus) / (2 * eps)
        
        # Compute B matrix (control Jacobian)
        for i in range(n_controls):
            # Perturb control
            u_plus = u0.copy()
            u_plus[i] += eps
            
            u_minus = u0.copy()
            u_minus[i] -= eps
            
            # Create perturbed controls
            control_plus = Control.from_array(u_plus)
            control_minus = Control.from_array(u_minus)
            
            # Compute derivatives
            f_plus = self._state_to_derivative_vector(
                self.derivatives(state, control_plus, environment)
            )
            f_minus = self._state_to_derivative_vector(
                self.derivatives(state, control_minus, environment)
            )
            
            # Central difference
            B[:, i] = (f_plus - f_minus) / (2 * eps)
        
        return A, B
    
    def _vector_to_state(self, x: np.ndarray, time: float) -> State:
        """Convert state vector to State object."""
        position = x[0:3]
        velocity = x[3:6]
        euler = x[6:9]
        angular_velocity = x[9:12]
        
        # Convert Euler angles to quaternion
        roll, pitch, yaw = euler
        attitude = Quaternion(axis=[0, 0, 1], angle=yaw) * \
                  Quaternion(axis=[0, 1, 0], angle=pitch) * \
                  Quaternion(axis=[1, 0, 0], angle=roll)
        
        return State(
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_velocity=angular_velocity,
            time=time
        )
    
    def _state_to_derivative_vector(self, derivatives: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert derivatives dict to vector."""
        # Convert quaternion derivative to Euler rate
        q_dot = derivatives['quaternion_dot']
        omega = derivatives['angular_velocity_dot']
        
        # Simplified conversion (would need proper Euler rate transformation)
        euler_dot = omega  # Approximation
        
        return np.concatenate([
            derivatives['position_dot'],
            derivatives['velocity_dot'],
            euler_dot,
            derivatives['angular_velocity_dot']
        ])
    
    def get_flight_envelope(self) -> Dict[str, Tuple[float, float]]:
        """Get aircraft flight envelope limits."""
        return {
            'airspeed': (20.0, 200.0),  # m/s
            'altitude': (0.0, 15000.0),  # m
            'load_factor': (-2.0, 4.0),  # g
            'angle_of_attack': (np.deg2rad(-10), np.deg2rad(20)),  # rad
            'sideslip': (np.deg2rad(-15), np.deg2rad(15)),  # rad
        }
    
    def check_flight_envelope(self, state: State, environment: Environment) -> Dict[str, bool]:
        """Check if current state is within flight envelope."""
        envelope = self.get_flight_envelope()
        
        # Compute current flight parameters
        airspeed = state.speed
        altitude = state.altitude
        
        # Compute load factor (simplified)
        accel_magnitude = np.linalg.norm(state.acceleration) if state.acceleration is not None else 0.0
        load_factor = accel_magnitude / 9.81
        
        # Get angle of attack and sideslip
        vel_body = state.attitude.inverse.rotate(state.velocity)
        if state.speed > 0.1:
            alpha = np.arctan2(vel_body[2], vel_body[0])
            beta = np.arcsin(np.clip(vel_body[1] / state.speed, -1, 1))
        else:
            alpha = beta = 0.0
        
        return {
            'airspeed_ok': envelope['airspeed'][0] <= airspeed <= envelope['airspeed'][1],
            'altitude_ok': envelope['altitude'][0] <= altitude <= envelope['altitude'][1],
            'load_factor_ok': envelope['load_factor'][0] <= load_factor <= envelope['load_factor'][1],
            'alpha_ok': envelope['angle_of_attack'][0] <= alpha <= envelope['angle_of_attack'][1],
            'beta_ok': envelope['sideslip'][0] <= beta <= envelope['sideslip'][1],
        } 