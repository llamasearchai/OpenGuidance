"""Propulsion models for OpenGuidance."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from numba import jit

from openguidance.core.types import Vehicle, VehicleType


class PropulsionModel:
    """Generic propulsion model for vehicles."""
    
    def __init__(self, vehicle: Vehicle):
        """Initialize propulsion model."""
        self.vehicle = vehicle
        self.max_thrust = vehicle.max_thrust or 1000.0  # N
        self.specific_impulse = vehicle.specific_impulse or 250.0  # s
        
        # Engine type from vehicle parameters
        self.engine_type = vehicle.parameters.get('engine_type', 'turbofan')
        
        # Thrust vector control capability
        self.has_tvc = vehicle.parameters.get('has_tvc', False)
        self.max_tvc_angle = vehicle.parameters.get('max_tvc_angle', np.radians(15))
        
        # Engine-specific parameters
        self.engine_params = vehicle.parameters.get('engine_params', {})
    
    def compute_thrust(
        self, 
        throttle: float, 
        airspeed: float, 
        altitude: float,
        tvc_pitch: float = 0.0,
        tvc_yaw: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute thrust force and moment."""
        # Atmospheric effects on thrust
        thrust_factor = self._compute_thrust_factor(airspeed, altitude)
        
        # Base thrust magnitude
        thrust_magnitude = throttle * self.max_thrust * thrust_factor
        
        if self.has_tvc:
            # Thrust vector control
            tvc_pitch = np.clip(tvc_pitch, -self.max_tvc_angle, self.max_tvc_angle)
            tvc_yaw = np.clip(tvc_yaw, -self.max_tvc_angle, self.max_tvc_angle)
            
            # Thrust components in body frame
            thrust_x = thrust_magnitude * np.cos(tvc_pitch) * np.cos(tvc_yaw)
            thrust_y = thrust_magnitude * np.sin(tvc_yaw)
            thrust_z = thrust_magnitude * np.sin(tvc_pitch) * np.cos(tvc_yaw)
            
            thrust_force = np.array([thrust_x, thrust_y, thrust_z])
            
            # Moments due to TVC (assuming engine at CG for simplicity)
            engine_offset = self.engine_params.get('offset', np.zeros(3))
            thrust_moment = np.cross(engine_offset, thrust_force)
            
        else:
            # Fixed thrust direction (along x-axis)
            thrust_force = np.array([thrust_magnitude, 0.0, 0.0])
            thrust_moment = np.zeros(3)
            
            # Add engine gyroscopic effects for turbofan
            if self.engine_type == 'turbofan':
                engine_rpm = throttle * 10000.0
                gyro_moment = self._compute_gyroscopic_moment(engine_rpm, airspeed)
                thrust_moment += gyro_moment
        
        return thrust_force, thrust_moment
    
    def _compute_thrust_factor(self, airspeed: float, altitude: float) -> float:
        """Compute thrust factor based on flight conditions."""
        if self.engine_type == 'turbofan':
            rho_ratio = np.exp(-altitude / 8500.0)
            mach_number = airspeed / 340.0
            
            altitude_factor = np.sqrt(rho_ratio)
            
            if mach_number < 0.8:
                mach_factor = 1.0
            else:
                mach_factor = 1.0 - 0.2 * (mach_number - 0.8)
            
            return altitude_factor * mach_factor
            
        elif self.engine_type == 'turbojet':
            rho_ratio = np.exp(-altitude / 8500.0)
            return rho_ratio
            
        elif self.engine_type == 'rocket':
            pressure_ratio = np.exp(-altitude / 8500.0)
            return 1.0 + 0.2 * (1.0 - pressure_ratio)
            
        else:
            return 1.0
    
    def _compute_gyroscopic_moment(self, rpm: float, airspeed: float) -> np.ndarray:
        """Compute gyroscopic moment from rotating engine components."""
        omega_engine = rpm * 2 * np.pi / 60.0
        engine_inertia = self.engine_params.get('inertia', 1.0)
        H_engine = engine_inertia * omega_engine
        
        pitch_rate = 0.0
        gyro_moment = np.array([0.0, 0.0, H_engine * pitch_rate])
        
        return gyro_moment
    
    def compute_fuel_flow(self, throttle: float, altitude: float) -> float:
        """Compute fuel flow rate (kg/s)."""
        base_flow = throttle * self.max_thrust / (self.specific_impulse * 9.81)
        
        if self.engine_type == 'turbofan':
            altitude_factor = np.exp(-altitude / 10000.0)
            return base_flow * altitude_factor
        elif self.engine_type == 'rocket':
            return base_flow
        else:
            return base_flow
    
    def get_engine_limits(self) -> Dict[str, Tuple[float, float]]:
        """Get engine operating limits."""
        limits = {
            'throttle': (0.0, 1.0),
            'thrust': (0.0, self.max_thrust),
        }
        
        if self.has_tvc:
            limits.update({
                'tvc_pitch': (-self.max_tvc_angle, self.max_tvc_angle),
                'tvc_yaw': (-self.max_tvc_angle, self.max_tvc_angle),
            })
        
        return limits
    
    @staticmethod
    @jit(nopython=True)
    def compute_exhaust_velocity(specific_impulse: float) -> float:
        """Compute exhaust velocity from specific impulse."""
        g0 = 9.81
        return specific_impulse * g0


class MultiEngineModel:
    """Multi-engine propulsion model."""
    
    def __init__(self, vehicle: Vehicle, engine_positions: np.ndarray):
        """Initialize multi-engine model."""
        self.vehicle = vehicle
        self.engine_positions = engine_positions
        self.num_engines = len(engine_positions)
        
        # Individual engine models
        self.engines = []
        for i in range(self.num_engines):
            engine_vehicle = Vehicle(
                name=f"Engine_{i}",
                type=vehicle.type,
                mass=vehicle.mass / self.num_engines,
                inertia=vehicle.inertia,
                max_thrust=(vehicle.max_thrust or 1000.0) / self.num_engines,
                specific_impulse=vehicle.specific_impulse,
                parameters=vehicle.parameters
            )
            self.engines.append(PropulsionModel(engine_vehicle))
    
    def compute_total_thrust(
        self, 
        throttles: np.ndarray,
        airspeed: float,
        altitude: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute total thrust force and moment from all engines."""
        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        
        for i, (engine, throttle) in enumerate(zip(self.engines, throttles)):
            force, moment = engine.compute_thrust(throttle, airspeed, altitude)
            total_force += force
            
            engine_moment = np.cross(self.engine_positions[i], force)
            total_moment += moment + engine_moment
        
        return total_force, total_moment
    
    def compute_differential_thrust_moment(
        self, 
        throttles: np.ndarray,
        airspeed: float,
        altitude: float
    ) -> np.ndarray:
        """Compute moment due to differential thrust."""
        total_moment = np.zeros(3)
        
        for i, (engine, throttle) in enumerate(zip(self.engines, throttles)):
            force, _ = engine.compute_thrust(throttle, airspeed, altitude)
            engine_moment = np.cross(self.engine_positions[i], force)
            total_moment += engine_moment
        
        return total_moment 