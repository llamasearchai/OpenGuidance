"""Environmental models for OpenGuidance."""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from numba import jit

from openguidance.core.types import Environment


class EnvironmentModel:
    """Advanced environmental model with atmospheric and wind effects."""
    
    def __init__(self, base_environment: Optional[Environment] = None):
        """Initialize environment model."""
        self.base_env = base_environment or Environment()
        
        # Standard atmosphere parameters
        self.sea_level_density = 1.225  # kg/m^3
        self.sea_level_pressure = 101325.0  # Pa
        self.sea_level_temperature = 288.15  # K
        self.temperature_lapse_rate = 0.0065  # K/m
        self.gas_constant = 287.0  # J/(kg*K)
        self.gamma = 1.4  # Specific heat ratio
        
        # Wind model parameters
        self.wind_layers = []  # List of wind layers
        self.turbulence_intensity = 0.1
        self.gust_model = "dryden"  # dryden, von_karman, discrete
        
    def get_atmospheric_properties(self, altitude: float) -> Dict[str, float]:
        """Get atmospheric properties at given altitude using standard atmosphere."""
        # Handle negative altitudes (below sea level)
        h = max(0.0, altitude)
        
        if h <= 11000.0:  # Troposphere
            # Linear temperature variation
            temperature = self.sea_level_temperature - self.temperature_lapse_rate * h
            
            # Pressure using hydrostatic equation
            pressure = self.sea_level_pressure * (temperature / self.sea_level_temperature) ** (9.81 / (self.gas_constant * self.temperature_lapse_rate))
            
            # Density from ideal gas law
            density = pressure / (self.gas_constant * temperature)
            
        elif h <= 20000.0:  # Lower stratosphere (isothermal)
            # Constant temperature
            temperature = 216.65  # K
            
            # Exponential pressure variation
            pressure = 22632.0 * np.exp(-9.81 * (h - 11000.0) / (self.gas_constant * temperature))
            
            # Density
            density = pressure / (self.gas_constant * temperature)
            
        else:  # Higher altitudes (simplified)
            # Exponential decay
            temperature = 216.65
            pressure = 22632.0 * np.exp(-9.81 * (h - 11000.0) / (self.gas_constant * temperature))
            density = pressure / (self.gas_constant * temperature)
        
        # Speed of sound
        speed_of_sound = np.sqrt(self.gamma * self.gas_constant * temperature)
        
        # Dynamic viscosity (Sutherland's law)
        mu_0 = 1.716e-5  # Pa*s at T_0 = 273.15 K
        T_0 = 273.15  # K
        S = 110.4  # K (Sutherland constant)
        dynamic_viscosity = mu_0 * (temperature / T_0)**(3/2) * (T_0 + S) / (temperature + S)
        
        return {
            'density': density,
            'pressure': pressure,
            'temperature': temperature,
            'speed_of_sound': speed_of_sound,
            'dynamic_viscosity': dynamic_viscosity,
            'kinematic_viscosity': dynamic_viscosity / density,
        }
    
    def get_wind_velocity(self, position: np.ndarray, time: float) -> np.ndarray:
        """Get wind velocity at position and time."""
        base_wind = self.base_env.wind_velocity.copy()
        
        # Add wind layers
        for layer in self.wind_layers:
            layer_wind = self._compute_wind_layer(position, time, layer)
            base_wind += layer_wind
        
        # Add turbulence
        if self.turbulence_intensity > 0:
            turbulence = self._compute_turbulence(position, time)
            base_wind += turbulence
        
        return base_wind
    
    def _compute_wind_layer(self, position: np.ndarray, time: float, layer: Dict[str, Any]) -> np.ndarray:
        """Compute wind contribution from a wind layer."""
        altitude = -position[2] if position[2] < 0 else position[2]  # Handle NED frame
        
        # Check if position is within layer bounds
        if altitude < layer.get('min_altitude', 0) or altitude > layer.get('max_altitude', 50000):
            return np.zeros(3)
        
        # Base wind for this layer
        layer_wind = np.array(layer.get('velocity', [0, 0, 0]))
        
        # Wind shear (linear variation with altitude)
        if 'shear' in layer:
            shear_rate = np.array(layer['shear'])  # m/s per m altitude
            altitude_ref = layer.get('reference_altitude', 0)
            layer_wind += shear_rate * (altitude - altitude_ref)
        
        # Spatial variation
        if 'spatial_variation' in layer:
            spatial_scale = layer['spatial_variation'].get('scale', 1000.0)  # m
            spatial_amplitude = layer['spatial_variation'].get('amplitude', 1.0)  # m/s
            
            # Simple sinusoidal variation
            spatial_factor = spatial_amplitude * np.sin(2 * np.pi * position[0] / spatial_scale)
            layer_wind[0] += spatial_factor
        
        # Temporal variation
        if 'temporal_variation' in layer:
            period = layer['temporal_variation'].get('period', 3600.0)  # s
            amplitude = layer['temporal_variation'].get('amplitude', 1.0)  # m/s
            
            temporal_factor = amplitude * np.sin(2 * np.pi * time / period)
            layer_wind += temporal_factor * np.array([1, 0, 0])  # Assume x-direction variation
        
        return layer_wind
    
    def _compute_turbulence(self, position: np.ndarray, time: float) -> np.ndarray:
        """Compute atmospheric turbulence."""
        if self.gust_model == "dryden":
            return self._dryden_turbulence(position, time)
        elif self.gust_model == "von_karman":
            return self._von_karman_turbulence(position, time)
        elif self.gust_model == "discrete":
            return self._discrete_gust(position, time)
        else:
            # Simple random turbulence
            return self.turbulence_intensity * np.random.randn(3)
    
    def _dryden_turbulence(self, position: np.ndarray, time: float) -> np.ndarray:
        """Dryden turbulence model (simplified)."""
        altitude = abs(position[2])
        
        # Turbulence intensity based on altitude
        if altitude < 150:  # Low altitude
            sigma_u = 2.0  # m/s
            sigma_v = 1.5
            sigma_w = 1.0
        elif altitude < 1500:  # Medium altitude
            sigma_u = 1.5
            sigma_v = 1.0
            sigma_w = 0.8
        else:  # High altitude
            sigma_u = 1.0
            sigma_v = 0.8
            sigma_w = 0.5
        
        # Scale lengths
        L_u = altitude if altitude > 150 else 150
        L_v = L_u / 2
        L_w = altitude if altitude > 150 else 150
        
        # Generate correlated turbulence (simplified)
        # In practice, would use proper spectral shaping filters
        white_noise = np.random.randn(3)
        
        # Apply intensity scaling
        turbulence = np.array([
            sigma_u * white_noise[0],
            sigma_v * white_noise[1],
            sigma_w * white_noise[2]
        ])
        
        return self.turbulence_intensity * turbulence
    
    def _von_karman_turbulence(self, position: np.ndarray, time: float) -> np.ndarray:
        """Von Karman turbulence model (simplified)."""
        # Similar to Dryden but with different spectral characteristics
        return self._dryden_turbulence(position, time) * 0.8
    
    def _discrete_gust(self, position: np.ndarray, time: float) -> np.ndarray:
        """Discrete gust model."""
        # Simple discrete gust implementation
        gust_probability = 0.001  # Probability of gust per time step
        
        if np.random.random() < gust_probability:
            # Generate discrete gust
            gust_magnitude = self.turbulence_intensity * 10.0  # Stronger than continuous turbulence
            gust_direction = np.random.randn(3)
            gust_direction /= np.linalg.norm(gust_direction)
            
            return gust_magnitude * gust_direction
        else:
            return np.zeros(3)
    
    def add_wind_layer(self, layer_config: Dict[str, Any]):
        """Add a wind layer to the environment."""
        self.wind_layers.append(layer_config)
    
    def get_gravity_vector(self, position: np.ndarray) -> np.ndarray:
        """Get gravity vector at position (can vary with altitude/location)."""
        # Simple model - constant gravity
        # In reality, would vary with latitude and altitude
        return self.base_env.gravity.copy()
    
    def get_magnetic_field(self, position: np.ndarray, time: float) -> np.ndarray:
        """Get magnetic field vector at position and time."""
        if self.base_env.magnetic_field is not None:
            return self.base_env.magnetic_field.copy()
        
        # Default Earth magnetic field (simplified)
        # In reality, would use IGRF model
        latitude = 0.0  # Would need to convert from position
        
        # Approximate magnetic field components (Tesla)
        B_north = 2e-5 * np.cos(np.deg2rad(latitude))
        B_east = 0.0
        B_down = 4e-5 * np.sin(np.deg2rad(latitude))
        
        return np.array([B_north, B_east, B_down])
    
    @staticmethod
    @jit(nopython=True)
    def compute_air_density(pressure: float, temperature: float) -> float:
        """Compute air density from pressure and temperature."""
        R = 287.0  # Specific gas constant for dry air
        return pressure / (R * temperature)
    
    @staticmethod
    @jit(nopython=True)
    def compute_mach_number(airspeed: float, temperature: float) -> float:
        """Compute Mach number from airspeed and temperature."""
        gamma = 1.4
        R = 287.0
        speed_of_sound = np.sqrt(gamma * R * temperature)
        return airspeed / speed_of_sound
    
    @staticmethod
    @jit(nopython=True)
    def compute_dynamic_pressure(density: float, airspeed: float) -> float:
        """Compute dynamic pressure."""
        return 0.5 * density * airspeed**2
    
    def create_standard_wind_profile(self, surface_wind_speed: float = 10.0, surface_wind_direction: float = 0.0):
        """Create a standard wind profile with altitude."""
        # Clear existing wind layers
        self.wind_layers = []
        
        # Surface layer (0-150m)
        self.add_wind_layer({
            'min_altitude': 0,
            'max_altitude': 150,
            'velocity': [
                surface_wind_speed * np.cos(np.deg2rad(surface_wind_direction)),
                surface_wind_speed * np.sin(np.deg2rad(surface_wind_direction)),
                0
            ],
            'shear': [0.02, 0, 0]  # Wind shear with altitude
        })
        
        # Boundary layer (150-1500m)
        self.add_wind_layer({
            'min_altitude': 150,
            'max_altitude': 1500,
            'velocity': [
                surface_wind_speed * 1.5 * np.cos(np.deg2rad(surface_wind_direction + 15)),
                surface_wind_speed * 1.5 * np.sin(np.deg2rad(surface_wind_direction + 15)),
                0
            ],
            'shear': [0.005, 0, 0]
        })
        
        # Free atmosphere (1500m+)
        self.add_wind_layer({
            'min_altitude': 1500,
            'max_altitude': 50000,
            'velocity': [
                surface_wind_speed * 2.0 * np.cos(np.deg2rad(surface_wind_direction + 30)),
                surface_wind_speed * 2.0 * np.sin(np.deg2rad(surface_wind_direction + 30)),
                0
            ]
        }) 