"""Aerodynamics models for OpenGuidance."""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from numba import jit

from openguidance.core.types import Vehicle, VehicleType


class AerodynamicsModel:
    """Aerodynamics model for aircraft and missiles."""
    
    def __init__(self, vehicle: Vehicle):
        """Initialize aerodynamics model."""
        self.vehicle = vehicle
        self.reference_area = vehicle.reference_area or 1.0
        self.reference_length = vehicle.reference_length or 1.0
        
        # Default aerodynamic coefficients
        self.aero_coeffs = vehicle.aero_coeffs or self._get_default_coeffs()
        
    def _get_default_coeffs(self) -> Dict[str, Any]:
        """Get default aerodynamic coefficients based on vehicle type."""
        if self.vehicle.type == VehicleType.AIRCRAFT:
            return {
                # Lift coefficients
                'CL0': 0.1,      # Zero-alpha lift coefficient
                'CLa': 5.7,      # Lift curve slope (1/rad)
                'CLde': 0.43,    # Elevator effectiveness
                'CLq': 7.95,     # Pitch rate damping
                
                # Drag coefficients  
                'CD0': 0.03,     # Zero-lift drag
                'CDa': 0.3,      # Drag due to alpha
                'CDa2': 0.0,     # Drag due to alpha^2
                'CDde': 0.0,     # Drag due to elevator
                
                # Side force coefficients
                'CYb': -0.98,    # Side force due to sideslip
                'CYdr': 0.17,    # Rudder effectiveness
                'CYp': 0.0,      # Roll rate coupling
                'CYr': 0.0,      # Yaw rate coupling
                
                # Rolling moment coefficients
                'Clb': -0.13,    # Dihedral effect
                'Clda': 0.25,    # Aileron effectiveness
                'Cldr': 0.024,   # Rudder coupling
                'Clp': -0.5,     # Roll damping
                'Clr': 0.15,     # Yaw-roll coupling
                
                # Pitching moment coefficients
                'Cm0': 0.045,    # Zero-alpha pitching moment
                'Cma': -0.7,     # Pitch stiffness
                'Cmde': -1.28,   # Elevator effectiveness
                'Cmq': -12.4,    # Pitch damping
                
                # Yawing moment coefficients
                'Cnb': 0.073,    # Weathercock stability
                'Cnda': -0.053,  # Aileron-yaw coupling
                'Cndr': -0.106,  # Rudder effectiveness
                'Cnp': -0.069,   # Roll-yaw coupling
                'Cnr': -0.095,   # Yaw damping
            }
        elif self.vehicle.type == VehicleType.MISSILE:
            return {
                # Simplified missile coefficients
                'CL0': 0.0,
                'CLa': 2.0,
                'CD0': 0.02,
                'CDa': 0.1,
                'CDa2': 0.5,
                'Cma': -1.0,
                'Cmq': -5.0,
                'Cnb': 0.5,
                'Cnr': -0.5,
            }
        else:
            # Basic coefficients for other vehicle types
            return {
                'CL0': 0.0, 'CLa': 2.0, 'CD0': 0.1, 'CDa': 0.2,
                'Cma': -0.5, 'Cmq': -2.0, 'Cnb': 0.2, 'Cnr': -0.2,
            }
    
    def compute_forces_moments(
        self, 
        airspeed: float,
        alpha: float,
        beta: float,
        angular_velocity: np.ndarray,
        control_surfaces: Dict[str, float],
        density: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute aerodynamic forces and moments."""
        # Dynamic pressure
        q_bar = 0.5 * density * airspeed**2
        
        # Extract angular rates (normalized by airspeed)
        p_hat = angular_velocity[0] * self.reference_length / (2 * airspeed) if airspeed > 0.1 else 0.0
        q_hat = angular_velocity[1] * self.reference_length / (2 * airspeed) if airspeed > 0.1 else 0.0
        r_hat = angular_velocity[2] * self.reference_length / (2 * airspeed) if airspeed > 0.1 else 0.0
        
        # Extract control surface deflections
        delta_e = control_surfaces.get('elevator', 0.0)
        delta_a = control_surfaces.get('aileron', 0.0)
        delta_r = control_surfaces.get('rudder', 0.0)
        
        # Compute aerodynamic coefficients
        CL = self._compute_lift_coefficient(alpha, delta_e, q_hat)
        CD = self._compute_drag_coefficient(alpha, delta_e)
        CY = self._compute_side_force_coefficient(beta, delta_r, p_hat, r_hat)
        
        Cl = self._compute_roll_moment_coefficient(beta, delta_a, delta_r, p_hat, r_hat)
        Cm = self._compute_pitch_moment_coefficient(alpha, delta_e, q_hat)
        Cn = self._compute_yaw_moment_coefficient(beta, delta_a, delta_r, p_hat, r_hat)
        
        # Convert to forces in wind frame
        L = q_bar * self.reference_area * CL  # Lift
        D = q_bar * self.reference_area * CD  # Drag
        Y = q_bar * self.reference_area * CY  # Side force
        
        # Transform to body frame
        # Wind frame: x-axis along velocity, z-axis perpendicular (lift direction)
        # Body frame: x-axis along fuselage
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        
        # Forces in body frame
        Fx = -D * cos_alpha * cos_beta + Y * sin_beta
        Fy = -D * sin_beta * cos_alpha + L * sin_alpha * sin_beta + Y * cos_beta
        Fz = -D * sin_alpha * cos_beta - L * cos_alpha
        
        forces = np.array([Fx, Fy, Fz])
        
        # Moments in body frame
        l = q_bar * self.reference_area * self.reference_length * Cl  # Roll moment
        m = q_bar * self.reference_area * self.reference_length * Cm  # Pitch moment
        n = q_bar * self.reference_area * self.reference_length * Cn  # Yaw moment
        
        moments = np.array([l, m, n])
        
        return forces, moments
    
    def _compute_lift_coefficient(self, alpha: float, delta_e: float, q_hat: float) -> float:
        """Compute lift coefficient."""
        CL0 = self.aero_coeffs.get('CL0', 0.0)
        CLa = self.aero_coeffs.get('CLa', 2.0)
        CLde = self.aero_coeffs.get('CLde', 0.0)
        CLq = self.aero_coeffs.get('CLq', 0.0)
        
        return CL0 + CLa * alpha + CLde * delta_e + CLq * q_hat
    
    def _compute_drag_coefficient(self, alpha: float, delta_e: float) -> float:
        """Compute drag coefficient."""
        CD0 = self.aero_coeffs.get('CD0', 0.1)
        CDa = self.aero_coeffs.get('CDa', 0.2)
        CDa2 = self.aero_coeffs.get('CDa2', 0.0)
        CDde = self.aero_coeffs.get('CDde', 0.0)
        
        return CD0 + CDa * abs(alpha) + CDa2 * alpha**2 + CDde * abs(delta_e)
    
    def _compute_side_force_coefficient(self, beta: float, delta_r: float, p_hat: float, r_hat: float) -> float:
        """Compute side force coefficient."""
        CYb = self.aero_coeffs.get('CYb', 0.0)
        CYdr = self.aero_coeffs.get('CYdr', 0.0)
        CYp = self.aero_coeffs.get('CYp', 0.0)
        CYr = self.aero_coeffs.get('CYr', 0.0)
        
        return CYb * beta + CYdr * delta_r + CYp * p_hat + CYr * r_hat
    
    def _compute_roll_moment_coefficient(self, beta: float, delta_a: float, delta_r: float, p_hat: float, r_hat: float) -> float:
        """Compute roll moment coefficient."""
        Clb = self.aero_coeffs.get('Clb', 0.0)
        Clda = self.aero_coeffs.get('Clda', 0.0)
        Cldr = self.aero_coeffs.get('Cldr', 0.0)
        Clp = self.aero_coeffs.get('Clp', 0.0)
        Clr = self.aero_coeffs.get('Clr', 0.0)
        
        return Clb * beta + Clda * delta_a + Cldr * delta_r + Clp * p_hat + Clr * r_hat
    
    def _compute_pitch_moment_coefficient(self, alpha: float, delta_e: float, q_hat: float) -> float:
        """Compute pitch moment coefficient."""
        Cm0 = self.aero_coeffs.get('Cm0', 0.0)
        Cma = self.aero_coeffs.get('Cma', -0.5)
        Cmde = self.aero_coeffs.get('Cmde', 0.0)
        Cmq = self.aero_coeffs.get('Cmq', 0.0)
        
        return Cm0 + Cma * alpha + Cmde * delta_e + Cmq * q_hat
    
    def _compute_yaw_moment_coefficient(self, beta: float, delta_a: float, delta_r: float, p_hat: float, r_hat: float) -> float:
        """Compute yaw moment coefficient."""
        Cnb = self.aero_coeffs.get('Cnb', 0.0)
        Cnda = self.aero_coeffs.get('Cnda', 0.0)
        Cndr = self.aero_coeffs.get('Cndr', 0.0)
        Cnp = self.aero_coeffs.get('Cnp', 0.0)
        Cnr = self.aero_coeffs.get('Cnr', 0.0)
        
        return Cnb * beta + Cnda * delta_a + Cndr * delta_r + Cnp * p_hat + Cnr * r_hat
    
    @staticmethod
    @jit(nopython=True)
    def compute_airdata(velocity_body: np.ndarray) -> Tuple[float, float, float]:
        """Compute airspeed, angle of attack, and sideslip angle."""
        u, v, w = velocity_body
        
        # Airspeed
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        
        if airspeed < 0.1:
            return airspeed, 0.0, 0.0
        
        # Angle of attack
        alpha = np.arctan2(w, u)
        
        # Sideslip angle
        beta = np.arcsin(np.clip(v / airspeed, -1.0, 1.0))
        
        return airspeed, alpha, beta 