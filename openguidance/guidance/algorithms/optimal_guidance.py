"""
Optimal guidance algorithms for aerospace applications.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum, auto
import logging

from openguidance.core.types import State, Environment

logger = logging.getLogger(__name__)


class OptimalGuidanceType(Enum):
    """Types of optimal guidance laws."""
    MINIMUM_TIME = auto()
    MINIMUM_ENERGY = auto()
    MINIMUM_FUEL = auto()
    MINIMUM_EFFORT = auto()
    MAXIMUM_RANGE = auto()


@dataclass
class OptimalGuidanceConfig:
    """Configuration for optimal guidance."""
    guidance_type: OptimalGuidanceType
    terminal_constraints: Dict[str, float]
    path_constraints: Dict[str, Tuple[float, float]]
    performance_index_weights: Dict[str, float]
    numerical_tolerance: float = 1e-6
    max_iterations: int = 100


class OptimalGuidance:
    """Optimal guidance law implementation using calculus of variations."""
    
    def __init__(self, config: OptimalGuidanceConfig):
        """Initialize optimal guidance system."""
        self.config = config
        self.guidance_type = config.guidance_type
        
        # Lagrange multipliers for terminal constraints
        self.lambda_terminal = np.zeros(6)  # Position and velocity constraints
        
        # Path constraint multipliers
        self.mu_path = {}
        
        # Guidance history for adaptive algorithms
        self.guidance_history = []
        
        logger.info(f"Optimal guidance initialized: {self.guidance_type.name}")
    
    def compute_command(
        self, 
        missile_state: State, 
        target_state: State,
        time_to_go: Optional[float] = None,
        environment: Optional[Environment] = None
    ) -> np.ndarray:
        """Compute optimal guidance acceleration command.
        
        Args:
            missile_state: Current missile state
            target_state: Current target state  
            time_to_go: Estimated time to intercept
            environment: Environment conditions
            
        Returns:
            Optimal acceleration command in inertial frame [m/s^2]
        """
        if time_to_go is None:
            time_to_go = self._estimate_time_to_go(missile_state, target_state)
        
        if time_to_go <= 0.1:  # Very close to target
            return np.zeros(3)
        
        # Select guidance law based on type
        if self.guidance_type == OptimalGuidanceType.MINIMUM_TIME:
            return self._minimum_time_guidance(missile_state, target_state, time_to_go)
        elif self.guidance_type == OptimalGuidanceType.MINIMUM_ENERGY:
            return self._minimum_energy_guidance(missile_state, target_state, time_to_go)
        elif self.guidance_type == OptimalGuidanceType.MINIMUM_FUEL:
            return self._minimum_fuel_guidance(missile_state, target_state, time_to_go)
        elif self.guidance_type == OptimalGuidanceType.MINIMUM_EFFORT:
            return self._minimum_effort_guidance(missile_state, target_state, time_to_go)
        elif self.guidance_type == OptimalGuidanceType.MAXIMUM_RANGE:
            return self._maximum_range_guidance(missile_state, target_state, time_to_go)
        else:
            logger.warning(f"Unknown guidance type: {self.guidance_type}")
            return np.zeros(3)
    
    def _minimum_time_guidance(
        self, 
        missile_state: State, 
        target_state: State, 
        time_to_go: float
    ) -> np.ndarray:
        """Minimum time optimal guidance law."""
        # Relative position and velocity
        rel_pos = target_state.position - missile_state.position
        rel_vel = target_state.velocity - missile_state.velocity
        
        # Range and closing velocity
        range_to_target = np.linalg.norm(rel_pos)
        if range_to_target < 1.0:
            return np.zeros(3)
        
        # Line-of-sight unit vector
        los_unit = rel_pos / range_to_target
        
        # Closing velocity
        v_closing = -np.dot(rel_vel, los_unit)
        if v_closing <= 0:
            return np.zeros(3)
        
        # Minimum time guidance (bang-bang control approximation)
        # For minimum time, use maximum available acceleration
        max_accel = self.config.path_constraints.get("max_acceleration", (0, 50))[1]
        
        # Direction: perpendicular to LOS towards zero miss distance
        los_rate = np.cross(rel_pos, rel_vel) / (range_to_target ** 2)
        
        if np.linalg.norm(los_rate) > 1e-6:
            accel_direction = los_rate / np.linalg.norm(los_rate)
            return -max_accel * accel_direction
        else:
            return np.zeros(3)
    
    def _minimum_energy_guidance(
        self, 
        missile_state: State, 
        target_state: State, 
        time_to_go: float
    ) -> np.ndarray:
        """Minimum energy optimal guidance law."""
        # Relative geometry
        rel_pos = target_state.position - missile_state.position
        rel_vel = target_state.velocity - missile_state.velocity
        
        # Zero effort miss (ZEM)
        zem = rel_pos + rel_vel * time_to_go
        
        # Zero effort miss velocity (ZEMV)
        zemv = rel_vel
        
        # Minimum energy guidance law
        # a* = -6*ZEM/t_go^2 - 4*ZEMV/t_go
        if time_to_go > 0.1:
            accel_cmd = -6.0 * zem / (time_to_go ** 2) - 4.0 * zemv / time_to_go
        else:
            accel_cmd = np.zeros(3)
        
        return accel_cmd
    
    def _minimum_fuel_guidance(
        self, 
        missile_state: State, 
        target_state: State, 
        time_to_go: float
    ) -> np.ndarray:
        """Minimum fuel optimal guidance law."""
        # Similar to minimum energy but with fuel consumption weighting
        rel_pos = target_state.position - missile_state.position
        rel_vel = target_state.velocity - missile_state.velocity
        
        # Estimate required acceleration for intercept
        zem = rel_pos + rel_vel * time_to_go
        
        # Fuel-optimal guidance (smooth control)
        if time_to_go > 0.1:
            # Softer control to minimize fuel consumption
            accel_cmd = -2.0 * zem / (time_to_go ** 2) - rel_vel / time_to_go
        else:
            accel_cmd = np.zeros(3)
        
        # Apply fuel consumption penalty
        fuel_penalty = self.config.performance_index_weights.get("fuel_weight", 1.0)
        accel_cmd *= (1.0 / fuel_penalty)
        
        return accel_cmd
    
    def _minimum_effort_guidance(
        self, 
        missile_state: State, 
        target_state: State, 
        time_to_go: float
    ) -> np.ndarray:
        """Minimum control effort optimal guidance law."""
        # Relative state
        rel_pos = target_state.position - missile_state.position
        rel_vel = target_state.velocity - missile_state.velocity
        
        # Zero effort miss
        zem = rel_pos + rel_vel * time_to_go
        
        # Minimum effort guidance (quadratic cost on control)
        if time_to_go > 0.1:
            # Smooth control minimizing acceleration magnitude
            accel_cmd = -3.0 * zem / (time_to_go ** 2) - 2.0 * rel_vel / time_to_go
        else:
            accel_cmd = np.zeros(3)
        
        # Apply effort weighting
        effort_weight = self.config.performance_index_weights.get("effort_weight", 1.0)
        accel_cmd *= effort_weight
        
        return accel_cmd
    
    def _maximum_range_guidance(
        self, 
        missile_state: State, 
        target_state: State, 
        time_to_go: float
    ) -> np.ndarray:
        """Maximum range optimal guidance law."""
        # For maximum range, minimize energy expenditure while maintaining intercept
        rel_pos = target_state.position - missile_state.position
        rel_vel = target_state.velocity - missile_state.velocity
        
        # Range consideration
        current_range = np.linalg.norm(missile_state.position)
        
        # Modified guidance for range optimization
        zem = rel_pos + rel_vel * time_to_go
        
        if time_to_go > 0.1:
            # Conservative guidance to preserve range
            range_factor = 1.0 / (1.0 + current_range / 10000.0)  # Reduce effort at long range
            accel_cmd = -2.0 * zem / (time_to_go ** 2) * range_factor
        else:
            accel_cmd = np.zeros(3)
        
        return accel_cmd
    
    def _estimate_time_to_go(self, missile_state: State, target_state: State) -> float:
        """Estimate time to intercept."""
        rel_pos = target_state.position - missile_state.position
        rel_vel = target_state.velocity - missile_state.velocity
        
        range_to_target = np.linalg.norm(rel_pos)
        
        if range_to_target < 1.0:
            return 0.0
        
        # Closing velocity
        closing_velocity = -np.dot(rel_vel, rel_pos) / range_to_target
        
        if closing_velocity > 0:
            return range_to_target / closing_velocity
        else:
            return float('inf')
    
    def get_performance_index(
        self, 
        missile_state: State, 
        target_state: State, 
        time_to_go: float
    ) -> float:
        """Compute current performance index value."""
        accel_cmd = self.compute_command(missile_state, target_state, time_to_go)
        accel_magnitude = np.linalg.norm(accel_cmd)
        
        if self.guidance_type == OptimalGuidanceType.MINIMUM_TIME:
            return float(time_to_go)
        elif self.guidance_type == OptimalGuidanceType.MINIMUM_ENERGY:
            return float(0.5 * accel_magnitude ** 2 * time_to_go)
        elif self.guidance_type == OptimalGuidanceType.MINIMUM_FUEL:
            # Approximate fuel consumption
            return float(accel_magnitude * time_to_go)
        elif self.guidance_type == OptimalGuidanceType.MINIMUM_EFFORT:
            return float(accel_magnitude ** 2)
        elif self.guidance_type == OptimalGuidanceType.MAXIMUM_RANGE:
            return float(-np.linalg.norm(missile_state.position))  # Negative for maximization
        else:
            return 0.0
    
    def check_terminal_constraints(
        self, 
        missile_state: State, 
        target_state: State
    ) -> Dict[str, float]:
        """Check terminal constraint violations."""
        rel_pos = target_state.position - missile_state.position
        rel_vel = target_state.velocity - missile_state.velocity
        
        miss_distance = np.linalg.norm(rel_pos)
        relative_speed = np.linalg.norm(rel_vel)
        
        constraints = {}
        
        # Position constraints
        max_miss = self.config.terminal_constraints.get("max_miss_distance", 1.0)
        constraints["miss_distance_violation"] = max(0, miss_distance - max_miss)
        
        # Velocity constraints
        max_rel_speed = self.config.terminal_constraints.get("max_relative_speed", 100.0)
        constraints["relative_speed_violation"] = max(0, relative_speed - max_rel_speed)
        
        return constraints
    
    def check_path_constraints(
        self, 
        missile_state: State, 
        accel_cmd: np.ndarray
    ) -> Dict[str, float]:
        """Check path constraint violations."""
        violations = {}
        
        # Acceleration constraints
        accel_magnitude = np.linalg.norm(accel_cmd)
        max_accel = self.config.path_constraints.get("max_acceleration", (0, 50))[1]
        violations["acceleration_violation"] = max(0, accel_magnitude - max_accel)
        
        # Altitude constraints
        altitude = -missile_state.position[2]  # NED frame
        min_alt, max_alt = self.config.path_constraints.get("altitude", (0, 50000))
        violations["altitude_violation"] = max(0, min_alt - altitude, altitude - max_alt)
        
        # Speed constraints
        speed = missile_state.speed
        min_speed, max_speed = self.config.path_constraints.get("speed", (0, 1000))
        violations["speed_violation"] = max(0, min_speed - speed, speed - max_speed)
        
        return violations
    
    def get_guidance_status(
        self, 
        missile_state: State, 
        target_state: State
    ) -> Dict[str, Any]:
        """Get detailed guidance status."""
        time_to_go = self._estimate_time_to_go(missile_state, target_state)
        accel_cmd = self.compute_command(missile_state, target_state, time_to_go)
        
        return {
            "guidance_type": self.guidance_type.name,
            "time_to_go": time_to_go,
            "acceleration_command": accel_cmd.tolist(),
            "acceleration_magnitude": np.linalg.norm(accel_cmd),
            "performance_index": self.get_performance_index(missile_state, target_state, time_to_go),
            "terminal_constraints": self.check_terminal_constraints(missile_state, target_state),
            "path_constraints": self.check_path_constraints(missile_state, accel_cmd),
            "is_feasible": all(v < self.config.numerical_tolerance 
                             for v in self.check_path_constraints(missile_state, accel_cmd).values())
        }


class AdaptiveOptimalGuidance(OptimalGuidance):
    """Adaptive optimal guidance with online parameter estimation."""
    
    def __init__(self, config: OptimalGuidanceConfig, adaptation_rate: float = 0.1):
        """Initialize adaptive optimal guidance."""
        super().__init__(config)
        self.adaptation_rate = adaptation_rate
        
        # Adaptive parameters
        self.estimated_target_acceleration = np.zeros(3)
        self.target_maneuver_history = []
        
        # Performance tracking
        self.performance_history = []
        self.miss_distance_history = []
    
    def compute_command(
        self, 
        missile_state: State, 
        target_state: State,
        time_to_go: Optional[float] = None,
        environment: Optional[Environment] = None
    ) -> np.ndarray:
        """Compute adaptive optimal guidance command."""
        # Update target maneuver estimate
        self._update_target_estimate(target_state)
        
        # Get base optimal command
        base_command = super().compute_command(missile_state, target_state, time_to_go, environment)
        
        # Add adaptive compensation
        adaptive_compensation = self._compute_adaptive_compensation(
            missile_state, target_state, time_to_go or self._estimate_time_to_go(missile_state, target_state)
        )
        
        return base_command + adaptive_compensation
    
    def _update_target_estimate(self, target_state: State) -> None:
        """Update target acceleration estimate."""
        # Store target state history
        self.target_maneuver_history.append({
            "time": target_state.time,
            "position": target_state.position.copy(),
            "velocity": target_state.velocity.copy()
        })
        
        # Keep only recent history
        if len(self.target_maneuver_history) > 10:
            self.target_maneuver_history.pop(0)
        
        # Estimate acceleration from velocity history
        if len(self.target_maneuver_history) >= 2:
            recent = self.target_maneuver_history[-1]
            previous = self.target_maneuver_history[-2]
            
            dt = recent["time"] - previous["time"]
            if dt > 0:
                accel_estimate = (recent["velocity"] - previous["velocity"]) / dt
                
                # Exponential moving average
                alpha = self.adaptation_rate
                self.estimated_target_acceleration = (
                    (1 - alpha) * self.estimated_target_acceleration + 
                    alpha * accel_estimate
                )
    
    def _compute_adaptive_compensation(
        self, 
        missile_state: State, 
        target_state: State, 
        time_to_go: float
    ) -> np.ndarray:
        """Compute adaptive compensation for target maneuvers."""
        if time_to_go <= 0.1:
            return np.zeros(3)
        
        # Compensate for estimated target acceleration
        # This is similar to Augmented Proportional Navigation
        target_accel_compensation = 0.5 * self.estimated_target_acceleration
        
        # Adaptive gain based on performance history
        if len(self.performance_history) > 5:
            recent_performance = np.mean(self.performance_history[-5:])
            if recent_performance > 1.0:  # Poor performance
                adaptive_gain = 1.5
            else:
                adaptive_gain = 1.0
        else:
            adaptive_gain = 1.0
        
        return adaptive_gain * target_accel_compensation


def create_minimum_time_guidance(
    max_acceleration: float = 50.0,
    max_miss_distance: float = 1.0
) -> OptimalGuidance:
    """Create minimum time optimal guidance configuration."""
    config = OptimalGuidanceConfig(
        guidance_type=OptimalGuidanceType.MINIMUM_TIME,
        terminal_constraints={"max_miss_distance": max_miss_distance},
        path_constraints={"max_acceleration": (0, max_acceleration)},
        performance_index_weights={"time_weight": 1.0}
    )
    return OptimalGuidance(config)


def create_minimum_energy_guidance(
    max_acceleration: float = 30.0,
    energy_weight: float = 1.0
) -> OptimalGuidance:
    """Create minimum energy optimal guidance configuration."""
    config = OptimalGuidanceConfig(
        guidance_type=OptimalGuidanceType.MINIMUM_ENERGY,
        terminal_constraints={"max_miss_distance": 1.0},
        path_constraints={"max_acceleration": (0, max_acceleration)},
        performance_index_weights={"energy_weight": energy_weight}
    )
    return OptimalGuidance(config) 