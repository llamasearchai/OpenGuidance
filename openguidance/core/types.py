"""Core data types for OpenGuidance."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
import numpy as np
from enum import Enum, auto
from pyquaternion import Quaternion


class VehicleType(Enum):
    """Types of vehicles supported."""
    AIRCRAFT = auto()
    MISSILE = auto()
    QUADROTOR = auto()
    SPACECRAFT = auto()
    GROUND = auto()
    MARINE = auto()
    CUSTOM = auto()


class ControlMode(Enum):
    """Control system modes."""
    MANUAL = auto()
    STABILIZE = auto()
    ALTITUDE_HOLD = auto()
    POSITION_HOLD = auto()
    GUIDED = auto()
    AUTO = auto()
    LAND = auto()
    RTL = auto()  # Return to Launch
    LOITER = auto()


class GuidanceMode(Enum):
    """Guidance system modes."""
    WAYPOINT = auto()
    TRAJECTORY_TRACKING = auto()
    TARGET_PURSUIT = auto()
    INTERCEPT = auto()
    RENDEZVOUS = auto()
    FORMATION = auto()
    TERRAIN_FOLLOWING = auto()


class NavigationMode(Enum):
    """Navigation system modes."""
    GPS_INS = auto()
    VISUAL_INERTIAL = auto()
    TERRAIN_RELATIVE = auto()
    CELESTIAL = auto()
    DEAD_RECKONING = auto()


@dataclass
class State:
    """Vehicle state representation."""
    # Position (m) - NED or ECEF or ECI depending on frame
    position: np.ndarray  # [x, y, z]
    
    # Velocity (m/s) - body or inertial frame
    velocity: np.ndarray  # [vx, vy, vz]
    
    # Attitude - quaternion representation
    attitude: Quaternion
    
    # Angular velocity (rad/s) - body frame
    angular_velocity: np.ndarray  # [p, q, r]
    
    # Time (s)
    time: float
    
    # Additional states
    acceleration: Optional[np.ndarray] = None  # [ax, ay, az]
    angular_acceleration: Optional[np.ndarray] = None  # [dp/dt, dq/dt, dr/dt]
    
    # Covariance matrix for uncertainty
    covariance: Optional[np.ndarray] = None  # 12x12 or larger
    
    # Reference frame
    frame: str = "NED"  # NED, ECEF, ECI, BODY
    
    def __post_init__(self):
        """Validate state data."""
        assert self.position.shape == (3,), f"Position must be 3D, got {self.position.shape}"
        assert self.velocity.shape == (3,), f"Velocity must be 3D, got {self.velocity.shape}"
        assert self.angular_velocity.shape == (3,), f"Angular velocity must be 3D"
        
    @property
    def euler_angles(self) -> np.ndarray:
        """Get Euler angles [roll, pitch, yaw] in radians."""
        # Convert quaternion to Euler angles (ZYX convention)
        return np.array(self.attitude.yaw_pitch_roll[::-1])  # Reverse to get [roll, pitch, yaw]
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from body to inertial frame."""
        return self.attitude.rotation_matrix
    
    @property
    def speed(self) -> float:
        """Get total speed magnitude."""
        return float(np.linalg.norm(self.velocity))
    
    @property
    def altitude(self) -> float:
        """Get altitude (negative of z in NED)."""
        if self.frame == "NED":
            return -self.position[2]
        else:
            # Would need proper conversion for other frames
            return self.position[2]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "attitude": [self.attitude.w, self.attitude.x, self.attitude.y, self.attitude.z],
            "angular_velocity": self.angular_velocity.tolist(),
            "time": self.time,
            "frame": self.frame,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "State":
        """Create from dictionary."""
        return cls(
            position=np.array(data["position"]),
            velocity=np.array(data["velocity"]),
            attitude=Quaternion(data["attitude"]),
            angular_velocity=np.array(data["angular_velocity"]),
            time=data["time"],
            frame=data.get("frame", "NED"),
        )
    
    def copy(self) -> "State":
        """Create a deep copy."""
        return State(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            attitude=Quaternion(self.attitude),
            angular_velocity=self.angular_velocity.copy(),
            time=self.time,
            acceleration=self.acceleration.copy() if self.acceleration is not None else None,
            angular_acceleration=self.angular_acceleration.copy() if self.angular_acceleration is not None else None,
            covariance=self.covariance.copy() if self.covariance is not None else None,
            frame=self.frame,
        )


@dataclass
class Control:
    """Control input representation."""
    # Primary controls (meaning depends on vehicle type)
    thrust: float = 0.0  # Thrust command (N or normalized)
    
    # Control surface deflections or motor commands
    aileron: float = 0.0  # Aileron deflection (rad) or roll command
    elevator: float = 0.0  # Elevator deflection (rad) or pitch command
    rudder: float = 0.0  # Rudder deflection (rad) or yaw command
    
    # Additional controls
    throttle: float = 0.0  # Engine throttle (0-1)
    flaps: float = 0.0  # Flap deflection (rad)
    speedbrake: float = 0.0  # Speedbrake deployment (0-1)
    gear: bool = False  # Landing gear state
    
    # Thrust vector control (for missiles/rockets)
    tvc_pitch: float = 0.0  # TVC pitch angle (rad)
    tvc_yaw: float = 0.0  # TVC yaw angle (rad)
    
    # Multi-rotor specific
    motor_commands: Optional[np.ndarray] = None  # Individual motor thrusts
    
    # Constraints
    constraints: Optional[Dict[str, Tuple[float, float]]] = None
    
    def __post_init__(self):
        """Apply constraints if specified."""
        if self.constraints:
            self.apply_constraints()
    
    def apply_constraints(self):
        """Apply control constraints."""
        if not self.constraints:
            return
            
        for control_name, (min_val, max_val) in self.constraints.items():
            if hasattr(self, control_name):
                current_val = getattr(self, control_name)
                constrained_val = np.clip(current_val, min_val, max_val)
                setattr(self, control_name, constrained_val)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        base_controls = np.array([
            self.thrust,
            self.aileron,
            self.elevator,
            self.rudder,
            self.throttle,
        ])
        
        if self.motor_commands is not None:
            return np.concatenate([base_controls, self.motor_commands])
        
        return base_controls
    
    @classmethod
    def from_array(cls, array: np.ndarray, vehicle_type: VehicleType = VehicleType.AIRCRAFT) -> "Control":
        """Create from numpy array."""
        control = cls()
        
        if len(array) >= 5:
            control.thrust = array[0]
            control.aileron = array[1]
            control.elevator = array[2]
            control.rudder = array[3]
            control.throttle = array[4]
        
        if len(array) > 5 and vehicle_type == VehicleType.QUADROTOR:
            control.motor_commands = array[5:]
        
        return control


@dataclass
class Trajectory:
    """Trajectory representation."""
    # Time points
    times: np.ndarray
    
    # States along trajectory
    states: List[State]
    
    # Optional control inputs
    controls: Optional[List[Control]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate trajectory data."""
        assert len(self.times) == len(self.states), "Times and states must have same length"
        if self.controls is not None:
            assert len(self.controls) == len(self.states) - 1 or len(self.controls) == len(self.states)
    
    @property
    def duration(self) -> float:
        """Get total trajectory duration."""
        return self.times[-1] - self.times[0]
    
    @property
    def num_points(self) -> int:
        """Get number of trajectory points."""
        return len(self.states)
    
    def get_state_at_time(self, t: float) -> State:
        """Interpolate state at given time."""
        if t <= self.times[0]:
            return self.states[0]
        if t >= self.times[-1]:
            return self.states[-1]
        
        # Find surrounding points
        idx = np.searchsorted(self.times, t)
        t0, t1 = self.times[idx-1], self.times[idx]
        alpha = (t - t0) / (t1 - t0)
        
        # Interpolate state
        s0, s1 = self.states[idx-1], self.states[idx]
        
        # Linear interpolation for position and velocity
        position = (1 - alpha) * s0.position + alpha * s1.position
        velocity = (1 - alpha) * s0.velocity + alpha * s1.velocity
        angular_velocity = (1 - alpha) * s0.angular_velocity + alpha * s1.angular_velocity
        
        # Quaternion SLERP for attitude
        attitude = Quaternion.slerp(s0.attitude, s1.attitude, alpha)
        
        return State(
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_velocity=angular_velocity,
            time=t,
            frame=s0.frame
        )
    
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to position, velocity, attitude arrays."""
        positions = np.array([s.position for s in self.states])
        velocities = np.array([s.velocity for s in self.states])
        attitudes = np.array([s.euler_angles for s in self.states])
        
        return positions, velocities, attitudes


@dataclass
class Vehicle:
    """Vehicle configuration and parameters."""
    # Basic properties
    name: str
    type: VehicleType
    
    # Mass properties
    mass: float  # kg
    inertia: np.ndarray  # 3x3 inertia tensor
    center_of_gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Geometric properties
    reference_area: Optional[float] = None  # m^2
    reference_length: Optional[float] = None  # m
    wingspan: Optional[float] = None  # m
    
    # Propulsion
    max_thrust: Optional[float] = None  # N
    specific_impulse: Optional[float] = None  # s
    fuel_mass: Optional[float] = None  # kg
    
    # Aerodynamic coefficients (for aircraft/missiles)
    aero_coeffs: Optional[Dict[str, Any]] = None
    
    # Control limits
    control_limits: Optional[Dict[str, Tuple[float, float]]] = None
    
    # Sensor configuration
    sensors: Optional[Dict[str, Any]] = None
    
    # Additional parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_mass(self) -> float:
        """Get total vehicle mass including fuel."""
        total = self.mass
        if self.fuel_mass is not None:
            total += self.fuel_mass
        return total
    
    def get_inertia_with_fuel(self) -> np.ndarray:
        """Get inertia tensor including fuel effects."""
        # Simplified - would need proper fuel slosh modeling
        return self.inertia
    
    def validate(self) -> bool:
        """Validate vehicle configuration."""
        checks = [
            self.mass > 0,
            self.inertia.shape == (3, 3),
            np.all(np.linalg.eigvals(self.inertia) > 0),  # Positive definite
        ]
        
        if self.reference_area is not None:
            checks.append(self.reference_area > 0)
        
        if self.max_thrust is not None:
            checks.append(self.max_thrust > 0)
        
        return all(checks)


@dataclass
class Mission:
    """Mission definition and objectives."""
    # Mission identification
    name: str
    id: str
    
    # Mission type
    type: str  # "intercept", "reconnaissance", "delivery", etc.
    
    # Objectives
    objectives: List[Dict[str, Any]]
    
    # Waypoints or target information
    waypoints: Optional[List[np.ndarray]] = None
    target_state: Optional[State] = None
    
    # Constraints
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Time constraints
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    max_duration: Optional[float] = None
    
    # Success criteria
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    
    # Mission phases
    phases: Optional[List[Dict[str, Any]]] = None
    
    def is_complete(self, state: State) -> bool:
        """Check if mission objectives are complete."""
        # Implement mission-specific completion logic
        if self.type == "waypoint":
            # Check if all waypoints visited
            pass
        elif self.type == "intercept":
            # Check if target intercepted
            pass
        
        return False
    
    def get_current_objective(self, state: State) -> Dict[str, Any]:
        """Get current active objective based on state."""
        # Return the current objective to pursue
        if self.objectives:
            return self.objectives[0]
        return {}


@dataclass
class Environment:
    """Environmental conditions."""
    # Atmospheric properties
    density: float = 1.225  # kg/m^3 (sea level)
    pressure: float = 101325.0  # Pa
    temperature: float = 288.15  # K
    speed_of_sound: float = 340.29  # m/s
    
    # Wind
    wind_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s
    wind_gust_intensity: float = 0.0
    
    # Gravity
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, 9.81]))  # m/s^2
    
    # Magnetic field
    magnetic_field: Optional[np.ndarray] = None  # Tesla
    
    def get_density_at_altitude(self, altitude: float) -> float:
        """Get atmospheric density at altitude using standard atmosphere."""
        # Simplified exponential model
        scale_height = 8500.0  # m
        return self.density * np.exp(-altitude / scale_height)
    
    def get_wind_at_position(self, position: np.ndarray, time: float) -> np.ndarray:
        """Get wind velocity at position including gusts."""
        base_wind = self.wind_velocity.copy()
        
        if self.wind_gust_intensity > 0:
            # Add Dryden wind gust model
            gust = self.wind_gust_intensity * np.random.randn(3)
            base_wind += gust
        
        return base_wind 