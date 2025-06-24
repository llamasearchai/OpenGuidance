"""Main simulation framework for OpenGuidance."""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from openguidance.core.types import State, Control, Vehicle, Environment, Mission
from openguidance.dynamics.models.aircraft import AircraftDynamics
from openguidance.dynamics.models.missile import MissileDynamics
from openguidance.dynamics.environment import EnvironmentModel
from openguidance.control.autopilot import Autopilot
from openguidance.guidance.algorithms.proportional_navigation import ProportionalNavigation


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""
    dt: float = 0.01  # Time step (s)
    max_time: float = 300.0  # Maximum simulation time (s)
    max_steps: int = 100000  # Maximum simulation steps
    
    # Logging
    log_frequency: int = 10  # Log every N steps
    save_trajectory: bool = True
    
    # Termination conditions
    termination_conditions: List[str] = field(default_factory=lambda: [
        "max_time", "max_steps", "ground_impact", "mission_complete"
    ])
    
    # Real-time options
    real_time_factor: float = 1.0  # 1.0 = real-time, 0.0 = as fast as possible
    enable_visualization: bool = False


# Alias for backward compatibility
SimulatorConfig = SimulationConfig


class SimulationResult:
    """Container for simulation results."""
    
    def __init__(self):
        """Initialize simulation result."""
        self.success = False
        self.termination_reason = ""
        self.final_time = 0.0
        self.final_state = None
        
        # Trajectory data
        self.times = []
        self.states = []
        self.controls = []
        self.guidance_commands = []
        
        # Performance metrics
        self.metrics = {}
        
        # Logs
        self.logs = []
    
    def add_data_point(self, time: float, state: State, control: Control, guidance_cmd: Optional[np.ndarray] = None):
        """Add a data point to the trajectory."""
        self.times.append(time)
        self.states.append(state.copy())
        self.controls.append(control)
        if guidance_cmd is not None:
            self.guidance_commands.append(guidance_cmd.copy())
    
    def compute_metrics(self):
        """Compute performance metrics from trajectory."""
        if not self.states:
            return
        
        # Basic metrics
        self.metrics['total_time'] = self.final_time
        if self.final_state:
            self.metrics['final_position'] = self.final_state.position.tolist()
            self.metrics['final_velocity'] = self.final_state.velocity.tolist()
            self.metrics['final_speed'] = self.final_state.speed
        else:
            self.metrics['final_position'] = [0.0, 0.0, 0.0]
            self.metrics['final_velocity'] = [0.0, 0.0, 0.0]
            self.metrics['final_speed'] = 0.0
        
        # Trajectory statistics
        positions = np.array([s.position for s in self.states])
        velocities = np.array([s.velocity for s in self.states])
        
        self.metrics['max_altitude'] = np.max(-positions[:, 2])  # NED frame
        self.metrics['max_speed'] = np.max([s.speed for s in self.states])
        self.metrics['distance_traveled'] = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        
        # Control effort
        if self.controls:
            control_arrays = np.array([c.to_array() for c in self.controls])
            self.metrics['max_control_magnitude'] = np.max(np.linalg.norm(control_arrays, axis=1))
            self.metrics['total_control_effort'] = np.sum(np.linalg.norm(control_arrays, axis=1))


class Simulator:
    """Main simulation engine for OpenGuidance."""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize simulator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        
        # Simulation state
        self.current_time = 0.0
        self.step_count = 0
        self.running = False
        
        # Components
        self.vehicles = {}  # Dict of vehicle_id -> (dynamics, autopilot, state)
        self.environment = EnvironmentModel()
        self.mission = None
        
        # Guidance systems
        self.guidance_systems = {}
        
        # Callbacks
        self.step_callbacks = []
        self.termination_callbacks = []
        
        # Results
        self.result = SimulationResult()
    
    def add_vehicle(
        self, 
        vehicle_id: str, 
        vehicle: Vehicle, 
        initial_state: State,
        autopilot: Optional[Autopilot] = None
    ):
        """Add a vehicle to the simulation.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            vehicle: Vehicle configuration
            initial_state: Initial state
            autopilot: Autopilot system (optional)
        """
        # Create dynamics model
        if vehicle.type.name == "AIRCRAFT":
            dynamics = AircraftDynamics(vehicle)
        elif vehicle.type.name == "MISSILE":
            dynamics = MissileDynamics(vehicle)
        else:
            raise ValueError(f"Unsupported vehicle type: {vehicle.type}")
        
        # Create autopilot if not provided
        if autopilot is None:
            autopilot = Autopilot(vehicle)
        
        # Store vehicle data
        self.vehicles[vehicle_id] = {
            'vehicle': vehicle,
            'dynamics': dynamics,
            'autopilot': autopilot,
            'state': initial_state.copy(),
            'control': Control()
        }
    
    def add_guidance_system(self, vehicle_id: str, guidance_system):
        """Add a guidance system for a vehicle."""
        self.guidance_systems[vehicle_id] = guidance_system
    
    def set_environment(self, environment: EnvironmentModel):
        """Set the environment model."""
        self.environment = environment
    
    def set_mission(self, mission: Mission):
        """Set the mission parameters."""
        self.mission = mission
    
    def add_step_callback(self, callback: Callable):
        """Add a callback function to be called each simulation step."""
        self.step_callbacks.append(callback)
    
    def add_termination_callback(self, callback: Callable) -> None:
        """Add a callback function to check termination conditions."""
        self.termination_callbacks.append(callback)
    
    def run(self) -> SimulationResult:
        """Run the simulation.
        
        Returns:
            Simulation results
        """
        self.running = True
        self.current_time = 0.0
        self.step_count = 0
        
        # Initialize result
        self.result = SimulationResult()
        
        try:
            while self.running:
                # Check termination conditions
                if self._check_termination():
                    break
                
                # Simulation step
                self._simulation_step()
                
                # Real-time delay
                if self.config.real_time_factor > 0:
                    time.sleep(self.config.dt / self.config.real_time_factor)
                
                # Increment counters
                self.current_time += self.config.dt
                self.step_count += 1
            
            # Finalize results
            self._finalize_results()
            
        except Exception as e:
            self.result.success = False
            self.result.termination_reason = f"Error: {str(e)}"
            self.result.logs.append(f"Simulation error: {str(e)}")
        
        return self.result
    
    def _simulation_step(self):
        """Execute one simulation step."""
        # Update each vehicle
        for vehicle_id, vehicle_data in self.vehicles.items():
            self._update_vehicle(vehicle_id, vehicle_data)
        
        # Log data
        if self.step_count % self.config.log_frequency == 0:
            self._log_step()
        
        # Execute callbacks
        for callback in self.step_callbacks:
            callback(self)
    
    def _update_vehicle(self, vehicle_id: str, vehicle_data: Dict[str, Any]):
        """Update a single vehicle for one time step."""
        state = vehicle_data['state']
        dynamics = vehicle_data['dynamics']
        autopilot = vehicle_data['autopilot']
        
        # Guidance command
        guidance_cmd = None
        if vehicle_id in self.guidance_systems:
            guidance_system = self.guidance_systems[vehicle_id]
            
            # Example: missile intercept guidance
            if hasattr(guidance_system, 'compute_command'):
                # Need target state for guidance
                target_state = self._get_target_state(vehicle_id)
                if target_state:
                    guidance_cmd = guidance_system.compute_command(state, target_state)
                    # Set autopilot command
                    autopilot.set_acceleration_command(guidance_cmd)
        
        # Autopilot control
        control = autopilot.update(state)
        
        # Dynamics integration
        new_state = dynamics.step(state, control, self.environment.base_env, self.config.dt)
        
        # Update vehicle data
        vehicle_data['state'] = new_state
        vehicle_data['control'] = control
        
        # Store in result
        if self.config.save_trajectory:
            self.result.add_data_point(self.current_time, new_state, control, guidance_cmd)
    
    def _get_target_state(self, vehicle_id: str) -> Optional[State]:
        """Get target state for guidance (simplified)."""
        # This is a simplified implementation
        # In practice, would have proper target tracking
        
        if self.mission and hasattr(self.mission, 'target_state'):
            return self.mission.target_state
        
        # Default: stationary target at origin
        from pyquaternion import Quaternion
        target_state = State(
            position=np.array([1000.0, 0.0, -100.0]),  # 1km away, 100m altitude
            velocity=np.array([0.0, 0.0, 0.0]),
            attitude=Quaternion(),  # Identity quaternion
            angular_velocity=np.zeros(3),
            time=self.current_time
        )
        
        return target_state
    
    def _check_termination(self) -> bool:
        """Check if simulation should terminate."""
        # Time limits
        if "max_time" in self.config.termination_conditions:
            if self.current_time >= self.config.max_time:
                self.result.termination_reason = "Maximum time reached"
                return True
        
        if "max_steps" in self.config.termination_conditions:
            if self.step_count >= self.config.max_steps:
                self.result.termination_reason = "Maximum steps reached"
                return True
        
        # Ground impact
        if "ground_impact" in self.config.termination_conditions:
            for vehicle_data in self.vehicles.values():
                state = vehicle_data['state']
                if state.position[2] > 0:  # Below ground in NED
                    self.result.termination_reason = "Ground impact"
                    return True
        
        # Mission complete
        if "mission_complete" in self.config.termination_conditions:
            if self.mission and self._is_mission_complete():
                self.result.termination_reason = "Mission complete"
                self.result.success = True
                return True
        
        # Custom termination callbacks
        for callback in self.termination_callbacks:
            if callback(self):
                self.result.termination_reason = "Custom termination condition"
                return True
        
        return False
    
    def _is_mission_complete(self) -> bool:
        """Check if mission objectives are complete."""
        if not self.mission:
            return False
        
        # Check each vehicle against mission objectives
        for vehicle_data in self.vehicles.values():
            state = vehicle_data['state']
            if self.mission.is_complete(state):
                return True
        
        return False
    
    def _log_step(self):
        """Log current simulation step."""
        log_entry = {
            'time': self.current_time,
            'step': self.step_count,
            'vehicles': {}
        }
        
        for vehicle_id, vehicle_data in self.vehicles.items():
            state = vehicle_data['state']
            log_entry['vehicles'][vehicle_id] = {
                'position': state.position.tolist(),
                'velocity': state.velocity.tolist(),
                'attitude': state.euler_angles.tolist(),
                'speed': state.speed,
                'altitude': state.altitude
            }
        
        self.result.logs.append(log_entry)
    
    def _finalize_results(self):
        """Finalize simulation results."""
        if self.vehicles:
            # Get final state from first vehicle
            first_vehicle = next(iter(self.vehicles.values()))
            self.result.final_state = first_vehicle['state']
            self.result.final_time = self.current_time
        
        # Compute metrics
        self.result.compute_metrics()
        
        # Set success if not already set
        if not self.result.termination_reason:
            self.result.success = True
            self.result.termination_reason = "Simulation completed successfully"
    
    def get_vehicle_state(self, vehicle_id: str) -> Optional[State]:
        """Get current state of a vehicle."""
        if vehicle_id in self.vehicles:
            return self.vehicles[vehicle_id]['state']
        return None
    
    def set_vehicle_state(self, vehicle_id: str, state: State):
        """Set state of a vehicle."""
        if vehicle_id in self.vehicles:
            self.vehicles[vehicle_id]['state'] = state.copy()
    
    def pause(self):
        """Pause the simulation."""
        self.running = False
    
    def resume(self):
        """Resume the simulation."""
        self.running = True
    
    def stop(self):
        """Stop the simulation."""
        self.running = False
        self.result.termination_reason = "Simulation stopped by user"


# Add alias for compatibility
SimulatorConfig = SimulationConfig 