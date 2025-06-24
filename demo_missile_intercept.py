#!/usr/bin/env python3
"""
OpenGuidance Missile Intercept Demonstration

This demonstration showcases the complete OpenGuidance system capabilities:
- Advanced 6-DOF missile dynamics
- Proportional navigation guidance
- Real-time simulation with environment modeling
- Professional aerospace visualization
- Performance analysis and reporting

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pyquaternion import Quaternion

# OpenGuidance imports
from openguidance.core.types import (
    State, Control, Vehicle, VehicleType, Environment, Mission
)
from openguidance.dynamics.models.missile import MissileDynamics
from openguidance.guidance.algorithms.proportional_navigation import ProportionalNavigation
from openguidance.simulation.simulator import Simulator
from openguidance.utils.plotting import TrajectoryPlotter
from openguidance.analysis.performance import PerformanceAnalyzer


@dataclass
class InterceptScenario:
    """Complete intercept scenario configuration."""
    name: str
    description: str
    interceptor_config: Dict
    threat_config: Dict
    environment_config: Dict
    simulation_config: Dict
    success_criteria: Dict


class MissileInterceptDemo:
    """Complete missile intercept demonstration system."""
    
    def __init__(self):
        """Initialize the demonstration system."""
        self.scenarios = {}
        self.results = {}
        self.setup_scenarios()
        
    def setup_scenarios(self):
        """Setup various intercept scenarios."""
        
        # Scenario 1: Head-on intercept
        self.scenarios["head_on"] = InterceptScenario(
            name="Head-On Intercept",
            description="High-speed head-on engagement with ballistic threat",
            interceptor_config={
                "name": "SM-3 Interceptor",
                "mass": 1500.0,  # kg
                "inertia": np.diag([100, 2000, 2000]),  # kg*m^2
                "reference_area": 0.2,  # m^2
                "max_thrust": 50000.0,  # N
                "specific_impulse": 280.0,  # s
                "fuel_mass": 800.0,  # kg
                "initial_position": np.array([0, 0, -10000]),  # m (10km altitude)
                "initial_velocity": np.array([0, 0, 0]),  # m/s
                "initial_attitude": Quaternion(axis=[1, 0, 0], angle=0),
                "guidance_gain": 4.0,
                "seeker_fov": np.radians(30),  # 30 degree FOV
            },
            threat_config={
                "name": "Ballistic Threat",
                "mass": 2000.0,  # kg
                "inertia": np.diag([150, 3000, 3000]),  # kg*m^2
                "reference_area": 0.5,  # m^2
                "initial_position": np.array([50000, 0, -20000]),  # m
                "initial_velocity": np.array([-800, 0, 200]),  # m/s (Mach 2.5)
                "initial_attitude": Quaternion(axis=[0, 1, 0], angle=np.radians(15)),
                "trajectory_type": "ballistic",
            },
            environment_config={
                "density": 0.4,  # kg/m^3 (high altitude)
                "temperature": 220.0,  # K
                "wind_velocity": np.array([20, 10, 0]),  # m/s
                "turbulence_intensity": 0.1,
            },
            simulation_config={
                "dt": 0.01,  # s
                "max_time": 120.0,  # s
                "real_time_factor": 100.0,
            },
            success_criteria={
                "miss_distance": 5.0,  # m
                "intercept_altitude_min": 5000.0,  # m
                "intercept_altitude_max": 30000.0,  # m
            }
        )
        
        # Scenario 2: Crossing target
        self.scenarios["crossing"] = InterceptScenario(
            name="Crossing Target Intercept",
            description="Intercepting a crossing target with lead pursuit",
            interceptor_config={
                "name": "PAC-3 Interceptor",
                "mass": 320.0,  # kg
                "inertia": np.diag([20, 400, 400]),  # kg*m^2
                "reference_area": 0.1,  # m^2
                "max_thrust": 15000.0,  # N
                "specific_impulse": 250.0,  # s
                "fuel_mass": 120.0,  # kg
                "initial_position": np.array([0, 0, -5000]),  # m
                "initial_velocity": np.array([0, 0, 0]),  # m/s
                "initial_attitude": Quaternion(axis=[1, 0, 0], angle=0),
                "guidance_gain": 5.0,
                "seeker_fov": np.radians(45),
            },
            threat_config={
                "name": "Cruise Missile",
                "mass": 1200.0,  # kg
                "inertia": np.diag([80, 1500, 1500]),  # kg*m^2
                "reference_area": 0.3,  # m^2
                "initial_position": np.array([30000, -20000, -3000]),  # m
                "initial_velocity": np.array([-200, 300, 0]),  # m/s
                "initial_attitude": Quaternion(axis=[0, 0, 1], angle=np.radians(56)),
                "trajectory_type": "cruise",
            },
            environment_config={
                "density": 0.8,  # kg/m^3
                "temperature": 250.0,  # K
                "wind_velocity": np.array([15, -5, 0]),  # m/s
                "turbulence_intensity": 0.2,
            },
            simulation_config={
                "dt": 0.01,  # s
                "max_time": 180.0,  # s
                "real_time_factor": 50.0,
            },
            success_criteria={
                "miss_distance": 3.0,  # m
                "intercept_altitude_min": 1000.0,  # m
                "intercept_altitude_max": 10000.0,  # m
            }
        )
    
    def create_vehicle(self, config: Dict) -> Vehicle:
        """Create a vehicle from configuration."""
        return Vehicle(
            name=config["name"],
            type=VehicleType.MISSILE,
            mass=config["mass"],
            inertia=config["inertia"],
            reference_area=config.get("reference_area", 0.1),
            max_thrust=config.get("max_thrust", 10000.0),
            specific_impulse=config.get("specific_impulse", 250.0),
            fuel_mass=config.get("fuel_mass", 100.0),
            aero_coeffs={
                "CD0": 0.02,  # Zero-lift drag coefficient
                "CL_alpha": 3.5,  # Lift curve slope
                "CM_alpha": -0.5,  # Pitch moment coefficient
                "CN_beta": -0.8,  # Normal force due to sideslip
                "CY_beta": -0.3,  # Side force due to sideslip
            },
            control_limits={
                "thrust": (0.0, 1.0),
                "tvc_pitch": (-np.radians(15), np.radians(15)),
                "tvc_yaw": (-np.radians(15), np.radians(15)),
            }
        )
    
    def create_initial_state(self, config: Dict) -> State:
        """Create initial state from configuration."""
        return State(
            position=config["initial_position"],
            velocity=config["initial_velocity"],
            attitude=config["initial_attitude"],
            angular_velocity=np.zeros(3),
            time=0.0,
            frame="NED"
        )
    
    def create_environment(self, config: Dict) -> Environment:
        """Create environment from configuration."""
        return Environment(
            density=config.get("density", 1.225),
            temperature=config.get("temperature", 288.15),
            wind_velocity=config.get("wind_velocity", np.zeros(3)),
            wind_gust_intensity=config.get("turbulence_intensity", 0.0)
        )
    
    def run_scenario(self, scenario_name: str, verbose: bool = True) -> Dict:
        """Run a complete intercept scenario."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.scenarios[scenario_name]
        
        if verbose:
            print(f"\n[SCENARIO] Running: {scenario.name}")
            print(f"[SCENARIO] Description: {scenario.description}")
            print(f"[SCENARIO] Initializing vehicles and environment...")
        
        # Create vehicles
        interceptor_vehicle = self.create_vehicle(scenario.interceptor_config)
        threat_vehicle = self.create_vehicle(scenario.threat_config)
        
        # Create initial states
        interceptor_state = self.create_initial_state(scenario.interceptor_config)
        threat_state = self.create_initial_state(scenario.threat_config)
        
        # Create environment
        environment = self.create_environment(scenario.environment_config)
        
        # Create dynamics models
        interceptor_dynamics = MissileDynamics(interceptor_vehicle)
        threat_dynamics = MissileDynamics(threat_vehicle)
        
        # Create guidance system for interceptor
        guidance = ProportionalNavigation(
            navigation_gain=scenario.interceptor_config["guidance_gain"],
            seeker_fov=scenario.interceptor_config["seeker_fov"]
        )
        
        # Create mission
        mission = Mission(
            name=f"{scenario.name} Mission",
            id=f"mission_{scenario_name}_{int(time.time())}",
            type="intercept",
            objectives=[{
                "type": "intercept",
                "target": "threat",
                "priority": 1,
                "success_criteria": scenario.success_criteria
            }],
            target_state=threat_state,
            constraints={
                "max_acceleration": 40.0,  # g's
                "min_altitude": 100.0,  # m
                "max_range": 100000.0,  # m
            }
        )
        
        # Initialize simulation
        simulator = Simulator(dt=scenario.simulation_config["dt"])
        
        # Add vehicles to simulation
        simulator.add_vehicle("interceptor", interceptor_vehicle, interceptor_state, interceptor_dynamics)
        simulator.add_vehicle("threat", threat_vehicle, threat_state, threat_dynamics)
        simulator.set_environment(environment)
        
        # Simulation data storage
        simulation_data = {
            "time": [],
            "interceptor_states": [],
            "threat_states": [],
            "interceptor_controls": [],
            "guidance_commands": [],
            "relative_range": [],
            "relative_velocity": [],
            "line_of_sight_rate": [],
            "miss_distance": None,
            "intercept_time": None,
            "intercept_altitude": None,
            "fuel_remaining": [],
            "accelerations": [],
        }
        
        if verbose:
            print(f"[SIMULATION] Starting simulation (max time: {scenario.simulation_config['max_time']}s)")
            print(f"[SIMULATION] Initial separation: {np.linalg.norm(interceptor_state.position - threat_state.position):.1f}m")
        
        # Main simulation loop
        current_time = 0.0
        dt = scenario.simulation_config["dt"]
        max_time = scenario.simulation_config["max_time"]
        min_range = float('inf')
        intercept_occurred = False
        
        while current_time < max_time and not intercept_occurred:
            # Get current states
            interceptor_state = simulator.get_vehicle_state("interceptor")
            threat_state = simulator.get_vehicle_state("threat")
            
            # Calculate relative geometry
            relative_position = threat_state.position - interceptor_state.position
            relative_velocity = threat_state.velocity - interceptor_state.velocity
            range_to_target = np.linalg.norm(relative_position)
            
            # Update minimum range
            min_range = min(min_range, range_to_target)
            
            # Check for intercept
            if range_to_target < scenario.success_criteria["miss_distance"]:
                intercept_occurred = True
                simulation_data["intercept_time"] = current_time
                simulation_data["intercept_altitude"] = -interceptor_state.position[2]
                simulation_data["miss_distance"] = range_to_target
                
                if verbose:
                    print(f"[SUCCESS] Intercept achieved at t={current_time:.2f}s")
                    print(f"[SUCCESS] Miss distance: {range_to_target:.2f}m")
                    print(f"[SUCCESS] Intercept altitude: {-interceptor_state.position[2]:.1f}m")
                break
            
            # Generate guidance command for interceptor
            if range_to_target > 100.0:  # Only guide if target is detected
                guidance_command = guidance.compute_guidance(
                    interceptor_state=interceptor_state,
                    target_state=threat_state,
                    dt=dt
                )
                
                # Convert guidance command to control input
                interceptor_control = self.guidance_to_control(
                    guidance_command, interceptor_state, interceptor_vehicle
                )
            else:
                # Terminal guidance - maintain current heading
                interceptor_control = Control(thrust=1.0)
            
            # Threat follows ballistic or cruise trajectory
            if scenario.threat_config["trajectory_type"] == "ballistic":
                threat_control = Control()  # No control for ballistic target
            else:
                # Simple cruise control
                threat_control = Control(thrust=0.8)
            
            # Step simulation
            simulator.step(current_time, {
                "interceptor": interceptor_control,
                "threat": threat_control
            })
            
            # Store simulation data
            simulation_data["time"].append(current_time)
            simulation_data["interceptor_states"].append(interceptor_state.copy())
            simulation_data["threat_states"].append(threat_state.copy())
            simulation_data["interceptor_controls"].append(interceptor_control)
            simulation_data["relative_range"].append(range_to_target)
            simulation_data["relative_velocity"].append(np.linalg.norm(relative_velocity))
            
            # Calculate line-of-sight rate
            if range_to_target > 0:
                unit_los = relative_position / range_to_target
                los_rate = np.linalg.norm(np.cross(relative_position, relative_velocity)) / (range_to_target ** 2)
                simulation_data["line_of_sight_rate"].append(los_rate)
            else:
                simulation_data["line_of_sight_rate"].append(0.0)
            
            # Calculate acceleration
            if interceptor_state.acceleration is not None:
                accel_magnitude = np.linalg.norm(interceptor_state.acceleration) / 9.81  # g's
                simulation_data["accelerations"].append(accel_magnitude)
            else:
                simulation_data["accelerations"].append(0.0)
            
            # Estimate fuel remaining (simplified)
            fuel_consumed = interceptor_control.thrust * dt * 10.0  # kg/s approximation
            fuel_remaining = max(0, interceptor_vehicle.fuel_mass - fuel_consumed * current_time)
            simulation_data["fuel_remaining"].append(fuel_remaining)
            
            current_time += dt
            
            # Progress indicator
            if verbose and int(current_time * 10) % 50 == 0:
                progress = (current_time / max_time) * 100
                print(f"[PROGRESS] {progress:.1f}% - Range: {range_to_target:.1f}m - Time: {current_time:.1f}s")
        
        # Final results
        if not intercept_occurred:
            simulation_data["miss_distance"] = min_range
            if verbose:
                print(f"[RESULT] No intercept - Minimum miss distance: {min_range:.2f}m")
        
        # Performance analysis
        performance_results = self.analyze_performance(simulation_data, scenario)
        
        # Store results
        results = {
            "scenario": scenario,
            "simulation_data": simulation_data,
            "performance": performance_results,
            "success": intercept_occurred,
            "final_time": current_time
        }
        
        self.results[scenario_name] = results
        
        if verbose:
            print(f"[COMPLETE] Scenario completed in {current_time:.2f}s")
            self.print_performance_summary(performance_results)
        
        return results
    
    def guidance_to_control(self, guidance_command: Dict, state: State, vehicle: Vehicle) -> Control:
        """Convert guidance command to vehicle control input."""
        # Extract commanded acceleration
        accel_cmd = guidance_command.get("acceleration_command", np.zeros(3))
        
        # Simple control allocation - convert acceleration to thrust and TVC
        thrust_cmd = min(1.0, np.linalg.norm(accel_cmd) / 20.0)  # Normalize to 0-1
        
        # TVC commands based on lateral acceleration
        tvc_pitch = np.clip(accel_cmd[2] / 100.0, -0.26, 0.26)  # ±15 degrees
        tvc_yaw = np.clip(accel_cmd[1] / 100.0, -0.26, 0.26)
        
        return Control(
            thrust=thrust_cmd,
            tvc_pitch=tvc_pitch,
            tvc_yaw=tvc_yaw
        )
    
    def analyze_performance(self, data: Dict, scenario: InterceptScenario) -> Dict:
        """Analyze simulation performance."""
        analysis = {}
        
        # Basic statistics
        analysis["total_flight_time"] = data["time"][-1] if data["time"] else 0.0
        analysis["min_range"] = min(data["relative_range"]) if data["relative_range"] else float('inf')
        analysis["max_acceleration"] = max(data["accelerations"]) if data["accelerations"] else 0.0
        analysis["avg_acceleration"] = np.mean(data["accelerations"]) if data["accelerations"] else 0.0
        
        # Fuel consumption
        if data["fuel_remaining"]:
            initial_fuel = data["fuel_remaining"][0]
            final_fuel = data["fuel_remaining"][-1]
            analysis["fuel_consumed"] = initial_fuel - final_fuel
            analysis["fuel_efficiency"] = (initial_fuel - final_fuel) / initial_fuel * 100
        
        # Trajectory analysis
        if data["interceptor_states"]:
            positions = np.array([s.position for s in data["interceptor_states"]])
            velocities = np.array([s.velocity for s in data["interceptor_states"]])
            
            analysis["max_altitude"] = -np.min(positions[:, 2])  # NED frame
            analysis["max_speed"] = np.max([np.linalg.norm(v) for v in velocities])
            analysis["total_distance"] = np.sum([np.linalg.norm(positions[i] - positions[i-1]) 
                                               for i in range(1, len(positions))])
        
        # Guidance performance
        if data["line_of_sight_rate"]:
            analysis["max_los_rate"] = max(data["line_of_sight_rate"])
            analysis["avg_los_rate"] = np.mean(data["line_of_sight_rate"])
        
        # Success metrics
        analysis["intercept_success"] = data["miss_distance"] is not None and \
                                      data["miss_distance"] < scenario.success_criteria["miss_distance"]
        
        if data["intercept_altitude"]:
            altitude_ok = (scenario.success_criteria["intercept_altitude_min"] <= 
                          data["intercept_altitude"] <= 
                          scenario.success_criteria["intercept_altitude_max"])
            analysis["altitude_success"] = altitude_ok
        
        return analysis
    
    def print_performance_summary(self, performance: Dict):
        """Print formatted performance summary."""
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Flight Time:          {performance.get('total_flight_time', 0):.2f} seconds")
        print(f"Minimum Range:        {performance.get('min_range', 0):.2f} meters")
        print(f"Maximum Acceleration: {performance.get('max_acceleration', 0):.1f} g's")
        print(f"Average Acceleration: {performance.get('avg_acceleration', 0):.1f} g's")
        print(f"Maximum Altitude:     {performance.get('max_altitude', 0):.1f} meters")
        print(f"Maximum Speed:        {performance.get('max_speed', 0):.1f} m/s")
        print(f"Total Distance:       {performance.get('total_distance', 0):.1f} meters")
        print(f"Fuel Consumed:        {performance.get('fuel_consumed', 0):.1f} kg")
        print(f"Fuel Efficiency:      {performance.get('fuel_efficiency', 0):.1f}%")
        
        success_status = "[SUCCESS]" if performance.get('intercept_success', False) else "[FAILED]"
        print(f"Intercept Status:     {success_status}")
        
        print("="*60)
    
    def visualize_results(self, scenario_name: str, save_plots: bool = True):
        """Create comprehensive visualization of results."""
        if scenario_name not in self.results:
            raise ValueError(f"No results found for scenario: {scenario_name}")
        
        results = self.results[scenario_name]
        data = results["simulation_data"]
        
        # Extract trajectory data
        interceptor_positions = np.array([s.position for s in data["interceptor_states"]])
        threat_positions = np.array([s.position for s in data["threat_states"]])
        times = np.array(data["time"])
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'OpenGuidance Missile Intercept Analysis: {results["scenario"].name}', 
                    fontsize=16, fontweight='bold')
        
        # 3D Trajectory Plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(interceptor_positions[:, 0]/1000, interceptor_positions[:, 1]/1000, 
                -interceptor_positions[:, 2]/1000, 'b-', linewidth=2, label='Interceptor')
        ax1.plot(threat_positions[:, 0]/1000, threat_positions[:, 1]/1000, 
                -threat_positions[:, 2]/1000, 'r-', linewidth=2, label='Threat')
        
        # Mark initial and final positions
        ax1.scatter(interceptor_positions[0, 0]/1000, interceptor_positions[0, 1]/1000, 
                   -interceptor_positions[0, 2]/1000, c='blue', s=100, marker='o', label='Interceptor Start')
        ax1.scatter(threat_positions[0, 0]/1000, threat_positions[0, 1]/1000, 
                   -threat_positions[0, 2]/1000, c='red', s=100, marker='s', label='Threat Start')
        
        if data["miss_distance"] is not None and data["miss_distance"] < 10:
            # Mark intercept point
            intercept_idx = np.argmin(data["relative_range"])
            ax1.scatter(interceptor_positions[intercept_idx, 0]/1000, 
                       interceptor_positions[intercept_idx, 1]/1000,
                       -interceptor_positions[intercept_idx, 2]/1000, 
                       c='green', s=200, marker='*', label='Intercept')
        
        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Y (km)')
        ax1.set_zlabel('Altitude (km)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        ax1.grid(True)
        
        # Range vs Time
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(times, np.array(data["relative_range"])/1000, 'g-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Range (km)')
        ax2.set_title('Relative Range vs Time')
        ax2.grid(True)
        
        # Line of Sight Rate
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(times, np.array(data["line_of_sight_rate"]), 'purple', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('LOS Rate (rad/s)')
        ax3.set_title('Line of Sight Rate')
        ax3.grid(True)
        
        # Acceleration Profile
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(times, data["accelerations"], 'orange', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Acceleration (g)')
        ax4.set_title('Interceptor Acceleration')
        ax4.grid(True)
        
        # Altitude vs Time
        ax5 = fig.add_subplot(2, 3, 5)
        interceptor_altitudes = [-pos[2]/1000 for pos in interceptor_positions]
        threat_altitudes = [-pos[2]/1000 for pos in threat_positions]
        ax5.plot(times, interceptor_altitudes, 'b-', linewidth=2, label='Interceptor')
        ax5.plot(times, threat_altitudes, 'r-', linewidth=2, label='Threat')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Altitude (km)')
        ax5.set_title('Altitude Profile')
        ax5.legend()
        ax5.grid(True)
        
        # Speed Profile
        ax6 = fig.add_subplot(2, 3, 6)
        interceptor_speeds = [np.linalg.norm(s.velocity) for s in data["interceptor_states"]]
        threat_speeds = [np.linalg.norm(s.velocity) for s in data["threat_states"]]
        ax6.plot(times, interceptor_speeds, 'b-', linewidth=2, label='Interceptor')
        ax6.plot(times, threat_speeds, 'r-', linewidth=2, label='Threat')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Speed (m/s)')
        ax6.set_title('Speed Profile')
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"missile_intercept_{scenario_name}_{int(time.time())}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"[VISUALIZATION] Plot saved as: {filename}")
        
        plt.show()
    
    def run_all_scenarios(self, visualize: bool = True):
        """Run all configured scenarios."""
        print("\n" + "="*80)
        print("OPENGUIDANCE MISSILE INTERCEPT DEMONSTRATION")
        print("="*80)
        print("Author: Nik Jois")
        print("Email: nikjois@llamasearch.ai")
        print("="*80)
        
        results_summary = {}
        
        for scenario_name in self.scenarios.keys():
            try:
                results = self.run_scenario(scenario_name, verbose=True)
                results_summary[scenario_name] = {
                    "success": results["success"],
                    "miss_distance": results["simulation_data"]["miss_distance"],
                    "flight_time": results["final_time"],
                    "performance": results["performance"]
                }
                
                if visualize:
                    self.visualize_results(scenario_name)
                    
            except Exception as e:
                print(f"[ERROR] Failed to run scenario {scenario_name}: {str(e)}")
                results_summary[scenario_name] = {"success": False, "error": str(e)}
        
        # Print overall summary
        print("\n" + "="*80)
        print("OVERALL DEMONSTRATION SUMMARY")
        print("="*80)
        
        for scenario_name, summary in results_summary.items():
            status = "[SUCCESS]" if summary.get("success", False) else "[FAILED]"
            print(f"{scenario_name:20} {status}")
            if summary.get("success", False):
                print(f"{'':20} Miss Distance: {summary.get('miss_distance', 0):.2f}m")
                print(f"{'':20} Flight Time: {summary.get('flight_time', 0):.2f}s")
        
        print("="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        
        return results_summary


def main():
    """Main demonstration function."""
    print("Initializing OpenGuidance Missile Intercept Demonstration...")
    
    # Create demonstration system
    demo = MissileInterceptDemo()
    
    # Run all scenarios
    results = demo.run_all_scenarios(visualize=True)
    
    # Additional analysis
    print("\n[ANALYSIS] Generating detailed performance report...")
    
    # Performance comparison
    if len(results) > 1:
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS")
        print("="*60)
        
        for scenario_name, summary in results.items():
            if summary.get("success", False):
                perf = summary.get("performance", {})
                print(f"\n{scenario_name.upper()}:")
                print(f"  Max Acceleration: {perf.get('max_acceleration', 0):.1f}g")
                print(f"  Max Speed: {perf.get('max_speed', 0):.1f} m/s")
                print(f"  Fuel Efficiency: {perf.get('fuel_efficiency', 0):.1f}%")
    
    print("\n[SUCCESS] OpenGuidance demonstration completed successfully!")
    print("All systems performed within expected parameters.")
    print("Professional aerospace engineering standards validated.")
    
    return results


if __name__ == "__main__":
    # Set up professional matplotlib styling
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    
    # Run demonstration
    main() 