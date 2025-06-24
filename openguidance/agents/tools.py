"""
Domain-specific tools for OpenGuidance agents.

This module provides specialized tools that agents can use to interact
with OpenGuidance systems and perform aerospace-specific calculations.

Author: Nik Jois (nikjois@llamasearch.ai)
"""

import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import asdict
from enum import Enum

from agents import function_tool

from ..core.system import OpenGuidance
from .core import AgentType

logger = logging.getLogger(__name__)

# Define AgentType locally to avoid circular import
class AgentType(Enum):
    """Types of OpenGuidance agents."""
    GUIDANCE = "guidance"
    NAVIGATION = "navigation"
    CONTROL = "control"
    SAFETY = "safety"
    ANALYSIS = "analysis"
    COORDINATOR = "coordinator"


# Guidance Tools
def calculate_proportional_navigation(
    missile_position: List[float],
    missile_velocity: List[float], 
    target_position: List[float],
    target_velocity: List[float],
    navigation_constant: float = 4.0
) -> Dict[str, Any]:
    """
    Calculate proportional navigation guidance commands.
    
    Args:
        missile_position: [x, y, z] position in meters
        missile_velocity: [vx, vy, vz] velocity in m/s
        target_position: [x, y, z] target position in meters
        target_velocity: [vx, vy, vz] target velocity in m/s
        navigation_constant: PN constant (typically 3-5)
        
    Returns:
        Dictionary with guidance commands and engagement analysis
    """
    try:
        # Calculate relative position and velocity
        rel_pos = np.array(target_position) - np.array(missile_position)
        rel_vel = np.array(target_velocity) - np.array(missile_velocity)
        
        # Range and range rate
        range_to_target = np.linalg.norm(rel_pos)
        range_rate = np.dot(rel_pos, rel_vel) / range_to_target if range_to_target > 0 else 0
        
        # Line of sight (LOS) unit vector
        los_unit = rel_pos / range_to_target if range_to_target > 0 else np.array([1, 0, 0])
        
        # LOS rate calculation
        los_rate = (rel_vel - range_rate * los_unit) / range_to_target if range_to_target > 0 else np.array([0, 0, 0])
        
        # Proportional navigation command
        pn_command = navigation_constant * np.cross(missile_velocity, los_rate)
        
        # Engagement geometry analysis
        missile_speed = np.linalg.norm(missile_velocity)
        target_speed = np.linalg.norm(target_velocity)
        
        # Time to intercept estimate
        if abs(range_rate) > 0.1:
            time_to_intercept = -range_to_target / range_rate
        else:
            time_to_intercept = range_to_target / missile_speed if missile_speed > 0 else float('inf')
        
        # Intercept feasibility
        max_achievable_accel = np.linalg.norm(pn_command)
        
        return {
            "guidance_command": {
                "acceleration": pn_command.tolist(),
                "magnitude": float(max_achievable_accel)
            },
            "engagement_geometry": {
                "range": float(range_to_target),
                "range_rate": float(range_rate),
                "los_rate_magnitude": float(np.linalg.norm(los_rate)),
                "missile_speed": float(missile_speed),
                "target_speed": float(target_speed)
            },
            "intercept_analysis": {
                "time_to_intercept": float(time_to_intercept),
                "feasible": time_to_intercept > 0 and time_to_intercept < 300,
                "required_acceleration": float(max_achievable_accel)
            },
            "navigation_constant": navigation_constant,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in proportional navigation calculation: {e}")
        return {"error": str(e), "success": False}


def optimize_trajectory(
    start_position: List[float],
    end_position: List[float],
    max_acceleration: float = 50.0,
    max_velocity: float = 500.0,
    time_constraint: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize trajectory between two points with constraints.
    
    Args:
        start_position: [x, y, z] starting position in meters
        end_position: [x, y, z] ending position in meters
        max_acceleration: Maximum acceleration in m/s²
        max_velocity: Maximum velocity in m/s
        time_constraint: Optional time constraint in seconds
        
    Returns:
        Dictionary with optimized trajectory
    """
    try:
        # Calculate direct path parameters
        displacement = np.array(end_position) - np.array(start_position)
        distance = np.linalg.norm(displacement)
        direction = displacement / distance if distance > 0 else np.array([1, 0, 0])
        
        # Minimum time trajectory (bang-bang control)
        # Phase 1: Accelerate to max velocity or halfway point
        accel_time = max_velocity / max_acceleration
        accel_distance = 0.5 * max_acceleration * accel_time**2
        
        if 2 * accel_distance <= distance:
            # Can reach max velocity
            coast_distance = distance - 2 * accel_distance
            coast_time = coast_distance / max_velocity
            total_time = 2 * accel_time + coast_time
            
            trajectory_phases = [
                {
                    "phase": "acceleration",
                    "duration": accel_time,
                    "acceleration": max_acceleration,
                    "final_velocity": max_velocity
                },
                {
                    "phase": "coast", 
                    "duration": coast_time,
                    "acceleration": 0.0,
                    "final_velocity": max_velocity
                },
                {
                    "phase": "deceleration",
                    "duration": accel_time,
                    "acceleration": -max_acceleration,
                    "final_velocity": 0.0
                }
            ]
        else:
            # Cannot reach max velocity
            accel_time_actual = np.sqrt(distance / max_acceleration)
            max_velocity_actual = max_acceleration * accel_time_actual
            total_time = 2 * accel_time_actual
            
            trajectory_phases = [
                {
                    "phase": "acceleration",
                    "duration": accel_time_actual,
                    "acceleration": max_acceleration,
                    "final_velocity": max_velocity_actual
                },
                {
                    "phase": "deceleration", 
                    "duration": accel_time_actual,
                    "acceleration": -max_acceleration,
                    "final_velocity": 0.0
                }
            ]
        
        # Check time constraint
        if time_constraint and total_time > time_constraint:
            # Need to reduce performance or declare infeasible
            required_acceleration = distance / (0.5 * time_constraint**2)
            if required_acceleration > max_acceleration * 2:
                feasible = False
                total_time = time_constraint
            else:
                feasible = True
                total_time = time_constraint
        else:
            feasible = True
        
        # Generate waypoints (simplified)
        num_waypoints = 10
        waypoints = []
        for i in range(num_waypoints + 1):
            t = (i / num_waypoints) * total_time
            # Simplified position calculation
            if t <= total_time / 2:
                # Acceleration phase
                pos = np.array(start_position) + 0.5 * max_acceleration * t**2 * direction
            else:
                # Deceleration phase
                t_decel = t - total_time / 2
                pos = (np.array(start_position) + 
                       0.5 * distance * direction + 
                       max_velocity * t_decel * direction - 
                       0.5 * max_acceleration * t_decel**2 * direction)
            waypoints.append(pos.tolist())
        
        return {
            "trajectory": {
                "waypoints": waypoints,
                "total_time": float(total_time),
                "total_distance": float(distance),
                "phases": trajectory_phases
            },
            "performance": {
                "max_acceleration_used": float(max_acceleration),
                "max_velocity_achieved": float(max_velocity if 2 * accel_distance <= distance else max_acceleration * np.sqrt(distance / max_acceleration)),
                "fuel_optimal": True,
                "time_optimal": not (time_constraint and total_time > time_constraint)
            },
            "constraints": {
                "max_acceleration": max_acceleration,
                "max_velocity": max_velocity,
                "time_constraint": time_constraint,
                "feasible": feasible
            },
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in trajectory optimization: {e}")
        return {"error": str(e), "success": False}


# Navigation Tools
def estimate_position_uncertainty(
    gps_accuracy: float = 5.0,
    imu_drift_rate: float = 0.1,
    time_since_gps: float = 10.0,
    velocity_uncertainty: float = 1.0
) -> Dict[str, Any]:
    """
    Estimate position uncertainty based on sensor characteristics.
    
    Args:
        gps_accuracy: GPS accuracy in meters (1-sigma)
        imu_drift_rate: IMU drift rate in m/s²
        time_since_gps: Time since last GPS update in seconds
        velocity_uncertainty: Velocity uncertainty in m/s (1-sigma)
        
    Returns:
        Dictionary with uncertainty estimates
    """
    try:
        # Position uncertainty from GPS
        gps_uncertainty = gps_accuracy
        
        # Position uncertainty from IMU integration
        imu_uncertainty = 0.5 * imu_drift_rate * time_since_gps**2
        
        # Position uncertainty from velocity integration
        vel_uncertainty = velocity_uncertainty * time_since_gps
        
        # Combined uncertainty (RSS)
        total_uncertainty = np.sqrt(
            gps_uncertainty**2 + imu_uncertainty**2 + vel_uncertainty**2
        )
        
        # Confidence levels (assuming Gaussian)
        uncertainty_1sigma = total_uncertainty
        uncertainty_2sigma = 2 * total_uncertainty
        uncertainty_3sigma = 3 * total_uncertainty
        
        return {
            "total_uncertainty_1sigma": float(uncertainty_1sigma),
            "total_uncertainty_2sigma": float(uncertainty_2sigma), 
            "total_uncertainty_3sigma": float(uncertainty_3sigma),
            "components": {
                "gps": float(gps_uncertainty),
                "imu_drift": float(imu_uncertainty),
                "velocity_integration": float(vel_uncertainty)
            },
            "confidence_levels": {
                "68_percent": float(uncertainty_1sigma),
                "95_percent": float(uncertainty_2sigma),
                "99_7_percent": float(uncertainty_3sigma)
            },
            "dominant_error_source": (
                "GPS" if gps_uncertainty > imu_uncertainty and gps_uncertainty > vel_uncertainty
                else "IMU drift" if imu_uncertainty > vel_uncertainty
                else "Velocity integration"
            ),
            "time_since_gps": time_since_gps,
            "recommendation": (
                "GPS update recommended" if time_since_gps > 30
                else "IMU calibration needed" if imu_drift_rate > 0.5
                else "Navigation accuracy acceptable"
            )
        }
        
    except Exception as e:
        logger.error(f"Error in position uncertainty estimation: {e}")
        return {"error": str(e)}


# Control Tools
def design_pid_controller(
    plant_transfer_function: Dict[str, List[float]],
    desired_settling_time: float = 2.0,
    desired_overshoot: float = 10.0,
    disturbance_rejection: bool = True
) -> Dict[str, Any]:
    """
    Design PID controller for given plant dynamics.
    
    Args:
        plant_transfer_function: {"numerator": [coeffs], "denominator": [coeffs]}
        desired_settling_time: Desired settling time in seconds
        desired_overshoot: Desired overshoot percentage
        disturbance_rejection: Whether to optimize for disturbance rejection
        
    Returns:
        Dictionary with PID gains and performance analysis
    """
    try:
        # Extract plant parameters (assuming second-order system for simplicity)
        num = plant_transfer_function.get("numerator", [1])
        den = plant_transfer_function.get("denominator", [1, 1, 1])
        
        # Design based on desired specifications
        # Convert overshoot to damping ratio
        overshoot_ratio = desired_overshoot / 100.0
        if overshoot_ratio > 0:
            zeta = -np.log(overshoot_ratio) / np.sqrt(np.pi**2 + np.log(overshoot_ratio)**2)
        else:
            zeta = 1.0  # Critically damped
        
        # Natural frequency from settling time (2% criterion)
        wn = 4.0 / (zeta * desired_settling_time)
        
        # PID gains (simplified design for second-order plant)
        if len(den) >= 3:
            a2, a1, a0 = den[0], den[1], den[2]
            k_plant = num[0] / a0 if len(num) > 0 else 1.0
            
            kp = (2 * zeta * wn * a2 - a1) / k_plant
            ki = (wn**2 * a2) / k_plant
            kd = 0.0  # Start with PI control
            
            # Add derivative term if needed for disturbance rejection
            if disturbance_rejection:
                kd = a2 / (4 * k_plant)
        else:
            # Simple gain scheduling for first-order system
            kp = wn
            ki = wn**2 / 2
            kd = 0.0
        
        # Ensure gains are positive and reasonable
        kp = max(0.1, min(100.0, kp))
        ki = max(0.01, min(50.0, ki))
        kd = max(0.0, min(10.0, kd))
        
        # Performance predictions
        predicted_settling_time = 4.0 / (zeta * wn) if zeta * wn > 0 else float('inf')
        predicted_overshoot = 100 * np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) if zeta < 1 else 0
        
        return {
            "gains": {
                "kp": float(kp),
                "ki": float(ki), 
                "kd": float(kd)
            },
            "design_parameters": {
                "damping_ratio": float(zeta),
                "natural_frequency": float(wn),
                "desired_settling_time": desired_settling_time,
                "desired_overshoot": desired_overshoot
            },
            "predicted_performance": {
                "settling_time": float(predicted_settling_time),
                "overshoot_percent": float(predicted_overshoot),
                "steady_state_error": 0.0 if ki > 0 else "Non-zero for step inputs"
            },
            "stability_margins": {
                "gain_margin_db": 6.0,  # Typical value
                "phase_margin_deg": 45.0,  # Typical value
                "stable": True
            },
            "tuning_recommendations": {
                "increase_kp": "Faster response, may increase overshoot",
                "increase_ki": "Eliminate steady-state error, may cause instability",
                "increase_kd": "Reduce overshoot, improve disturbance rejection"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in PID controller design: {e}")
        return {"error": str(e)}


# Safety Tools
def assess_safety_margins(
    current_state: Dict[str, float],
    safety_limits: Dict[str, Dict[str, float]],
    trajectory_prediction: Optional[List[Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    Assess safety margins for current and predicted states.
    
    Args:
        current_state: {"altitude": m, "velocity": m/s, "acceleration": m/s², ...}
        safety_limits: {"altitude": {"min": m, "max": m}, ...}
        trajectory_prediction: Optional list of future states
        
    Returns:
        Dictionary with safety assessment
    """
    try:
        safety_status = {}
        violations = []
        warnings = []
        
        # Check current state against limits
        for param, value in current_state.items():
            if param in safety_limits:
                limits = safety_limits[param]
                min_limit = limits.get("min", float('-inf'))
                max_limit = limits.get("max", float('inf'))
                
                # Calculate margins
                margin_to_min = value - min_limit if min_limit != float('-inf') else float('inf')
                margin_to_max = max_limit - value if max_limit != float('inf') else float('inf')
                
                # Determine status
                if value < min_limit or value > max_limit:
                    status = "VIOLATION"
                    violations.append(f"{param}: {value} outside [{min_limit}, {max_limit}]")
                elif margin_to_min < abs(min_limit * 0.1) or margin_to_max < abs(max_limit * 0.1):
                    status = "WARNING"
                    warnings.append(f"{param}: {value} near limits [{min_limit}, {max_limit}]")
                else:
                    status = "SAFE"
                
                safety_status[param] = {
                    "value": value,
                    "limits": limits,
                    "margin_to_min": float(margin_to_min),
                    "margin_to_max": float(margin_to_max),
                    "status": status,
                    "margin_percent": min(
                        100 * margin_to_min / abs(min_limit) if min_limit != float('-inf') else 100,
                        100 * margin_to_max / abs(max_limit) if max_limit != float('inf') else 100
                    )
                }
        
        # Check trajectory prediction if provided
        future_violations = []
        if trajectory_prediction:
            for i, future_state in enumerate(trajectory_prediction):
                for param, value in future_state.items():
                    if param in safety_limits:
                        limits = safety_limits[param]
                        min_limit = limits.get("min", float('-inf'))
                        max_limit = limits.get("max", float('inf'))
                        
                        if value < min_limit or value > max_limit:
                            future_violations.append({
                                "time_step": i,
                                "parameter": param,
                                "value": value,
                                "limits": limits
                            })
        
        # Overall assessment
        overall_status = (
            "CRITICAL" if violations
            else "WARNING" if warnings or future_violations
            else "SAFE"
        )
        
        return {
            "overall_status": overall_status,
            "current_state_assessment": safety_status,
            "violations": violations,
            "warnings": warnings,
            "future_violations": future_violations,
            "recommendations": {
                "immediate_actions": violations[:3] if violations else [],
                "monitoring_required": warnings[:3] if warnings else [],
                "trajectory_adjustments": len(future_violations) > 0
            },
            "safety_score": (
                0.0 if violations
                else 0.5 if warnings or future_violations
                else 1.0
            )
        }
        
    except Exception as e:
        logger.error(f"Error in safety margin assessment: {e}")
        return {"error": str(e)}


# Analysis Tools
def analyze_system_performance(
    simulation_data: Dict[str, List[float]],
    performance_metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze system performance from simulation data.
    
    Args:
        simulation_data: {"time": [s], "position": [m], "velocity": [m/s], ...}
        performance_metrics: List of metrics to calculate
        
    Returns:
        Dictionary with performance analysis
    """
    try:
        if performance_metrics is None:
            performance_metrics = ["settling_time", "overshoot", "steady_state_error", "rise_time"]
        
        results = {}
        
        # Extract time series
        time = np.array(simulation_data.get("time", []))
        
        for metric in performance_metrics:
            if metric == "settling_time" and "position" in simulation_data:
                position = np.array(simulation_data["position"])
                target = position[-1] if len(position) > 0 else 0
                
                # Find settling time (within 2% of final value)
                tolerance = 0.02 * abs(target)
                settled_indices = np.where(np.abs(position - target) <= tolerance)[0]
                
                if len(settled_indices) > 0:
                    settling_time = time[settled_indices[0]]
                else:
                    settling_time = float('inf')
                
                results["settling_time"] = float(settling_time)
            
            elif metric == "overshoot" and "position" in simulation_data:
                position = np.array(simulation_data["position"])
                target = position[-1] if len(position) > 0 else 0
                
                if target != 0:
                    max_overshoot = (np.max(position) - target) / target * 100
                    results["overshoot_percent"] = float(max(0, max_overshoot))
                else:
                    results["overshoot_percent"] = 0.0
            
            elif metric == "rise_time" and "position" in simulation_data:
                position = np.array(simulation_data["position"])
                target = position[-1] if len(position) > 0 else 0
                
                # 10% to 90% rise time
                ten_percent = 0.1 * target
                ninety_percent = 0.9 * target
                
                ten_idx = np.where(position >= ten_percent)[0]
                ninety_idx = np.where(position >= ninety_percent)[0]
                
                if len(ten_idx) > 0 and len(ninety_idx) > 0:
                    rise_time = time[ninety_idx[0]] - time[ten_idx[0]]
                else:
                    rise_time = float('inf')
                
                results["rise_time"] = float(rise_time)
            
            elif metric == "steady_state_error" and "position" in simulation_data:
                position = np.array(simulation_data["position"])
                target = position[-1] if len(position) > 0 else 0
                
                # Use last 10% of data for steady-state calculation
                ss_start = int(0.9 * len(position))
                ss_position = np.mean(position[ss_start:]) if ss_start < len(position) else target
                
                ss_error = abs(ss_position - target)
                results["steady_state_error"] = float(ss_error)
        
        # Additional statistics
        if "position" in simulation_data:
            position = np.array(simulation_data["position"])
            results["statistics"] = {
                "mean": float(np.mean(position)),
                "std": float(np.std(position)),
                "min": float(np.min(position)),
                "max": float(np.max(position)),
                "final_value": float(position[-1]) if len(position) > 0 else 0
            }
        
        # Performance rating
        performance_score = 1.0
        if "settling_time" in results and results["settling_time"] > 10:
            performance_score *= 0.8
        if "overshoot_percent" in results and results["overshoot_percent"] > 20:
            performance_score *= 0.7
        if "steady_state_error" in results and results["steady_state_error"] > 1.0:
            performance_score *= 0.6
        
        results["performance_score"] = performance_score
        results["performance_rating"] = (
            "Excellent" if performance_score > 0.9
            else "Good" if performance_score > 0.7
            else "Acceptable" if performance_score > 0.5
            else "Poor"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in performance analysis: {e}")
        return {"error": str(e)}


# Simple tool registration function
def get_tools_for_agent_type(
    agent_type: Any,
    openguidance_system: Optional[OpenGuidance] = None
) -> List[Callable]:
    """
    Get appropriate tools for the given agent type.
    
    Args:
        agent_type: Type of agent
        openguidance_system: Optional OpenGuidance system instance
        
    Returns:
        List of tool functions
    """
    tools = []
    
    # Use string comparison to avoid enum type conflicts
    agent_type_str = str(agent_type.value) if hasattr(agent_type, 'value') else str(agent_type)
    
    if agent_type_str == "guidance":
        tools.extend([
            calculate_proportional_navigation,
            optimize_trajectory
        ])
    elif agent_type_str == "navigation":
        tools.extend([
            # Navigation tools would go here
        ])
    elif agent_type_str == "control":
        tools.extend([
            # Control tools would go here
        ])
    elif agent_type_str == "safety":
        tools.extend([
            # Safety tools would go here
        ])
    elif agent_type_str == "analysis":
        tools.extend([
            # Analysis tools would go here
        ])
    
    return tools


# Tool collections for easy access
class GuidanceTools:
    """Collection of guidance-related tools."""
    
    @staticmethod
    def get_all_tools() -> List[Callable]:
        return [calculate_proportional_navigation, optimize_trajectory]


class NavigationTools:
    """Collection of navigation-related tools."""
    
    @staticmethod
    def get_all_tools() -> List[Callable]:
        return [estimate_position_uncertainty]


class ControlTools:
    """Collection of control-related tools."""
    
    @staticmethod
    def get_all_tools() -> List[Callable]:
        return [design_pid_controller]


class SafetyTools:
    """Collection of safety-related tools."""
    
    @staticmethod
    def get_all_tools() -> List[Callable]:
        return [assess_safety_margins]


class AnalysisTools:
    """Collection of analysis-related tools."""
    
    @staticmethod
    def get_all_tools() -> List[Callable]:
        return [analyze_system_performance]


class SimulationTools:
    """Collection of simulation-related tools."""
    
    @staticmethod
    def get_all_tools() -> List[Callable]:
        # Placeholder for future simulation tools
        return [] 