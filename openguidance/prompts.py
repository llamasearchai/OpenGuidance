"""
Advanced prompt management system with dynamic templating and optimization.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import re
import json
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Advanced prompt template with variable substitution and validation."""
    
    name: str
    template: str
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Extract variables from template after initialization."""
        self.variables = self._extract_variables()
    
    def _extract_variables(self) -> List[str]:
        """Extract variable names from template using regex."""
        pattern = r'\{\{(\w+)\}\}'
        return list(set(re.findall(pattern, self.template)))
    
    def render(self, **kwargs) -> str:
        """Render template with provided variables."""
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        rendered = self.template
        for var, value in kwargs.items():
            rendered = rendered.replace(f"{{{{{var}}}}}", str(value))
        
        return rendered
    
    def validate_variables(self, **kwargs) -> Dict[str, Any]:
        """Validate provided variables against template requirements."""
        result = {
            "valid": True,
            "missing": [],
            "extra": [],
            "errors": []
        }
        
        provided_vars = set(kwargs.keys())
        required_vars = set(self.variables)
        
        result["missing"] = list(required_vars - provided_vars)
        result["extra"] = list(provided_vars - required_vars)
        
        if result["missing"]:
            result["valid"] = False
            result["errors"].append(f"Missing variables: {result['missing']}")
        
        return result


class DynamicPrompt:
    """Dynamic prompt that adapts based on context and conditions."""
    
    def __init__(self, base_template: PromptTemplate):
        self.base_template = base_template
        self.conditions: List[Callable] = []
        self.transformations: List[Callable] = []
        self.adaptations: Dict[str, Dict[str, Any]] = {}
    
    def add_condition(self, condition: Callable[[Dict], bool]) -> 'DynamicPrompt':
        """Add condition for dynamic adaptation."""
        self.conditions.append(condition)
        return self
    
    def add_transformation(self, transform: Callable[[str], str]) -> 'DynamicPrompt':
        """Add transformation function."""
        self.transformations.append(transform)
        return self
    
    def add_adaptation(self, name: str, template: PromptTemplate, condition: Callable) -> 'DynamicPrompt':
        """Add conditional template adaptation."""
        adaptation_data = {
            "template": template,
            "condition": condition
        }
        self.adaptations[name] = adaptation_data
        return self
    
    def render(self, context: Dict[str, Any], **kwargs) -> str:
        """Render dynamic prompt based on context."""
        # Select appropriate template
        selected_template = self.base_template
        
        for name, adaptation in self.adaptations.items():
            if adaptation["condition"](context):
                selected_template = adaptation["template"]
                logger.debug(f"Using adaptation: {name}")
                break
        
        # Render base content
        content = selected_template.render(**kwargs)
        
        # Apply transformations
        for transform in self.transformations:
            content = transform(content)
        
        return content


class PromptManager:
    """Comprehensive prompt management system with versioning and optimization."""
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.templates: Dict[str, Dict[str, PromptTemplate]] = {}
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the prompt manager."""
        if self.is_initialized:
            logger.warning("PromptManager already initialized")
            return
        
        # Load all predefined templates (system + GNC framework)
        for template in ALL_TEMPLATES.values():
            self.register_template(template)
        
        self.is_initialized = True
        logger.info("PromptManager initialization completed")
    
    async def cleanup(self) -> None:
        """Cleanup prompt manager resources."""
        if not self.is_initialized:
            return
        
        # Cleanup any resources here
        self.is_initialized = False
        logger.info("PromptManager cleanup completed")
    
    def register_template(self, template: PromptTemplate) -> None:
        """Register a new prompt template."""
        if template.name not in self.templates:
            self.templates[template.name] = {}
        
        self.templates[template.name][template.version] = template
        self.usage_stats[f"{template.name}:{template.version}"] = {
            "usage_count": 0,
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "last_used": None
        }
        
        logger.info(f"Registered template: {template.name} v{template.version}")
    
    def get_template(self, name: str, version: Optional[str] = None) -> PromptTemplate:
        """Retrieve prompt template by name and version."""
        if name not in self.templates:
            raise ValueError(f"Template not found: {name}")
        
        if version is None:
            # Get latest version
            version = max(self.templates[name].keys())
        
        if version not in self.templates[name]:
            raise ValueError(f"Template version not found: {name} v{version}")
        
        return self.templates[name][version]
    
    def list_templates(self) -> Dict[str, List[str]]:
        """List all templates with their versions."""
        return {name: list(versions.keys()) for name, versions in self.templates.items()}
    
    def record_usage(self, name: str, version: str, success: bool, response_time: float) -> None:
        """Record template usage for analytics."""
        key = f"{name}:{version}"
        if key not in self.usage_stats:
            return
        
        stats = self.usage_stats[key]
        stats["usage_count"] += 1
        stats["last_used"] = datetime.utcnow().isoformat()
        
        # Update success rate
        current_successes = stats["success_rate"] * (stats["usage_count"] - 1)
        new_successes = current_successes + (1 if success else 0)
        stats["success_rate"] = new_successes / stats["usage_count"]
        
        # Update average response time
        current_total_time = stats["avg_response_time"] * (stats["usage_count"] - 1)
        new_total_time = current_total_time + response_time
        stats["avg_response_time"] = new_total_time / stats["usage_count"]
    
    def get_best_template(self, name: str, metric: str = "success_rate") -> PromptTemplate:
        """Get best performing version of a template."""
        if name not in self.templates:
            raise ValueError(f"Template not found: {name}")
        
        best_version = None
        best_score = -1
        
        for version in self.templates[name].keys():
            key = f"{name}:{version}"
            if key in self.usage_stats and self.usage_stats[key]["usage_count"] > 0:
                score = self.usage_stats[key].get(metric, 0)
                if score > best_score:
                    best_score = score
                    best_version = version
        
        if best_version is None:
            best_version = max(self.templates[name].keys())
        
        return self.templates[name][best_version]
    
    def export_templates(self, filename: str) -> None:
        """Export templates to JSON file."""
        export_data = {}
        for name, versions in self.templates.items():
            export_data[name] = {}
            for version, template in versions.items():
                export_data[name][version] = {
                    "template": template.template,
                    "variables": template.variables,
                    "metadata": template.metadata,
                    "version": template.version,
                    "created_at": template.created_at.isoformat(),
                    "tags": template.tags
                }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Templates exported to: {filename}")
    
    def get_export_data(self) -> Dict[str, Any]:
        """Get template export data without writing to file."""
        export_data = {}
        for name, versions in self.templates.items():
            export_data[name] = {}
            for version, template in versions.items():
                export_data[name][version] = {
                    "template": template.template,
                    "variables": template.variables,
                    "metadata": template.metadata,
                    "version": template.version,
                    "created_at": template.created_at.isoformat(),
                    "tags": template.tags
                }
        return export_data
    
    def import_templates(self, filename: str) -> None:
        """Import templates from JSON file."""
        with open(filename, 'r') as f:
            import_data = json.load(f)
        
        for name, versions in import_data.items():
            for version, data in versions.items():
                template = PromptTemplate(
                    name=name,
                    template=data["template"],
                    metadata=data.get("metadata", {}),
                    version=data.get("version", version),
                    created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
                    tags=data.get("tags", [])
                )
                self.register_template(template)
        
        logger.info(f"Templates imported from: {filename}")


# Predefined professional templates
SYSTEM_TEMPLATES = {
    "code_review": PromptTemplate(
        name="code_review",
        template="""You are an expert code reviewer. Please analyze the following code:

{{code}}

Provide feedback on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Security implications
5. Suggestions for improvement

Language: {{language}}
Context: {{context}}""",
        tags=["code", "review", "analysis"]
    ),
    
    "technical_documentation": PromptTemplate(
        name="technical_documentation",
        template="""Create comprehensive technical documentation for:

Topic: {{topic}}
Audience: {{audience}}
Complexity Level: {{complexity}}

Requirements:
- Clear explanations with examples
- Proper structure and formatting
- Code samples where applicable
- Best practices and guidelines

Additional Context: {{context}}""",
        tags=["documentation", "technical", "writing"]
    ),
    
    "problem_solving": PromptTemplate(
        name="problem_solving",
        template="""Analyze and solve the following problem:

Problem: {{problem}}
Domain: {{domain}}
Constraints: {{constraints}}

Please provide:
1. Problem analysis and breakdown
2. Proposed solution approach
3. Implementation details
4. Potential challenges and mitigations
5. Alternative approaches

Context: {{context}}""",
        tags=["problem-solving", "analysis", "solution"]
    ),
    
    "system_design": PromptTemplate(
        name="system_design",
        template="""Design a system for the following requirements:

System: {{system_name}}
Requirements: {{requirements}}
Scale: {{scale}}
Constraints: {{constraints}}

Provide:
1. High-level architecture
2. Component breakdown
3. Data flow and storage
4. Scalability considerations
5. Technology recommendations
6. Trade-offs and alternatives

Context: {{context}}""",
        tags=["architecture", "design", "system"]
    )
}

# ---------------------------------------------------------------------------
# OpenGuidance GNC Framework Prompt Templates (auto-generated)
# ---------------------------------------------------------------------------

GNC_PROMPTS = {
    "core_data_structures_validation": PromptTemplate(
        name="core_data_structures_validation",
        template="""Prompt 1: Core Data Structures Validation
Implement comprehensive unit tests for the core State, Control, Vehicle, and Mission data classes, focusing on validation of quaternion operations, coordinate frame transformations, state interpolation methods, and serialization/deserialization functionality, ensuring all edge cases like gimbal lock, quaternion normalization, and frame conversion accuracy are thoroughly tested with both synthetic and real-world aerospace data scenarios.""",
        tags=["gnc", "testing", "core"],
    ),
    "configuration_management_system": PromptTemplate(
        name="configuration_management_system",
        template="""Prompt 2: Configuration Management System
Develop a robust configuration management system with hierarchical YAML/JSON support, environment variable overrides, schema validation using Pydantic, and dynamic configuration updates, including comprehensive testing of configuration inheritance, validation error handling, type coercion, and performance benchmarking to ensure sub-millisecond configuration access times suitable for real-time control systems.""",
        tags=["gnc", "configuration"],
    ),
    "safety_critical_type_system": PromptTemplate(
        name="safety_critical_type_system",
        template="""Prompt 3: Safety-Critical Type System
Create a type-safe wrapper system around all numerical computations with automatic unit checking, bounds validation, and overflow protection, implementing custom decorators and context managers that enforce aerospace safety standards, validate physical constraints (like maximum g-forces), and provide detailed error reporting with stack traces for debugging safety-critical control law failures.""",
        tags=["gnc", "safety", "types"],
    ),
    "aircraft_dynamics_engine": PromptTemplate(
        name="aircraft_dynamics_engine",
        template="""Prompt 4: 6-DOF Aircraft Dynamics Engine
Implement a high-fidelity 6-DOF aircraft dynamics model with full aerodynamic coefficient lookup tables, engine thrust modeling, landing gear dynamics, and atmospheric effects, ensuring numerical stability through adaptive Runge-Kutta integration, validation against NASA test data, and performance optimization using Numba JIT compilation to achieve real-time execution speeds faster than 1kHz simulation rates.""",
        tags=["gnc", "dynamics", "aircraft"],
    ),
    "missile_dynamics_tvc": PromptTemplate(
        name="missile_dynamics_tvc",
        template="""Prompt 5: Missile Dynamics with TVC
Develop a comprehensive missile dynamics model incorporating thrust vector control, fin aerodynamics, mass depletion effects, and flexible body dynamics, with extensive validation against published missile performance data, Monte Carlo analysis of manufacturing tolerances, and integration testing with guidance algorithms to ensure closed-loop stability and accurate intercept predictions.""",
        tags=["gnc", "dynamics", "missile"],
    ),
    "spacecraft_orbital_mechanics": PromptTemplate(
        name="spacecraft_orbital_mechanics",
        template="""Prompt 6: Spacecraft Orbital Mechanics
Create a precise spacecraft dynamics model with J2-J6 gravitational harmonics, atmospheric drag modeling, solar radiation pressure, and multi-body gravitational effects, implementing both Cartesian and orbital element propagation methods, validating against STK/GMAT results, and optimizing for long-duration orbital simulations with numerical stability over mission lifetimes.""",
        tags=["gnc", "dynamics", "spacecraft"],
    ),
    "quadrotor_multi_body_dynamics": PromptTemplate(
        name="quadrotor_multi_body_dynamics",
        template="""Prompt 7: Quadrotor Multi-Body Dynamics
Implement detailed quadrotor dynamics including rotor aerodynamics, motor dynamics, flexible frame effects, and ground effect modeling, with extensive validation through hardware-in-the-loop testing, parameter identification algorithms for system identification, and integration with advanced control algorithms like geometric controllers and adaptive schemes.""",
        tags=["gnc", "dynamics", "quadrotor"],
    ),
    "proportional_navigation_family": PromptTemplate(
        name="proportional_navigation_family",
        template="""Prompt 8: Proportional Navigation Family
Develop a complete suite of proportional navigation variants (PN, TPN, APN, OPN) with optimal navigation constant selection, noise robustness analysis, and performance comparison framework, including extensive Monte Carlo validation against maneuvering targets, miss distance statistics, and computational efficiency benchmarking to ensure real-time performance in embedded systems.""",
        tags=["gnc", "guidance", "navigation"],
    ),
    "optimal_guidance_laws": PromptTemplate(
        name="optimal_guidance_laws",
        template="""Prompt 9: Optimal Guidance Laws
Implement advanced optimal guidance algorithms including minimum-time, minimum-energy, and constrained guidance laws using direct and indirect optimization methods, with comprehensive validation against analytical solutions, performance comparison with classical methods, and robustness analysis under system uncertainties and modeling errors.""",
        tags=["gnc", "guidance", "optimal"],
    ),
    "trajectory_optimization_engine": PromptTemplate(
        name="trajectory_optimization_engine",
        template="""Prompt 10: Trajectory Optimization Engine
Create a sophisticated trajectory optimization framework supporting direct collocation, multiple shooting, and pseudospectral methods, with automatic mesh refinement, constraint handling, and multi-objective optimization capabilities, validating against known optimal control problems and benchmarking performance against commercial tools like OTIS and POST2.""",
        tags=["gnc", "optimization", "trajectory"],
    ),
    "path_following_waypoint_navigation": PromptTemplate(
        name="path_following_waypoint_navigation",
        template="""Prompt 11: Path Following and Waypoint Navigation
Develop robust path following algorithms including Dubins paths, Clothoid curves, and 3D trajectory following with collision avoidance, implementing lookahead distance optimization, crosstrack error minimization, and dynamic obstacle avoidance, with extensive testing in complex 3D environments and validation against flight test data.""",
        tags=["gnc", "guidance", "path"],
    ),
    "extended_kalman_filter_framework": PromptTemplate(
        name="extended_kalman_filter_framework",
        template="""Prompt 12: Extended Kalman Filter Framework
Implement a flexible EKF framework supporting multiple vehicle types with automatic Jacobian computation, adaptive noise tuning, and numerical stability enhancements, including comprehensive validation against known nonlinear estimation problems, comparison with particle filters, and performance analysis under various sensor failure scenarios.""",
        tags=["gnc", "navigation", "ekf"],
    ),
    "unscented_kalman_filter": PromptTemplate(
        name="unscented_kalman_filter",
        template="""Prompt 13: Unscented Kalman Filter Implementation
Develop a robust UKF implementation with sigma point selection optimization, scaled unscented transform, and square-root filtering for numerical stability, extensively testing against highly nonlinear systems, comparing performance with EKF, and validating estimation accuracy for spacecraft attitude determination and aircraft navigation scenarios.""",
        tags=["gnc", "navigation", "ukf"],
    ),
    "multi_sensor_fusion_architecture": PromptTemplate(
        name="multi_sensor_fusion_architecture",
        template="""Prompt 14: Multi-Sensor Fusion Architecture
Create a sophisticated sensor fusion system supporting IMU, GPS, vision, radar, and LiDAR sensors with asynchronous measurement processing, sensor fault detection and isolation, and adaptive weighting schemes, implementing comprehensive testing with synthetic and real sensor data, and validating navigation accuracy in GPS-denied environments.""",
        tags=["gnc", "navigation", "fusion"],
    ),
    "slam_integration_module": PromptTemplate(
        name="slam_integration_module",
        template="""Prompt 15: SLAM Integration Module
Implement simultaneous localization and mapping algorithms tailored for aerospace applications, including visual-inertial SLAM for UAVs and landmark-based SLAM for planetary navigation, with extensive validation in challenging environments, computational efficiency optimization, and integration with existing navigation filters.""",
        tags=["gnc", "navigation", "slam"],
    ),
    "advanced_pid_controller_suite": PromptTemplate(
        name="advanced_pid_controller_suite",
        template="""Prompt 16: Advanced PID Controller Suite
Develop a comprehensive PID controller framework with auto-tuning capabilities, gain scheduling, anti-windup protection, and derivative filtering, implementing extensive tuning algorithms (Ziegler-Nichols, Cohen-Coon, genetic algorithms), validation against control benchmarks, and performance testing under actuator saturation and sensor noise conditions.""",
        tags=["gnc", "control", "pid"],
    ),
    "lqr_system": PromptTemplate(
        name="lqr_system",
        template="""Prompt 17: Linear Quadratic Regulator System
Create a robust LQR implementation with infinite and finite horizon variants, Kalman filter integration for LQG control, and gain scheduling for nonlinear systems, including comprehensive stability analysis, robustness margins calculation, and validation against classical control problems with performance comparison to industry-standard tools.""",
        tags=["gnc", "control", "lqr"],
    ),
    "mpc_engine": PromptTemplate(
        name="mpc_engine",
        template="""Prompt 18: Model Predictive Control Engine
Implement a high-performance MPC framework using CasADi optimization with real-time feasibility, constraint handling, and warm-start capabilities, extensively testing computational performance for various prediction horizons, validating trajectory tracking accuracy, and implementing adaptive MPC variants for uncertain systems.""",
        tags=["gnc", "control", "mpc"],
    ),
    "adaptive_and_robust_control": PromptTemplate(
        name="adaptive_and_robust_control",
        template="""Prompt 19: Adaptive and Robust Control
Develop advanced adaptive control algorithms including MRAC, L1 adaptive control, and robust H-infinity controllers with uncertainty quantification, extensive stability proofs, and validation against time-varying and uncertain systems, ensuring graceful degradation under modeling errors and disturbances.""",
        tags=["gnc", "control", "adaptive"],
    ),
    "high_fidelity_simulation_core": PromptTemplate(
        name="high_fidelity_simulation_core",
        template="""Prompt 20: High-Fidelity Simulation Core
Create a modular simulation engine supporting multiple integration methods, real-time execution, and hardware-in-the-loop capabilities, implementing comprehensive timing analysis, deterministic execution modes, and extensive validation of numerical accuracy against analytical solutions and commercial simulators.""",
        tags=["gnc", "simulation"],
    ),
    "monte_carlo_analysis_framework": PromptTemplate(
        name="monte_carlo_analysis_framework",
        template="""Prompt 21: Monte Carlo Analysis Framework
Develop a sophisticated Monte Carlo simulation system with Latin hypercube sampling, importance sampling, and parallel execution capabilities, implementing statistical analysis tools, confidence interval calculation, and automated report generation, with validation against known probability distributions and aerospace uncertainty models.""",
        tags=["gnc", "simulation", "monte_carlo"],
    ),
    "visualization_and_animation": PromptTemplate(
        name="visualization_and_animation",
        template="""Prompt 22: 3D Visualization and Animation
Implement advanced visualization capabilities using Plotly, Matplotlib, and OpenGL for real-time 3D trajectory display, vehicle animation, and data plotting, creating interactive dashboards, VR/AR integration capabilities, and automated video generation for presentation and analysis purposes.""",
        tags=["gnc", "visualization"],
    ),
    "hardware_in_loop_interface": PromptTemplate(
        name="hardware_in_loop_interface",
        template="""Prompt 23: Hardware-in-Loop Interface
Create a comprehensive HIL testing framework supporting multiple communication protocols (serial, UDP, TCP, CAN), real-time synchronization, and automated test sequence execution, implementing extensive protocol validation, timing analysis, and integration testing with commercial autopilot hardware.""",
        tags=["gnc", "hil"],
    ),
    "multi_disciplinary_optimization": PromptTemplate(
        name="multi_disciplinary_optimization",
        template="""Prompt 24: Multi-Disciplinary Optimization
Develop OpenMDAO integration with gradient-based and gradient-free optimization algorithms, supporting parallel execution, design space exploration, and Pareto frontier generation, implementing comprehensive benchmarking against known optimization problems and validation of convergence properties.""",
        tags=["gnc", "optimization", "mdao"],
    ),
    "trajectory_optimization_solvers": PromptTemplate(
        name="trajectory_optimization_solvers",
        template="""Prompt 25: Trajectory Optimization Solvers
Create interfaces to multiple optimization solvers (IPOPT, SNOPT, GEKKO) with automatic solver selection, constraint scaling, and convergence monitoring, implementing extensive performance comparison, numerical conditioning analysis, and robustness testing under various problem formulations.""",
        tags=["gnc", "optimization", "solvers"],
    ),
    "parameter_identification_system": PromptTemplate(
        name="parameter_identification_system",
        template="""Prompt 26: Parameter Identification System
Implement sophisticated parameter identification algorithms for system identification using flight test data, including maximum likelihood estimation, least squares methods, and Bayesian approaches, with extensive validation against known system parameters and uncertainty quantification.""",
        tags=["gnc", "system_id"],
    ),
    "safety_monitor_implementation": PromptTemplate(
        name="safety_monitor_implementation",
        template="""Prompt 27: Safety Monitor Implementation
Develop a comprehensive safety monitoring system with geofencing, envelope protection, and automatic recovery modes, implementing fault detection algorithms, redundancy management, and extensive testing of failure scenarios with automated safety system validation and certification support.""",
        tags=["gnc", "safety", "monitor"],
    ),
    "formal_verification_tools": PromptTemplate(
        name="formal_verification_tools",
        template="""Prompt 28: Formal Verification Tools
Create formal verification capabilities for control law validation using reachability analysis, model checking, and theorem proving techniques, implementing automated test case generation, property verification, and integration with safety-critical software development standards.""",
        tags=["gnc", "verification"],
    ),
    "fault_detection_isolation": PromptTemplate(
        name="fault_detection_isolation",
        template="""Prompt 29: Fault Detection and Isolation
Implement advanced FDI algorithms including analytical redundancy, model-based detection, and machine learning approaches, with extensive testing using fault injection, validation against known failure modes, and integration with reconfigurable control systems.""",
        tags=["gnc", "safety", "fdi"],
    ),
    "code_generator": PromptTemplate(
        name="code_generator",
        template="""Prompt 30: C/C++ Code Generator
Develop automatic code generation capabilities for real-time embedded systems, supporting MISRA-C compliance, fixed-point arithmetic, and memory optimization, implementing extensive validation of generated code against Python reference implementations and performance benchmarking on target hardware.""",
        tags=["gnc", "codegen"],
    ),
    "real_time_execution_framework": PromptTemplate(
        name="real_time_execution_framework",
        template="""Prompt 31: Real-Time Execution Framework
Create a real-time control executive with deterministic scheduling, memory management, and inter-task communication, implementing comprehensive timing analysis, jitter measurement, and validation of real-time constraints under worst-case execution scenarios.""",
        tags=["gnc", "rtos"],
    ),
    "embedded_system_integration": PromptTemplate(
        name="embedded_system_integration",
        template="""Prompt 32: Embedded System Integration
Implement support for various embedded platforms (ARM Cortex, DSPs, FPGAs) with automatic cross-compilation, hardware abstraction layers, and device driver integration, including extensive hardware validation, performance optimization, and power consumption analysis.""",
        tags=["gnc", "embedded"],
    ),
    "automated_test_suite": PromptTemplate(
        name="automated_test_suite",
        template="""Prompt 33: Automated Test Suite
Develop a comprehensive automated testing framework with unit tests, integration tests, regression tests, and performance benchmarks, implementing continuous integration workflows, automated report generation, and comparison with baseline results for quality assurance.""",
        tags=["gnc", "testing", "automation"],
    ),
    "benchmark_validation_suite": PromptTemplate(
        name="benchmark_validation_suite",
        template="""Prompt 34: Benchmark Validation Suite
Create extensive validation against published aerospace benchmarks, NASA test cases, and commercial simulator results, implementing automated comparison tools, statistical significance testing, and detailed error analysis to ensure framework accuracy and reliability.""",
        tags=["gnc", "validation", "benchmark"],
    ),
    "performance_profiling_system": PromptTemplate(
        name="performance_profiling_system",
        template="""Prompt 35: Performance Profiling System
Implement sophisticated performance analysis tools with execution time profiling, memory usage analysis, and computational bottleneck identification, creating automated optimization recommendations and performance regression detection for maintaining real-time execution requirements.""",
        tags=["gnc", "profiling"],
    ),
    "interactive_tutorial_system": PromptTemplate(
        name="interactive_tutorial_system",
        template="""Prompt 36: Interactive Tutorial System
Develop comprehensive Jupyter notebook tutorials covering all framework capabilities with interactive widgets, 3D visualizations, and progressive complexity levels, implementing automated testing of tutorial code, user feedback collection, and adaptive learning path recommendations.""",
        tags=["gnc", "documentation", "tutorial"],
    ),
    "api_documentation_generator": PromptTemplate(
        name="api_documentation_generator",
        template="""Prompt 37: API Documentation Generator
Create automated documentation generation with code examples, mathematical derivations, and cross-references, implementing documentation testing, version control integration, and interactive API exploration tools for enhanced user experience.""",
        tags=["gnc", "documentation", "api"],
    ),
    "command_line_interface": PromptTemplate(
        name="command_line_interface",
        template="""Prompt 38: Command-Line Interface
Implement a sophisticated CLI with rich output formatting, progress bars, interactive demos, and batch processing capabilities, creating extensive usability testing, command validation, and integration with popular workflow tools for enhanced productivity.""",
        tags=["gnc", "cli"],
    ),
    "mission_simulation": PromptTemplate(
        name="mission_simulation",
        template="""Prompt 39: End-to-End Mission Simulation
Develop comprehensive mission-level simulations combining all framework components with realistic scenarios, environmental models, and failure injection, implementing automated mission analysis, performance metrics calculation, and comparison with mission requirements.""",
        tags=["gnc", "simulation", "mission"],
    ),
    "cross_platform_compatibility": PromptTemplate(
        name="cross_platform_compatibility",
        template="""Prompt 40: Cross-Platform Compatibility
Ensure framework compatibility across multiple operating systems, Python versions, and hardware architectures, implementing extensive compatibility testing, automated build systems, and package distribution mechanisms for broad accessibility and reliability.""",
        tags=["gnc", "compatibility"],
    ),
}

# Merge system and GNC templates for easy loading
ALL_TEMPLATES = {
    **SYSTEM_TEMPLATES,
    **GNC_PROMPTS,
}