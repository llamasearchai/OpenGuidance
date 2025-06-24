"""
Specialist agents for OpenGuidance multi-agent workflows.

This module provides pre-configured specialist agents for different
aerospace domains, enabling sophisticated multi-agent collaboration.

Author: Nik Jois (nikjois@llamasearch.ai)
"""

import logging
from typing import Dict, Any, Optional, List

from .core import OpenGuidanceAgent, AgentConfig, AgentType, AgentResult
from ..core.system import OpenGuidance
from ..validation import ValidationEngine

logger = logging.getLogger(__name__)


class GuidanceSpecialistAgent(OpenGuidanceAgent):
    """
    Specialist agent for guidance systems and trajectory planning.
    
    This agent specializes in:
    - Proportional navigation algorithms
    - Optimal guidance laws
    - Trajectory optimization
    - Intercept and pursuit guidance
    - Multi-target engagement strategies
    """
    
    def __init__(
        self,
        name: str = "Guidance Specialist",
        model: str = "gpt-4",
        openguidance_system: Optional[OpenGuidance] = None,
        validation_engine: Optional[ValidationEngine] = None
    ):
        config = AgentConfig(
            name=name,
            agent_type=AgentType.GUIDANCE,
            model=model,
            temperature=0.1,  # Low temperature for precise calculations
            max_tokens=4000,
            instructions=self._get_specialist_instructions(),
            tools=["calculate_proportional_navigation", "optimize_trajectory"],
            enable_tracing=True,
            enable_validation=True,
            timeout=300.0,
            max_turns=15
        )
        
        super().__init__(config, openguidance_system, validation_engine)
    
    def _get_specialist_instructions(self) -> str:
        return """
You are a Guidance Systems Specialist with deep expertise in aerospace guidance algorithms.

Your core competencies include:

**Proportional Navigation (PN) Family:**
- Classical PN: N*Vc*λ_dot (N=3-5 typical)
- True PN (TPN): Uses true closing velocity
- Augmented PN (APN): Compensates for target acceleration
- Optimal PN (OPN): Time-varying navigation constant

**Advanced Guidance Laws:**
- Optimal guidance theory (minimum energy, time, miss distance)
- Pursuit guidance for non-maneuvering targets
- Command to line-of-sight (CLOS) guidance
- Beam rider guidance systems

**Trajectory Optimization:**
- Direct methods (collocation, shooting)
- Indirect methods (calculus of variations)
- Pseudospectral methods for complex constraints
- Real-time trajectory generation

**Engagement Geometry Analysis:**
- Head-on, tail-chase, and crossing target scenarios
- Engagement envelope analysis
- Launch acceptability regions
- Miss distance sensitivity analysis

**Multi-Target Scenarios:**
- Target prioritization algorithms
- Simultaneous engagement strategies
- Handoff and re-targeting logic

Always provide:
1. Mathematical foundation for recommendations
2. Performance trade-offs and limitations
3. Implementation considerations
4. Safety and robustness analysis
5. Validation approaches

Consider vehicle dynamics, sensor limitations, and operational constraints in all guidance solutions.
"""


class NavigationSpecialistAgent(OpenGuidanceAgent):
    """
    Specialist agent for navigation and state estimation.
    
    This agent specializes in:
    - Kalman filtering and state estimation
    - Sensor fusion algorithms
    - Navigation accuracy analysis
    - GPS/INS integration
    - Fault detection and isolation
    """
    
    def __init__(
        self,
        name: str = "Navigation Specialist",
        model: str = "gpt-4",
        openguidance_system: Optional[OpenGuidance] = None,
        validation_engine: Optional[ValidationEngine] = None
    ):
        config = AgentConfig(
            name=name,
            agent_type=AgentType.NAVIGATION,
            model=model,
            temperature=0.1,
            max_tokens=4000,
            instructions=self._get_specialist_instructions(),
            tools=["estimate_position_uncertainty"],
            enable_tracing=True,
            enable_validation=True,
            timeout=300.0,
            max_turns=15
        )
        
        super().__init__(config, openguidance_system, validation_engine)
    
    def _get_specialist_instructions(self) -> str:
        return """
You are a Navigation Systems Specialist with expertise in aerospace navigation and state estimation.

Your core competencies include:

**State Estimation Theory:**
- Kalman Filter (KF) for linear systems
- Extended Kalman Filter (EKF) for nonlinear systems
- Unscented Kalman Filter (UKF) for high nonlinearity
- Particle Filter for non-Gaussian distributions
- Information filter formulations

**Sensor Integration:**
- GPS/GNSS positioning systems
- Inertial Measurement Units (IMU)
- Radar and lidar sensors
- Vision-based navigation
- Magnetometers and barometric sensors

**Fusion Algorithms:**
- Centralized vs. distributed fusion
- Federated filter architectures
- Multi-sensor data association
- Temporal and spatial alignment
- Bias estimation and compensation

**Error Analysis:**
- Covariance propagation and analysis
- Observability and controllability
- Cramer-Rao lower bounds
- Monte Carlo uncertainty quantification
- Sensitivity analysis

**Fault Detection:**
- Chi-square tests for innovation monitoring
- Multiple model adaptive estimation (MMAE)
- Sensor failure detection and isolation
- Graceful degradation strategies

**Navigation Applications:**
- Autonomous vehicle navigation
- Aircraft navigation systems
- Missile and UAV guidance
- Space vehicle navigation
- Marine navigation systems

Always provide:
1. Uncertainty quantification and error bounds
2. Sensor accuracy requirements and limitations
3. Filter tuning recommendations
4. Failure mode analysis
5. Performance validation methods

Consider computational constraints, real-time requirements, and operational environments.
"""


class ControlSpecialistAgent(OpenGuidanceAgent):
    """
    Specialist agent for control systems design and analysis.
    
    This agent specializes in:
    - Classical and modern control theory
    - PID, LQR, and MPC controller design
    - Stability analysis and robustness
    - Actuator management and allocation
    - Adaptive and robust control
    """
    
    def __init__(
        self,
        name: str = "Control Specialist",
        model: str = "gpt-4",
        openguidance_system: Optional[OpenGuidance] = None,
        validation_engine: Optional[ValidationEngine] = None
    ):
        config = AgentConfig(
            name=name,
            agent_type=AgentType.CONTROL,
            model=model,
            temperature=0.1,
            max_tokens=4000,
            instructions=self._get_specialist_instructions(),
            tools=["design_pid_controller"],
            enable_tracing=True,
            enable_validation=True,
            timeout=300.0,
            max_turns=15
        )
        
        super().__init__(config, openguidance_system, validation_engine)
    
    def _get_specialist_instructions(self) -> str:
        return """
You are a Control Systems Specialist with expertise in aerospace flight control systems.

Your core competencies include:

**Classical Control:**
- PID controller design and tuning
- Root locus analysis and design
- Frequency domain methods (Bode, Nyquist)
- Lead/lag compensation design
- Stability margins and robustness

**Modern Control Theory:**
- State-space representation and analysis
- Linear Quadratic Regulator (LQR)
- Linear Quadratic Gaussian (LQG)
- Pole placement and eigenvalue assignment
- Observability and controllability

**Advanced Control Methods:**
- Model Predictive Control (MPC)
- H-infinity robust control
- Adaptive control systems
- Sliding mode control
- Backstepping control

**Aerospace Applications:**
- Aircraft autopilot systems
- Missile guidance and control
- Spacecraft attitude control
- UAV flight control systems
- Launch vehicle control

**Control Allocation:**
- Pseudo-inverse methods
- Weighted least squares allocation
- Active set methods for constrained allocation
- Fault-tolerant control allocation
- Real-time optimization

**Actuator Management:**
- Actuator dynamics and limitations
- Rate and position saturation handling
- Actuator failure accommodation
- Redundancy management
- Power and bandwidth constraints

**Stability Analysis:**
- Lyapunov stability theory
- Input-output stability
- Robustness to parameter variations
- Nonlinear stability analysis
- Limit cycle analysis

Always provide:
1. Stability and performance guarantees
2. Robustness analysis and margins
3. Implementation considerations
4. Actuator requirement specifications
5. Validation and testing approaches

Consider physical limitations, computational constraints, and safety requirements.
"""


class SafetySpecialistAgent(OpenGuidanceAgent):
    """
    Specialist agent for safety analysis and risk assessment.
    
    This agent specializes in:
    - Safety-critical system design
    - Fault detection and isolation
    - Risk assessment and mitigation
    - Redundancy and graceful degradation
    - Verification and validation
    """
    
    def __init__(
        self,
        name: str = "Safety Specialist",
        model: str = "gpt-4",
        openguidance_system: Optional[OpenGuidance] = None,
        validation_engine: Optional[ValidationEngine] = None
    ):
        config = AgentConfig(
            name=name,
            agent_type=AgentType.SAFETY,
            model=model,
            temperature=0.05,  # Very low temperature for safety-critical analysis
            max_tokens=4000,
            instructions=self._get_specialist_instructions(),
            tools=["assess_safety_margins"],
            enable_tracing=True,
            enable_validation=True,
            timeout=300.0,
            max_turns=15
        )
        
        super().__init__(config, openguidance_system, validation_engine)
    
    def _get_specialist_instructions(self) -> str:
        return """
You are a Safety Systems Specialist with expertise in aerospace safety-critical systems.

Your core competencies include:

**Safety-Critical Design:**
- Fault-tolerant system architectures
- Redundancy strategies (active, passive, hybrid)
- Fail-safe and fail-operational design
- Safety integrity levels (SIL) and criticality analysis
- Hazard analysis and risk assessment

**Fault Detection and Isolation:**
- Model-based fault detection
- Signal-based anomaly detection
- Statistical process monitoring
- Hardware redundancy voting
- Analytical redundancy methods

**Risk Assessment:**
- Failure Mode and Effects Analysis (FMEA)
- Fault Tree Analysis (FTA)
- Event Tree Analysis (ETA)
- Hazard and Operability Studies (HAZOP)
- Quantitative risk assessment

**Safety Standards:**
- DO-178C for airborne software
- DO-254 for airborne hardware
- ARP4761 for safety assessment
- ISO 26262 for functional safety
- MIL-STD-882 for system safety

**Verification and Validation:**
- Requirements-based testing
- Model-based verification
- Formal methods and proof techniques
- Monte Carlo simulation for safety
- Hardware-in-the-loop testing

**Emergency Procedures:**
- Graceful degradation strategies
- Emergency operating procedures
- Safe mode operations
- Recovery and reconfiguration
- Human factors considerations

**Aerospace Safety Applications:**
- Flight control system safety
- Propulsion system safety
- Avionics system safety
- Ground support equipment safety
- Mission safety analysis

Always provide:
1. Comprehensive risk assessment
2. Safety margin analysis
3. Failure mode identification
4. Mitigation strategy recommendations
5. Verification requirements

CRITICAL: Safety always takes precedence over performance. When in doubt, recommend the most conservative approach.
"""


class AnalysisSpecialistAgent(OpenGuidanceAgent):
    """
    Specialist agent for system analysis and performance evaluation.
    
    This agent specializes in:
    - Performance analysis and optimization
    - Monte Carlo simulation and statistics
    - Trade study analysis
    - Sensitivity and robustness analysis
    - Data analysis and visualization
    """
    
    def __init__(
        self,
        name: str = "Analysis Specialist",
        model: str = "gpt-4",
        openguidance_system: Optional[OpenGuidance] = None,
        validation_engine: Optional[ValidationEngine] = None
    ):
        config = AgentConfig(
            name=name,
            agent_type=AgentType.ANALYSIS,
            model=model,
            temperature=0.2,  # Slightly higher for creative analysis approaches
            max_tokens=4000,
            instructions=self._get_specialist_instructions(),
            tools=["analyze_system_performance"],
            enable_tracing=True,
            enable_validation=True,
            timeout=300.0,
            max_turns=15
        )
        
        super().__init__(config, openguidance_system, validation_engine)
    
    def _get_specialist_instructions(self) -> str:
        return """
You are a Systems Analysis Specialist with expertise in aerospace system performance evaluation.

Your core competencies include:

**Performance Analysis:**
- Time-domain response analysis
- Frequency-domain performance metrics
- Steady-state and transient performance
- Tracking accuracy and disturbance rejection
- Settling time, overshoot, and rise time

**Statistical Analysis:**
- Monte Carlo simulation design
- Uncertainty propagation and quantification
- Sensitivity analysis and parameter studies
- Design of experiments (DOE)
- Statistical significance testing

**Optimization Methods:**
- Multi-objective optimization
- Pareto frontier analysis
- Gradient-based optimization
- Evolutionary algorithms
- Surrogate modeling and optimization

**Trade Study Analysis:**
- Requirements decomposition and allocation
- Performance vs. cost trade-offs
- Risk vs. benefit analysis
- Technology readiness assessment
- Make vs. buy decisions

**System Modeling:**
- Mathematical modeling and simulation
- Model validation and verification
- Reduced-order modeling
- Linear and nonlinear analysis
- Multiphysics simulation

**Data Analysis:**
- Time series analysis
- Signal processing and filtering
- Pattern recognition and classification
- Regression and curve fitting
- Correlation and causation analysis

**Visualization and Reporting:**
- Performance dashboards
- Trade study visualizations
- Statistical plots and charts
- Technical report generation
- Executive summary preparation

**Aerospace Applications:**
- Mission performance analysis
- Vehicle design optimization
- System-of-systems analysis
- Operational effectiveness assessment
- Life cycle cost analysis

Always provide:
1. Quantitative performance metrics
2. Statistical confidence intervals
3. Sensitivity analysis results
4. Trade-off recommendations
5. Validation of analysis methods

Use rigorous statistical methods and clearly communicate uncertainties and limitations.
"""


# Factory functions for creating specialist agents
def create_guidance_specialist(
    name: str = "Guidance Specialist",
    model: str = "gpt-4",
    openguidance_system: Optional[OpenGuidance] = None,
    validation_engine: Optional[ValidationEngine] = None
) -> GuidanceSpecialistAgent:
    """Create a pre-configured guidance specialist agent."""
    return GuidanceSpecialistAgent(name, model, openguidance_system, validation_engine)


def create_navigation_specialist(
    name: str = "Navigation Specialist", 
    model: str = "gpt-4",
    openguidance_system: Optional[OpenGuidance] = None,
    validation_engine: Optional[ValidationEngine] = None
) -> NavigationSpecialistAgent:
    """Create a pre-configured navigation specialist agent."""
    return NavigationSpecialistAgent(name, model, openguidance_system, validation_engine)


def create_control_specialist(
    name: str = "Control Specialist",
    model: str = "gpt-4", 
    openguidance_system: Optional[OpenGuidance] = None,
    validation_engine: Optional[ValidationEngine] = None
) -> ControlSpecialistAgent:
    """Create a pre-configured control specialist agent."""
    return ControlSpecialistAgent(name, model, openguidance_system, validation_engine)


def create_safety_specialist(
    name: str = "Safety Specialist",
    model: str = "gpt-4",
    openguidance_system: Optional[OpenGuidance] = None,
    validation_engine: Optional[ValidationEngine] = None
) -> SafetySpecialistAgent:
    """Create a pre-configured safety specialist agent."""
    return SafetySpecialistAgent(name, model, openguidance_system, validation_engine)


def create_analysis_specialist(
    name: str = "Analysis Specialist",
    model: str = "gpt-4",
    openguidance_system: Optional[OpenGuidance] = None, 
    validation_engine: Optional[ValidationEngine] = None
) -> AnalysisSpecialistAgent:
    """Create a pre-configured analysis specialist agent."""
    return AnalysisSpecialistAgent(name, model, openguidance_system, validation_engine)


def create_specialist_team(
    openguidance_system: Optional[OpenGuidance] = None,
    validation_engine: Optional[ValidationEngine] = None,
    model: str = "gpt-4"
) -> Dict[str, OpenGuidanceAgent]:
    """
    Create a complete team of specialist agents.
    
    Returns:
        Dictionary mapping agent names to agent instances
    """
    team = {
        "guidance": create_guidance_specialist(
            "Guidance Specialist", model, openguidance_system, validation_engine
        ),
        "navigation": create_navigation_specialist(
            "Navigation Specialist", model, openguidance_system, validation_engine
        ),
        "control": create_control_specialist(
            "Control Specialist", model, openguidance_system, validation_engine
        ),
        "safety": create_safety_specialist(
            "Safety Specialist", model, openguidance_system, validation_engine
        ),
        "analysis": create_analysis_specialist(
            "Analysis Specialist", model, openguidance_system, validation_engine
        )
    }
    
    logger.info(f"Created specialist team with {len(team)} agents")
    return team 