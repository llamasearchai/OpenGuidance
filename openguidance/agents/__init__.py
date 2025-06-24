"""
OpenAI Agents SDK Integration for OpenGuidance.

This module provides integration with the OpenAI Agents SDK, enabling
sophisticated multi-agent workflows for guidance, navigation, and control tasks.

Author: Nik Jois (nikjois@llamasearch.ai)
"""

from .core import (
    AgentType,
    OpenGuidanceAgent,
    GuidanceAgentRunner,
    AgentConfig,
    AgentResult,
    AgentError
)

from .tools import (
    GuidanceTools,
    NavigationTools,
    ControlTools,
    SimulationTools,
    AnalysisTools
)

from .workflows import (
    MultiAgentGuidanceWorkflow,
    TrajectoryOptimizationWorkflow,
    MissionPlanningWorkflow,
    SafetyValidationWorkflow,
    create_multi_agent_guidance_workflow,
    create_trajectory_optimization_workflow,
    create_mission_planning_workflow,
    create_safety_validation_workflow
)

from .specialists import (
    GuidanceSpecialistAgent,
    NavigationSpecialistAgent,
    ControlSpecialistAgent,
    SafetySpecialistAgent,
    AnalysisSpecialistAgent
)

__all__ = [
    # Core components
    "AgentType",
    "OpenGuidanceAgent",
    "GuidanceAgentRunner", 
    "AgentConfig",
    "AgentResult",
    "AgentError",
    
    # Tools
    "GuidanceTools",
    "NavigationTools",
    "ControlTools", 
    "SimulationTools",
    "AnalysisTools",
    
    # Workflows
    "MultiAgentGuidanceWorkflow",
    "TrajectoryOptimizationWorkflow",
    "MissionPlanningWorkflow",
    "SafetyValidationWorkflow",
    "create_multi_agent_guidance_workflow",
    "create_trajectory_optimization_workflow", 
    "create_mission_planning_workflow",
    "create_safety_validation_workflow",
    
    # Specialist agents
    "GuidanceSpecialistAgent",
    "NavigationSpecialistAgent", 
    "ControlSpecialistAgent",
    "SafetySpecialistAgent",
    "AnalysisSpecialistAgent"
] 