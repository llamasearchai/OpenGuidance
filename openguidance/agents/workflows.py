"""
Multi-agent workflows for OpenGuidance.

This module provides sophisticated multi-agent workflows that orchestrate
specialist agents to solve complex aerospace system problems.

Author: Nik Jois (nikjois@llamasearch.ai)
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from agents.tracing import trace

from .core import GuidanceAgentRunner, AgentResult, AgentError, AgentType
from .specialists import (
    create_guidance_specialist,
    create_navigation_specialist,
    create_control_specialist,
    create_safety_specialist,
    create_analysis_specialist
)
from ..core.system import OpenGuidance
from ..validation import ValidationEngine

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowResult:
    """Result from a workflow execution."""
    workflow_name: str
    status: WorkflowStatus
    execution_time: float
    agent_results: List[AgentResult]
    summary: str
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseWorkflow:
    """Base class for multi-agent workflows."""
    
    def __init__(
        self,
        name: str,
        openguidance_system: Optional[OpenGuidance] = None,
        validation_engine: Optional[ValidationEngine] = None,
        enable_tracing: bool = True
    ):
        self.name = name
        self.openguidance_system = openguidance_system
        self.validation_engine = validation_engine
        self.enable_tracing = enable_tracing
        
        # Initialize agent runner
        self.runner = GuidanceAgentRunner(
            openguidance_system=openguidance_system,
            validation_engine=validation_engine,
            enable_tracing=enable_tracing
        )
        
        # Workflow state
        self.status = WorkflowStatus.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        logger.info(f"Initialized workflow: {name}")
    
    async def execute(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> WorkflowResult:
        """Execute the workflow."""
        self.status = WorkflowStatus.RUNNING
        self.start_time = time.time()
        
        try:
            if self.enable_tracing:
                with trace(
                    f"Workflow: {self.name}",
                    metadata={
                        "request": request[:200] + "..." if len(request) > 200 else request,
                        "session_id": session_id
                    }
                ):
                    result = await self._execute_workflow(request, context, session_id)
            else:
                result = await self._execute_workflow(request, context, session_id)
            
            self.status = WorkflowStatus.COMPLETED
            return result
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            logger.error(f"Workflow {self.name} failed: {e}", exc_info=True)
            
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.FAILED,
                execution_time=time.time() - self.start_time,
                agent_results=[],
                summary=f"Workflow failed: {str(e)}",
                error=str(e)
            )
        finally:
            self.end_time = time.time()
    
    async def _execute_workflow(
        self,
        request: str,
        context: Optional[Dict[str, Any]],
        session_id: Optional[str]
    ) -> WorkflowResult:
        """Override this method in subclasses."""
        raise NotImplementedError("Subclasses must implement _execute_workflow")
    
    def _create_summary(self, agent_results: List[AgentResult]) -> str:
        """Create a summary from agent results."""
        successful_agents = [r for r in agent_results if r.success]
        failed_agents = [r for r in agent_results if not r.success]
        
        summary_parts = [
            f"Workflow '{self.name}' completed with {len(successful_agents)} successful agents"
        ]
        
        if failed_agents:
            summary_parts.append(f"and {len(failed_agents)} failed agents")
        
        # Add key insights from successful agents
        for result in successful_agents[:3]:  # Top 3 results
            if result.content:
                summary_parts.append(f"\n{result.agent_name}: {result.content[:200]}...")
        
        return ". ".join(summary_parts)


class MultiAgentGuidanceWorkflow(BaseWorkflow):
    """
    Comprehensive multi-agent workflow for guidance system analysis.
    
    This workflow orchestrates guidance, navigation, control, safety, and analysis
    specialists to provide a complete assessment of a guidance system problem.
    """
    
    def __init__(
        self,
        openguidance_system: Optional[OpenGuidance] = None,
        validation_engine: Optional[ValidationEngine] = None,
        enable_tracing: bool = True,
        model: str = "gpt-4"
    ):
        super().__init__(
            "Multi-Agent Guidance Analysis",
            openguidance_system,
            validation_engine,
            enable_tracing
        )
        
        self.model = model
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all specialist agents."""
        # Create specialist agents
        self.guidance_agent = create_guidance_specialist(
            model=self.model,
            openguidance_system=self.openguidance_system,
            validation_engine=self.validation_engine
        )
        
        self.navigation_agent = create_navigation_specialist(
            model=self.model,
            openguidance_system=self.openguidance_system,
            validation_engine=self.validation_engine
        )
        
        self.control_agent = create_control_specialist(
            model=self.model,
            openguidance_system=self.openguidance_system,
            validation_engine=self.validation_engine
        )
        
        self.safety_agent = create_safety_specialist(
            model=self.model,
            openguidance_system=self.openguidance_system,
            validation_engine=self.validation_engine
        )
        
        self.analysis_agent = create_analysis_specialist(
            model=self.model,
            openguidance_system=self.openguidance_system,
            validation_engine=self.validation_engine
        )
        
        # Register agents with runner
        for agent in [self.guidance_agent, self.navigation_agent, self.control_agent,
                      self.safety_agent, self.analysis_agent]:
            self.runner.agents[agent.config.name] = agent
    
    async def _execute_workflow(
        self,
        request: str,
        context: Optional[Dict[str, Any]],
        session_id: Optional[str]
    ) -> WorkflowResult:
        """Execute the multi-agent guidance workflow."""
        
        # Phase 1: Parallel analysis by all specialists
        logger.info("Phase 1: Parallel specialist analysis")
        
        phase1_results = await self.runner.run_parallel_agents(
            request=request,
            agent_names=[
                "Guidance Specialist",
                "Navigation Specialist", 
                "Control Specialist",
                "Safety Specialist"
            ],
            context=context,
            session_id=session_id
        )
        
        # Phase 2: Analysis agent synthesizes results
        logger.info("Phase 2: Synthesis and analysis")
        
        synthesis_context = (context or {}).copy()
        synthesis_context["specialist_results"] = [
            {"agent": r.agent_name, "content": r.content, "success": r.success}
            for r in phase1_results
        ]
        
        synthesis_request = f"""
        Analyze and synthesize the following specialist assessments for the request: "{request}"
        
        Provide a comprehensive analysis that:
        1. Integrates findings from all specialists
        2. Identifies key trade-offs and conflicts
        3. Provides actionable recommendations
        4. Assesses overall system feasibility
        5. Highlights critical risks and mitigations
        """
        
        synthesis_result = await self.runner.run_agent(
            agent_name="Analysis Specialist",
            request=synthesis_request,
            context=synthesis_context,
            session_id=session_id
        )
        
        # Combine all results
        all_results = phase1_results + [synthesis_result]
        
        # Generate recommendations
        recommendations = self._extract_recommendations(all_results)
        
        # Calculate execution time
        execution_time = time.time() - self.start_time
        
        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            execution_time=execution_time,
            agent_results=all_results,
            summary=self._create_summary(all_results),
            recommendations=recommendations,
            metadata={
                "phases": 2,
                "parallel_agents": 4,
                "synthesis_agent": 1
            }
        )
    
    def _extract_recommendations(self, results: List[AgentResult]) -> List[str]:
        """Extract actionable recommendations from agent results."""
        recommendations = []
        
        for result in results:
            if result.success and "recommend" in result.content.lower():
                # Simple extraction of recommendation sentences
                sentences = result.content.split('.')
                for sentence in sentences:
                    if "recommend" in sentence.lower():
                        recommendations.append(f"{result.agent_name}: {sentence.strip()}")
        
        return recommendations[:10]  # Limit to top 10


class TrajectoryOptimizationWorkflow(BaseWorkflow):
    """
    Workflow for trajectory optimization problems.
    
    This workflow combines guidance and analysis specialists to solve
    complex trajectory optimization problems with multiple constraints.
    """
    
    def __init__(
        self,
        openguidance_system: Optional[OpenGuidance] = None,
        validation_engine: Optional[ValidationEngine] = None,
        enable_tracing: bool = True,
        model: str = "gpt-4"
    ):
        super().__init__(
            "Trajectory Optimization",
            openguidance_system,
            validation_engine,
            enable_tracing
        )
        
        self.model = model
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize agents for trajectory optimization."""
        self.guidance_agent = create_guidance_specialist(
            model=self.model,
            openguidance_system=self.openguidance_system,
            validation_engine=self.validation_engine
        )
        
        self.analysis_agent = create_analysis_specialist(
            model=self.model,
            openguidance_system=self.openguidance_system,
            validation_engine=self.validation_engine
        )
        
        # Register agents
        self.runner.agents["Guidance Specialist"] = self.guidance_agent
        self.runner.agents["Analysis Specialist"] = self.analysis_agent
    
    async def _execute_workflow(
        self,
        request: str,
        context: Optional[Dict[str, Any]],
        session_id: Optional[str]
    ) -> WorkflowResult:
        """Execute trajectory optimization workflow."""
        
        # Phase 1: Initial trajectory design
        logger.info("Phase 1: Initial trajectory design")
        
        guidance_result = await self.runner.run_agent(
            agent_name="Guidance Specialist",
            request=f"Design an optimal trajectory for: {request}",
            context=context,
            session_id=session_id
        )
        
        # Phase 2: Performance analysis
        logger.info("Phase 2: Performance analysis")
        
        analysis_context = (context or {}).copy()
        analysis_context["trajectory_design"] = guidance_result.content
        
        analysis_result = await self.runner.run_agent(
            agent_name="Analysis Specialist",
            request=f"Analyze the performance of the proposed trajectory for: {request}",
            context=analysis_context,
            session_id=session_id
        )
        
        # Phase 3: Iterative refinement if needed
        logger.info("Phase 3: Refinement analysis")
        
        refinement_context = analysis_context.copy()
        refinement_context["performance_analysis"] = analysis_result.content
        
        refinement_result = await self.runner.run_agent(
            agent_name="Guidance Specialist",
            request=f"Refine the trajectory based on performance analysis for: {request}",
            context=refinement_context,
            session_id=session_id
        )
        
        all_results = [guidance_result, analysis_result, refinement_result]
        execution_time = time.time() - self.start_time
        
        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            execution_time=execution_time,
            agent_results=all_results,
            summary=self._create_summary(all_results),
            recommendations=self._extract_recommendations(all_results),
            metadata={
                "phases": 3,
                "iterative_refinement": True
            }
        )


class MissionPlanningWorkflow(BaseWorkflow):
    """
    Comprehensive mission planning workflow.
    
    This workflow coordinates all specialists to develop a complete
    mission plan with guidance, navigation, control, and safety considerations.
    """
    
    def __init__(
        self,
        openguidance_system: Optional[OpenGuidance] = None,
        validation_engine: Optional[ValidationEngine] = None,
        enable_tracing: bool = True,
        model: str = "gpt-4"
    ):
        super().__init__(
            "Mission Planning",
            openguidance_system,
            validation_engine,
            enable_tracing
        )
        
        self.model = model
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents for mission planning."""
        agents = {
            "Guidance Specialist": create_guidance_specialist(
                model=self.model,
                openguidance_system=self.openguidance_system,
                validation_engine=self.validation_engine
            ),
            "Navigation Specialist": create_navigation_specialist(
                model=self.model,
                openguidance_system=self.openguidance_system,
                validation_engine=self.validation_engine
            ),
            "Control Specialist": create_control_specialist(
                model=self.model,
                openguidance_system=self.openguidance_system,
                validation_engine=self.validation_engine
            ),
            "Safety Specialist": create_safety_specialist(
                model=self.model,
                openguidance_system=self.openguidance_system,
                validation_engine=self.validation_engine
            ),
            "Analysis Specialist": create_analysis_specialist(
                model=self.model,
                openguidance_system=self.openguidance_system,
                validation_engine=self.validation_engine
            )
        }
        
        for name, agent in agents.items():
            self.runner.agents[name] = agent
    
    async def _execute_workflow(
        self,
        request: str,
        context: Optional[Dict[str, Any]],
        session_id: Optional[str]
    ) -> WorkflowResult:
        """Execute mission planning workflow."""
        
        # Sequential workflow with dependencies
        results = []
        
        # Phase 1: Mission requirements analysis
        logger.info("Phase 1: Mission requirements analysis")
        
        analysis_result = await self.runner.run_agent(
            agent_name="Analysis Specialist",
            request=f"Analyze mission requirements for: {request}",
            context=context,
            session_id=session_id
        )
        results.append(analysis_result)
        
        # Phase 2: Guidance system design
        logger.info("Phase 2: Guidance system design")
        
        guidance_context = (context or {}).copy()
        guidance_context["mission_requirements"] = analysis_result.content
        
        guidance_result = await self.runner.run_agent(
            agent_name="Guidance Specialist",
            request=f"Design guidance system for mission: {request}",
            context=guidance_context,
            session_id=session_id
        )
        results.append(guidance_result)
        
        # Phase 3: Navigation system design
        logger.info("Phase 3: Navigation system design")
        
        nav_context = guidance_context.copy()
        nav_context["guidance_design"] = guidance_result.content
        
        nav_result = await self.runner.run_agent(
            agent_name="Navigation Specialist",
            request=f"Design navigation system for mission: {request}",
            context=nav_context,
            session_id=session_id
        )
        results.append(nav_result)
        
        # Phase 4: Control system design
        logger.info("Phase 4: Control system design")
        
        control_context = nav_context.copy()
        control_context["navigation_design"] = nav_result.content
        
        control_result = await self.runner.run_agent(
            agent_name="Control Specialist",
            request=f"Design control system for mission: {request}",
            context=control_context,
            session_id=session_id
        )
        results.append(control_result)
        
        # Phase 5: Safety assessment
        logger.info("Phase 5: Safety assessment")
        
        safety_context = control_context.copy()
        safety_context["control_design"] = control_result.content
        
        safety_result = await self.runner.run_agent(
            agent_name="Safety Specialist",
            request=f"Assess safety for mission: {request}",
            context=safety_context,
            session_id=session_id
        )
        results.append(safety_result)
        
        execution_time = time.time() - self.start_time
        
        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            execution_time=execution_time,
            agent_results=results,
            summary=self._create_summary(results),
            recommendations=self._extract_recommendations(results),
            metadata={
                "phases": 5,
                "sequential_design": True,
                "full_system_integration": True
            }
        )


class SafetyValidationWorkflow(BaseWorkflow):
    """
    Comprehensive safety validation workflow.
    
    This workflow focuses on safety analysis and validation,
    coordinating safety and analysis specialists for thorough assessment.
    """
    
    def __init__(
        self,
        openguidance_system: Optional[OpenGuidance] = None,
        validation_engine: Optional[ValidationEngine] = None,
        enable_tracing: bool = True,
        model: str = "gpt-4"
    ):
        super().__init__(
            "Safety Validation",
            openguidance_system,
            validation_engine,
            enable_tracing
        )
        
        self.model = model
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize agents for safety validation."""
        self.safety_agent = create_safety_specialist(
            model=self.model,
            openguidance_system=self.openguidance_system,
            validation_engine=self.validation_engine
        )
        
        self.analysis_agent = create_analysis_specialist(
            model=self.model,
            openguidance_system=self.openguidance_system,
            validation_engine=self.validation_engine
        )
        
        self.runner.agents["Safety Specialist"] = self.safety_agent
        self.runner.agents["Analysis Specialist"] = self.analysis_agent
    
    async def _execute_workflow(
        self,
        request: str,
        context: Optional[Dict[str, Any]],
        session_id: Optional[str]
    ) -> WorkflowResult:
        """Execute safety validation workflow."""
        
        results = []
        
        # Phase 1: Initial safety assessment
        logger.info("Phase 1: Initial safety assessment")
        
        safety_result = await self.runner.run_agent(
            agent_name="Safety Specialist",
            request=f"Perform comprehensive safety assessment for: {request}",
            context=context,
            session_id=session_id
        )
        results.append(safety_result)
        
        # Phase 2: Quantitative risk analysis
        logger.info("Phase 2: Quantitative risk analysis")
        
        analysis_context = (context or {}).copy()
        analysis_context["safety_assessment"] = safety_result.content
        
        analysis_result = await self.runner.run_agent(
            agent_name="Analysis Specialist",
            request=f"Perform quantitative risk analysis for: {request}",
            context=analysis_context,
            session_id=session_id
        )
        results.append(analysis_result)
        
        # Phase 3: Mitigation strategies
        logger.info("Phase 3: Mitigation strategies")
        
        mitigation_context = analysis_context.copy()
        mitigation_context["risk_analysis"] = analysis_result.content
        
        mitigation_result = await self.runner.run_agent(
            agent_name="Safety Specialist",
            request=f"Develop risk mitigation strategies for: {request}",
            context=mitigation_context,
            session_id=session_id
        )
        results.append(mitigation_result)
        
        execution_time = time.time() - self.start_time
        
        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            execution_time=execution_time,
            agent_results=results,
            summary=self._create_summary(results),
            recommendations=self._extract_recommendations(results),
            metadata={
                "phases": 3,
                "safety_focused": True,
                "risk_quantification": True
            }
        )


# Factory functions for creating workflows
def create_multi_agent_guidance_workflow(
    openguidance_system: Optional[OpenGuidance] = None,
    validation_engine: Optional[ValidationEngine] = None,
    model: str = "gpt-4"
) -> MultiAgentGuidanceWorkflow:
    """Create a multi-agent guidance workflow."""
    return MultiAgentGuidanceWorkflow(
        openguidance_system=openguidance_system,
        validation_engine=validation_engine,
        model=model
    )


def create_trajectory_optimization_workflow(
    openguidance_system: Optional[OpenGuidance] = None,
    validation_engine: Optional[ValidationEngine] = None,
    model: str = "gpt-4"
) -> TrajectoryOptimizationWorkflow:
    """Create a trajectory optimization workflow."""
    return TrajectoryOptimizationWorkflow(
        openguidance_system=openguidance_system,
        validation_engine=validation_engine,
        model=model
    )


def create_mission_planning_workflow(
    openguidance_system: Optional[OpenGuidance] = None,
    validation_engine: Optional[ValidationEngine] = None,
    model: str = "gpt-4"
) -> MissionPlanningWorkflow:
    """Create a mission planning workflow."""
    return MissionPlanningWorkflow(
        openguidance_system=openguidance_system,
        validation_engine=validation_engine,
        model=model
    )


def create_safety_validation_workflow(
    openguidance_system: Optional[OpenGuidance] = None,
    validation_engine: Optional[ValidationEngine] = None,
    model: str = "gpt-4"
) -> SafetyValidationWorkflow:
    """Create a safety validation workflow."""
    return SafetyValidationWorkflow(
        openguidance_system=openguidance_system,
        validation_engine=validation_engine,
        model=model
    ) 