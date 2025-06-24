"""
Core OpenAI Agents SDK integration for OpenGuidance.

This module provides the fundamental integration between OpenGuidance
and the OpenAI Agents SDK, enabling sophisticated multi-agent workflows.

Author: Nik Jois (nikjois@llamasearch.ai)
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from openai import OpenAI
from agents import Agent, Runner, function_tool, ModelSettings
from agents import trace, add_trace_processor
from agents.tracing.processors import default_processor

from ..core.system import OpenGuidance
from ..core.config import Config
from ..models.execution import ExecutionResult, ExecutionStatus
from ..validation import ValidationEngine

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of OpenGuidance agents."""
    GUIDANCE = "guidance"
    NAVIGATION = "navigation"
    CONTROL = "control"
    SAFETY = "safety"
    ANALYSIS = "analysis"
    COORDINATOR = "coordinator"


@dataclass
class AgentConfig:
    """Configuration for OpenGuidance agents."""
    name: str
    agent_type: AgentType
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 4000
    instructions: str = ""
    tools: List[str] = field(default_factory=list)
    enable_tracing: bool = True
    enable_validation: bool = True
    timeout: float = 300.0
    max_turns: int = 20
    
    def to_model_settings(self) -> ModelSettings:
        """Convert to OpenAI Agents SDK ModelSettings."""
        return ModelSettings(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tool_choice="auto",
            parallel_tool_calls=True
        )


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_name: str
    agent_type: AgentType
    content: str
    success: bool
    execution_time: float
    token_usage: Dict[str, int]
    validation_score: Optional[float] = None
    validation_passed: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    trace_id: Optional[str] = None


class AgentError(Exception):
    """Custom exception for agent-related errors."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, agent_type: Optional[AgentType] = None):
        super().__init__(message)
        self.agent_name = agent_name
        self.agent_type = agent_type


class OpenGuidanceAgent:
    """
    Enhanced OpenAI Agent with OpenGuidance integration.
    
    This class wraps the OpenAI Agents SDK Agent with OpenGuidance-specific
    functionality including validation, tracing, and domain-specific tools.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        openguidance_system: Optional[OpenGuidance] = None,
        validation_engine: Optional[ValidationEngine] = None
    ):
        self.config = config
        self.openguidance_system = openguidance_system
        self.validation_engine = validation_engine
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Create the OpenAI Agent
        self.agent = Agent(
            name=config.name,
            instructions=config.instructions or self._get_default_instructions(),
            model=config.model,
            tools=self.tools,
            model_settings=config.to_model_settings()
        )
        
        logger.info(f"OpenGuidanceAgent '{config.name}' initialized with type {config.agent_type}")
    
    def _get_default_instructions(self) -> str:
        """Get default instructions based on agent type."""
        instructions = {
            AgentType.GUIDANCE: """
You are a Guidance Specialist Agent for aerospace systems. Your expertise includes:
- Proportional Navigation (PN, TPN, APN, OPN)
- Optimal guidance laws and trajectory shaping
- Intercept and pursuit guidance algorithms
- Waypoint navigation and path following
- Multi-target engagement strategies

Provide precise, mathematically sound guidance solutions with clear explanations.
Always consider safety constraints and performance requirements.
""",
            AgentType.NAVIGATION: """
You are a Navigation Specialist Agent for aerospace systems. Your expertise includes:
- State estimation and sensor fusion
- Kalman filtering (EKF, UKF, Particle filters)
- IMU, GPS, radar, and vision-based navigation
- SLAM and localization algorithms
- Navigation accuracy analysis

Provide robust navigation solutions with uncertainty quantification.
Always validate sensor data and handle failure modes gracefully.
""",
            AgentType.CONTROL: """
You are a Control Systems Specialist Agent for aerospace systems. Your expertise includes:
- Classical control (PID, LQR, LQG)
- Modern control (MPC, H-infinity, Adaptive)
- Autopilot design and implementation
- Control allocation and actuator management
- Stability analysis and robustness

Provide stable, robust control solutions with performance guarantees.
Always consider actuator limits and system constraints.
""",
            AgentType.SAFETY: """
You are a Safety Specialist Agent for aerospace systems. Your expertise includes:
- Safety-critical system design
- Fault detection and isolation
- Redundancy and graceful degradation
- Risk assessment and mitigation
- Verification and validation

Provide comprehensive safety analysis with clear risk assessments.
Always prioritize safety over performance when conflicts arise.
""",
            AgentType.ANALYSIS: """
You are an Analysis Specialist Agent for aerospace systems. Your expertise includes:
- Performance analysis and optimization
- Monte Carlo simulation and uncertainty quantification
- Stability and robustness analysis
- Trade studies and design optimization
- Data analysis and visualization

Provide thorough analysis with statistical rigor and clear visualizations.
Always quantify uncertainties and validate results.
""",
            AgentType.COORDINATOR: """
You are a Coordinator Agent for multi-agent aerospace system workflows. Your role is to:
- Orchestrate specialist agents for complex tasks
- Synthesize results from multiple agents
- Ensure consistency and coherence across analyses
- Manage workflow dependencies and timing
- Provide comprehensive final reports

Coordinate effectively while maintaining technical accuracy across all domains.
Always validate cross-domain consistency and resolve conflicts.
"""
        }
        
        return instructions.get(self.config.agent_type, "You are a helpful aerospace systems assistant.")
    
    def _initialize_tools(self) -> List[Any]:
        """Initialize tools based on agent configuration."""
        from .tools import get_tools_for_agent_type
        return get_tools_for_agent_type(self.config.agent_type, self.openguidance_system)
    
    async def execute(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> AgentResult:
        """
        Execute the agent with the given request.
        
        Args:
            request: The request to process
            context: Additional context information
            session_id: Session identifier for tracing
            
        Returns:
            AgentResult with execution details
        """
        start_time = time.time()
        trace_id = None
        
        try:
            # Create enhanced request with context
            enhanced_request = self._enhance_request(request, context)
            
            # Execute with tracing if enabled
            if self.config.enable_tracing:
                with trace(
                    f"{self.config.name} Execution",
                    metadata={
                        "agent_type": self.config.agent_type.value,
                        "session_id": session_id,
                        "request": request[:200] + "..." if len(request) > 200 else request
                    }
                ) as span:
                    trace_id = span.trace_id
                    result = await Runner.run(
                        self.agent,
                        enhanced_request,
                        max_turns=self.config.max_turns
                    )
            else:
                result = await Runner.run(
                    self.agent,
                    enhanced_request,
                    max_turns=self.config.max_turns
                )
            
            execution_time = time.time() - start_time
            
            # Validate result if validation is enabled
            validation_score = None
            validation_passed = None
            
            if self.config.enable_validation and self.validation_engine:
                validation_result = await self.validation_engine.validate_response(
                    result.final_output,
                    context or {}
                )
                validation_score = validation_result.overall_score
                validation_passed = validation_result.passed
            
            return AgentResult(
                agent_name=self.config.name,
                agent_type=self.config.agent_type,
                content=result.final_output,
                success=True,
                execution_time=execution_time,
                token_usage=getattr(result, 'token_usage', {}),
                validation_score=validation_score,
                validation_passed=validation_passed,
                trace_id=trace_id,
                metadata={
                    "turns": getattr(result, 'turns', 0),
                    "model": self.config.model
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Agent execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return AgentResult(
                agent_name=self.config.name,
                agent_type=self.config.agent_type,
                content="",
                success=False,
                execution_time=execution_time,
                token_usage={},
                error=error_msg,
                trace_id=trace_id
            )
    
    def _enhance_request(self, request: str, context: Optional[Dict[str, Any]]) -> str:
        """Enhance the request with context and domain-specific information."""
        if not context:
            return request
        
        enhanced_parts = [request]
        
        # Add context information
        if context:
            enhanced_parts.append("\nContext:")
            for key, value in context.items():
                enhanced_parts.append(f"- {key}: {value}")
        
        # Add agent-specific guidance
        if self.config.agent_type == AgentType.GUIDANCE:
            enhanced_parts.append("\nConsider: vehicle dynamics, target behavior, engagement geometry")
        elif self.config.agent_type == AgentType.NAVIGATION:
            enhanced_parts.append("\nConsider: sensor accuracy, environmental conditions, failure modes")
        elif self.config.agent_type == AgentType.CONTROL:
            enhanced_parts.append("\nConsider: actuator limits, stability margins, disturbance rejection")
        elif self.config.agent_type == AgentType.SAFETY:
            enhanced_parts.append("\nConsider: failure modes, risk assessment, safety margins")
        elif self.config.agent_type == AgentType.ANALYSIS:
            enhanced_parts.append("\nConsider: statistical significance, uncertainty quantification, validation")
        
        return "\n".join(enhanced_parts)


class GuidanceAgentRunner:
    """
    High-level runner for OpenGuidance agent workflows.
    
    This class provides convenient methods for running single agents
    or orchestrating multi-agent workflows.
    """
    
    def __init__(
        self,
        openguidance_system: Optional[OpenGuidance] = None,
        validation_engine: Optional[ValidationEngine] = None,
        enable_tracing: bool = True
    ):
        self.openguidance_system = openguidance_system
        self.validation_engine = validation_engine
        self.enable_tracing = enable_tracing
        
        # Initialize tracing if enabled
        if enable_tracing:
            add_trace_processor(default_processor())
        
        self.agents: Dict[str, OpenGuidanceAgent] = {}
        
        logger.info("GuidanceAgentRunner initialized")
    
    def create_agent(self, config: AgentConfig) -> OpenGuidanceAgent:
        """Create and register a new agent."""
        agent = OpenGuidanceAgent(
            config=config,
            openguidance_system=self.openguidance_system,
            validation_engine=self.validation_engine
        )
        
        self.agents[config.name] = agent
        logger.info(f"Created and registered agent: {config.name}")
        
        return agent
    
    async def run_agent(
        self,
        agent_name: str,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> AgentResult:
        """Run a specific agent by name."""
        if agent_name not in self.agents:
            raise AgentError(f"Agent '{agent_name}' not found")
        
        agent = self.agents[agent_name]
        return await agent.execute(request, context, session_id)
    
    async def run_multi_agent_workflow(
        self,
        workflow_name: str,
        request: str,
        agent_sequence: List[str],
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> List[AgentResult]:
        """
        Run a multi-agent workflow with specified agent sequence.
        
        Args:
            workflow_name: Name of the workflow for tracing
            request: Initial request
            agent_sequence: List of agent names to execute in order
            context: Shared context
            session_id: Session identifier
            
        Returns:
            List of results from each agent
        """
        results = []
        current_context = context or {}
        
        if self.enable_tracing:
            with trace(
                f"Multi-Agent Workflow: {workflow_name}",
                metadata={
                    "agents": agent_sequence,
                    "session_id": session_id
                }
            ):
                for agent_name in agent_sequence:
                    # Update context with previous results
                    if results:
                        current_context["previous_results"] = [
                            {"agent": r.agent_name, "content": r.content}
                            for r in results
                        ]
                    
                    result = await self.run_agent(
                        agent_name, request, current_context, session_id
                    )
                    results.append(result)
                    
                    # Stop if any agent fails critically
                    if not result.success and result.error:
                        logger.error(f"Workflow stopped due to agent failure: {result.error}")
                        break
        else:
            for agent_name in agent_sequence:
                if results:
                    current_context["previous_results"] = [
                        {"agent": r.agent_name, "content": r.content}
                        for r in results
                    ]
                
                result = await self.run_agent(
                    agent_name, request, current_context, session_id
                )
                results.append(result)
                
                if not result.success and result.error:
                    logger.error(f"Workflow stopped due to agent failure: {result.error}")
                    break
        
        return results
    
    async def run_parallel_agents(
        self,
        request: str,
        agent_names: List[str],
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> List[AgentResult]:
        """Run multiple agents in parallel."""
        tasks = []
        
        for agent_name in agent_names:
            task = self.run_agent(agent_name, request, context, session_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    AgentResult(
                        agent_name=agent_names[i],
                        agent_type=AgentType.COORDINATOR,  # Default
                        content="",
                        success=False,
                        execution_time=0.0,
                        token_usage={},
                        error=str(result)
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered agents."""
        status = {}
        
        for name, agent in self.agents.items():
            status[name] = {
                "type": agent.config.agent_type.value,
                "model": agent.config.model,
                "tools": len(agent.tools),
                "validation_enabled": agent.config.enable_validation,
                "tracing_enabled": agent.config.enable_tracing
            }
        
        return status 