"""
Enhanced API routes for OpenGuidance with OpenAI Agents SDK integration.

This module provides comprehensive API endpoints for both traditional
OpenGuidance functionality and advanced multi-agent workflows.

Author: Nik Jois (nikjois@llamasearch.ai)
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.system import OpenGuidance
from ..agents import (
    create_multi_agent_guidance_workflow,
    create_trajectory_optimization_workflow,
    create_mission_planning_workflow,
    create_safety_validation_workflow,
    GuidanceAgentRunner,
    AgentConfig
)
from ..agents.core import AgentType
from .dependencies import get_system, get_current_user, rate_limiter

logger = logging.getLogger(__name__)

# Create routers
guidance_router = APIRouter(prefix="/guidance", tags=["guidance"])
agents_router = APIRouter(prefix="/agents", tags=["agents"])
workflows_router = APIRouter(prefix="/workflows", tags=["workflows"])
admin_router = APIRouter(prefix="/admin", tags=["admin"])
monitoring_router = APIRouter(prefix="/monitoring", tags=["monitoring"])


# Pydantic models for agents and workflows
class AgentRequest(BaseModel):
    """Request model for single agent execution."""
    agent_type: str = Field(..., description="Type of agent (guidance, navigation, control, safety, analysis)")
    request: str = Field(..., min_length=1, max_length=10000, description="Request to process")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    model: str = Field("gpt-4", description="Model to use for the agent")
    session_id: Optional[str] = Field(None, description="Session ID for context")


class WorkflowRequest(BaseModel):
    """Request model for multi-agent workflow execution."""
    workflow_type: str = Field(..., description="Type of workflow (guidance, trajectory, mission, safety)")
    request: str = Field(..., min_length=1, max_length=10000, description="Request to process")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    model: str = Field("gpt-4", description="Model to use for agents")
    session_id: Optional[str] = Field(None, description="Session ID for context")


class AgentResponse(BaseModel):
    """Response model for agent execution."""
    agent_name: str
    agent_type: str
    content: str
    success: bool
    execution_time: float
    token_usage: Dict[str, int]
    validation_score: Optional[float] = None
    validation_passed: Optional[bool] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class WorkflowResponse(BaseModel):
    """Response model for workflow execution."""
    workflow_name: str
    status: str
    execution_time: float
    agent_results: List[AgentResponse]
    summary: str
    recommendations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# Original guidance endpoints
@guidance_router.post("/", response_model=Dict[str, Any])
async def process_guidance(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system),
    _: bool = Depends(rate_limiter)
):
    """Process a guidance request using the traditional OpenGuidance system."""
    start_time = time.time()
    session_id = request.get("session_id", f"session_{int(start_time)}")
    
    try:
        result = await system.process_request(
            request=request.get("message", ""),
            session_id=session_id,
            context=request.get("context", {}),
            timeout=request.get("timeout", 300.0)
        )
        
        execution_time = time.time() - start_time
        
        return {
            "content": result.content,
            "session_id": session_id,
            "execution_time": execution_time,
            "token_usage": getattr(result, 'token_usage', {}),
            "validation_score": getattr(result, 'validation_score', None),
            "validation_passed": getattr(result, 'validation_passed', None),
            "metadata": {
                "system_type": "traditional",
                "user": current_user
            }
        }
        
    except Exception as e:
        logger.error(f"Guidance processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@guidance_router.post("/stream")
async def stream_guidance(
    request: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system),
    _: bool = Depends(rate_limiter)
):
    """Stream guidance responses."""
    session_id = request.get("session_id", f"session_{int(time.time())}")
    
    async def generate_stream():
        try:
            async for chunk in system.process_streaming_request(
                request=request.get("message", ""),
                session_id=session_id,
                context=request.get("context", {})
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: {{'error': '{str(e)}'}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )


# New agent endpoints
@agents_router.post("/execute", response_model=AgentResponse)
async def execute_agent(
    request: AgentRequest,
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system),
    _: bool = Depends(rate_limiter)
):
    """Execute a single specialist agent."""
    try:
        # Validate agent type
        try:
            agent_type = AgentType(request.agent_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type: {request.agent_type}. Valid types: {[t.value for t in AgentType]}"
            )
        
        # Create agent runner
        runner = GuidanceAgentRunner(
            openguidance_system=system,
            enable_tracing=True
        )
        
        # Create agent configuration
        config = AgentConfig(
            name=f"{agent_type.value.title()} Specialist",
            agent_type=agent_type,
            model=request.model,
            enable_tracing=True,
            enable_validation=True
        )
        
        # Create and register agent
        agent = runner.create_agent(config)
        
        # Execute agent
        result = await agent.execute(
            request=request.request,
            context=request.context,
            session_id=request.session_id
        )
        
        return AgentResponse(
            agent_name=result.agent_name,
            agent_type=result.agent_type.value,
            content=result.content,
            success=result.success,
            execution_time=result.execution_time,
            token_usage=result.token_usage,
            validation_score=result.validation_score,
            validation_passed=result.validation_passed,
            trace_id=result.trace_id,
            metadata=result.metadata,
            error=result.error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@agents_router.get("/types", response_model=List[str])
async def get_agent_types():
    """Get available agent types."""
    return [agent_type.value for agent_type in AgentType]


@agents_router.get("/status", response_model=Dict[str, Any])
async def get_agent_status(
    current_user: str = Depends(get_current_user)
):
    """Get status of agent system."""
    return {
        "available_types": [t.value for t in AgentType],
        "default_model": "gpt-4",
        "tracing_enabled": True,
        "validation_enabled": True
    }


# Workflow endpoints
@workflows_router.post("/execute", response_model=WorkflowResponse)
async def execute_workflow(
    request: WorkflowRequest,
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system),
    _: bool = Depends(rate_limiter)
):
    """Execute a multi-agent workflow."""
    try:
        # Create appropriate workflow
        workflow = None
        
        if request.workflow_type == "guidance":
            workflow = create_multi_agent_guidance_workflow(
                openguidance_system=system,
                model=request.model
            )
        elif request.workflow_type == "trajectory":
            workflow = create_trajectory_optimization_workflow(
                openguidance_system=system,
                model=request.model
            )
        elif request.workflow_type == "mission":
            workflow = create_mission_planning_workflow(
                openguidance_system=system,
                model=request.model
            )
        elif request.workflow_type == "safety":
            workflow = create_safety_validation_workflow(
                openguidance_system=system,
                model=request.model
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid workflow type: {request.workflow_type}. Valid types: guidance, trajectory, mission, safety"
            )
        
        # Execute workflow
        result = await workflow.execute(
            request=request.request,
            context=request.context,
            session_id=request.session_id
        )
        
        # Convert agent results to response format
        agent_responses = []
        for agent_result in result.agent_results:
            agent_responses.append(AgentResponse(
                agent_name=agent_result.agent_name,
                agent_type=agent_result.agent_type.value,
                content=agent_result.content,
                success=agent_result.success,
                execution_time=agent_result.execution_time,
                token_usage=agent_result.token_usage,
                validation_score=agent_result.validation_score,
                validation_passed=agent_result.validation_passed,
                trace_id=agent_result.trace_id,
                metadata=agent_result.metadata,
                error=agent_result.error
            ))
        
        return WorkflowResponse(
            workflow_name=result.workflow_name,
            status=result.status.value,
            execution_time=result.execution_time,
            agent_results=agent_responses,
            summary=result.summary,
            recommendations=result.recommendations,
            metadata=result.metadata,
            error=result.error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@workflows_router.get("/types", response_model=List[str])
async def get_workflow_types():
    """Get available workflow types."""
    return ["guidance", "trajectory", "mission", "safety"]


@workflows_router.get("/status", response_model=Dict[str, Any])
async def get_workflow_status(
    current_user: str = Depends(get_current_user)
):
    """Get status of workflow system."""
    return {
        "available_workflows": ["guidance", "trajectory", "mission", "safety"],
        "default_model": "gpt-4",
        "multi_agent_enabled": True,
        "tracing_enabled": True
    }


# Session management endpoints
@guidance_router.get("/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    limit: int = Query(50, ge=1, le=1000),
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Get conversation history for a session."""
    try:
        if system.memory_manager:
            history = await system.memory_manager.get_conversation_history(
                session_id, limit=limit
            )
            return {
                "session_id": session_id,
                "history": history,
                "count": len(history)
            }
        else:
            return {
                "session_id": session_id,
                "history": [],
                "count": 0,
                "message": "Memory manager not enabled"
            }
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@guidance_router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Clear session data."""
    try:
        if system.memory_manager:
            await system.memory_manager.clear_session(session_id)
            return {
                "session_id": session_id,
                "status": "cleared",
                "message": "Session data cleared successfully"
            }
        else:
            return {
                "session_id": session_id,
                "status": "no_action",
                "message": "Memory manager not enabled"
            }
    except Exception as e:
        logger.error(f"Failed to clear session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Memory management endpoints
@guidance_router.get("/sessions/{session_id}/memory")
async def get_session_memory(
    session_id: str,
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Get memory items for a session."""
    try:
        if system.memory_manager:
            memories = await system.memory_manager.get_session_memories(session_id)
            return {
                "session_id": session_id,
                "memories": memories,
                "count": len(memories)
            }
        else:
            return {
                "session_id": session_id,
                "memories": [],
                "count": 0,
                "message": "Memory manager not enabled"
            }
    except Exception as e:
        logger.error(f"Failed to get session memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@guidance_router.post("/memory")
async def store_memory(
    memory_data: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Store a memory item."""
    try:
        if system.memory_manager:
            memory_id = await system.memory_manager.store_memory(memory_data)
            return {
                "memory_id": memory_id,
                "status": "stored",
                "message": "Memory stored successfully"
            }
        else:
            return {
                "status": "not_stored",
                "message": "Memory manager not enabled"
            }
    except Exception as e:
        logger.error(f"Failed to store memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin endpoints
@admin_router.get("/config")
async def get_config(
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Get system configuration."""
    try:
        return {
            "model_name": system.config.model_name,
            "temperature": system.config.temperature,
            "max_tokens": system.config.max_tokens,
            "enable_memory": system.config.enable_memory,
            "enable_code_execution": system.config.enable_code_execution,
            "enable_validation": system.config.enable_validation,
            "environment": system.config.environment,
            "debug": system.config.debug
        }
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/stats")
async def get_detailed_stats(
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Get detailed system statistics."""
    try:
        stats = system.get_system_stats()
        return {
            "system_stats": stats,
            "agents_enabled": True,
            "workflows_enabled": True,
            "tracing_enabled": True,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/export")
async def export_system_state(
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Export system state."""
    try:
        export_data = await system.export_system_state()
        return {
            "export_id": str(uuid4()),
            "timestamp": time.time(),
            "data": export_data,
            "status": "exported"
        }
    except Exception as e:
        logger.error(f"Failed to export system state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/import")
async def import_system_state(
    import_data: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Import system state."""
    try:
        result = await system.import_system_state(import_data)
        return {
            "status": "imported",
            "imported_items": result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to import system state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/system/restart")
async def restart_system(
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Restart the system."""
    try:
        await system.cleanup()
        await system.initialize()
        return {
            "status": "restarted",
            "timestamp": time.time(),
            "message": "System restarted successfully"
        }
    except Exception as e:
        logger.error(f"Failed to restart system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Monitoring endpoints
@monitoring_router.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "OpenGuidance API",
        "version": "1.0.0"
    }


@monitoring_router.get("/metrics")
async def get_metrics(
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Get system metrics."""
    try:
        stats = system.get_system_stats()
        return {
            "metrics": stats,
            "agents": {
                "enabled": True,
                "types_available": len(AgentType),
                "workflows_available": 4
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/status")
async def get_detailed_status(
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Get detailed system status."""
    try:
        return {
            "system": {
                "initialized": system.is_initialized,
                "uptime": time.time() - system.start_time,
                "memory_enabled": system.config.enable_memory,
                "execution_enabled": system.config.enable_code_execution,
                "validation_enabled": system.config.enable_validation
            },
            "agents": {
                "enabled": True,
                "types": [t.value for t in AgentType],
                "tracing_enabled": True
            },
            "workflows": {
                "enabled": True,
                "types": ["guidance", "trajectory", "mission", "safety"]
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=str(e))