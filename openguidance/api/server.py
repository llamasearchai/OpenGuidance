"""
FastAPI server implementation for OpenGuidance API.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from ..core.system import OpenGuidance
from ..core.config import Config
from .dependencies import get_system, get_current_user, rate_limiter
from .middleware import RequestLoggingMiddleware, SecurityMiddleware
from .routes import guidance_router, admin_router, monitoring_router

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class GuidanceRequest(BaseModel):
    """Request model for guidance processing."""
    message: str = Field(..., min_length=1, max_length=10000, description="User message to process")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    stream: bool = Field(False, description="Whether to stream the response")


class GuidanceResponse(BaseModel):
    """Response model for guidance processing."""
    content: str = Field(..., description="Generated response content")
    session_id: str = Field(..., description="Session ID used")
    execution_time: float = Field(..., description="Processing time in seconds")
    token_usage: Dict[str, int] = Field(..., description="Token usage statistics")
    validation_score: float = Field(..., description="Response validation score")
    validation_passed: bool = Field(..., description="Whether validation passed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health status")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


# Global application state
class AppState:
    def __init__(self):
        self.system: Optional[OpenGuidance] = None
        self.start_time: float = time.time()
        self.request_count: int = 0


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting OpenGuidance API server...")
    
    try:
        # Initialize the guidance system
        config = Config()
        app_state.system = OpenGuidance(config)
        await app_state.system.initialize()
        
        logger.info("OpenGuidance system initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenGuidance system: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down OpenGuidance API server...")
        if app_state.system:
            await app_state.system.cleanup()
        logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="OpenGuidance API",
    description="Advanced AI assistant framework with memory, code execution, and intelligent prompting",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer(auto_error=False)

# Include routers
app.include_router(guidance_router)
app.include_router(admin_router)
app.include_router(monitoring_router)

# Core endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "service": "OpenGuidance API",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    current_time = time.time()
    uptime = current_time - app_state.start_time
    
    components = {
        "api": {"status": "healthy", "uptime": uptime},
        "system": {"status": "unknown", "initialized": False}
    }
    
    if app_state.system:
        try:
            # Check system health
            system_stats = app_state.system.get_system_stats()
            components["system"] = {
                "status": "healthy",
                "initialized": True,
                "stats": system_stats
            }
            
            # Check memory manager
            if hasattr(app_state.system, 'memory_manager') and app_state.system.memory_manager:
                components["memory"] = {
                    "status": "healthy",
                    "available": True
                }
            
        except Exception as e:
            components["system"] = {
                "status": "error",
                "error": str(e)
            }
    
    return HealthResponse(
        status="healthy",
        timestamp=current_time,
        version="1.0.0",
        uptime=uptime,
        components=components
    )


@app.post("/guidance", response_model=GuidanceResponse)
async def process_guidance(
    request: GuidanceRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system),
    _: bool = Depends(rate_limiter)
):
    """Process a guidance request."""
    start_time = time.time()
    session_id = request.session_id or f"session_{int(start_time)}"
    
    try:
        # Process the request
        result = await system.process_request(
            request.message,
            session_id,
            context=request.context
        )
        
        execution_time = time.time() - start_time
        
        # Log analytics in background
        background_tasks.add_task(
            log_request_analytics,
            current_user,
            session_id,
            execution_time,
            True
        )
        
        return GuidanceResponse(
            content=result.content,
            session_id=session_id,
            execution_time=execution_time,
            token_usage=result.metadata.get("token_usage", {}),
            validation_score=result.metadata.get("validation_score", 1.0),
            validation_passed=result.metadata.get("validation_passed", True),
            metadata=result.metadata
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        # Log error analytics in background
        background_tasks.add_task(
            log_request_analytics,
            current_user,
            session_id,
            execution_time,
            False,
            str(e)
        )
        
        logger.error(f"Guidance processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_failed",
                "message": str(e),
                "request_id": session_id
            }
        )


@app.post("/guidance/stream")
async def stream_guidance(
    request: GuidanceRequest,
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system),
    _: bool = Depends(rate_limiter)
):
    """Stream a guidance response."""
    session_id = request.session_id or f"session_{int(time.time())}"
    
    async def generate_stream():
        try:
            async for chunk in system.process_streaming_request(
                request.message,
                session_id,
                context=request.context
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        finally:
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.get("/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    limit: int = 50,
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Get session conversation history."""
    try:
        if not system.memory_manager:
            return {"history": [], "message": "Memory not available"}
        
        memories = await system.memory_manager.get_session_memories(session_id)
        
        # Convert to history format
        history = []
        for memory in memories[-limit:]:
            history.append({
                "timestamp": memory.created_at.isoformat() if hasattr(memory.created_at, 'isoformat') else str(memory.created_at),
                "content": memory.content,
                "type": memory.content_type,
                "metadata": memory.metadata
            })
        
        return {"history": history, "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Clear a session's data."""
    try:
        if system.memory_manager:
            await system.memory_manager.delete_session(session_id)
        
        return {"status": "cleared", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Failed to clear session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_api_stats(
    current_user: str = Depends(get_current_user),
    system: OpenGuidance = Depends(get_system)
):
    """Get API usage statistics."""
    try:
        system_stats = system.get_system_stats()
        
        api_stats = {
            "uptime": time.time() - app_state.start_time,
            "total_requests": app_state.request_count,
            "system_stats": system_stats
        }
        
        return api_stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def log_request_analytics(
    user: str,
    session_id: str,
    execution_time: float,
    success: bool,
    error: Optional[str] = None
):
    """Log request analytics (background task)."""
    app_state.request_count += 1
    
    log_data = {
        "user": user,
        "session_id": session_id,
        "execution_time": execution_time,
        "success": success,
        "timestamp": time.time(),
        "error": error
    }
    
    logger.info(f"Request analytics: {log_data}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An internal error occurred"
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "openguidance.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )