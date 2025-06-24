"""
Dependency injection functions for OpenGuidance API.
"""

import time
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting state
rate_limit_state: Dict[str, Dict[str, Any]] = {}


async def get_system():
    """
    Dependency to get the OpenGuidance system instance.
    This is a placeholder that will be replaced with actual system instance.
    """
    from .server import app_state
    
    if not app_state.system:
        raise HTTPException(
            status_code=503,
            detail="OpenGuidance system not available"
        )
    
    return app_state.system


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    Dependency to get the current authenticated user.
    In production, this would validate JWT tokens or API keys.
    """
    # For development, allow unauthenticated access
    if not credentials:
        return "anonymous"
    
    # Simple token validation (replace with proper JWT validation in production)
    token = credentials.credentials
    
    # Basic API key validation
    valid_tokens = {
        "dev-token-123": "developer",
        "test-token-456": "tester",
        "admin-token-789": "admin"
    }
    
    user = valid_tokens.get(token)
    if not user:
        # For development, accept any token as a user identifier
        return f"user_{hash(token) % 10000}"
    
    return user


async def get_admin_user(current_user: str = Depends(get_current_user)) -> str:
    """
    Dependency to ensure the current user has admin privileges.
    """
    admin_users = {"admin", "developer"}
    
    if current_user not in admin_users:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    
    return current_user


async def rate_limiter(
    request: Request,
    current_user: str = Depends(get_current_user)
) -> bool:
    """
    Rate limiting dependency.
    Implements token bucket algorithm for rate limiting.
    """
    # Rate limit configuration
    rate_limits = {
        "anonymous": {"requests_per_minute": 10, "burst_size": 20},
        "developer": {"requests_per_minute": 100, "burst_size": 200},
        "admin": {"requests_per_minute": 1000, "burst_size": 2000},
        "default": {"requests_per_minute": 60, "burst_size": 120}
    }
    
    # Get rate limit config for user
    user_config = rate_limits.get(current_user, rate_limits["default"])
    requests_per_minute = user_config["requests_per_minute"]
    burst_size = user_config["burst_size"]
    
    # Initialize user state if not exists
    current_time = time.time()
    if current_user not in rate_limit_state:
        rate_limit_state[current_user] = {
            "tokens": burst_size,
            "last_update": current_time
        }
    
    user_state = rate_limit_state[current_user]
    
    # Calculate tokens to add based on time elapsed
    time_elapsed = current_time - user_state["last_update"]
    tokens_to_add = time_elapsed * (requests_per_minute / 60.0)
    
    # Add tokens (up to burst_size)
    user_state["tokens"] = min(burst_size, user_state["tokens"] + tokens_to_add)
    user_state["last_update"] = current_time
    
    # Check if request can be processed
    if user_state["tokens"] >= 1:
        user_state["tokens"] -= 1
        return True
    else:
        # Rate limit exceeded
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": "Too many requests",
                "retry_after": 60 / requests_per_minute
            }
        )


async def get_request_context(request: Request) -> Dict[str, Any]:
    """
    Dependency to extract request context information.
    """
    return {
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "request_id": getattr(request.state, "request_id", None),
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers)
    }


async def validate_content_type(request: Request) -> bool:
    """
    Dependency to validate request content type for POST requests.
    """
    if request.method in ["POST", "PUT", "PATCH"]:
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            raise HTTPException(
                status_code=415,
                detail="Content-Type must be application/json"
            )
    
    return True


class DatabaseSession:
    """
    Database session dependency (placeholder for actual database integration).
    """
    
    def __init__(self):
        self.is_active = True
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        self.is_active = False


async def get_db_session() -> DatabaseSession:
    """
    Dependency to get database session.
    In production, this would return actual database session.
    """
    session = DatabaseSession()
    try:
        yield session
    finally:
        await session.close()


async def get_cache():
    """
    Dependency to get cache instance.
    In production, this would return Redis or other cache instance.
    """
    # Placeholder cache implementation
    class SimpleCache:
        def __init__(self):
            self._cache = {}
        
        async def get(self, key: str) -> Optional[Any]:
            return self._cache.get(key)
        
        async def set(self, key: str, value: Any, ttl: int = 300) -> None:
            self._cache[key] = {
                "value": value,
                "expires_at": time.time() + ttl
            }
        
        async def delete(self, key: str) -> None:
            self._cache.pop(key, None)
        
        async def clear_expired(self) -> None:
            current_time = time.time()
            expired_keys = [
                key for key, data in self._cache.items()
                if data["expires_at"] < current_time
            ]
            for key in expired_keys:
                del self._cache[key]
    
    cache = SimpleCache()
    await cache.clear_expired()
    return cache


async def get_metrics_collector():
    """
    Dependency to get metrics collector instance.
    """
    # Placeholder metrics collector
    class MetricsCollector:
        def __init__(self):
            self.metrics = {}
        
        def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
            if name not in self.metrics:
                self.metrics[name] = {"count": 0, "total": 0.0}
            self.metrics[name]["count"] += 1
            self.metrics[name]["total"] += value
        
        def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
            self.metrics[name] = {"current": value}
        
        def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
            if name not in self.metrics:
                self.metrics[name] = {"values": []}
            self.metrics[name]["values"].append(value)
        
        def get_metrics(self) -> Dict[str, Any]:
            return self.metrics.copy()
    
    return MetricsCollector()


# Configuration dependency
class APIConfig:
    """API configuration settings."""
    
    def __init__(self):
        import os
        
        self.debug = os.getenv("OPENGUIDANCE_DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("OPENGUIDANCE_LOG_LEVEL", "INFO")
        self.api_host = os.getenv("OPENGUIDANCE_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("OPENGUIDANCE_PORT", "8000"))
        self.cors_origins = os.getenv("OPENGUIDANCE_CORS_ORIGINS", "*").split(",")
        self.max_request_size = int(os.getenv("OPENGUIDANCE_MAX_REQUEST_SIZE", "10485760"))  # 10MB
        self.request_timeout = int(os.getenv("OPENGUIDANCE_REQUEST_TIMEOUT", "30"))
        
        # Rate limiting
        self.rate_limit_enabled = os.getenv("OPENGUIDANCE_RATE_LIMIT", "true").lower() == "true"
        self.default_rate_limit = int(os.getenv("OPENGUIDANCE_DEFAULT_RATE_LIMIT", "60"))
        
        # Security
        self.require_auth = os.getenv("OPENGUIDANCE_REQUIRE_AUTH", "false").lower() == "true"
        self.jwt_secret = os.getenv("OPENGUIDANCE_JWT_SECRET", "development-secret-key")
        self.jwt_algorithm = os.getenv("OPENGUIDANCE_JWT_ALGORITHM", "HS256")


async def get_api_config() -> APIConfig:
    """Dependency to get API configuration."""
    return APIConfig()