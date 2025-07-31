"""
Authentication middleware and workflow monitoring for Knowledge Graph Agent API.

This module implements comprehensive security and monitoring features including:
- API key authentication middleware
- Request logging and response time tracking
- Rate limiting and request validation
- CORS configuration
- Workflow progress tracking with real-time status updates
- Health monitoring with LangGraph workflow status
"""

import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Global middleware state
request_stats: Dict[str, Any] = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "average_response_time": 0.0,
    "active_requests": 0,
    "rate_limit_violations": 0
}

# Rate limiting storage
rate_limit_storage: Dict[str, deque] = defaultdict(lambda: deque())

# Active API keys (in production, this would be from a database)
VALID_API_KEYS = {
    "kg-agent-key-001": {
        "name": "Development Key",
        "permissions": ["read", "write", "admin"],
        "rate_limit": 1000,  # requests per hour
        "created_at": datetime.now(),
        "last_used": None
    },
    "kg-agent-key-002": {
        "name": "Production Key", 
        "permissions": ["read", "write"],
        "rate_limit": 500,
        "created_at": datetime.now(),
        "last_used": None
    }
}


class APIKeyAuthentication(HTTPBearer):
    """
    API Key authentication handler.
    
    Validates API keys provided in the Authorization header or X-API-Key header
    and manages key-based permissions and rate limiting.
    """
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.settings = get_settings()
    
    async def __call__(self, request: Request) -> Optional[str]:
        """
        Authenticate API key from request headers.
        
        Args:
            request: FastAPI request object
            
        Returns:
            API key if valid, None if invalid
            
        Raises:
            HTTPException: If authentication fails and auto_error is True
        """
        # Skip authentication for health check and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return None
        
        # Try to get API key from different headers
        api_key = None
        
        # Check X-API-Key header first
        api_key = request.headers.get("X-API-Key")
        
        # If not found, check Authorization header
        if not api_key:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                api_key = auth_header.replace("Bearer ", "")
        
        if not api_key:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required. Provide in X-API-Key header or Authorization Bearer token.",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            return None
        
        return await self.validate_api_key(api_key, request)
    
    async def validate_api_key(self, api_key: str, request: Request) -> str:
        """
        Validate API key and update usage statistics.
        
        Args:
            api_key: API key to validate
            request: FastAPI request object
            
        Returns:
            Valid API key
            
        Raises:
            HTTPException: If API key is invalid or rate limited
        """
        if api_key not in VALID_API_KEYS:
            logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        key_info = VALID_API_KEYS[api_key]
        
        # Check rate limiting
        client_ip = self.get_client_ip(request)
        if not await self.check_rate_limit(api_key, client_ip, key_info["rate_limit"]):
            rate_limit_storage[api_key].clear()  # Reset on violation
            request_stats["rate_limit_violations"] += 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {key_info['rate_limit']} requests per hour."
            )
        
        # Update key usage
        key_info["last_used"] = datetime.now()
        
        # Store key info in request state for later use
        request.state.api_key = api_key
        request.state.api_key_info = key_info
        
        logger.info(f"API key authenticated: {key_info['name']} from {client_ip}")
        return api_key
    
    async def check_rate_limit(self, api_key: str, client_ip: str, limit: int) -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            api_key: API key making the request
            client_ip: Client IP address
            limit: Rate limit per hour
            
        Returns:
            True if within limits, False if exceeded
        """
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        # Clean old requests
        key_requests = rate_limit_storage[api_key]
        while key_requests and key_requests[0] < one_hour_ago:
            key_requests.popleft()
        
        # Check if under limit
        if len(key_requests) < limit:
            key_requests.append(now)
            return True
        
        return False
    
    def get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client IP address
        """
        # Check for forwarded headers first (behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging and response time tracking middleware.
    
    Logs all requests with detailed information including:
    - Request details (method, path, headers, body size)
    - Response details (status code, response time, body size)
    - Authentication information
    - Error details for failed requests
    """
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Process request and log details.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response object
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        
        # Log request details
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": self.get_client_ip(request),
                "user_agent": request.headers.get("User-Agent", ""),
                "content_length": request.headers.get("Content-Length", 0)
            }
        )
        
        # Update global stats
        request_stats["total_requests"] += 1
        request_stats["active_requests"] += 1
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update statistics
            if response.status_code < 400:
                request_stats["successful_requests"] += 1
            else:
                request_stats["failed_requests"] += 1
            
            # Update average response time
            total_requests = request_stats["total_requests"]
            current_avg = request_stats["average_response_time"]
            request_stats["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
            
            # Log response details
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "content_length": response.headers.get("Content-Length", 0)
                }
            )
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = str(response_time)
            
            return response
            
        except Exception as e:
            # Calculate response time for errors
            response_time = time.time() - start_time
            
            # Update error statistics
            request_stats["failed_requests"] += 1
            
            # Log error details
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "response_time": response_time
                }
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Response-Time": str(response_time)
                }
            )
        
        finally:
            request_stats["active_requests"] -= 1
    
    def get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class WorkflowMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Workflow monitoring and progress tracking middleware.
    
    Provides real-time monitoring of LangGraph workflows including:
    - Workflow execution metrics
    - Progress tracking and status updates
    - Performance monitoring
    - Error tracking and recovery
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.workflow_metrics: Dict[str, Any] = {
            "active_workflows": 0,
            "completed_workflows": 0,
            "failed_workflows": 0,
            "average_workflow_time": 0.0,
            "workflow_queue_size": 0
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Monitor workflow-related requests.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response object with workflow monitoring
        """
        # Check if this is a workflow-related request
        is_workflow_request = any(
            endpoint in request.url.path
            for endpoint in ["/index", "/query", "/workflows"]
        )
        
        if is_workflow_request:
            return await self.monitor_workflow_request(request, call_next)
        
        return await call_next(request)
    
    async def monitor_workflow_request(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Monitor workflow requests with detailed tracking.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response with workflow monitoring headers
        """
        start_time = time.time()
        
        # Determine workflow type
        workflow_type = self.determine_workflow_type(request)
        
        # Update workflow queue metrics
        if workflow_type:
            self.workflow_metrics["workflow_queue_size"] += 1
            if workflow_type in ["indexing", "query"]:
                self.workflow_metrics["active_workflows"] += 1
        
        try:
            response = await call_next(request)
            
            # Calculate workflow time
            workflow_time = time.time() - start_time
            
            # Update success metrics
            if response.status_code < 400 and workflow_type:
                self.workflow_metrics["completed_workflows"] += 1
                self.update_average_workflow_time(workflow_time)
            
            # Add workflow monitoring headers
            if workflow_type:
                response.headers["X-Workflow-Type"] = workflow_type
                response.headers["X-Workflow-Time"] = str(workflow_time)
                response.headers["X-Active-Workflows"] = str(self.workflow_metrics["active_workflows"])
            
            return response
            
        except Exception as e:
            # Update failure metrics
            if workflow_type:
                self.workflow_metrics["failed_workflows"] += 1
            
            logger.error(f"Workflow request failed: {e}")
            raise
        
        finally:
            # Update queue and active workflow counts
            if workflow_type:
                self.workflow_metrics["workflow_queue_size"] = max(0, self.workflow_metrics["workflow_queue_size"] - 1)
                if workflow_type in ["indexing", "query"]:
                    self.workflow_metrics["active_workflows"] = max(0, self.workflow_metrics["active_workflows"] - 1)
    
    def determine_workflow_type(self, request: Request) -> Optional[str]:
        """
        Determine workflow type from request path.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Workflow type or None
        """
        path = request.url.path
        
        if "/index" in path:
            return "indexing"
        elif "/query" in path:
            return "query"
        elif "/workflows" in path:
            return "monitoring"
        
        return None
    
    def update_average_workflow_time(self, workflow_time: float):
        """
        Update average workflow execution time.
        
        Args:
            workflow_time: Time taken for workflow execution
        """
        completed = self.workflow_metrics["completed_workflows"]
        current_avg = self.workflow_metrics["average_workflow_time"]
        
        self.workflow_metrics["average_workflow_time"] = (
            (current_avg * (completed - 1) + workflow_time) / completed
        )


class HealthMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Health monitoring middleware for system components.
    
    Monitors the health of:
    - LangGraph workflows
    - LangChain components
    - Vector stores
    - LLM providers
    - System resources
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.health_checks = {
            "last_check": datetime.now(),
            "component_status": {
                "workflows": True,
                "vector_store": True,
                "llm_provider": True,
                "embedding_service": True
            },
            "system_metrics": {
                "memory_usage": 0.0,
                "cpu_usage": 0.0,
                "disk_usage": 0.0
            }
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Monitor system health during request processing.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response with health monitoring headers
        """
        # Perform health check if needed (every 60 seconds)
        now = datetime.now()
        if (now - self.health_checks["last_check"]).seconds > 60:
            await self.perform_health_check()
            self.health_checks["last_check"] = now
        
        response = await call_next(request)
        
        # Add health status headers
        overall_health = all(self.health_checks["component_status"].values())
        response.headers["X-System-Health"] = "healthy" if overall_health else "degraded"
        response.headers["X-Health-Check-Time"] = self.health_checks["last_check"].isoformat()
        
        return response
    
    async def perform_health_check(self):
        """
        Perform comprehensive system health check.
        
        Updates component status and system metrics.
        """
        try:
            # Check workflow availability
            # In production, this would actually test workflow instances
            self.health_checks["component_status"]["workflows"] = True
            
            # Check vector store connectivity
            # In production, this would test actual vector store connection
            self.health_checks["component_status"]["vector_store"] = True
            
            # Check LLM provider
            # In production, this would test OpenAI API connectivity
            self.health_checks["component_status"]["llm_provider"] = True
            
            # Check embedding service
            # In production, this would test embedding service
            self.health_checks["component_status"]["embedding_service"] = True
            
            # Update system metrics (simplified)
            self.health_checks["system_metrics"].update({
                "memory_usage": 0.0,  # Would get actual memory usage
                "cpu_usage": 0.0,     # Would get actual CPU usage
                "disk_usage": 0.0     # Would get actual disk usage
            })
            
            logger.debug("Health check completed successfully")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            # Mark components as unhealthy on failure
            for component in self.health_checks["component_status"]:
                self.health_checks["component_status"][component] = False


# Utility functions for middleware configuration

def configure_cors_middleware(app, settings):
    """
    Configure CORS middleware for the application.
    
    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Response-Time", 
            "X-Workflow-Type",
            "X-Workflow-Time",
            "X-Active-Workflows",
            "X-System-Health",
            "X-Health-Check-Time"
        ]
    )


def get_request_statistics() -> Dict[str, Any]:
    """
    Get current request statistics.
    
    Returns:
        Dictionary containing request statistics
    """
    return request_stats.copy()


def get_workflow_metrics() -> Dict[str, Any]:
    """
    Get current workflow metrics.
    
    Returns:
        Dictionary containing workflow metrics
    """
    # This would typically be retrieved from a workflow monitoring instance
    return {
        "active_workflows": request_stats.get("active_requests", 0),
        "completed_workflows": request_stats.get("successful_requests", 0),
        "failed_workflows": request_stats.get("failed_requests", 0),
        "average_workflow_time": request_stats.get("average_response_time", 0.0)
    }


def reset_statistics():
    """Reset all middleware statistics (useful for testing)."""
    global request_stats, rate_limit_storage
    
    request_stats.update({
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "average_response_time": 0.0,
        "active_requests": 0,
        "rate_limit_violations": 0
    })
    
    rate_limit_storage.clear()
