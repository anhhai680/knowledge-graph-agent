"""
FastAPI main application with LangGraph workflow integration.

This module sets up the FastAPI application for the Knowledge Graph Agent,
providing comprehensive REST API endpoints for repository indexing, query
processing, and workflow management with full LangGraph integration.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.workflows.indexing_workflow import IndexingWorkflow
from src.workflows.query_workflow import QueryWorkflow
from src.vectorstores.store_factory import VectorStoreFactory
from src.llm.llm_factory import LLMFactory
from src.llm.embedding_factory import EmbeddingFactory

logger = get_logger(__name__)

# Global workflow instances for dependency injection
workflow_instances: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events for the FastAPI application,
    initializing workflow instances and cleaning up resources.
    """
    # Startup
    logger.info("Starting Knowledge Graph Agent API...")
    
    try:
        settings = get_settings()
        
        # Initialize core factories
        logger.info("Initializing core factories...")
        vector_store_factory = VectorStoreFactory()
        llm_factory = LLMFactory()
        embedding_factory = EmbeddingFactory()
        
        # Initialize workflow instances
        logger.info("Initializing workflow instances...")
        workflow_instances["indexing"] = IndexingWorkflow(
            vector_store_factory=vector_store_factory,
            llm_factory=llm_factory,
            embedding_factory=embedding_factory
        )
        
        workflow_instances["query"] = QueryWorkflow(
            vector_store_factory=vector_store_factory,
            llm_factory=llm_factory,
            embedding_factory=embedding_factory
        )
        
        # Validate configurations
        logger.info("Validating component configurations...")
        await _validate_components()
        
        logger.info("Knowledge Graph Agent API started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise HTTPException(status_code=500, detail=f"Application startup failed: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Knowledge Graph Agent API...")
    workflow_instances.clear()
    logger.info("Knowledge Graph Agent API shut down complete!")


async def _validate_components():
    """
    Validate that all core components are properly configured.
    
    Raises:
        HTTPException: If any component validation fails
    """
    try:
        settings = get_settings()
        
        # Validate required settings
        required_settings = [
            "OPENAI_API_KEY",
            "VECTOR_STORE_TYPE",
            "GITHUB_TOKEN"
        ]
        
        missing_settings = []
        for setting in required_settings:
            if not getattr(settings, setting, None):
                missing_settings.append(setting)
        
        if missing_settings:
            raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")
        
        # Test workflow initialization
        if "indexing" not in workflow_instances or "query" not in workflow_instances:
            raise ValueError("Failed to initialize workflow instances")
        
        logger.info("All component validations passed")
        
    except Exception as e:
        logger.error(f"Component validation failed: {e}")
        raise


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title="Knowledge Graph Agent API",
        description="""
        **Knowledge Graph Agent REST API**
        
        This API provides comprehensive endpoints for:
        - Repository indexing with LangGraph workflows
        - Intelligent query processing with adaptive RAG
        - Workflow status monitoring and management
        - Vector store operations and metadata retrieval
        
        **Key Features:**
        - LangGraph workflow integration for stateful processing
        - Multi-repository support with GitHub integration
        - Language-aware document chunking (.NET, React, Python)
        - Vector store abstraction (Chroma, Pinecone)
        - Real-time workflow progress tracking
        - Comprehensive error handling and retry mechanisms
        
        **Authentication:**
        All endpoints require API key authentication via the `X-API-Key` header.
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    
    # Add comprehensive middleware stack
    # app.add_middleware(HealthMonitoringMiddleware)
    # app.add_middleware(WorkflowMonitoringMiddleware)
    # app.add_middleware(RequestLoggingMiddleware)
    
    # Configure CORS (simplified for now)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, configure properly
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router, prefix="/api/v1")
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Global exception handler caught: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        try:
            settings = get_settings()
            return {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "workflows": {
                        "indexing": "indexing" in workflow_instances,
                        "query": "query" in workflow_instances
                    },
                    "vector_store": settings.database_type.value,
                    "llm_provider": "openai"
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e)
                }
            )
    
    return app


def get_indexing_workflow() -> IndexingWorkflow:
    """
    Dependency to get the indexing workflow instance.
    
    Returns:
        IndexingWorkflow: The global indexing workflow instance
        
    Raises:
        HTTPException: If workflow instance is not available
    """
    if "indexing" not in workflow_instances:
        raise HTTPException(
            status_code=503,
            detail="Indexing workflow not available"
        )
    return workflow_instances["indexing"]


def get_query_workflow() -> QueryWorkflow:
    """
    Dependency to get the query workflow instance.
    
    Returns:
        QueryWorkflow: The global query workflow instance
        
    Raises:
        HTTPException: If workflow instance is not available
    """
    if "query" not in workflow_instances:
        raise HTTPException(
            status_code=503,
            detail="Query workflow not available"
        )
    return workflow_instances["query"]


def get_vector_store():
    """
    Dependency to get the vector store instance.
    
    Returns:
        BaseStore: The vector store instance
        
    Raises:
        HTTPException: If vector store instance is not available
    """
    try:
        vector_store_factory = VectorStoreFactory()
        return vector_store_factory.create()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store not available: {str(e)}"
        )


# Create the FastAPI app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug" if settings.app_env.value == "development" else "info"
    )
