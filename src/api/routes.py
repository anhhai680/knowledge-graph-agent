"""
API routes for the Knowledge Graph Agent with comprehensive LangGraph integration.

This module implements all MVP endpoints for repository indexing, query processing,
workflow management, and system monitoring with full LangGraph workflow integration.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query

from src.api.models import (
    IndexRepositoryRequest,
    QueryRequest,
    QueryResponse,
    IndexingResponse,
    BatchIndexingResponse,
    RepositoriesResponse,
    StatsResponse,
    HealthResponse,
    WorkflowState,
    WorkflowStatus,
    RepositoryInfo,
    DocumentResult,
    DocumentMetadata,
    QueryIntent,
    SearchStrategy
)
from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.workflows.indexing_workflow import IndexingWorkflow
from src.workflows.query_workflow import QueryWorkflow


logger = get_logger(__name__)

# Global workflow tracking
active_workflows: Dict[str, Dict] = {}


def _map_workflow_intent_to_api(workflow_intent: str) -> QueryIntent:
    """Map workflow QueryIntent to API QueryIntent."""
    mapping = {
        "code_search": QueryIntent.CODE_SEARCH,
        "documentation": QueryIntent.DOCUMENTATION,
        "explanation": QueryIntent.GENERAL,  # Map explanation to general
        "debugging": QueryIntent.DEBUGGING,
        "architecture": QueryIntent.GENERAL,  # Map architecture to general
    }
    return mapping.get(workflow_intent, QueryIntent.GENERAL)


def _map_workflow_strategy_to_api(workflow_strategy: str) -> SearchStrategy:
    """Map workflow SearchStrategy to API SearchStrategy."""
    mapping = {
        "semantic": SearchStrategy.SEMANTIC,
        "keyword": SearchStrategy.KEYWORD,
        "hybrid": SearchStrategy.HYBRID,
        "metadata_filtered": SearchStrategy.HYBRID,  # Map metadata_filtered to hybrid
    }
    return mapping.get(workflow_strategy, SearchStrategy.HYBRID)


# Create router instance
router = APIRouter(
    tags=["Knowledge Graph Agent API"],
    responses={
        404: {"description": "Not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)


def get_indexing_workflow() -> IndexingWorkflow:
    """Dependency injection for indexing workflow."""
    # This will be implemented in main.py as a dependency
    from src.api.main import get_indexing_workflow
    return get_indexing_workflow()


def get_query_workflow() -> QueryWorkflow:
    """Dependency injection for query workflow.""" 
    # This will be implemented in main.py as a dependency
    from src.api.main import get_query_workflow
    return get_query_workflow()


def get_vector_store():
    """Dependency injection for vector store."""
    # This will be implemented in main.py as a dependency
    from src.api.main import get_vector_store
    return get_vector_store()


@router.get("/")
async def root():
    """Welcome endpoint with API information."""
    return {
        "message": "Welcome to the Knowledge Graph Agent API!",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "indexing": "/api/v1/index",
            "query": "/api/v1/query",
            "repositories": "/api/v1/repositories",
            "stats": "/api/v1/stats",
            "workflows": "/api/v1/workflows"
        }
    }


@router.post("/index", response_model=BatchIndexingResponse)
async def index_all_repositories(
    background_tasks: BackgroundTasks,
    indexing_workflow: IndexingWorkflow = Depends(get_indexing_workflow)
):
    """
    Trigger LangGraph indexing workflow for all repositories from appSettings.json.
    
    This endpoint reads the repositories from appSettings.json and initiates
    indexing workflows for all configured repositories in parallel.
    """
    try:
        logger.info("Starting batch indexing for all repositories from appSettings.json")
        
        # Load repositories from appSettings.json
        settings = get_settings()
        with open("appSettings.json", "r") as f:
            app_settings = json.load(f)
        
        repositories = app_settings.get("repositories", [])
        if not repositories:
            raise HTTPException(
                status_code=400,
                detail="No repositories found in appSettings.json"
            )
        
        batch_id = str(uuid.uuid4())
        workflows = []
        
        for repo_config in repositories:
            workflow_id = str(uuid.uuid4())
            
            # Create indexing request from config
            repo_request = IndexRepositoryRequest(
                repository_url=repo_config["url"],
                branch=repo_config.get("branch", "main"),
                file_extensions=repo_config.get("file_extensions"),
                max_files=repo_config.get("max_files")
            )
            
            # Start background indexing workflow
            background_tasks.add_task(
                _run_indexing_workflow,
                workflow_id,
                repo_request,
                indexing_workflow
            )
            
            # Track workflow
            active_workflows[workflow_id] = {
                "id": workflow_id,
                "type": "indexing",
                "status": WorkflowStatus.PENDING,
                "repository": repo_request.repository_url,
                "started_at": datetime.now(),
                "batch_id": batch_id
            }
            
            workflows.append(IndexingResponse(
                workflow_id=workflow_id,
                repository=repo_request.repository_url,
                status=WorkflowStatus.PENDING,
                estimated_duration="10-30 minutes",
                message=f"Indexing workflow queued for {repo_request.repository_url}"
            ))
        
        logger.info(f"Started batch indexing for {len(workflows)} repositories")
        
        return BatchIndexingResponse(
            workflows=workflows,
            batch_id=batch_id,
            total_repositories=len(workflows),
            message=f"Batch indexing started for {len(workflows)} repositories"
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="appSettings.json file not found"
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in appSettings.json"
        )
    except Exception as e:
        logger.error(f"Batch indexing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch indexing failed: {str(e)}"
        )


@router.post("/index/repository", response_model=IndexingResponse)
async def index_repository(
    request: IndexRepositoryRequest,
    background_tasks: BackgroundTasks,
    indexing_workflow: IndexingWorkflow = Depends(get_indexing_workflow)
):
    """
    Trigger LangGraph indexing workflow for a specific repository.
    
    This endpoint initiates the indexing workflow for a single repository,
    processing all files and generating vector embeddings for storage.
    """
    try:
        workflow_id = str(uuid.uuid4())
        
        logger.info(f"Starting indexing workflow for repository: {request.repository_url}")
        
        # Start background indexing workflow
        background_tasks.add_task(
            _run_indexing_workflow,
            workflow_id,
            request,
            indexing_workflow
        )
        
        # Track workflow
        active_workflows[workflow_id] = {
            "id": workflow_id,
            "type": "indexing",
            "status": WorkflowStatus.PENDING,
            "repository": request.repository_url,
            "started_at": datetime.now()
        }
        
        return IndexingResponse(
            workflow_id=workflow_id,
            repository=request.repository_url,
            status=WorkflowStatus.PENDING,
            estimated_duration="10-30 minutes",
            message=f"Indexing workflow started for {request.repository_url}"
        )
        
    except Exception as e:
        logger.error(f"Repository indexing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Repository indexing failed: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    query_workflow: QueryWorkflow = Depends(get_query_workflow),
):
    """
    Execute LangGraph query workflow with adaptive RAG processing.
    
    This endpoint processes user queries using the complete LangGraph query
    workflow, including intent analysis, document retrieval, and response generation.
    """
    try:
        start_time = datetime.now()
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Execute query workflow using the workflow's proper state creation
        result_state = await query_workflow.run(
            query=request.query,
            repositories=request.repositories,
            languages=request.language_filter,
            k=request.top_k or 5
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert workflow results to API response
        document_results = []
        for doc in result_state.get("context_documents", []):
            # Create DocumentMetadata object properly
            doc_metadata = DocumentMetadata(
                source=doc.get("metadata", {}).get("source", ""),
                repository=doc.get("metadata", {}).get("repository", ""),
                language=doc.get("metadata", {}).get("language"),
                file_type=doc.get("metadata", {}).get("file_type"),
                symbols=doc.get("metadata", {}).get("symbols", []),
                last_modified=doc.get("metadata", {}).get("last_modified"),
                size=doc.get("metadata", {}).get("size")
            )
            
            document_results.append(DocumentResult(
                content=doc.get("content", ""),  # Fixed: changed from "page_content" to "content"
                metadata=doc_metadata,
                score=0.0,  # Fixed: Set to 0.0 since similarity scores aren't computed in current workflow
                chunk_index=doc.get("metadata", {}).get("chunk_index")
            ))
        
        # Calculate confidence score with proper fallback
        confidence_score = result_state.get("response_confidence")
        if confidence_score is None:  
            # Calculate a basic confidence score based on retrieved documents  
            doc_count = len(document_results)  
            if doc_count == 0:  
                confidence_score = 0.0  
            else:  
                # Simple confidence calculation based on document count and content  
                base_confidence = min(doc_count / 5.0, 1.0)  # More docs = higher confidence  
                total_content_length = sum(len(doc.content) for doc in document_results)  
                content_confidence = min(total_content_length / 2000.0, 1.0)  # More content = higher confidence  
                confidence_score = (base_confidence * 0.6 + content_confidence * 0.4) # Weighted average
        
        response = QueryResponse(
            query=request.query,
            intent=_map_workflow_intent_to_api(str(result_state.get("query_intent", "general") or "general")),  # Fixed: ensure string type
            strategy=_map_workflow_strategy_to_api(result_state.get("search_strategy", "hybrid")),  # Fixed: use mapping function
            results=document_results,
            total_results=len(document_results),
            processing_time=processing_time,
            confidence_score=confidence_score,
            suggestions=result_state.get("suggestions", [])
        )
        
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@router.get("/repositories", response_model=RepositoriesResponse)
async def list_repositories(
    vector_store=Depends(get_vector_store)
):
    """
    List indexed repositories with metadata from vector store.
    
    Returns comprehensive information about all indexed repositories,
    including file counts, languages, and indexing status.
    """
    try:
        logger.info("Retrieving repository list with metadata from vector store")
        
        # Check for dimension mismatch first
        try:
            is_compatible, compatibility_msg = vector_store.check_embedding_dimension_compatibility()
            logger.info(f"Embedding compatibility check: {is_compatible}, message: {compatibility_msg}")
            if not is_compatible:
                logger.warning(f"Dimension mismatch detected: {compatibility_msg}")
                logger.info("Falling back to configuration-based repository list due to dimension mismatch")
                # Fall through to appSettings.json fallback
                repository_metadata = []
            else:
                # Get repository information from vector store
                logger.info("Getting repository metadata from vector store...")
                repository_metadata = vector_store.get_repository_metadata()
                logger.info(f"Retrieved {len(repository_metadata)} repositories from vector store")
        except Exception as e:
            logger.warning(f"Could not check embedding compatibility: {str(e)}")
            # Try to get repository metadata anyway, let the method handle the error
            logger.info("Trying to get repository metadata despite compatibility check failure...")
            repository_metadata = vector_store.get_repository_metadata()
            logger.info(f"Retrieved {len(repository_metadata)} repositories from vector store")
        
        # Convert repository metadata to RepositoryInfo objects
        repositories = []
        logger.info(f"Processing {len(repository_metadata)} repository metadata entries")
        for repo_data in repository_metadata:
            try:
                # Parse last_indexed date if it's a string
                last_indexed = repo_data.get("last_indexed")
                if isinstance(last_indexed, str):
                    try:
                        from dateutil import parser
                        last_indexed = parser.parse(last_indexed)
                    except Exception:
                        last_indexed = datetime.now()
                elif not last_indexed:
                    last_indexed = datetime.now()
                
                logger.info(f"Processing repository: {repo_data.get('name')} - files: {repo_data.get('file_count')}, docs: {repo_data.get('document_count')}")
                
                repository_info = RepositoryInfo(
                    name=repo_data.get("name", "Unknown"),
                    url=repo_data.get("url", ""),
                    branch=repo_data.get("branch", "main"),
                    last_indexed=last_indexed,
                    file_count=repo_data.get("file_count", 0),
                    document_count=repo_data.get("document_count", 0),
                    languages=repo_data.get("languages", []),
                    size_mb=repo_data.get("size_mb", 0.0)
                )
                repositories.append(repository_info)
                
            except Exception as e:
                logger.warning(f"Error processing repository metadata: {e}")
                # Skip malformed repository entries
                continue
        
        # If no repositories found, try to check from appSettings.json as fallback
        if not repositories:
            logger.info("No repositories found in vector store, checking appSettings.json")
            try:
                with open("appSettings.json", "r") as f:
                    app_settings = json.load(f)
                    
                configured_repos = app_settings.get("repositories", [])
                for repo_config in configured_repos:
                    # Create placeholder entries for configured but not yet indexed repositories
                    repo_url = repo_config.get("url", "")
                    repo_name = repo_url.split("/")[-2:] if "/" in repo_url else [repo_url]
                    if len(repo_name) >= 2:
                        repo_name = f"{repo_name[-2]}/{repo_name[-1]}"
                    else:
                        repo_name = repo_url
                    
                    repositories.append(RepositoryInfo(
                        name=repo_name,
                        url=repo_url,
                        branch=repo_config.get("branch", "main"),
                        last_indexed=datetime.now(),  # Placeholder
                        file_count=0,  # Not indexed yet
                        document_count=0,  # Not indexed yet
                        languages=[],  # Unknown until indexed
                        size_mb=0.0  # Unknown until indexed
                    ))
                    
            except FileNotFoundError:
                logger.warning("No appSettings.json found and no repositories in vector store")
            except Exception as e:
                logger.error(f"Error reading appSettings.json: {e}")
        
        return RepositoriesResponse(
            repositories=repositories,
            total_count=len(repositories),
            last_updated=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to list repositories: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list repositories: {str(e)}"
        )
        logger.error(f"Failed to list repositories: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list repositories: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    indexing_workflow: IndexingWorkflow = Depends(get_indexing_workflow),
    query_workflow: QueryWorkflow = Depends(get_query_workflow),
    vector_store=Depends(get_vector_store)
):
    """
    Health check with LangGraph workflow status and vector store component health.
    
    Provides comprehensive system health information including workflow
    availability, component status, and system metrics.
    """
    try:
        # Check vector store health
        vector_store_healthy = True
        try:
            is_healthy, _ = vector_store.health_check()
            vector_store_healthy = is_healthy
        except Exception:
            vector_store_healthy = False
        
        components = {
            "workflows": {
                "indexing": indexing_workflow is not None,
                "query": query_workflow is not None
            },
            "vector_store": vector_store_healthy,
            "llm_provider": True,  # Would check OpenAI API connection
            "embedding_provider": True  # Would check embedding service
        }
        
        all_healthy = all(
            component if isinstance(component, bool) else all(component.values())
            for component in components.values()
        )
        
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            version="1.0.0",
            components=components,
            uptime_seconds=None,  # Would track actual uptime
            last_check=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            components={},
            uptime_seconds=None,
            last_check=datetime.now()
        )

@router.get("/stats", response_model=StatsResponse)
async def get_statistics(
    vector_store=Depends(get_vector_store)
):
    """
    Index statistics and repository metrics from vector store.
    
    Returns comprehensive statistics about the indexed content,
    including document counts, language distribution, and system metrics.
    """
    try:
        logger.info("Retrieving system statistics from vector store")
        
        # Check for dimension mismatch first
        try:
            is_compatible, compatibility_msg = vector_store.check_embedding_dimension_compatibility()
            if not is_compatible:
                logger.warning(f"Dimension mismatch detected: {compatibility_msg}")
                # Return degraded statistics with warning
                return StatsResponse(
                    total_repositories=0,
                    total_documents=0,
                    total_files=0,
                    index_size_mb=0.0,
                    languages={},
                    recent_queries=0,
                    active_workflows=0,
                    system_health="degraded"
                )
        except Exception as e:
            logger.warning(f"Could not check embedding compatibility: {str(e)}")
        
        # Get collection statistics
        collection_stats = vector_store.get_collection_stats()
        total_documents = collection_stats.get("count", 0)
        
        # Get repository metadata
        repository_metadata = vector_store.get_repository_metadata()
        total_repositories = len(repository_metadata)
        
        # Aggregate statistics from repository metadata
        total_files = sum(repo.get("file_count", 0) for repo in repository_metadata)
        total_size_mb = sum(repo.get("size_mb", 0.0) for repo in repository_metadata)
        
        # Aggregate language distribution
        language_counts = {}
        for repo in repository_metadata:
            for language in repo.get("languages", []):
                if language:
                    language_counts[language] = language_counts.get(language, 0) + repo.get("document_count", 0)
        
        # Get active workflows count
        active_workflow_count = len([w for w in active_workflows.values() 
                                   if w["status"] == WorkflowStatus.RUNNING])
        
        # Determine system health based on vector store availability
        try:
            is_healthy, health_message = vector_store.health_check()
            system_health = "healthy" if is_healthy else "degraded"
        except Exception:
            system_health = "degraded"
        
        stats = StatsResponse(
            total_repositories=total_repositories,
            total_documents=total_documents,
            total_files=total_files,
            index_size_mb=round(total_size_mb, 2),
            languages=language_counts,
            recent_queries=0,  # TODO: Implement query tracking
            active_workflows=active_workflow_count,
            system_health=system_health
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/workflows/{workflow_id}/status", response_model=WorkflowState)
async def get_workflow_status(workflow_id: str):
    """
    Get LangGraph workflow execution status and progress.
    
    Returns detailed information about a specific workflow's current
    state, progress, and execution details.
    """
    try:
        if workflow_id not in active_workflows:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found"
            )
        
        workflow_data = active_workflows[workflow_id]
        
        return WorkflowState(
            workflow_id=workflow_id,
            workflow_type=workflow_data["type"],
            status=WorkflowStatus(workflow_data["status"]),
            progress=None,  # Add missing progress field
            started_at=workflow_data["started_at"],
            completed_at=workflow_data.get("completed_at"),
            error_message=workflow_data.get("error_message"),
            metadata=workflow_data.get("metadata", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow status: {str(e)}"
        )


@router.get("/workflows", response_model=List[WorkflowState])
async def list_workflows(
    status: Optional[WorkflowStatus] = Query(None, description="Filter by workflow status"),
    workflow_type: Optional[str] = Query(None, description="Filter by workflow type"),
    limit: int = Query(100, description="Maximum number of workflows to return")
):
    """
    List all workflows with optional filtering.
    
    Returns a list of workflow states with optional filtering by status
    and type, useful for monitoring and management.
    """
    try:
        workflows = []
        
        for workflow_id, workflow_data in active_workflows.items():
            # Apply filters
            if status and workflow_data["status"] != status:
                continue
            if workflow_type and workflow_data["type"] != workflow_type:
                continue
            
            workflows.append(WorkflowState(
                workflow_id=workflow_id,
                workflow_type=workflow_data["type"],
                status=WorkflowStatus(workflow_data["status"]),
                progress=None,  # Add missing progress field
                started_at=workflow_data["started_at"],
                completed_at=workflow_data.get("completed_at"),
                error_message=workflow_data.get("error_message"),
                metadata=workflow_data.get("metadata", {})
            ))
            
            if len(workflows) >= limit:
                break
        
        return workflows
        
    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list workflows: {str(e)}"
        )


async def _run_indexing_workflow(
    workflow_id: str,
    request: IndexRepositoryRequest,
    indexing_workflow: IndexingWorkflow
):
    """
    Background task to run indexing workflow.
    
    Args:
        workflow_id: Unique workflow identifier
        request: Indexing request parameters
        indexing_workflow: Indexing workflow instance
    """
    try:
        logger.info(f"Starting indexing workflow {workflow_id} for {request.repository_url}")
        
        # Update workflow status
        active_workflows[workflow_id]["status"] = WorkflowStatus.RUNNING
        
        # Create indexing state as dictionary (TypedDict)
        indexing_state = {
            "workflow_id": workflow_id,
            "workflow_type": "indexing", 
            "status": "in_progress",
            "created_at": datetime.now().timestamp(),
            "updated_at": datetime.now().timestamp(),
            "progress_percentage": 0.0,
            "current_step": "initializing",
            "errors": [],
            "metadata": {},
            "repositories": [request.repository_url],
            "current_repo": request.repository_url,
            "processed_files": 0,
            "total_files": 0,
            "embeddings_generated": 0,
            "repository_states": {},
            "file_processing_states": [],
            "vector_store_type": "chroma",
            "collection_name": "knowledge_graph",
            "batch_size": 100,
            "total_chunks": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "documents_per_second": None,
            "embeddings_per_second": None,
            "total_processing_time": None
        }
        
        # Execute indexing workflow (simplified for now)
        result_state = await indexing_workflow.ainvoke(indexing_state)
        
        # Update workflow completion
        active_workflows[workflow_id].update({
            "status": WorkflowStatus.COMPLETED,
            "completed_at": datetime.now(),
            "metadata": {
                "processed_files": result_state.get("processed_files", 0),
                "embeddings_generated": result_state.get("embeddings_generated", 0),
                "errors": result_state.get("errors", [])
            }
        })
        
        logger.info(f"Indexing workflow {workflow_id} completed successfully")
        
    except Exception as e:
        # TODO: Implement proper workflow execution
        logger.error(f"Indexing workflow {workflow_id} failed: {e}")
        
        # Update workflow failure
        active_workflows[workflow_id].update({
            "status": WorkflowStatus.FAILED,
            "completed_at": datetime.now(),
            "error_message": str(e)
        })


@router.post("/fix/chroma-dimension")
async def fix_chroma_dimension(
    vector_store=Depends(get_vector_store)
):
    """
    Fix Chroma vector store dimension mismatch issues.
    
    This endpoint automatically detects and fixes dimension mismatches
    between the current embedding model and the Chroma collection.
    """
    try:
        logger.info("Checking for Chroma dimension mismatch")
        
        # Check if there's a dimension mismatch
        is_compatible, compatibility_msg = vector_store.check_embedding_dimension_compatibility()
        
        if is_compatible:
            return {
                "status": "success",
                "message": "No dimension mismatch detected",
                "details": compatibility_msg,
                "action_taken": "none"
            }
        
        logger.warning(f"Dimension mismatch detected: {compatibility_msg}")
        
        # Recreate the collection with correct dimensions
        success = vector_store.recreate_collection_with_correct_dimension()
        
        if success:
            logger.info("Chroma collection recreated successfully with correct dimensions")
            return {
                "status": "success", 
                "message": "Dimension mismatch fixed successfully",
                "details": f"Fixed: {compatibility_msg}",
                "action_taken": "collection_recreated",
                "warning": "All existing data has been cleared. You will need to re-index your repositories."
            }
        else:
            logger.error("Failed to recreate Chroma collection")
            return {
                "status": "error",
                "message": "Failed to fix dimension mismatch",
                "details": compatibility_msg,
                "action_taken": "none"
            }
            
    except Exception as e:
        logger.error(f"Error fixing Chroma dimension: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fixing Chroma dimension: {str(e)}"
        )


@router.get("/diagnose/chroma-dimension")
async def diagnose_chroma_dimension(
    vector_store=Depends(get_vector_store)
):
    """
    Diagnose Chroma vector store dimension compatibility.
    
    This endpoint checks for dimension mismatches and provides detailed
    information about the current state without making any changes.
    """
    try:
        logger.info("Diagnosing Chroma dimension compatibility")
        
        # Check dimension compatibility
        is_compatible, compatibility_msg = vector_store.check_embedding_dimension_compatibility()
        
        # Get current embedding model info
        test_embedding = vector_store.embeddings.embed_query("test")
        current_dimension = len(test_embedding)
        
        # Get collection info
        try:
            collection_info = vector_store.client.get_collection(vector_store.collection_name)
            collection_exists = True
            collection_metadata = collection_info.metadata if hasattr(collection_info, "metadata") else None
            expected_dimension = None
            if collection_metadata:
                expected_dimension = collection_metadata.get("dimension")
        except Exception as e:
            collection_exists = False
            collection_metadata = None
            expected_dimension = None
        
        diagnosis = {
            "compatible": is_compatible,
            "message": compatibility_msg,
            "current_embedding_model": getattr(vector_store.embeddings, 'model', 'unknown'),
            "current_dimension": current_dimension,
            "collection_exists": collection_exists,
            "expected_dimension": expected_dimension,
            "collection_metadata": collection_metadata,
            "collection_name": vector_store.collection_name
        }
        
        if not is_compatible:
            diagnosis["recommended_action"] = "Use POST /fix/chroma-dimension to automatically fix this issue"
            diagnosis["warning"] = "Fixing will recreate the collection and clear all existing data"
        
        return {
            "status": "success",
            "diagnosis": diagnosis
        }
        
    except Exception as e:
        logger.error(f"Error diagnosing Chroma dimension: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error diagnosing Chroma dimension: {str(e)}"
        )
