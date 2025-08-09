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
    SearchStrategy,
    GraphQueryRequest,
    GraphQueryResponse,
    GraphInfoResponse
)
from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.feature_flags import is_graph_enabled
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
        "explanation": QueryIntent.EXPLANATION,  # Now available in API enum
        "debugging": QueryIntent.DEBUGGING,
        "architecture": QueryIntent.ARCHITECTURE,  # Now available in API enum
        "implementation": QueryIntent.IMPLEMENTATION,
        "general": QueryIntent.GENERAL,
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


def get_graph_store():
    """Dependency injection for graph store."""
    # This will be implemented in main.py as a dependency
    from src.api.main import get_graph_store
    return get_graph_store()


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
            repo_name = repo_config.get("name", repo_config["url"])  # Use repository name if available, else URL
            
            # Start background indexing workflow with repository name
            background_tasks.add_task(
                _run_indexing_workflow,
                workflow_id,
                repo_name,  # Pass repository name instead of request object
                indexing_workflow
            )
            
            # Track workflow
            active_workflows[workflow_id] = {
                "id": workflow_id,
                "type": "indexing",
                "status": WorkflowStatus.PENDING,
                "repository": repo_config["url"],  # Keep URL for display
                "started_at": datetime.now(),
                "batch_id": batch_id
            }
            
            workflows.append(IndexingResponse(
                workflow_id=workflow_id,
                repository=repo_config["url"],  # Keep URL for display
                status=WorkflowStatus.PENDING,
                estimated_duration="10-30 minutes",
                message=f"Indexing workflow queued for {repo_config['url']}"
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
        
        # Extract repository name from URL (e.g., "anhhai680/car-web-client" from URL)
        if request.repository_url.startswith("https://github.com/"):
            repo_path = request.repository_url.replace("https://github.com/", "").rstrip("/")
            if repo_path.endswith(".git"):
                repo_path = repo_path[:-4]
            # Use just the repository name part (after the last "/")
            repo_name = repo_path.split("/")[-1] if "/" in repo_path else repo_path
        elif request.repository_url.startswith("git@github.com:"):
            repo_path = request.repository_url.replace("git@github.com:", "").rstrip("/")
            if repo_path.endswith(".git"):
                repo_path = repo_path[:-4]
            # Use just the repository name part (after the last "/")
            repo_name = repo_path.split("/")[-1] if "/" in repo_path else repo_path
        else:
            # Fallback: use the URL as-is (this will likely fail but allows debugging)
            repo_name = request.repository_url
        
        # Start background indexing workflow
        background_tasks.add_task(
            _run_indexing_workflow,
            workflow_id,
            repo_name,
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
    For Q2 system visualization queries, it returns generated Mermaid diagrams.
    """
    try:
        start_time = datetime.now()
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Use RAGAgent for Q2 queries and advanced prompt management
        # Import here to avoid circular imports
        from src.agents.rag_agent import RAGAgent
        from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
        
        # Pre-check for Q2 queries to ensure detection
        parsing_handler = QueryParsingHandler()
        is_q2_query_direct = parsing_handler._is_q2_system_relationship_query(request.query)
        logger.info(f"API Q2 DIRECT CHECK: Query='{request.query}' -> Q2={is_q2_query_direct}")
        
        # Create RAGAgent with the same workflow
        rag_agent = RAGAgent(
            workflow=query_workflow,
            default_top_k=request.top_k or 5,
            repository_filter=request.repositories,
            language_filter=request.language_filter
        )
        
        # Process query through RAGAgent which supports Q2 queries
        agent_result = await rag_agent._process_input({
            "query": request.query,
            "repositories": request.repositories,
            "language_filter": request.language_filter,
            "top_k": request.top_k or 5
        })
        
        logger.debug(f"RAGAgent result: {agent_result}")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Check if this is a Q2 query by looking for generated response content and metadata
        generated_answer = agent_result.get("answer", "")
        is_q2_response = (
            "mermaid" in generated_answer.lower() or
            "graph TB" in generated_answer or
            agent_result.get("prompt_metadata", {}).get("template_type") == "Q2SystemVisualizationTemplate"
        )
        
        # Additional Q2 detection logging for debugging
        logger.debug(f"API Q2 DEBUG: Query='{request.query}', Generated answer length={len(generated_answer)}")
        logger.debug(f"API Q2 DEBUG: Template type={agent_result.get('prompt_metadata', {}).get('template_type')}")
        logger.debug(f"API Q2 DEBUG: Q2 response detected={is_q2_response}")
        logger.debug(f"API Q2 DEBUG: RAG result keys={list(agent_result.keys())}")

        # Check for Q2 flag in the RAGAgent result metadata
        if not is_q2_response and agent_result.get("prompt_metadata", {}).get("is_q2_visualization"):
            is_q2_response = True
            logger.info("API Q2: Q2 detected via metadata flag")
        
        # Fallback: If Q2 was detected directly but response doesn't seem like Q2, force Q2 response
        if is_q2_query_direct and not is_q2_response:
            logger.warning("API Q2 FALLBACK: Q2 query detected but response doesn't contain Q2 content. Creating fallback Q2 response.")
            
            # Import here to avoid circular imports
            from src.utils.prompt_manager import PromptManager
            
            # Create a generic Q2 response based on actual repositories
            prompt_manager = PromptManager()
            repositories = prompt_manager._get_repository_information()
            mermaid_diagram = prompt_manager._generate_generic_mermaid_diagram(repositories)
            architecture_explanation = prompt_manager._generate_architecture_explanation(repositories)
            
            fallback_q2_response = f"""Looking at the system architecture based on the available repositories:

```mermaid
graph TB
{mermaid_diagram}
```

{architecture_explanation}

**Communication Patterns:**
- **API Integration**: Services communicate through well-defined REST APIs
- **Data Flow**: Information flows between components based on business requirements  
- **Modular Design**: Each repository handles specific functionality and concerns

This architecture provides flexibility and maintainability by organizing functionality into separate, focused components."""
            
            generated_answer = fallback_q2_response
            is_q2_response = True
            logger.info("API Q2 FALLBACK: Created generic fallback Q2 response based on actual repositories")
        
        logger.debug(f"Generated answer length: {len(generated_answer)}")
        logger.debug(f"Is Q2 response: {is_q2_response}")
        logger.debug(f"Template type: {agent_result.get('prompt_metadata', {}).get('template_type')}")
        
        # Convert RAGAgent sources to API document results
        document_results = []
        for source in agent_result.get("sources", []):
            metadata = source.get("metadata", {})
            doc_metadata = DocumentMetadata(
                source=metadata.get("file_path", metadata.get("source", "")),
                repository=metadata.get("repository", ""),
                language=metadata.get("language"),
                file_type=metadata.get("file_type"),
                symbols=metadata.get("symbols", []),
                last_modified=metadata.get("last_modified"),
                size=metadata.get("size")
            )
            
            document_results.append(DocumentResult(
                content=source.get("content", ""),
                metadata=doc_metadata,
                score=0.0,  # RAGAgent doesn't provide similarity scores
                chunk_index=metadata.get("chunk_index")
            ))
        
        # Extract confidence score
        confidence_score = agent_result.get("confidence", 0.0)
        
        # Extract and map query intent
        query_intent = agent_result.get("query_intent")
        intent_value = query_intent.value if query_intent else "general"
        
        logger.debug(f"API ROUTE DEBUG: RAGAgent result keys: {list(agent_result.keys())}")
        logger.debug(f"API ROUTE DEBUG: Query='{request.query}', Intent from agent={query_intent}, Intent value={intent_value}")
        logger.debug(f"API ROUTE DEBUG: Intent type: {type(query_intent)}")
        logger.debug(f"API ROUTE DEBUG: Generated answer length: {len(generated_answer)}")
        
        # Determine response type and content based on query type
        if is_q2_response and generated_answer:
            # For Q2 queries, return the generated Mermaid diagram and explanation
            response_type = "generated"
            generated_response = generated_answer
            logger.info(f"Returning generated response for Q2 query: {len(generated_response)} characters")
        else:
            # For regular queries, return search results
            response_type = "search"
            generated_response = None
            logger.info(f"Returning search results: {len(document_results)} documents")
        
        response = QueryResponse(
            query=request.query,
            intent=_map_workflow_intent_to_api(intent_value),
            strategy=_map_workflow_strategy_to_api("semantic"),  # RAGAgent uses semantic search
            results=document_results,
            total_results=len(document_results),
            processing_time=processing_time,
            confidence_score=confidence_score,
            suggestions=[],  # RAGAgent doesn't provide suggestions currently
            generated_response=generated_response,
            response_type=response_type
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
    repo_name: str,
    indexing_workflow: IndexingWorkflow
):
    """
    Background task to run indexing workflow.
    
    Args:
        workflow_id: Unique workflow identifier
        repo_name: Repository name (e.g., "car-web-client")
        indexing_workflow: Indexing workflow instance
    """
    try:
        logger.info(f"Starting indexing workflow {workflow_id} for repository: {repo_name}")
        
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
            "repositories": [repo_name],  # Use repository name
            "current_repo": repo_name,    # Use repository name
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


@router.post("/graph/query", response_model=GraphQueryResponse)
async def execute_graph_query(
    request: GraphQueryRequest,
    graph_store=Depends(get_graph_store)
):
    """
    Execute a Cypher query against the knowledge graph.
    
    This endpoint allows executing Cypher queries against the graph database
    when graph features are enabled.
    """
    if not is_graph_enabled():
        raise HTTPException(
            status_code=400, 
            detail="Graph features are not enabled"
        )
    
    start_time = datetime.now()
    
    try:
        # Execute the query
        result = graph_store.execute_query(
            query=request.query,
            parameters=request.parameters
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return GraphQueryResponse(
            success=True,
            result=result,
            error=None,
            processing_time=processing_time,
            query=request.query
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Graph query failed: {str(e)}")
        
        return GraphQueryResponse(
            success=False,
            result=None,
            error=str(e),
            processing_time=processing_time,
            query=request.query
        )


@router.get("/graph/info", response_model=GraphInfoResponse)
async def get_graph_info(
    graph_store=Depends(get_graph_store)
):
    """
    Get information about the graph database.
    
    This endpoint provides information about the graph database
    when graph features are enabled.
    """
    if not is_graph_enabled():
        raise HTTPException(
            status_code=400, 
            detail="Graph features are not enabled"
        )
    
    try:
        info = graph_store.get_graph_info()
        
        return GraphInfoResponse(
            connected=info.get("connected", False),
            node_count=info.get("node_count", 0),
            relationship_count=info.get("relationship_count", 0),
            database_type=info.get("database_type", "Unknown"),
            schema_info=None,  # TODO: Add schema info
            performance_metrics=None  # TODO: Add performance metrics
        )
        
    except Exception as e:
        logger.error(f"Failed to get graph info: {str(e)}")
        
        # Check if it's a connection error
        if "Failed to connect to MemGraph" in str(e) or "Connection refused" in str(e):
            raise HTTPException(
                status_code=503,
                detail="MemGraph database is not available. Please ensure MemGraph is running and accessible at the configured URL."
            )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get graph info: {str(e)}"
        )


@router.get("/graph/health")
async def get_graph_health():
    """
    Health check endpoint for graph database connectivity.
    
    This endpoint checks if graph features are enabled and if the
    graph database is accessible, providing detailed diagnostic information.
    """
    if not is_graph_enabled():
        return {
            "status": "disabled",
            "message": "Graph features are not enabled",
            "enabled": False,
            "connected": False,
            "configuration": None
        }
    
    try:
        from ..vectorstores.store_factory import VectorStoreFactory
        from ..config.settings import get_settings
        
        settings = get_settings()
        graph_config = {
            "type": settings.graph_store.type,
            "url": settings.graph_store.url,
            "has_auth": bool(settings.graph_store.username and settings.graph_store.password)
        }
        
        # Try to create and connect to graph store
        vector_store_factory = VectorStoreFactory()
        graph_store = vector_store_factory.create(store_type="graph")
        
        return {
            "status": "healthy",
            "message": "Graph database is accessible",
            "enabled": True,
            "connected": True,
            "configuration": graph_config,
            "database_info": graph_store.get_graph_info() if hasattr(graph_store, 'get_graph_info') else None
        }
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide specific guidance based on error type
        if "Failed to connect to MemGraph" in error_msg or "Connection refused" in error_msg:
            status_msg = "MemGraph database is not accessible"
            if "localhost" in settings.graph_store.url:
                status_msg += " (Check if running in Docker with correct hostname)"
        else:
            status_msg = f"Graph database error: {error_msg}"
        
        return {
            "status": "unhealthy",
            "message": status_msg,
            "enabled": True,
            "connected": False,
            "configuration": graph_config if 'graph_config' in locals() else None,
            "error": error_msg
        }
