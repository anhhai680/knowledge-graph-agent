"""
Workflow state schemas for LangGraph workflows.

This module defines TypedDict schemas for different workflow states including
indexing and query workflows, providing type safety and structure validation.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union
from enum import Enum
import time

try:
    from src.config.settings import AppSettings
    _app_settings_available = True
except ImportError:
    AppSettings = None
    _app_settings_available = False


def get_openai_model_name() -> str:
    """Get OpenAI model name with fallback."""
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        return settings.openai.model
    except Exception:
        return "gpt-4o-mini"


def get_openai_temperature() -> float:
    """Get OpenAI temperature with fallback."""
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        return settings.openai.temperature
    except Exception:
        return 0.7


def get_openai_max_tokens() -> int:
    """Get OpenAI max tokens with fallback."""
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        return settings.openai.max_tokens
    except Exception:
        return 4000


class WorkflowType(str, Enum):
    """Workflow type enumeration."""

    INDEXING = "indexing"
    QUERY = "query"
    MAINTENANCE = "maintenance"


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class QueryIntent(str, Enum):
    """Query intent enumeration for query workflows."""

    CODE_SEARCH = "code_search"
    DOCUMENTATION = "documentation"
    EXPLANATION = "explanation"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    EVENT_FLOW = "event_flow"
    GENERIC_QA = "generic_qa"


class SearchStrategy(str, Enum):
    """Search strategy enumeration for query workflows."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    METADATA_FILTERED = "metadata_filtered"


# Base workflow state schema
class BaseWorkflowState(TypedDict):
    """
    Base workflow state schema.

    Contains common fields shared across all workflow types.
    """

    workflow_id: str
    workflow_type: WorkflowType
    status: ProcessingStatus
    created_at: float
    updated_at: float
    progress_percentage: float
    current_step: Optional[str]
    errors: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# Repository processing state
class RepositoryState(TypedDict):
    """Repository processing state schema."""

    name: str
    url: str
    branch: str
    status: ProcessingStatus
    processed_files: int
    total_files: int
    embeddings_generated: int
    errors: List[str]
    processing_start_time: Optional[float]
    processing_end_time: Optional[float]


# File processing state
class FileProcessingState(TypedDict):
    """File processing state schema."""

    file_path: str
    language: str
    status: ProcessingStatus
    chunk_count: int
    embedding_count: int
    processing_time: Optional[float]
    error_message: Optional[str]


# Indexing workflow state schema
class IndexingState(BaseWorkflowState):
    """
    Indexing workflow state schema.

    Extends base workflow state with indexing-specific fields for tracking
    repository processing, document chunking, and embedding generation.
    """

    repositories: List[str]
    current_repo: str
    processed_files: int
    total_files: int
    embeddings_generated: int
    repository_states: Dict[str, RepositoryState]
    file_processing_states: List[FileProcessingState]
    vector_store_type: str
    collection_name: str
    batch_size: int

    # Processing statistics
    total_chunks: int
    successful_embeddings: int
    failed_embeddings: int

    # Performance metrics
    documents_per_second: Optional[float]
    embeddings_per_second: Optional[float]
    total_processing_time: Optional[float]
    
    # Graph storage statistics
    graph_storage_stats: Optional[Dict[str, Any]]


# Document retrieval state
class DocumentRetrievalState(TypedDict):
    """Document retrieval state schema."""

    query_vector: Optional[List[float]]
    search_strategy: SearchStrategy
    top_k: int
    similarity_threshold: float
    metadata_filters: Dict[str, Any]
    retrieved_documents: List[Dict[str, Any]]
    retrieval_time: Optional[float]
    relevance_scores: List[float]
    status: ProcessingStatus


# LLM generation state
class LLMGenerationState(TypedDict):
    """LLM generation state schema."""

    prompt_template: str
    context_documents: List[Dict[str, Any]]
    generated_response: Optional[str]
    generation_time: Optional[float]
    token_usage: Dict[str, int]
    model_name: str
    temperature: float
    max_tokens: int
    status: ProcessingStatus


# Query workflow state schema
class QueryState(BaseWorkflowState):
    """
    Query workflow state schema.

    Extends base workflow state with query-specific fields for tracking
    query processing, document retrieval, and response generation.
    """

    original_query: str
    processed_query: str
    query_intent: Optional[QueryIntent]
    search_strategy: SearchStrategy

    # Repository and language filtering
    target_repositories: Optional[List[str]]
    target_languages: Optional[List[str]]
    target_file_types: Optional[List[str]]

    # Retrieval configuration
    retrieval_config: Dict[str, Any]
    document_retrieval: DocumentRetrievalState

    # Context preparation
    context_size: int
    context_documents: List[Dict[str, Any]]
    context_preparation_time: Optional[float]

    # LLM generation
    llm_generation: LLMGenerationState

    # Response quality
    response_quality_score: Optional[float]
    response_confidence: Optional[float]
    response_sources: List[Dict[str, Any]]

    # Performance metrics
    total_query_time: Optional[float]
    retrieval_time: Optional[float]
    generation_time: Optional[float]
    start_time: Optional[float]


# Maintenance workflow state schema
class MaintenanceState(BaseWorkflowState):
    """
    Maintenance workflow state schema.

    For cleanup, optimization, and maintenance operations.
    """

    operation_type: str
    target_collections: List[str]
    items_processed: int
    items_deleted: int
    items_updated: int
    space_freed: Optional[int]
    optimization_metrics: Dict[str, Any]


# Union type for all workflow states
WorkflowState = Union[IndexingState, QueryState, MaintenanceState]


# Helper functions for state creation and validation
def create_base_workflow_state(
    workflow_id: str, workflow_type: WorkflowType
) -> BaseWorkflowState:
    """
    Create base workflow state.

    Args:
        workflow_id: Unique workflow identifier
        workflow_type: Type of workflow

    Returns:
        Base workflow state
    """
    current_time = time.time()
    return BaseWorkflowState(
        workflow_id=workflow_id,
        workflow_type=workflow_type,
        status=ProcessingStatus.NOT_STARTED,
        created_at=current_time,
        updated_at=current_time,
        progress_percentage=0.0,
        current_step=None,
        errors=[],
        metadata={},
    )


def create_indexing_state(
    workflow_id: str,
    repositories: List[str],
    vector_store_type: str = "chroma",
    collection_name: str = "knowledge-base-graph",
    batch_size: int = 50,
) -> IndexingState:
    """
    Create indexing workflow state.

    Args:
        workflow_id: Unique workflow identifier
        repositories: List of repositories to index
        vector_store_type: Type of vector store to use
        collection_name: Collection name for vector storage
        batch_size: Batch size for processing

    Returns:
        Indexing workflow state
    """
    base_state = create_base_workflow_state(workflow_id, WorkflowType.INDEXING)

    return IndexingState(
        **base_state,
        repositories=repositories,
        current_repo="",
        processed_files=0,
        total_files=0,
        embeddings_generated=0,
        repository_states={},
        file_processing_states=[],
        vector_store_type=vector_store_type,
        collection_name=collection_name,
        batch_size=batch_size,
        total_chunks=0,
        successful_embeddings=0,
        failed_embeddings=0,
        documents_per_second=None,
        embeddings_per_second=None,
        total_processing_time=None,
        graph_storage_stats=None,
    )


def create_repository_state(
    name: str, url: str, branch: str = "main"
) -> RepositoryState:
    """
    Create repository processing state.

    Args:
        name: Repository name
        url: Repository URL
        branch: Repository branch

    Returns:
        Repository state
    """
    return RepositoryState(
        name=name,
        url=url,
        branch=branch,
        status=ProcessingStatus.NOT_STARTED,
        processed_files=0,
        total_files=0,
        embeddings_generated=0,
        errors=[],
        processing_start_time=None,
        processing_end_time=None,
    )


def create_query_state(
    workflow_id: str,
    original_query: str,
    search_strategy: SearchStrategy = SearchStrategy.SEMANTIC,
    target_repositories: Optional[List[str]] = None,
    target_languages: Optional[List[str]] = None,
    target_file_types: Optional[List[str]] = None,
    retrieval_config: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
) -> QueryState:
    """
    Create query workflow state.

    Args:
        workflow_id: Unique workflow identifier
        original_query: Original user query
        search_strategy: Search strategy to use
        top_k: Number of documents to retrieve

    Returns:
        Query workflow state
    """
    base_state = create_base_workflow_state(workflow_id, WorkflowType.QUERY)
    current_time = time.time()

    query_state: QueryState = {
        "workflow_id": base_state["workflow_id"],
        "workflow_type": base_state["workflow_type"],
        "status": base_state["status"],
        "created_at": base_state["created_at"],
        "updated_at": base_state["updated_at"],
        "progress_percentage": base_state["progress_percentage"],
        "current_step": base_state["current_step"],
        "errors": base_state["errors"],
        "metadata": base_state["metadata"],
        "original_query": original_query,
        "processed_query": original_query,
        "query_intent": None,
        "search_strategy": search_strategy,
        "target_repositories": target_repositories,
        "target_languages": target_languages,
        "target_file_types": target_file_types,
        "retrieval_config": retrieval_config or {
            "top_k": top_k,
            "similarity_threshold": 0.7,
            "metadata_filters": {},
        },
        "document_retrieval": {
            "query_vector": None,
            "search_strategy": search_strategy,
            "top_k": top_k,
            "similarity_threshold": 0.7,
            "metadata_filters": {},
            "retrieved_documents": [],
            "retrieval_time": None,
            "relevance_scores": [],
            "status": ProcessingStatus.NOT_STARTED,
        },
        "context_size": 0,
        "context_documents": [],
        "context_preparation_time": None,
        "llm_generation": {
            "prompt_template": "",
            "context_documents": [],
            "generated_response": None,
            "generation_time": None,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model_name": get_openai_model_name(),
            "temperature": get_openai_temperature(),
            "max_tokens": get_openai_max_tokens(),
            "status": ProcessingStatus.NOT_STARTED,
        },
        "response_quality_score": None,
        "response_confidence": None,
        "response_sources": [],
        "total_query_time": None,
        "retrieval_time": None,
        "generation_time": None,
        "start_time": current_time,
        "total_query_time": None
    }
    return query_state


def create_file_processing_state(file_path: str, language: str) -> FileProcessingState:
    """
    Create file processing state.

    Args:
        file_path: Path to the file being processed
        language: Programming language of the file

    Returns:
        File processing state
    """
    return FileProcessingState(
        file_path=file_path,
        language=language,
        status=ProcessingStatus.NOT_STARTED,
        chunk_count=0,
        embedding_count=0,
        processing_time=None,
        error_message=None,
    )


def update_workflow_progress(
    state: WorkflowState, progress_percentage: float, current_step: Optional[str] = None
) -> WorkflowState:
    """
    Update workflow progress.

    Args:
        state: Workflow state to update
        progress_percentage: Progress percentage (0-100)
        current_step: Optional current step name

    Returns:
        Updated workflow state
    """
    state["progress_percentage"] = max(0.0, min(100.0, progress_percentage))
    state["updated_at"] = time.time()

    if current_step is not None:
        state["current_step"] = current_step

    return state


def add_workflow_error(
    state: WorkflowState,
    error_message: str,
    step: Optional[str] = None,
    error_details: Optional[Dict[str, Any]] = None,
) -> WorkflowState:
    """
    Add error to workflow state.

    Args:
        state: Workflow state to update
        error_message: Error message
        step: Optional step where error occurred
        error_details: Optional additional error details

    Returns:
        Updated workflow state
    """
    error_entry = {
        "message": error_message,
        "timestamp": time.time(),
        "step": step or state.get("current_step"),
        "details": error_details or {},
    }

    state["errors"].append(error_entry)
    state["updated_at"] = time.time()

    return state


def is_workflow_complete(state: WorkflowState) -> bool:
    """
    Check if workflow is complete.

    Args:
        state: Workflow state to check

    Returns:
        True if workflow is complete, False otherwise
    """
    return state["status"] == ProcessingStatus.COMPLETED


def is_workflow_failed(state: WorkflowState) -> bool:
    """
    Check if workflow has failed.

    Args:
        state: Workflow state to check

    Returns:
        True if workflow has failed, False otherwise
    """
    return state["status"] == ProcessingStatus.FAILED


def get_workflow_duration(state: WorkflowState) -> Optional[float]:
    """
    Get workflow duration in seconds.

    Args:
        state: Workflow state

    Returns:
        Duration in seconds if available, None otherwise
    """
    if "created_at" in state and "updated_at" in state:
        return state["updated_at"] - state["created_at"]
    return None
