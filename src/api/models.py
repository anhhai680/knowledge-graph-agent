"""
Pydantic models for API request/response handling with LangGraph workflow integration.

This module defines all the data models used for API endpoints, including
request validation, response formatting, and workflow state representations.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# Import GraphQueryResult from core graph domain
from ..graphstores.base_graph_store import GraphQueryResult


class QueryIntent(str, Enum):
    """Query intent enumeration for adaptive processing."""
    CODE_SEARCH = "code_search"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    EXPLANATION = "explanation"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    EVENT_FLOW = "event_flow"
    GENERAL = "general"


class WorkflowStatus(str, Enum):
    """Workflow execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SearchStrategy(str, Enum):
    """Search strategy enumeration for query processing."""
    SIMILARITY = "similarity"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"


# Request Models

class IndexRepositoryRequest(BaseModel):
    """Request model for repository indexing operations."""
    
    repository_url: str = Field(
        ...,
        description="GitHub repository URL to index",
        json_schema_extra={"example": "https://github.com/user/repo"}
    )
    branch: Optional[str] = Field(
        default="main",
        description="Branch to index",
        json_schema_extra={"example": "main"}
    )
    file_extensions: Optional[List[str]] = Field(
        default=None,
        description="File extensions to include (if not specified, uses default set)",
        json_schema_extra={"example": [".py", ".js", ".ts", ".md"]}
    )
    max_files: Optional[int] = Field(
        default=None,
        description="Maximum number of files to process",
        json_schema_extra={"example": 1000}
    )
    force_reindex: bool = Field(
        default=False,
        description="Force reindexing even if repository is already indexed"
    )
    incremental: bool = Field(
        default=False,
        description="Enable incremental re-indexing based on git commit history"
    )
    dry_run: bool = Field(
        default=False,
        description="Analyze changes without performing actual indexing (incremental mode only)"
    )
    
    @field_validator('repository_url')
    @classmethod
    def validate_repository_url(cls, v):
        """Validate GitHub repository URL format."""
        if not v.startswith(('https://github.com/', 'git@github.com:')):
            raise ValueError('Repository URL must be a valid GitHub URL')
        return v
    
    @field_validator('dry_run')
    @classmethod
    def validate_dry_run(cls, v, info):
        """Validate that dry_run is only used with incremental mode."""
        if v and not info.data.get('incremental', False):
            raise ValueError('dry_run can only be used with incremental=True')
        return v


class BatchIndexRequest(BaseModel):
    """Request model for batch repository indexing."""
    
    repositories: List[IndexRepositoryRequest] = Field(
        ...,
        description="List of repositories to index",
        min_length=1
    )
    parallel_jobs: Optional[int] = Field(
        default=3,
        description="Number of parallel indexing jobs",
        ge=1,
        le=10
    )


class QueryRequest(BaseModel):
    """Request model for query processing operations."""
    
    query: str = Field(
        ...,
        description="User query text",
        min_length=1,
        max_length=2000,
        json_schema_extra={"example": "How to implement authentication in FastAPI?"}
    )
    intent: Optional[QueryIntent] = Field(
        default=None,
        description="Query intent (auto-detected if not provided)"
    )
    repositories: Optional[List[str]] = Field(
        default=None,
        description="Repository names to search within",
        json_schema_extra={"example": ["user/repo1", "user/repo2"]}
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Number of top results to return",
        ge=1,
        le=50
    )
    search_strategy: Optional[SearchStrategy] = Field(
        default=SearchStrategy.HYBRID,
        description="Search strategy to use"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include document metadata in results"
    )
    language_filter: Optional[List[str]] = Field(
        default=None,
        description="Programming languages to filter by",
        json_schema_extra={"example": ["python", "typescript", "javascript"]}
    )


# Response Models

class DocumentMetadata(BaseModel):
    """Document metadata model."""
    
    source: str = Field(..., description="Document source path")
    repository: str = Field(..., description="Repository name")
    language: Optional[str] = Field(None, description="Programming language")
    file_type: Optional[str] = Field(None, description="File type/extension")
    symbols: Optional[List[str]] = Field(None, description="Extracted code symbols")
    last_modified: Optional[datetime] = Field(None, description="Last modification date")
    size: Optional[int] = Field(None, description="File size in bytes")


class DocumentResult(BaseModel):
    """Individual document search result."""
    
    content: str = Field(..., description="Document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)
    chunk_index: Optional[int] = Field(None, description="Chunk index within document")


class QueryResponse(BaseModel):
    """Response model for query processing results."""
    
    query: str = Field(..., description="Original query text")
    intent: QueryIntent = Field(..., description="Detected or provided query intent")
    strategy: SearchStrategy = Field(..., description="Search strategy used")
    results: List[DocumentResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    processing_time: float = Field(..., description="Query processing time in seconds")
    confidence_score: float = Field(..., description="Overall confidence score", ge=0.0, le=1.0)
    suggestions: Optional[List[str]] = Field(None, description="Query improvement suggestions")
    # New field for Q2 system visualization and other generated responses
    generated_response: Optional[str] = Field(None, description="Generated response content (e.g., Mermaid diagrams for Q2 queries)")
    response_type: Optional[str] = Field(None, description="Type of response: 'search' for document results, 'generated' for Q2/chat responses")


class WorkflowProgress(BaseModel):
    """Workflow progress information."""
    
    current_step: str = Field(..., description="Current workflow step")
    completed_steps: int = Field(..., description="Number of completed steps")
    total_steps: int = Field(..., description="Total number of steps")
    progress_percentage: float = Field(..., description="Progress percentage", ge=0.0, le=100.0)
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class WorkflowState(BaseModel):
    """Workflow state information."""
    
    workflow_id: str = Field(..., description="Unique workflow identifier")
    workflow_type: str = Field(..., description="Type of workflow (indexing/query)")
    status: WorkflowStatus = Field(..., description="Current workflow status")
    progress: Optional[WorkflowProgress] = Field(None, description="Progress information")
    started_at: datetime = Field(..., description="Workflow start time")
    completed_at: Optional[datetime] = Field(None, description="Workflow completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional workflow metadata")


class IndexingResponse(BaseModel):
    """Response model for indexing operations."""
    
    workflow_id: str = Field(..., description="Workflow identifier for tracking")
    repository: str = Field(..., description="Repository being indexed")
    status: WorkflowStatus = Field(..., description="Initial workflow status")
    estimated_duration: Optional[str] = Field(None, description="Estimated completion time")
    message: str = Field(..., description="Response message")
    incremental: bool = Field(default=False, description="Whether incremental mode was used")
    dry_run: bool = Field(default=False, description="Whether this was a dry-run")
    change_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of changes (incremental mode only)")


class BatchIndexingResponse(BaseModel):
    """Response model for batch indexing operations."""
    
    workflows: List[IndexingResponse] = Field(..., description="Individual workflow responses")
    batch_id: str = Field(..., description="Batch operation identifier")
    total_repositories: int = Field(..., description="Total number of repositories")
    message: str = Field(..., description="Batch operation message")


class RepositoryInfo(BaseModel):
    """Repository information model."""
    
    name: str = Field(..., description="Repository name")
    url: str = Field(..., description="Repository URL")
    branch: str = Field(..., description="Indexed branch")
    last_indexed: datetime = Field(..., description="Last indexing time")
    file_count: int = Field(..., description="Number of indexed files")
    document_count: int = Field(..., description="Number of document chunks")
    languages: List[str] = Field(..., description="Programming languages found")
    size_mb: float = Field(..., description="Total size in megabytes")


class RepositoriesResponse(BaseModel):
    """Response model for repository listing."""
    
    repositories: List[RepositoryInfo] = Field(..., description="List of indexed repositories")
    total_count: int = Field(..., description="Total number of repositories")
    last_updated: datetime = Field(..., description="Last update time")


class StatsResponse(BaseModel):
    """Response model for system statistics."""
    
    total_repositories: int = Field(..., description="Total indexed repositories")
    total_documents: int = Field(..., description="Total document chunks")
    total_files: int = Field(..., description="Total indexed files")
    index_size_mb: float = Field(..., description="Total index size in MB")
    languages: Dict[str, int] = Field(..., description="Language distribution")
    recent_queries: int = Field(..., description="Recent query count")
    active_workflows: int = Field(..., description="Currently active workflows")
    system_health: str = Field(..., description="Overall system health status")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    components: Dict[str, Any] = Field(..., description="Component health status")
    uptime_seconds: Optional[float] = Field(None, description="System uptime in seconds")
    last_check: datetime = Field(default_factory=datetime.now, description="Last health check time")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


# Graph Query Models

class GraphQueryRequest(BaseModel):
    """Request model for graph query operations."""
    
    query: str = Field(
        ...,
        description="Cypher query to execute",
        min_length=1,
        max_length=5000,
        json_schema_extra={"example": "MATCH (f:File) RETURN f LIMIT 10"}
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Query parameters",
        json_schema_extra={"example": {"file_path": "src/main.py"}}
    )
    timeout_seconds: Optional[int] = Field(
        default=30,
        description="Query timeout in seconds",
        ge=1,
        le=300
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Validate Cypher query format."""
        if not v.strip().upper().startswith(('MATCH', 'CREATE', 'MERGE', 'RETURN', 'WITH')):
            raise ValueError('Query must be a valid Cypher query starting with MATCH, CREATE, MERGE, RETURN, or WITH')
        return v


class GraphQueryResponse(BaseModel):
    """Response model for graph query operations."""
    
    success: bool = Field(..., description="Query execution success")
    result: Optional[GraphQueryResult] = Field(None, description="Query result")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(..., description="Total processing time in seconds")
    query: str = Field(..., description="Original query text")


class GraphInfoResponse(BaseModel):
    """Response model for graph database information."""
    
    connected: bool = Field(..., description="Graph database connection status")
    node_count: int = Field(..., description="Total number of nodes")
    relationship_count: int = Field(..., description="Total number of relationships")
    database_type: str = Field(..., description="Graph database type")
    schema_info: Optional[Dict[str, Any]] = Field(None, description="Graph schema information")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")


# Generic Q&A Models

class QuestionCategory(str, Enum):
    """Question categories for generic project Q&A."""
    
    BUSINESS_CAPABILITY = "business_capability"
    API_ENDPOINTS = "api_endpoints"
    DATA_MODELING = "data_modeling"
    WORKFLOWS = "workflows"
    ARCHITECTURE = "architecture"


class GenericQARequest(BaseModel):
    """Request model for generic Q&A operations."""
    
    question: str = Field(
        ...,
        description="Question about project architecture or implementation",
        min_length=1,
        max_length=1000,
        json_schema_extra={"example": "What business capability does this service own?"}
    )
    category: Optional[QuestionCategory] = Field(
        default=None,
        description="Question category (auto-detected if not provided)"
    )
    template: Optional[str] = Field(
        default=None,
        description="Project template type (e.g., 'python_fastapi', 'dotnet_clean_architecture')",
        json_schema_extra={"example": "python_fastapi"}
    )
    include_analysis: bool = Field(
        default=True,
        description="Include project analysis in response"
    )
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        """Validate question content."""
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class ProjectTemplate(BaseModel):
    """Project template configuration model."""
    
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    supported_categories: List[QuestionCategory] = Field(..., description="Supported question categories")
    architecture_patterns: List[str] = Field(..., description="Architecture patterns")
    examples: Optional[Dict[str, List[str]]] = Field(None, description="Example questions by category")


class ProjectAnalysisRequest(BaseModel):
    """Request model for project analysis operations."""
    
    project_path: Optional[str] = Field(
        default=None,
        description="Local path to project directory"
    )
    repository_url: Optional[str] = Field(
        default=None,
        description="URL to project repository"
    )
    template_hint: Optional[str] = Field(
        default=None,
        description="Hint about project template type"
    )
    analysis_depth: str = Field(
        default="standard",
        description="Analysis depth: 'basic', 'standard', or 'comprehensive'",
        json_schema_extra={"example": "standard"}
    )
    
    @field_validator('analysis_depth')
    @classmethod
    def validate_analysis_depth(cls, v):
        """Validate analysis depth."""
        if v not in ["basic", "standard", "comprehensive"]:
            raise ValueError('Analysis depth must be "basic", "standard", or "comprehensive"')
        return v


class GenericQAResponse(BaseModel):
    """Response model for generic Q&A operations."""
    
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    category: QuestionCategory = Field(..., description="Detected or provided question category")
    template: str = Field(..., description="Project template used")
    confidence_score: float = Field(..., description="Answer confidence score", ge=0.0, le=1.0)
    project_analysis: Optional[Dict[str, Any]] = Field(None, description="Project analysis results")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TemplateListResponse(BaseModel):
    """Response model for template listing."""
    
    templates: List[ProjectTemplate] = Field(..., description="Available project templates")
    total_count: int = Field(..., description="Total number of templates")
    default_template: str = Field(..., description="Default template name")


class CategoryListResponse(BaseModel):
    """Response model for category listing."""
    
    categories: List[Dict[str, Any]] = Field(..., description="Supported question categories")
    examples: Dict[str, List[str]] = Field(..., description="Example questions by category")


class ProjectAnalysisResponse(BaseModel):
    """Response model for project analysis operations."""
    
    success: bool = Field(..., description="Analysis success status")
    detected_template: Optional[str] = Field(None, description="Detected project template")
    architecture_patterns: List[str] = Field(default_factory=list, description="Detected architecture patterns")
    business_capabilities: List[str] = Field(default_factory=list, description="Identified business capabilities")
    api_endpoints: List[Dict[str, Any]] = Field(default_factory=list, description="Discovered API endpoints")
    data_models: List[Dict[str, Any]] = Field(default_factory=list, description="Analyzed data models")
    operational_patterns: Dict[str, Any] = Field(default_factory=dict, description="Operational patterns")
    confidence: float = Field(..., description="Analysis confidence score", ge=0.0, le=1.0)
    analysis_timestamp: float = Field(..., description="Analysis timestamp")
    error: Optional[str] = Field(None, description="Error message if analysis failed")


# Configuration Models

class APIConfig(BaseModel):
    """API configuration model."""
    
    max_query_length: int = Field(default=2000, description="Maximum query length")
    default_top_k: int = Field(default=5, description="Default number of results")
    max_top_k: int = Field(default=50, description="Maximum number of results")
    default_timeout: int = Field(default=30, description="Default request timeout in seconds")
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    max_parallel_workflows: int = Field(default=10, description="Maximum parallel workflows")
