"""
Environment-based configuration module for the Knowledge Graph Agent.

This module provides configuration management for the application using Pydantic models
to validate environment variables and application settings.
"""

import json
import os
from enum import Enum
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
import logging
from .git_settings import GitSettings

# Load environment variables from .env file
load_dotenv()


class LogLevel(str, Enum):
    """Log level enum."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AppEnvironment(str, Enum):
    """Application environment enum."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LLMProvider(str, Enum):
    """LLM provider enum."""

    OPENAI = "openai"
    OLLAMA = "ollama"


class EmbeddingProvider(str, Enum):
    """Embedding provider enum."""

    OPENAI = "openai"


class DatabaseType(str, Enum):
    """Vector database type enum."""

    CHROMA = "chroma"
    PINECONE = "pinecone"


class GraphStoreType(str, Enum):
    """Graph database type enum."""

    MEMGRAPH = "memgraph"
    NEO4J = "neo4j"


class WorkflowStateBackend(str, Enum):
    """Workflow state backend enum."""

    MEMORY = "memory"
    DATABASE = "database"


class OpenAISettings(BaseModel):
    """OpenAI API settings."""

    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field("gpt-4o-mini", description="OpenAI model to use")
    temperature: float = Field(0.7, description="Temperature for OpenAI API calls")
    max_tokens: int = Field(4000, description="Maximum tokens for OpenAI API calls")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate OpenAI API key."""
        if not v or v == "your_openai_api_key":
            raise ValueError("OpenAI API key is required")
        return v


class ChromaSettings(BaseModel):
    """Chroma database settings."""

    host: str = Field("localhost", description="Chroma server host")
    port: int = Field(8000, description="Chroma server port")
    collection_name: str = Field(
        "knowledge-base-graph", description="Chroma collection name"
    )


class PineconeSettings(BaseModel):
    """Pinecone database settings."""

    api_key: str = Field(..., description="Pinecone API key")
    collection_name: str = Field(
        "knowledge-base-graph", description="Pinecone collection name"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate Pinecone API key."""
        if not v or v == "pinecone-api-key":
            raise ValueError("Pinecone API key is required when using Pinecone")
        return v


class GraphStoreSettings(BaseModel):
    """Graph database settings."""

    store_type: GraphStoreType = Field(
        GraphStoreType.MEMGRAPH, description="Graph database type"
    )
    url: str = Field("bolt://localhost:7687", description="Graph database URL")
    username: str = Field("", description="Graph database username")
    password: str = Field("", description="Graph database password")
    database: str = Field("memgraph", description="Graph database name")


class GitHubSettings(BaseModel):
    """GitHub API settings."""

    token: str = Field(..., description="GitHub API token")
    file_extensions: List[str] = Field(
        [
            ".cs",
            ".csproj",
            ".py",
            ".php",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".html",
            ".cshtml",
            ".md",
            ".txt",
            ".json",
            ".yml",
            ".yaml",
            ".csv",
            "dockerfile",
            ".config",
            ".sh",
            ".bash",
        ],
        description="File extensions to process",
    )

    @field_validator("token")
    @classmethod
    def validate_token(cls, v: str) -> str:
        """Validate GitHub token."""
        if not v or v == "your_github_token":
            raise ValueError("GitHub token is required")
        return v


class EmbeddingSettings(BaseModel):
    """Embedding settings."""

    provider: EmbeddingProvider = Field(
        EmbeddingProvider.OPENAI, description="Embedding provider"
    )
    model: str = Field("text-embedding-ada-002", description="Embedding model")
    batch_size: int = Field(50, description="Batch size for embedding generation")
    max_tokens_per_batch: int = Field(250000, description="Maximum tokens per batch")
    embedding_api_key: Optional[str] = Field(
        None, description="API key for embedding provider (if required)"
    )


class DocumentProcessingSettings(BaseModel):
    """Document processing settings."""

    chunk_size: int = Field(1000, description="Document chunk size")
    chunk_overlap: int = Field(200, description="Document chunk overlap")


class WorkflowSettings(BaseModel):
    """Workflow settings."""

    state_persistence: bool = Field(
        True, description="Enable workflow state persistence"
    )
    retry_attempts: int = Field(3, description="Number of retry attempts for workflows")
    retry_delay_seconds: int = Field(
        5, description="Delay between retry attempts in seconds"
    )
    timeout_seconds: int = Field(3600, description="Workflow timeout in seconds")
    parallel_repos: int = Field(
        2, description="Number of repositories to process in parallel"
    )
    state_backend: WorkflowStateBackend = Field(
        WorkflowStateBackend.MEMORY,
        description="Workflow state backend (memory or database)",
    )


class LangChainSettings(BaseModel):
    """LangChain settings."""

    tracing: bool = Field(False, description="Enable LangChain tracing")
    api_key: Optional[str] = Field(None, description="LangSmith API key")
    project: str = Field("knowledge-graph-agent", description="LangSmith project name")


class RepositoryConfig(BaseModel):
    """GitHub repository configuration."""

    name: str = Field(..., description="Repository name")
    url: str = Field(..., description="Repository URL")
    branch: str = Field("main", description="Repository branch")
    description: Optional[str] = Field(None, description="Repository description")


class AppSettings(BaseModel):
    """Application settings."""

    app_env: AppEnvironment = Field(
        AppEnvironment.DEVELOPMENT, description="Application environment"
    )
    log_level: LogLevel = Field(LogLevel.INFO, description="Log level")
    llm_provider: LLMProvider = Field(LLMProvider.OPENAI, description="LLM provider")
    llm_api_base_url: Optional[str] = Field(
        "https://api.openai.com/v1",
        description="LLM API base URL (for non-OpenAI providers)",
    )
    database_type: DatabaseType = Field(
        DatabaseType.CHROMA, description="Vector database type"
    )

    # Component settings
    openai: OpenAISettings
    chroma: ChromaSettings
    pinecone: Optional[PineconeSettings] = None
    graph_store: GraphStoreSettings = GraphStoreSettings()
    github: GitHubSettings
    git: GitSettings = GitSettings()
    embedding: EmbeddingSettings
    document_processing: DocumentProcessingSettings
    workflow: WorkflowSettings
    langchain: LangChainSettings

    # Configuration flags
    use_git_loader: bool = Field(
        True, 
        description="Use Git-based loader instead of API-based loader"
    )
    
    # Graph feature flags
    enable_graph_features: bool = Field(
        False, 
        description="Enable graph database features"
    )
    enable_hybrid_search: bool = Field(
        False, 
        description="Enable hybrid search (vector + graph)"
    )
    enable_graph_visualization: bool = Field(
        False, 
        description="Enable graph visualization features"
    )

    # Repository configurations from appSettings.json
    repositories: List[RepositoryConfig] = Field(
        [], description="GitHub repositories to process"
    )

    @model_validator(mode="after")
    def validate_database_settings(self) -> "AppSettings":
        """Validate database settings based on the selected database type."""
        if self.database_type == DatabaseType.PINECONE and not self.pinecone:
            raise ValueError(
                "Pinecone settings are required when using Pinecone database"
            )
        return self


def load_app_settings_from_json(file_path: str) -> List[RepositoryConfig]:
    """
    Load repository configurations from appSettings.json file.

    Args:
        file_path: Path to appSettings.json file

    Returns:
        List of RepositoryConfig objects
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        repos = []
        for repo_data in data.get("repositories", []):
            repos.append(RepositoryConfig(**repo_data))
        return repos
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.error(f"Error loading appSettings.json: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error loading appSettings.json: {str(e)}")
        return []


def get_settings() -> AppSettings:
    """
    Get application settings from environment variables and appSettings.json.

    Returns:
        AppSettings object with configuration values
    """
    try:
        # Check if we're in test environment
        app_env = os.getenv("APP_ENV", AppEnvironment.DEVELOPMENT.value)
        is_testing = app_env == AppEnvironment.TESTING.value
        
        # Load basic settings from environment
        settings_dict = {
            "app_env": app_env,
            "log_level": os.getenv("LOG_LEVEL", LogLevel.INFO.value),
            "llm_provider": os.getenv("LLM_PROVIDER", LLMProvider.OPENAI.value),
            "llm_api_base_url": os.getenv(
                "LLM_API_BASE_URL", "https://api.openai.com/v1"
            ),
            "database_type": os.getenv("DATABASE_TYPE", DatabaseType.CHROMA.value),
            # Graph feature flags
            "enable_graph_features": os.getenv("ENABLE_GRAPH_FEATURES", "false").lower() == "true",
            "enable_hybrid_search": os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() == "true",
            "enable_graph_visualization": os.getenv("ENABLE_GRAPH_VISUALIZATION", "false").lower() == "true",
            # OpenAI settings
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", "test_openai_key" if is_testing else ""),
                "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                "temperature": float(os.getenv("TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("MAX_TOKENS", "4000")),
            },
            # Chroma settings
            "chroma": {
                "host": os.getenv("CHROMA_HOST", "localhost"),
                "port": int(os.getenv("CHROMA_PORT", "8000")),
                "collection_name": os.getenv(
                    "CHROMA_COLLECTION_NAME", "knowledge-base-graph"
                ),
            },
            # Pinecone settings (optional)
            "pinecone": (
                {
                    "api_key": os.getenv("PINECONE_API_KEY", "test_pinecone_key" if is_testing else ""),
                    "collection_name": os.getenv(
                        "PINECONE_COLLECTION_NAME", "knowledge-base-graph"
                    ),
                }
                if os.getenv("DATABASE_TYPE") == DatabaseType.PINECONE.value
                else None
            ),
            # GitHub settings
            "github": {
                "token": os.getenv("GITHUB_TOKEN", "test_github_token" if is_testing else ""),
                "file_extensions": json.loads(
                    os.getenv("GITHUB_FILE_EXTENSIONS", "[]")
                ),
            },
            # Embedding settings
            "embedding": {
                "provider": os.getenv(
                    "EMBEDDING_PROVIDER", EmbeddingProvider.OPENAI.value
                ),
                "model": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
                "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "50")),
                "max_tokens_per_batch": int(
                    os.getenv("MAX_TOKENS_PER_BATCH", "250000")
                ),
                "embedding_api_key": os.getenv("EMBEDDING_API_KEY", "test_embedding_key" if is_testing else None),
            },
            # Document processing settings
            "document_processing": {
                "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
                "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
            },
            # Workflow settings
            "workflow": {
                "state_persistence": os.getenv(
                    "WORKFLOW_STATE_PERSISTENCE", "true"
                ).lower()
                == "true",
                "retry_attempts": int(os.getenv("WORKFLOW_RETRY_ATTEMPTS", "3")),
                "retry_delay_seconds": int(
                    os.getenv("WORKFLOW_RETRY_DELAY_SECONDS", "5")
                ),
                "timeout_seconds": int(os.getenv("WORKFLOW_TIMEOUT_SECONDS", "3600")),
                "parallel_repos": int(os.getenv("WORKFLOW_PARALLEL_REPOS", "2")),
                "state_backend": os.getenv(
                    "WORKFLOW_STATE_BACKEND", WorkflowStateBackend.MEMORY.value
                ),
            },
            # Graph store settings
            "graph_store": {
                "store_type": os.getenv("GRAPH_STORE_TYPE", GraphStoreType.MEMGRAPH.value),
                "url": os.getenv("GRAPH_STORE_URL", "bolt://localhost:7687"),
                "username": os.getenv("GRAPH_STORE_USER", ""),
                "password": os.getenv("GRAPH_STORE_PASSWORD", ""),
                "database": os.getenv("GRAPH_STORE_DATABASE", "memgraph"),
            },
            # LangChain settings
            "langchain": {
                "tracing": os.getenv("LANGCHAIN_TRACING", "false").lower() == "true",
                "api_key": os.getenv("LANGCHAIN_API_KEY", "test_langchain_key" if is_testing else None),
                "project": os.getenv("LANGCHAIN_PROJECT", "knowledge-graph-agent"),
            },
        }

        # Load repository configurations from appSettings.json
        try:
            repositories = load_app_settings_from_json("appSettings.json")
            settings_dict["repositories"] = [repo.model_dump() for repo in repositories]
        except Exception:
            # If appSettings.json doesn't exist or can't be loaded, use empty list for tests
            settings_dict["repositories"] = []

        # Create and validate settings object
        settings = AppSettings(**settings_dict)
        return settings

    except ValidationError as e:
        detailed_errors = []
        for error in e.errors():
            loc = " -> ".join(str(loc_item) for loc_item in error["loc"])
            detailed_errors.append(f"{loc}: {error['msg']}")

        error_message = f"Configuration validation errors:\n" + "\n".join(
            detailed_errors
        )
        logging.error(error_message)
        raise ValueError(error_message)

    except Exception as e:
        error_message = f"Error loading configuration: {str(e)}"
        logging.error(error_message)
        raise ValueError(error_message)


# Singleton instance of settings
settings = get_settings()
