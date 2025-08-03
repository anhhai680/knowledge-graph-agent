# Technical Context - Knowledge Graph Agent

**Document Created:** July 30, 2025  
**Last Updated:** August 3, 2025  

## Technology Stack

### Core Fr│   ├── workflows/                # LangGraph workflow definitions
│       ├── base_workflow.py      # Base workflow interface
│       ├── indexing_workflow.py  # Repository indexing workflow
│       ├── query_workflow.py     # Refactored modular query workflow (253 lines)
│       ├── query/                # ✨ NEW: Modular query components (August 3, 2025)
│       │   ├── handlers/          # Specialized query processing handlers
│       │   │   ├── query_parsing_handler.py      # Query parsing and intent analysis (116 lines)
│       │   │   ├── vector_search_handler.py      # Document retrieval and ranking (241 lines)
│       │   │   ├── context_processing_handler.py # Context preparation (164 lines)
│       │   │   └── llm_generation_handler.py     # LLM interaction and response generation (199 lines)
│       │   └── orchestrator/      # Query workflow orchestration
│       │       └── query_orchestrator.py         # Modular workflow coordination (267 lines)
│       ├── state_manager.py      # Workflow state management
│       └── workflow_states.py    # State type definitionsk
- **Python 3.11+**: Primary programming language with modern async support
- **LangChain**: AI/ML pipeline framework for document processing, embeddings, and RAG
- **LangGraph**: Stateful workflow orchestration with error recovery and state management
- **FastAPI**: Modern web framework for REST API with automatic documentation and CORS

### AI/ML Dependencies
- **OpenAI API**: Primary LLM provider for embeddings and chat completions
- **langchain-openai**: LangChain wrappers for OpenAI integration
- **tiktoken**: Token counting and text encoding for OpenAI models

### Git-Based Loading System
- **GitPython**: Python Git library for repository operations (if needed for advanced operations)
- **chardet**: Character encoding detection for file content reading
- **pathlib**: Modern path handling for file system operations
- **subprocess**: Secure Git command execution with timeout handling

### Vector Storage
- **Chroma**: Local vector database for development and self-hosted deployments
- **Pinecone**: Cloud vector database for production scalability
- **langchain-community**: Community integrations for vector stores
- **langchain-chroma**: Chroma-specific LangChain integration

### Web & API
- **uvicorn**: ASGI server for FastAPI applications
- **httpx**: Modern HTTP client for external API calls
- **python-multipart**: Form data parsing for file uploads
- **python-dotenv**: Environment variable management
- **fastapi[all]**: Full FastAPI installation with all optional dependencies

### Development & Testing
- **pytest**: Testing framework with async support
- **pytest-asyncio**: Async test support
- **pytest-mock**: Mocking capabilities for testing
- **black**: Code formatting
- **isort**: Import sorting
- **setuptools**: Package management and building

## Development Setup

### Environment Requirements
```bash
# Python version requirement
Python 3.11 or higher

# Virtual environment setup
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Configuration Files

#### Environment Variables (.env)
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Database Configuration
DATABASE_TYPE=chroma  # or 'pinecone'
CHROMA_PERSIST_DIRECTORY=./chroma_db
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=knowledge-graph-index

# GitHub Configuration (Optional - for private repos)
GITHUB_TOKEN=your_github_personal_access_token

# Git Configuration
GIT_TEMP_REPO_PATH=./temp_repo
GIT_COMMAND_TIMEOUT=300
GIT_FORCE_FRESH_CLONE=false
GIT_CLEANUP_AFTER_PROCESSING=false

# API Configuration
HOST=0.0.0.0
PORT=8000

# Application Settings
LOG_LEVEL=INFO
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIES=3
RETRY_DELAY=1.0
```

#### Repository Configuration (appSettings.json)
```json
{
    "repositories": [
        {
            "name": "example-repo",
            "url": "https://github.com/owner/repository",
            "description": "Repository description",
            "branch": "main"
        }
    ]
}
```

### Project Structure
```
knowledge-graph-agent/
├── src/                          # Source code
│   ├── agents/                   # RAG agent implementations
│   │   ├── base_agent.py         # Base agent interface
│   │   └── rag_agent.py          # RAG agent with query workflow
│   ├── api/                      # FastAPI routes and models
│   │   ├── main.py               # FastAPI application setup
│   │   ├── routes.py             # API endpoint definitions
│   │   └── models.py             # Pydantic request/response models
│   ├── config/                   # Configuration management
│   │   ├── settings.py           # Main settings with Pydantic validation
│   │   └── git_settings.py       # Git-specific configuration
│   ├── llm/                      # LLM and embedding factories
│   │   ├── llm_factory.py        # LLM provider abstraction
│   │   ├── embedding_factory.py  # Embedding provider abstraction
│   │   └── openai_provider.py    # OpenAI-specific implementation
│   ├── loaders/                  # Git-based document loaders
│   │   ├── enhanced_github_loader.py      # Main Git-based loader
│   │   ├── git_repository_manager.py      # Repository lifecycle management
│   │   ├── git_command_executor.py        # Safe Git command execution
│   │   ├── file_system_processor.py       # File discovery and reading
│   │   ├── git_metadata_extractor.py      # Git metadata extraction
│   │   ├── repository_url_handler.py      # URL and authentication handling
│   │   ├── git_error_handler.py           # Git operation error recovery
│   │   ├── loader_migration_manager.py    # API to Git loader migration
│   │   └── github_loader.py               # Legacy API-based loader
│   ├── processors/               # Document processing and chunking
│   │   ├── document_processor.py  # Main document processing
│   │   ├── chunking_strategy.py   # Language-aware chunking
│   │   └── metadata_extractor.py  # Document metadata extraction
│   ├── utils/                    # Utilities and helpers
│   │   ├── prompt_manager.py     # Prompt template management
│   │   ├── logging.py            # Structured logging setup
│   │   └── helpers.py            # Common utility functions
│   ├── vectorstores/             # Vector store implementations
│   └── workflows/                # LangGraph workflow definitions
│       ├── base_workflow.py      # Base workflow interface
│       ├── indexing_workflow.py  # Repository indexing workflow
│       ├── query_workflow.py     # RAG query workflow
│       ├── state_manager.py      # Workflow state management
│       └── workflow_states.py    # State type definitions
├── tests/                        # Test suites
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── workflows/                # ✨ ENHANCED: Workflow-specific tests (August 3, 2025)
│   │   ├── query/                # Modular query workflow tests
│   │   │   ├── test_orchestrator.py               # Orchestrator tests (284 lines)
│   │   │   ├── test_query_parsing_handler.py      # Parsing handler tests (202 lines)
│   │   │   └── test_vector_search_handler.py      # Search handler tests (224 lines)
│   │   ├── test_performance_comparison.py          # Performance validation tests (245 lines)
│   │   └── test_query_workflow_refactored.py       # Backward compatibility tests (263 lines)
│   ├── test_git_implementation.py # Git system integration tests
│   ├── test_query_workflow.py    # Query workflow tests
│   └── test_repository_listing.py # Repository processing tests
├── docs/                         # Documentation
├── memory-bank/                  # Memory bank files
├── web/                          # Web UI components
├── temp_repo/                    # Local Git repositories cache
├── chroma_db/                    # Local Chroma database
├── main.py                       # Application entry point
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── pyproject.toml               # Project configuration
└── docker-compose.yml          # Docker development setup
```

## Technical Constraints

### Performance Constraints
- **Memory Usage**: Large repositories can consume significant RAM during Git operations
- **Processing Time**: Document chunking and embedding generation can be time-intensive
- **Storage Limits**: Vector storage costs scale with document volume
- **Concurrent Processing**: Limited by system resources and Git operation locks
- **Repository Size**: Very large repositories may require chunked processing
- **Git Operations**: Network connectivity required for initial clone and updates

### Security Constraints
- **API Keys**: Secure storage and rotation of OpenAI service credentials
- **GitHub Access**: Private repository access requires appropriate token permissions
- **Data Privacy**: Sensitive code content sent to external AI services
- **Local Storage**: Git repositories cached locally require disk space management
- **Git Authentication**: SSH keys and HTTPS tokens must be securely managed

### Git System Constraints
- **Git Installation**: Requires Git to be installed and accessible in system PATH
- **Network Access**: Initial repository cloning requires internet connectivity
- **Disk Space**: Local repository caching consumes storage space
- **File System**: Cross-platform path handling and encoding considerations
- **Repository Access**: Authentication required for private repositories
- **Concurrent Access**: Multiple processes accessing same repository require coordination

## Dependencies Deep Dive

### Core LangChain Components
```python
# Document Processing with Git-based Loading
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Git-based Enhanced GitHub Loader
from src.loaders.enhanced_github_loader import EnhancedGitHubLoader
from src.loaders.git_repository_manager import GitRepositoryManager
from src.loaders.git_command_executor import GitCommandExecutor

# Embeddings and Vector Stores
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore

# LLM Integration
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Prompt Management
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from src.utils.prompt_manager import PromptManager
```

### LangGraph Workflow Components
```python
# Workflow State Management
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# State Definitions
from typing import TypedDict, Annotated, List, Dict, Any
from src.workflows.workflow_states import IndexingState, QueryState

# Workflow Implementations
from src.workflows.indexing_workflow import IndexingWorkflow
from src.workflows.query_workflow import QueryWorkflow
from src.workflows.state_manager import StateManager

# Error Handling and Retry Logic
from langgraph.prebuilt import ToolNode
from src.workflows.base_workflow import BaseWorkflow
```

### FastAPI and Web Components
```python
# API Framework
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Request/Response Models
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from src.api.models import IndexRequest, QueryRequest, StatusResponse

# Async Support
import asyncio
import httpx
from src.api.main import app, get_application
```

### Git Operations and File Processing
```python
# Git Command Execution
import subprocess
from pathlib import Path
from src.loaders.git_command_executor import GitCommandExecutor, GitCommandResult

# File System Processing
import chardet
from src.loaders.file_system_processor import FileSystemProcessor

# Repository Management
from src.loaders.git_repository_manager import GitRepositoryManager
from src.loaders.repository_url_handler import RepositoryUrlHandler

# Metadata Extraction
from src.loaders.git_metadata_extractor import GitMetadataExtractor
from src.processors.metadata_extractor import MetadataExtractor
```

## Development Workflow

### Local Development
1. **Environment Setup**: Create and activate virtual environment
2. **Dependencies**: Install requirements with `pip install -r requirements.txt`
3. **Configuration**: Copy `.env.example` to `.env` and configure
4. **Database**: Initialize vector store (Chroma creates automatically)
5. **Testing**: Run tests with `pytest` before committing changes
6. **Code Quality**: Use `black` and `isort` for code formatting

### Docker Development
```yaml
# docker-compose.yml for local development
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_TYPE=chroma
      - CHROMA_PERSIST_DIRECTORY=/app/chroma_db
      - GIT_TEMP_REPO_PATH=/app/temp_repo
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./temp_repo:/app/temp_repo
      - ./.env:/app/.env
```

### Testing Strategy
```python
# Unit Testing with pytest
@pytest.mark.asyncio
async def test_git_repository_manager():
    manager = GitRepositoryManager()
    repo_path = manager.get_local_repo_path("owner", "repo")
    assert "temp_repo/owner/repo" in repo_path

@pytest.mark.asyncio
async def test_enhanced_github_loader():
    loader = EnhancedGitHubLoader("microsoft", "vscode", branch="main")
    validation = loader.validate_setup()
    assert validation["git_available"] is True

# Integration Testing
@pytest.mark.integration
async def test_indexing_workflow():
    workflow = IndexingWorkflow()
    config = {
        "repository": {"owner": "test", "name": "repo"},
        "processing_config": {"chunk_size": 1000}
    }
    # Test workflow execution with mocked Git operations
    result = await workflow.run(config)
    assert result["status"] == "completed"

# Git System Testing
def test_git_command_executor():
    executor = GitCommandExecutor()
    assert executor.validate_git_installation() is True
```

## Technical Decisions

### Why LangChain + LangGraph?
- **Proven Patterns**: Battle-tested components for AI/ML pipelines
- **Extensive Integrations**: Rich ecosystem of providers and tools
- **Stateful Workflows**: LangGraph provides robust state management with error recovery
- **Error Recovery**: Built-in retry and error handling mechanisms
- **Community Support**: Large community and extensive documentation
- **Workflow Visualization**: Clear representation of processing pipelines

### Why Git-Based Loading Instead of GitHub API?
- **Rate Limit Elimination**: No more GitHub API constraints (5000 requests/hour)
- **Performance Enhancement**: 10x faster file loading through direct file system access
- **Rich Metadata**: Complete Git history, commit information, and repository statistics
- **Offline Capability**: Cached repositories enable offline processing
- **Simplified Architecture**: Removes complex API rate limiting and retry logic
- **Better Scalability**: Handle large repositories without API throttling

### Why Dual Vector Storage?
- **Development Flexibility**: Chroma for local development, no external dependencies
- **Production Scalability**: Pinecone for cloud deployment and scale
- **Cost Optimization**: Choose storage based on deployment requirements
- **Vendor Independence**: Avoid lock-in to single vector database provider
- **Migration Path**: Easy switching between providers with unified interface

### Why FastAPI?
- **Modern Python**: Native async/await support for better performance
- **Automatic Documentation**: OpenAPI/Swagger documentation generation
- **Type Safety**: Pydantic integration for request/response validation
- **Production Ready**: High performance with uvicorn ASGI server
- **Middleware Support**: Easy CORS handling and request processing

### Configuration Strategy
- **Environment Variables**: Runtime configuration without code changes
- **JSON Configuration**: Repository settings for easy modification
- **Validation**: Pydantic models ensure configuration correctness
- **Defaults**: Sensible defaults reduce configuration complexity
- **Git Settings**: Specialized configuration for Git operations and performance tuning

## Integration Points

### External Services
1. **Git Repositories**: Direct repository cloning and file system access (replaces GitHub API)
2. **OpenAI API**: Embeddings generation and LLM responses with rate limiting
3. **Pinecone API**: Cloud vector storage and similarity search (optional)
4. **Local Chroma**: File-based vector storage for development and self-hosted deployments

### Internal Integrations
1. **Workflow Orchestration**: LangGraph manages processing state across Git operations
2. **Component Communication**: LangChain interfaces provide abstraction layers
3. **Configuration Management**: Centralized settings with Git-specific configuration
4. **Logging Integration**: Structured logging across all components including Git operations
5. **Error Recovery**: Comprehensive error handling for Git operations and workflow failures

### Git System Integration
1. **Repository Management**: Local cloning, pulling, and validation of Git repositories
2. **File System Access**: Direct file reading with encoding detection and metadata extraction
3. **Metadata Extraction**: Git command execution for commit history and repository statistics
4. **Authentication**: Support for HTTPS tokens and SSH key authentication
5. **Caching Strategy**: Intelligent repository caching with configurable cleanup policies

### Performance Optimizations
1. **Local Repository Caching**: Reuse cloned repositories for subsequent processing
2. **Parallel File Processing**: Concurrent document processing within repositories
3. **Intelligent Updates**: Only pull repository changes when necessary
4. **Memory Management**: Efficient file processing with streaming and chunking
5. **Resource Cleanup**: Configurable repository cleanup and disk space management

This technical foundation provides a robust, scalable, and maintainable implementation of the Knowledge Graph Agent with significant performance improvements over API-based approaches. The Git-based loading system eliminates external dependencies and rate limiting while providing richer metadata and faster processing capabilities. Recent modular refactoring (August 3, 2025) has enhanced maintainability and testability with 76% complexity reduction in the query workflow architecture.
