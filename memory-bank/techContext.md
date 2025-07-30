# Technical Context - Knowledge Graph Agent

**Document Created:** July 30, 2025  
**Last Updated:** July 30, 2025  

## Technology Stack

### Core Framework
- **Python 3.11+**: Primary programming language with modern async support
- **LangChain**: AI/ML pipeline framework for document processing, embeddings, and RAG
- **LangGraph**: Stateful workflow orchestration with error recovery and state management
- **FastAPI**: Modern web framework for REST API with automatic documentation

### AI/ML Dependencies
- **OpenAI API**: Primary LLM provider for embeddings and chat completions
- **langchain-openai**: LangChain wrappers for OpenAI integration
- **tiktoken**: Token counting and text encoding for OpenAI models

### Vector Storage
- **Chroma**: Local vector database for development and self-hosted deployments
- **Pinecone**: Cloud vector database for production scalability
- **langchain-community**: Community integrations for vector stores

### Web & API
- **uvicorn**: ASGI server for FastAPI applications
- **httpx**: Modern HTTP client for external API calls
- **python-multipart**: Form data parsing for file uploads
- **python-dotenv**: Environment variable management

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

# GitHub Configuration
GITHUB_TOKEN=your_github_personal_access_token

# API Configuration
API_KEY=your_api_key_for_authentication
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
│   ├── agents/                   # Agent implementations
│   ├── api/                      # FastAPI routes and middleware
│   ├── config/                   # Configuration management
│   ├── llm/                      # LLM and embedding factories
│   ├── loaders/                  # Document loaders (GitHub)
│   ├── processors/               # Document processing and chunking
│   ├── utils/                    # Utilities and helpers
│   ├── vectorstores/             # Vector store implementations
│   └── workflows/                # LangGraph workflow definitions
├── tests/                        # Test suites
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── docs/                         # Documentation
├── memory-bank/                  # Memory bank files
├── web/                          # Web UI components
├── main.py                       # Application entry point
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── pyproject.toml               # Project configuration
└── docker-compose.yml          # Docker development setup
```

## Technical Constraints

### API Limitations
- **OpenAI API**: Rate limits and token limits per request
- **GitHub API**: 5000 requests/hour for authenticated users, file size limits
- **Pinecone**: Request rate limits and index size constraints
- **Token Limits**: GPT-4 context window limitations for large documents

### Performance Constraints
- **Memory Usage**: Large repositories can consume significant RAM during processing
- **Processing Time**: Document chunking and embedding generation can be time-intensive
- **Storage Limits**: Vector storage costs scale with document volume
- **Concurrent Processing**: Limited by API rate limits and system resources

### Security Constraints
- **API Keys**: Secure storage and rotation of service credentials
- **GitHub Access**: Private repository access requires appropriate token permissions
- **Data Privacy**: Sensitive code content sent to external AI services
- **Authentication**: API endpoint security and access control

## Dependencies Deep Dive

### Core LangChain Components
```python
# Document Processing
from langchain.document_loaders import GitHubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Embeddings and Vector Stores
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone

# LLM Integration
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Prompt Management
from langchain.prompts import PromptTemplate, ChatPromptTemplate
```

### LangGraph Workflow Components
```python
# Workflow State Management
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# State Definitions
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages

# Error Handling and Retry Logic
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import MessageGraph
```

### FastAPI and Web Components
```python
# API Framework
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

# Request/Response Models
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

# Async Support
import asyncio
import httpx
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
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./.env:/app/.env
```

### Testing Strategy
```python
# Unit Testing with pytest
@pytest.mark.asyncio
async def test_document_processing():
    processor = DocumentProcessor()
    document = create_test_document()
    result = await processor.process(document)
    assert result.chunks is not None

# Integration Testing
@pytest.mark.integration
async def test_indexing_workflow():
    workflow = IndexingWorkflow()
    config = load_test_config()
    result = await workflow.run(config)
    assert result.status == "completed"
```

## Technical Decisions

### Why LangChain + LangGraph?
- **Proven Patterns**: Battle-tested components for AI/ML pipelines
- **Extensive Integrations**: Rich ecosystem of providers and tools
- **Stateful Workflows**: LangGraph provides robust state management
- **Error Recovery**: Built-in retry and error handling mechanisms
- **Community Support**: Large community and extensive documentation

### Why Dual Vector Storage?
- **Development Flexibility**: Chroma for local development, no external dependencies
- **Production Scalability**: Pinecone for cloud deployment and scale
- **Cost Optimization**: Choose storage based on deployment requirements
- **Vendor Independence**: Avoid lock-in to single vector database provider

### Why FastAPI?
- **Modern Python**: Native async/await support for better performance
- **Automatic Documentation**: OpenAPI/Swagger documentation generation
- **Type Safety**: Pydantic integration for request/response validation
- **Production Ready**: High performance with uvicorn ASGI server

### Configuration Strategy
- **Environment Variables**: Runtime configuration without code changes
- **JSON Configuration**: Repository settings for easy modification
- **Validation**: Pydantic models ensure configuration correctness
- **Defaults**: Sensible defaults reduce configuration complexity

## Integration Points

### External Services
1. **GitHub API**: Repository content fetching and authentication
2. **OpenAI API**: Embeddings generation and LLM responses
3. **Pinecone API**: Cloud vector storage and similarity search
4. **Local Chroma**: File-based vector storage for development

### Internal Integrations
1. **Workflow Orchestration**: LangGraph manages processing state
2. **Component Communication**: LangChain interfaces provide abstraction
3. **Configuration Management**: Centralized settings with validation
4. **Logging Integration**: Structured logging across all components

This technical foundation provides a robust, scalable, and maintainable implementation of the Knowledge Graph Agent while supporting future enhancements and provider additions.
