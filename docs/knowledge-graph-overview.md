# Knowledge Graph Agent Document

## Overview
This document outlines the overview of an AI Agent with RAG (Retrieval-Augmented Generation) capabilities to index and query codebases from GitHub repositories. The agent will provide intelligent responses based on the indexed knowledge base.

## Requirements
- Load all settings from environment file (.env)
- Index all documents and files in GitHub Repository based on list of URLs provided in `appSettings.json`
- Using Python language for AI Agent
- Using LangChain framework to build AI Agent, and LangGraph to manage workflows
- Integration with LLM models (e.g.OpenAI...)
- Using vector database with Chroma and Pinecone. Allow switch between them based on `DATABASE_TYPE` environment setting.

## Architecture Overview

### System Components
1. **Document Loaders**: Extract content from various sources (GitHub, Github Enterprise)
2. **Text Processors**: Clean, chunk, and prepare documents for vectorization
3. **Vector Store**: Store and retrieve document embeddings
4. **LLM Interface**: Handle queries and generate responses
5. **RAG Pipeline**: Orchestrate retrieval and generation processes
6. **API Layer**: Provide REST endpoints for interaction

### Data Flow

#### System Architecture Diagram
```mermaid
graph TB
    %% Data Sources
    subgraph "Data Sources"
        GH[GitHub Repository]
    end
    
    %% Document Processing Pipeline
    subgraph "Document Processing Pipeline"
        DL[Document Loaders]
        TP[Text Processor]
        CHUNK[Chunking Strategy]
        EM[Embeddings Generator]
    end
    
    %% Vector Storage
    subgraph "Vector Storage"
        PC[Pinecone]
    end
    
    %% AI Agent Core
    subgraph "AI Agent Core"
        RA[RAG Agent]
        QP[Query Processor]
        CR[Context Retriever]
        LLM[LLM Interface]
        PM[Prompt Manager]
    end
    
    %% API Layer
    subgraph "API Layer"
        API[REST API]
        AUTH[Authentication]
    end
    
    %% Monitoring & Logging
    subgraph "Monitoring"
        LOG[Logger]
        HEALTH[Health Check]
    end
    
    %% Data Flow - Document Processing
    GH --> DL
    DL --> TP
    TP --> CHUNK
    CHUNK --> EM
    EM --> PC
    
    %% Data Flow - Query Processing
    API --> AUTH
    AUTH --> QP
    QP --> CR
    CR --> PC
    CR --> RA
    RA --> PM
    PM --> LLM
    LLM --> PM
    PM --> RA
    RA --> API
    
    %% Monitoring Connections
    API --> LOG
    API --> HEALTH
```

#### Query Processing Flow
```mermaid
sequenceDiagram
    participant U as User
    participant API as API Gateway
    participant A as AI Agent
    participant QP as Query Processor
    participant VS as Vector Store
    participant LLM as LLM Service
    participant PM as Prompt Manager
    
    U->>API: Submit Query
    API->>A: Process Query Request
    A->>QP: Parse & Validate Query
    QP->>VS: Retrieve Similar Documents
    VS-->>QP: Return Relevant Chunks
    QP->>PM: Prepare Context
    PM->>LLM: Send Prompt with Context
    LLM-->>PM: Generate Response
    PM->>A: Format Response
    A->>API: Return Structured Result
    API-->>U: Send Response
```

#### Document Indexing Flow
```mermaid
flowchart TD
    START([Start Indexing]) --> SOURCE{Select Source Type}
    
    SOURCE -->|GitHub| GH[GitHub Loader]
    
    GH --> EXTRACT[Extract Content]
    
    EXTRACT --> CLEAN[Clean Text]
    CLEAN --> CHUNK[Split into Chunks]
    CHUNK --> EMBED[Generate Embeddings]
    
    EMBED --> STORE{Storage Type}
    STORE --> PINECONE[Store in Pinecone]
    
    PINECONE --> INDEX[Update Index]
    
    INDEX --> NOTIFY[Notify Completion]
    NOTIFY --> END([End])
```

## Implementation Details

### 1. Project Structure
```
knowledge-graph-agent/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── rag_agent.py
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── github_loader.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   └── chunking_strategy.py
│   ├── vectorstores/
│   │   ├── __init__.py
│   │   ├── base_store.py
│   │   ├── chroma_store.py
│   │   ├── pinecone_store.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm_factory.py
│   │   ├── embedding_factory.py
│   │   ├── openai_llm.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   └── helpers.py
│   └── api/
│       ├── __init__.py
│       ├── routes.py
│       ├── models.py
│       └── middleware.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── performance/
├── docs/
│   ├── api/
│   ├── deployment/
│   └── user-guide/
├── scripts/
│   ├── setup.sh
│   ├── deploy.sh
│   └── migrate.py
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── docker-compose.yml
├── docker-compose.prod.yml
├── Dockerfile
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
└── main.py
```

### 2. Environment Configuration

#### .env Configuration
```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_BASE_URL=https://api.openai.com/v1 #Just use for ollama only

OPENAI_API_KEY=your_openai_api_key

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_BATCH_SIZE=50
MAX_TOKENS_PER_BATCH=250000

# Vector Database Configuration
DATABASE_TYPE=chroma #This configuration to allow switch vector store between chroma and pinecone
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=knowledge-base-graph

PINECONE_API_KEY=pinecone-api-key
PINECONE_COLLECTION_NAME=knowledge-base-graph

# GitHub Configuration
GITHUB_TOKEN=your_github_token
GITHUB_FILE_EXTENSIONS=[".cs",".csproj",".py",".php",".js",".jsx",".ts",".tsx",".html",".cshtml",".md", ".txt",".json",".yml",".yaml",".csv","dockerfile", ".config",".sh",".bash"]

# Application Configuration
APP_ENV=development #production
LOG_LEVEL=INFO #DEBUG
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=4000
TEMPERATURE=0.7
```

## References
- https://python.langchain.com/docs/tutorials/rag/
- https://langchain-ai.github.io/langgraph/tutorials/workflows/#building-blocks-the-augmented-llm
- https://docs.pinecone.io/guides/get-started/quickstart

