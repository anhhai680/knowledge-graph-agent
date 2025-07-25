# LangGraph Workflows - Detailed Flow Analysis

## Overview

This document clarifies the specific LangGraph workflows and LangChain usage in the Knowledge Graph Agent project.

## LangGraph Workflow Architecture

### 1. LangGraph Indexing Workflow

The **Indexing Workflow** handles the stateful process of loading, processing, and storing repository content:

```mermaid
flowchart TD
    %% Indexing Workflow States
    START([Start Indexing]) --> INIT[Initialize State]
    INIT --> LOAD_REPOS[Load Repositories]
    LOAD_REPOS --> VALIDATE_REPOS{Validate Repos}
    
    VALIDATE_REPOS -->|Valid| LOAD_FILES[Load Files from GitHub]
    VALIDATE_REPOS -->|Invalid| ERROR_REPOS[Log Repository Errors]
    ERROR_REPOS --> LOAD_FILES
    
    LOAD_FILES --> PROCESS_DOCS[Process Documents]
    PROCESS_DOCS --> CHUNK_DOCS[Language-Aware Chunking]
    CHUNK_DOCS --> EXTRACT_META[Extract Metadata]
    EXTRACT_META --> GENERATE_EMBED[Generate Embeddings]
    GENERATE_EMBED --> STORE_VECTORS[Store in Vector DB]
    
    STORE_VECTORS --> UPDATE_STATE[Update Workflow State]
    UPDATE_STATE --> CHECK_COMPLETE{All Repos Processed?}
    
    CHECK_COMPLETE -->|No| LOAD_FILES
    CHECK_COMPLETE -->|Yes| FINALIZE[Finalize Index]
    FINALIZE --> END([End Successfully])
    
    %% Error Handling States
    LOAD_FILES --> ERROR_FILES[Handle File Errors]
    PROCESS_DOCS --> ERROR_PROCESSING[Handle Processing Errors]
    GENERATE_EMBED --> ERROR_EMBED[Handle Embedding Errors]
    STORE_VECTORS --> ERROR_STORAGE[Handle Storage Errors]
    
    ERROR_FILES --> LOAD_FILES
    ERROR_PROCESSING --> PROCESS_DOCS
    ERROR_EMBED --> GENERATE_EMBED
    ERROR_STORAGE --> STORE_VECTORS
    
    %% State Persistence
    UPDATE_STATE -.-> STATE_DB[(Workflow State)]
    ERROR_FILES -.-> STATE_DB
    ERROR_PROCESSING -.-> STATE_DB
    ERROR_EMBED -.-> STATE_DB
    ERROR_STORAGE -.-> STATE_DB
```

**Key LangGraph Features Used:**
- **State Management**: Tracks progress across repositories, files processed, errors encountered
- **Error Recovery**: Automatic retry mechanisms with exponential backoff
- **Progress Tracking**: Persistent state allows resuming interrupted indexing
- **Parallel Processing**: Multiple repositories can be processed concurrently

### 2. LangGraph Query Workflow

The **Query Workflow** handles the stateful RAG query processing:

```mermaid
flowchart TD
    %% Query Workflow States
    START([Query Request]) --> PARSE[Parse Query]
    PARSE --> VALIDATE{Validate Query}
    
    VALIDATE -->|Valid| ANALYZE[Analyze Query Intent]
    VALIDATE -->|Invalid| RETURN_ERROR[Return Validation Error]
    
    ANALYZE --> DETERMINE_STRATEGY[Determine Search Strategy]
    DETERMINE_STRATEGY --> RETRIEVE[Vector Search]
    RETRIEVE --> FILTER_RESULTS[Filter & Rank Results]
    
    FILTER_RESULTS --> CHECK_CONTEXT{Sufficient Context?}
    CHECK_CONTEXT -->|No| EXPAND_SEARCH[Expand Search Parameters]
    CHECK_CONTEXT -->|Yes| PREPARE_CONTEXT[Prepare LLM Context]
    
    EXPAND_SEARCH --> RETRIEVE
    
    PREPARE_CONTEXT --> GENERATE_PROMPT[Generate Contextual Prompt]
    GENERATE_PROMPT --> LLM_CALL[Call LLM]
    LLM_CALL --> FORMAT_RESPONSE[Format Response]
    FORMAT_RESPONSE --> VALIDATE_RESPONSE{Response Quality Check}
    
    VALIDATE_RESPONSE -->|Good| RETURN_SUCCESS[Return Successful Response]
    VALIDATE_RESPONSE -->|Poor| RETRY_GENERATION[Retry with Different Context]
    
    RETRY_GENERATION --> PREPARE_CONTEXT
    RETURN_SUCCESS --> END([End Successfully])
    
    %% Error Handling
    RETRIEVE --> ERROR_RETRIEVAL[Handle Retrieval Errors]
    LLM_CALL --> ERROR_LLM[Handle LLM Errors]
    
    ERROR_RETRIEVAL --> FALLBACK_SEARCH[Fallback Search Strategy]
    ERROR_LLM --> RETRY_LLM[Retry LLM Call]
    
    FALLBACK_SEARCH --> FILTER_RESULTS
    RETRY_LLM --> LLM_CALL
    
    %% State Tracking
    ANALYZE -.-> QUERY_STATE[(Query State)]
    RETRIEVE -.-> QUERY_STATE
    LLM_CALL -.-> QUERY_STATE
```

**Key LangGraph Features Used:**
- **Query State Tracking**: Maintains context throughout the query processing
- **Adaptive Search**: Adjusts search strategy based on initial results
- **Response Quality Control**: Validates and retries poor responses
- **Error Recovery**: Handles vector DB and LLM failures gracefully

## LangChain Usage in the Project

### **Where do we use LangChain?**

LangChain is used throughout the project as the foundational framework:

```mermaid
flowchart TB
    %% LangChain Components
    subgraph "LangChain Framework Integration"
        
        subgraph "Document Processing (LangChain)"
            LC_LOADERS[LangChain Document Loaders]
            LC_SPLITTERS[LangChain Text Splitters]
            LC_EMBEDDINGS[LangChain Embeddings]
        end
        
        subgraph "Vector Stores (LangChain)"
            LC_VECTORSTORE[LangChain VectorStore Interface]
            LC_CHROMA[LangChain Chroma Integration]
            LC_PINECONE[LangChain Pinecone Integration]
        end
        
        subgraph "LLM Integration (LangChain)"
            LC_LLM[LangChain LLM Interface]
            LC_OPENAI[LangChain OpenAI Integration]
            LC_PROMPTS[LangChain Prompt Templates]
        end
        
        subgraph "RAG Chain (LangChain)"
            LC_RETRIEVER[LangChain Retriever]
            LC_RAG_CHAIN[LangChain RAG Chain]
            LC_RUNNABLE[LangChain Runnable Interface]
        end
    end
    
    %% LangGraph Orchestration
    subgraph "LangGraph Workflows"
        LG_INDEX_WORKFLOW[Indexing Workflow]
        LG_QUERY_WORKFLOW[Query Workflow]
        LG_STATE[Workflow State Management]
    end
    
    %% Project Components
    subgraph "Project Implementation"
        GITHUB_LOADER[GitHub Loader]
        DOC_PROCESSOR[Document Processor]
        VECTOR_FACTORY[Vector Store Factory]
        RAG_AGENT[RAG Agent]
        API_ROUTES[API Routes]
    end
    
    %% Connections
    LC_LOADERS --> GITHUB_LOADER
    LC_SPLITTERS --> DOC_PROCESSOR
    LC_EMBEDDINGS --> DOC_PROCESSOR
    
    LC_VECTORSTORE --> VECTOR_FACTORY
    LC_CHROMA --> VECTOR_FACTORY
    LC_PINECONE --> VECTOR_FACTORY
    
    LC_LLM --> RAG_AGENT
    LC_OPENAI --> RAG_AGENT
    LC_PROMPTS --> RAG_AGENT
    
    LC_RETRIEVER --> RAG_AGENT
    LC_RAG_CHAIN --> RAG_AGENT
    LC_RUNNABLE --> RAG_AGENT
    
    LG_INDEX_WORKFLOW --> GITHUB_LOADER
    LG_INDEX_WORKFLOW --> DOC_PROCESSOR
    LG_INDEX_WORKFLOW --> VECTOR_FACTORY
    
    LG_QUERY_WORKFLOW --> RAG_AGENT
    LG_QUERY_WORKFLOW --> VECTOR_FACTORY
    
    RAG_AGENT --> API_ROUTES
```

### Specific LangChain Components Used:

#### 1. **Document Processing Chain**
```python
# LangChain components in src/loaders/github_loader.py
from langchain.document_loaders import BaseLoader
from langchain.schema import Document

# LangChain components in src/processors/document_processor.py  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
```

#### 2. **Vector Store Integration**
```python
# LangChain components in src/vectorstores/
from langchain.vectorstores import Chroma, Pinecone
from langchain.vectorstores.base import VectorStore
```

#### 3. **LLM and RAG Chain**
```python
# LangChain components in src/agents/rag_agent.py
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
```

#### 4. **Integration with LangGraph**
```python
# LangChain Runnable integration with LangGraph workflows
from langgraph import StateGraph
from langchain.schema.runnable import Runnable

# Workflows inherit from Runnable for seamless integration
class IndexingWorkflow(Runnable):
    # LangChain components orchestrated by LangGraph
```

## Key Architectural Decisions

### 1. **Why LangGraph + LangChain?**
- **LangChain**: Provides rich ecosystem of components (loaders, splitters, vector stores, LLMs)
- **LangGraph**: Adds stateful workflow orchestration and error recovery capabilities
- **Integration**: LangGraph workflows orchestrate LangChain components with persistent state

### 2. **Workflow State Management**
```python
# Example workflow state structure
class IndexingState(TypedDict):
    repositories: List[str]
    current_repo: str
    processed_files: int
    total_files: int
    errors: List[str]
    embeddings_generated: int
    status: str
```

### 3. **Error Recovery and Retry Logic**
- **LangGraph**: Manages retry logic and state persistence
- **LangChain**: Provides underlying component reliability
- **Combined**: Robust error handling with automatic recovery

## Benefits of This Architecture

1. **Stateful Processing**: LangGraph maintains state across long-running operations
2. **Component Reusability**: LangChain components are orchestrated by workflows
3. **Error Recovery**: Automatic retry and recovery mechanisms
4. **Scalability**: Parallel processing within workflows
5. **Observability**: Complete workflow tracing and state inspection
6. **Extensibility**: Easy to add new steps or modify existing workflows

This architecture ensures reliable, observable, and maintainable AI agent workflows while leveraging the rich LangChain ecosystem.
