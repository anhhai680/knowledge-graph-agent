# GitHub Copilot Instructions for Knowledge Graph Agent

## Code Implementation Guidelines

This is an AI-powered RAG system using **FastAPI + LangChain/LangGraph + OpenAI**. When implementing features or fixing bugs, follow these project-specific patterns and conventions.

### Core Architecture Patterns

**1. LangGraph Workflow Pattern** - All complex operations use stateful workflows:
```python
from src.workflows.base_workflow import BaseWorkflow, WorkflowStep
from src.workflows.workflow_states import WorkflowState

class MyWorkflow(BaseWorkflow):
    async def process_step(self, state: Dict[str, Any], step: WorkflowStep) -> Dict[str, Any]:
        # Always include error handling and state updates
        try:
            result = await self._execute_operation(state)
            return self._update_state(state, result)
        except Exception as e:
            return self._handle_error(state, e, step)
```

**2. FastAPI Async Pattern** - All API endpoints must be async:
```python
from fastapi import APIRouter, HTTPException, Depends
from src.api.models import QueryRequest, QueryResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> QueryResponse:
    try:
        result = await workflow_instance.execute({"query": request.query})
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**3. Vector Store Factory Pattern** - Always use the factory for vector operations:
```python
from src.vectorstores.store_factory import VectorStoreFactory
from src.config.settings import get_settings

settings = get_settings()
vector_store = VectorStoreFactory.create_store(settings.database_type)
```

### Configuration & Environment Patterns

**1. Settings Validation** - Use Pydantic for all configuration:
```python
from src.config.settings import get_settings
from pydantic import BaseModel, Field, field_validator

class MyConfig(BaseModel):
    timeout_seconds: int = Field(default=300, ge=1, le=7200)
    
    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v):
        if v > 3600:
            logger.warning(f"High timeout value: {v}s")
        return v
```

**2. Environment Variables** - Always check required keys:
```python
import os
from src.config.settings import get_settings

# Required environment variables for this project:
# OPENAI_API_KEY, GITHUB_TOKEN, DATABASE_TYPE (chroma|pinecone)
settings = get_settings()
if not settings.openai_api_key:
    raise ValueError("OPENAI_API_KEY is required")
```

### Error Handling & Logging Patterns

**1. Structured Logging** - Use loguru, not print statements:
```python
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Good patterns:
logger.info(f"Processing repository: {repo_name}")
logger.error(f"Failed to index {file_path}: {e}", exc_info=True)
logger.debug(f"State transition: {old_state} -> {new_state}")

# Avoid: print statements or basic logging
```

**2. Exception Handling** - Project-specific error types:
```python
from src.utils.defensive_programming import safe_execute
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def robust_operation():
    try:
        result = await risky_operation()
        return safe_execute(lambda: process_result(result))
    except SpecificProjectError as e:
        logger.error(f"Known error pattern: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
```

### Data Processing Patterns

**1. Document Chunking** - Use project's chunking strategy:
```python
from src.processors.chunking_strategy import ChunkingStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter

strategy = ChunkingStrategy(
    chunk_size=settings.chunk_size,  # Default: 1000
    chunk_overlap=settings.chunk_overlap,  # Default: 200
    file_extension=file_ext
)
chunks = await strategy.process_document(content, metadata)
```

**2. Metadata Enrichment** - Always include context:
```python
metadata = {
    "source": file_path,
    "repository": repo_name,
    "file_type": file_extension,
    "chunk_index": chunk_idx,
    "timestamp": datetime.utcnow().isoformat(),
    "language": detect_language(file_extension),
    "symbols": extract_code_symbols(content)  # For code files
}
```

### Testing Patterns

**CRITICAL:** Always set PYTHONPATH when running tests:
```bash
PYTHONPATH=. pytest tests/unit/test_my_feature.py -v
```

**1. Async Test Pattern**:
```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_workflow_execution():
    # Mock external dependencies
    with patch('src.vectorstores.store_factory.VectorStoreFactory.create_store') as mock_store:
        mock_store.return_value = AsyncMock()
        
        workflow = MyWorkflow()
        result = await workflow.execute({"test": "data"})
        
        assert result["status"] == "completed"
        mock_store.assert_called_once()
```

**2. Integration Test Pattern**:
```python
@pytest.mark.integration
async def test_end_to_end_workflow():
    # These tests require Docker services running: make docker-up
    settings = get_settings()
    vector_store = VectorStoreFactory.create_store(settings.database_type)
    
    # Test with real vector store
    result = await index_document("test content", {"source": "test.py"})
    assert result is not None
```

### Performance & Scalability Patterns

**1. Batch Processing** - Use project's batch patterns:
```python
from src.loaders.enhanced_github_loader import EnhancedGitHubLoader

# Process in configurable batches
batch_size = settings.embedding_batch_size  # Default: 50
for batch in chunk_list(documents, batch_size):
    embeddings = await embedding_provider.embed_batch(batch)
    await vector_store.add_documents(batch, embeddings)
```

**2. Workflow State Management**:
```python
from src.workflows.state_manager import WorkflowStateManager

# Always persist state for long-running operations
state_manager = WorkflowStateManager(backend=settings.workflow_state_backend)
workflow_id = str(uuid.uuid4())

await state_manager.save_state(workflow_id, {
    "step": "processing",
    "progress": {"completed": 10, "total": 100},
    "last_updated": datetime.utcnow()
})
```

## Common Implementation Patterns

### Adding New API Endpoints
1. Define request/response models in `src/api/models.py`
2. Implement route in appropriate router file
3. Add workflow if complex processing needed
4. Include proper error handling and logging
5. Add tests with PYTHONPATH set

### Adding New Workflow Steps
1. Extend `BaseWorkflow` class
2. Implement required abstract methods
3. Add proper state transitions
4. Include retry logic with tenacity
5. Add comprehensive error handling

### Integrating New Vector Stores
1. Implement `BaseVectorStore` interface
2. Add to `VectorStoreFactory`
3. Update settings configuration
4. Add environment variables to `.env.example`
5. Update Docker compose if needed

## Critical Development Rules

1. **Always use async/await** - This is an async-first codebase
2. **Type hints everywhere** - Use Python 3.11+ type annotations
3. **Environment-driven config** - No hardcoded values
4. **Structured logging** - Use loguru with context
5. **Error resilience** - Include retry logic and proper error handling
6. **Test with PYTHONPATH** - Set `PYTHONPATH=.` for all pytest runs
7. **Docker for integration** - Use `make docker-up` for external services
8. **Format before commit** - Run `make format` before any commit

## Quick Reference

**Running tests:** `PYTHONPATH=. pytest tests/unit/ -v`
**Code formatting:** `make format && make lint`
**Start services:** `make docker-up`
**Environment setup:** `cp .env.example .env` then edit
**Workflow debugging:** Set `LOG_LEVEL=DEBUG` in `.env`

**Remember** to always clean up all debugging-related code or temporary files before completing the task.
