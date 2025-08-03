# Chroma Dimension Mismatch Fixes

This document explains the fixes implemented for Chroma vector store dimension mismatch issues.

## Problem Description

The system was experiencing two main issues with the Chroma vector store:

1. **`'NoneType' object has no attribute 'get'`** - This error occurred when trying to access metadata from a Chroma collection that had `None` metadata.

2. **`Collection expecting embedding with dimension of 1536, got 384`** - This error occurred when there was a mismatch between the embedding model's output dimension and what the Chroma collection expected.

## Root Cause Analysis

### Issue 1: None Metadata Error
- The `get_collection_stats()` method was trying to call `.get()` on `collection_info.metadata` without checking if it was `None`
- Chroma collections can have `None` metadata, especially when created without explicit metadata

### Issue 2: Dimension Mismatch
- The system is configured to use `text-embedding-ada-002` which produces 1536-dimensional embeddings
- However, the Chroma collection was created with a different embedding model that produces 384-dimensional embeddings
- This mismatch prevents proper vector operations

## Implemented Fixes

### 1. Safe Metadata Access
```python
# Before (problematic)
dimension = collection_info.metadata.get("dimension")

# After (safe)
dimension = None
if hasattr(collection_info, "metadata") and collection_info.metadata is not None:
    dimension = collection_info.metadata.get("dimension")
```

### 2. Dimension Compatibility Checking
Added `check_embedding_dimension_compatibility()` method that:
- Tests the current embedding model's output dimension
- Compares it with the collection's expected dimension
- Returns detailed compatibility information

### 3. Automatic Collection Recreation
Added `recreate_collection_with_correct_dimension()` method that:
- Deletes the existing collection
- Creates a new collection with the correct dimension metadata
- Reinitializes the LangChain Chroma instance

### 4. Enhanced Error Handling
- Updated `get_repository_metadata()` to check dimension compatibility before querying
- Updated `health_check()` to include dimension compatibility checks
- Added graceful degradation when dimension mismatches are detected

### 5. API Endpoints for Fixes
- Added `/fix/chroma-dimension` endpoint to automatically fix dimension mismatches
- Updated `/stats` endpoint to handle dimension mismatches gracefully
- Enhanced `/health` endpoint to detect dimension issues

## Usage

### Automatic Fix via API
```bash
curl -X POST http://localhost:8000/fix/chroma-dimension
```

### Manual Fix via Diagnostic Script
```bash
python debug/fix_chroma_dimension.py
```

### Programmatic Fix
```python
from src.vectorstores.chroma_store import ChromaStore

store = ChromaStore()

# Check for issues
is_compatible, msg = store.check_embedding_dimension_compatibility()
if not is_compatible:
    # Fix the issue
    success = store.recreate_collection_with_correct_dimension()
    if success:
        print("Collection recreated successfully")
```

## Configuration

### Embedding Model Configuration
The system is configured to use `text-embedding-ada-002` which produces 1536-dimensional embeddings:

```python
# In settings.py
class EmbeddingSettings(BaseModel):
    model: str = Field("text-embedding-ada-002", description="Embedding model")
```

### Chroma Collection Configuration
Collections are created with dimension metadata:

```python
# When creating a new collection
self.collection = self.client.create_collection(
    name=self.collection_name,
    metadata={"dimension": current_dimension}
)
```

## Testing

Run the tests to verify the fixes:

```bash
pytest tests/test_chroma_fixes.py -v
```

## Monitoring

### Health Check
The health check now includes dimension compatibility:

```python
is_healthy, msg = store.health_check()
# Returns False if dimension mismatch is detected
```

### Statistics Endpoint
The statistics endpoint gracefully handles dimension mismatches:

```python
# Returns degraded statistics when dimension mismatch is detected
stats = store.get_collection_stats()
```

## Troubleshooting

### Common Issues

1. **Collection still has wrong dimension after fix**
   - Ensure the application is restarted after the fix
   - Check that the embedding model configuration is correct

2. **Fix endpoint returns error**
   - Check Chroma server connectivity
   - Verify collection permissions

3. **Dimension mismatch persists**
   - Manually delete the Chroma collection
   - Restart the application to create a new collection

### Manual Resolution Steps

1. Stop the application
2. Delete the Chroma collection:
   - Local: Delete `chroma_db/` directory
   - Server: Use Chroma client to delete collection
3. Restart the application
4. Re-index repositories if needed

## Future Improvements

1. **Automatic Dimension Detection**: Automatically detect and use the correct embedding model based on collection metadata
2. **Migration Tools**: Tools to migrate existing collections to new embedding models
3. **Configuration Validation**: Validate embedding model configuration against collection requirements
4. **Monitoring Alerts**: Alert when dimension mismatches are detected

## Related Files

- `src/vectorstores/chroma_store.py` - Main fixes implementation
- `src/api/routes.py` - API endpoint updates
- `debug/fix_chroma_dimension.py` - Diagnostic script
- `tests/test_chroma_fixes.py` - Test coverage
- `src/config/settings.py` - Configuration settings 