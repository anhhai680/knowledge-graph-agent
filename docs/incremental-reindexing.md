# Git-based Incremental Re-indexing

This document describes the git-based incremental re-indexing feature implemented for the Knowledge Graph Agent.

## Overview

The incremental re-indexing feature allows the system to efficiently update the vector database by processing only files that have changed since the last indexing operation, based on git commit history. This significantly reduces processing time and compute resources for large repositories.

## Key Features

- **Git-based Change Detection**: Uses git commit diffs to identify exactly which files have changed
- **Per-Repository Commit Tracking**: Tracks the last indexed commit for each repository/branch combination
- **Automatic Vector Cleanup**: Removes stale vectors for deleted, modified, and renamed files
- **Dry-run Analysis**: Preview changes without performing actual indexing
- **Seamless Fallback**: Automatically falls back to full indexing when needed
- **Thread-safe Persistence**: JSON-based commit tracking with thread safety

## Components

### 1. CommitTracker (`src/utils/commit_tracker.py`)

Manages persistent storage of the last indexed commit hash for each repository/branch combination.

**Key Methods:**
- `get_last_indexed_commit(repository, branch)` - Get last indexed commit
- `update_last_indexed_commit(repository, commit_hash, branch, metadata)` - Update tracking
- `has_been_indexed(repository, branch)` - Check if repository was previously indexed
- `remove_repository(repository, branch)` - Remove tracking (forces full re-index)

### 2. GitDiffService (`src/loaders/git_diff_service.py`)

Computes file changes between git commits and categorizes them for processing.

**Key Methods:**
- `get_changes_between_commits(repo_path, from_commit, to_commit)` - Compute diff
- `get_files_to_process(diff_result)` - Files that need indexing
- `get_files_to_remove(diff_result)` - Files to remove from vector store
- `validate_commits(repo_path, *commits)` - Validate commit hashes exist

### 3. Enhanced IndexingWorkflow

Extended with new workflow steps for incremental processing:

**New Steps:**
- `DETERMINE_CHANGED_FILES` - Analyze changes since last index
- `CLEANUP_STALE_VECTORS` - Remove outdated vectors before storing new ones

**New Parameters:**
- `incremental: bool` - Enable incremental mode
- `dry_run: bool` - Only analyze changes, don't perform indexing

## API Usage

### Basic Incremental Re-indexing

```bash
POST /api/v1/index/repository
{
    "repository_url": "https://github.com/owner/repo",
    "branch": "main",
    "incremental": true
}
```

### Dry-run Analysis

```bash
POST /api/v1/index/repository  
{
    "repository_url": "https://github.com/owner/repo",
    "incremental": true,
    "dry_run": true
}
```

### Management Endpoints

```bash
# View commit tracking status
GET /api/v1/commits/tracking

# Preview incremental changes
GET /api/v1/commits/diff/owner/repo?branch=main

# Remove commit tracking (force full re-index next time)
DELETE /api/v1/commits/tracking/owner/repo?branch=main
```

## Workflow Logic

### Decision Tree

1. **Check if incremental mode enabled**
   - If no: Perform full indexing
   - If yes: Continue to step 2

2. **Check for previous index**
   - If no previous index: Perform full indexing
   - If previous index exists: Continue to step 3

3. **Get current commit and compare**
   - If same as last indexed: Skip (no changes)
   - If different: Continue to step 4

4. **Validate commits and compute diff**
   - If commits invalid: Fall back to full indexing
   - If valid: Perform incremental processing

5. **Process changes**
   - Remove stale vectors for deleted/modified files
   - Load and process only changed files
   - Update commit tracking on success

### File Change Types

- **Added**: New files to be indexed
- **Modified**: Changed files (old vectors removed, new ones added)
- **Deleted**: Files to be removed from vector store
- **Renamed**: Old path removed, new path indexed
- **Copied**: New path indexed (original kept)

## Benefits

### Performance Improvements

- **Faster Processing**: Only changed files are processed
- **Reduced Compute**: Significant resource savings for large repositories
- **Efficient Storage**: Automatic cleanup prevents vector store bloat

### Reliability Features

- **Consistent State**: Vector store always reflects actual repository state
- **Safe Operations**: Dry-run capability for impact assessment
- **Robust Fallbacks**: Graceful handling of edge cases

### Operational Benefits

- **Detailed Reporting**: Change summaries show exactly what was processed
- **Flexible Control**: Support for both full and incremental modes
- **Easy Management**: Simple API endpoints for monitoring and control

## Configuration

The feature uses existing configuration settings:

```python
# File extensions to process (from settings)
settings.github.file_extensions

# GitHub token for repository access
settings.github.token
```

## Storage

Commit tracking data is stored in `indexed_commits.json` in the application root directory. The format is:

```json
{
  "owner/repo#main": {
    "repository": "owner/repo",
    "branch": "main", 
    "last_commit": "abc123def456",
    "last_indexed_at": "2025-01-15T10:30:00",
    "metadata": {
      "workflow_id": "workflow-123",
      "files_processed": 150,
      "processing_mode": "incremental"
    }
  }
}
```

## Error Handling

The implementation includes comprehensive error handling:

- **Invalid commits**: Falls back to full indexing
- **Git operations failures**: Graceful degradation
- **Repository access issues**: Clear error reporting
- **Corrupted tracking data**: Automatic recovery

## Testing

Comprehensive test coverage includes:

- **Unit tests** for CommitTracker and GitDiffService
- **Integration tests** for workflow steps
- **Error condition testing** for robustness
- **Thread safety verification** for concurrent operations

## Migration

The feature is backward compatible:

- **Existing workflows** continue to work unchanged
- **Default behavior** remains full indexing
- **Gradual adoption** possible by enabling incremental mode per repository
- **No breaking changes** to existing API endpoints

## Performance Characteristics

### Time Complexity
- **Change detection**: O(changed files) vs O(all files)
- **Vector operations**: O(changed vectors) vs O(all vectors)  
- **Storage access**: O(1) for commit lookups

### Space Complexity
- **Memory usage**: Proportional to changed files only
- **Storage overhead**: Minimal JSON tracking file
- **Network efficiency**: Only changed files downloaded

## Future Enhancements

Potential improvements for future versions:

- **Webhook integration** for real-time change detection
- **Parallel repository processing** for multiple repos
- **Advanced conflict resolution** for concurrent modifications
- **Metrics and monitoring** integration
- **Database-backed tracking** for enterprise deployments

## Troubleshooting

### Common Issues

1. **No changes detected**: Check if commits are different
2. **Full indexing triggered**: Verify commit hashes are valid
3. **Permission errors**: Ensure GitHub token has repository access
4. **Stale data**: Use DELETE endpoint to reset tracking

### Debug Information

Enable debug logging to see detailed workflow execution:

```python
import logging
logging.getLogger('src.utils.commit_tracker').setLevel(logging.DEBUG)
logging.getLogger('src.loaders.git_diff_service').setLevel(logging.DEBUG)
```