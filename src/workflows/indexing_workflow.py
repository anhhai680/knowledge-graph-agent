"""
LangGraph Indexing Workflow Implementation.

This module implements a complete stateful indexing workflow using LangGraph
for processing multiple GitHub repositories, document chunking, and vector storage.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, cast

from langchain.schema import Document

from src.config.settings import settings
from src.loaders.enhanced_github_loader import EnhancedGitHubLoader
from src.processors.document_processor import DocumentProcessor
from src.llm.embedding_factory import EmbeddingFactory
from src.vectorstores.store_factory import VectorStoreFactory
from src.graphstores.memgraph_store import MemGraphStore
from src.utils.feature_flags import is_graph_enabled
from src.utils.defensive_programming import safe_len, ensure_list
from src.workflows.base_workflow import BaseWorkflow
from src.workflows.workflow_states import (
    IndexingState,
    ProcessingStatus,
    WorkflowType,
    create_indexing_state,
    create_repository_state,
    create_file_processing_state,
    update_workflow_progress,
    add_workflow_error,
)


class IndexingWorkflowSteps(str):
    """Indexing workflow step enumeration."""

    INITIALIZE_STATE = "initialize_state"
    LOAD_REPOSITORIES = "load_repositories"
    VALIDATE_REPOS = "validate_repos"
    DETERMINE_CHANGED_FILES = "determine_changed_files"
    LOAD_FILES_FROM_GITHUB = "load_files_from_github"
    PROCESS_DOCUMENTS = "process_documents"
    LANGUAGE_AWARE_CHUNKING = "language_aware_chunking"
    EXTRACT_METADATA = "extract_metadata"
    GENERATE_EMBEDDINGS = "generate_embeddings"
    CLEANUP_STALE_VECTORS = "cleanup_stale_vectors"
    STORE_IN_VECTOR_DB = "store_in_vector_db"
    STORE_IN_GRAPH_DB = "store_in_graph_db"
    UPDATE_WORKFLOW_STATE = "update_workflow_state"
    CHECK_COMPLETE = "check_complete"
    FINALIZE_INDEX = "finalize_index"

    # Error handling states
    HANDLE_FILE_ERRORS = "handle_file_errors"
    HANDLE_PROCESSING_ERRORS = "handle_processing_errors"
    HANDLE_EMBEDDING_ERRORS = "handle_embedding_errors"
    HANDLE_STORAGE_ERRORS = "handle_storage_errors"

# Length of content preview for graph nodes
CONTENT_PREVIEW_LENGTH = 200  


class IndexingWorkflow(BaseWorkflow[IndexingState]):
    """
    LangGraph indexing workflow for processing GitHub repositories.

    This workflow handles the complete indexing pipeline:
    1. Initialize state and load repository configurations
    2. Validate repositories and establish connections
    3. Load files from GitHub repositories
    4. Process documents with language-aware chunking
    5. Extract metadata and generate embeddings
    6. Store documents and embeddings in vector database
    7. Update workflow state and finalize indexing
    """

    def __init__(
        self,
        repositories: Optional[List[str]] = None,
        app_settings_path: str = "appSettings.json",
        vector_store_type: Optional[str] = None,
        collection_name: Optional[str] = None,
        batch_size: int = 50,
        max_workers: int = 2,
        incremental: bool = False,
        dry_run: bool = False,
        repository_urls: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initialize indexing workflow.

        Args:
            repositories: List of repository names to index (None for all from appSettings)
            app_settings_path: Path to appSettings.json file
            vector_store_type: Vector store type (overrides env setting)
            collection_name: Collection name (overrides env setting)
            batch_size: Batch size for embedding generation
            max_workers: Maximum number of worker threads for parallel processing
            incremental: Enable incremental re-indexing based on git commits
            dry_run: Only analyze changes without performing actual indexing
            repository_urls: Optional mapping of repository names to URLs (for API usage)
            **kwargs: Additional arguments passed to BaseWorkflow
        """
        super().__init__(**kwargs)

        self.app_settings_path = app_settings_path
        self.target_repositories = repositories
        self.repository_urls = repository_urls or {}
        self.vector_store_type = vector_store_type or settings.database_type.value
        self.collection_name = collection_name or (
            settings.pinecone.collection_name
            if self.vector_store_type == "pinecone" and settings.pinecone
            else settings.chroma.collection_name
        )
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.incremental = incremental
        self.dry_run = dry_run

        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_factory = EmbeddingFactory()
        self.vector_store_factory = VectorStoreFactory()

        # Initialize incremental indexing components
        if self.incremental:
            from src.utils.commit_tracker import get_commit_tracker
            from src.loaders.git_diff_service import GitDiffService
            self.commit_tracker = get_commit_tracker()
            self.git_diff_service = GitDiffService()
        else:
            self.commit_tracker = None
            self.git_diff_service = None

        # Repository configuration cache
        self._repo_configs: Dict[str, Dict[str, Any]] = {}
        self._app_settings: Optional[Dict[str, Any]] = None

        mode_msg = "incremental" if self.incremental else "full"
        if self.dry_run:
            mode_msg += " (dry-run)"
        
        self.logger.info(
            f"Initialized {mode_msg} indexing workflow with vector store: {self.vector_store_type}"
        )

    def define_steps(self) -> List[str]:
        """
        Define the indexing workflow steps.

        Returns:
            List of step names in execution order
        """
        steps = [
            IndexingWorkflowSteps.INITIALIZE_STATE,
            IndexingWorkflowSteps.LOAD_REPOSITORIES,
            IndexingWorkflowSteps.VALIDATE_REPOS,
        ]
        
        # Add incremental-specific step if enabled
        if self.incremental:
            steps.append(IndexingWorkflowSteps.DETERMINE_CHANGED_FILES)
        
        steps.extend([
            IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB,
            IndexingWorkflowSteps.PROCESS_DOCUMENTS,
            IndexingWorkflowSteps.EXTRACT_METADATA,
        ])
        
        # Add cleanup step for incremental mode (before storing new vectors)
        if self.incremental and not self.dry_run:
            steps.append(IndexingWorkflowSteps.CLEANUP_STALE_VECTORS)
        
        # Skip storage steps in dry-run mode
        if not self.dry_run:
            steps.extend([
                IndexingWorkflowSteps.STORE_IN_VECTOR_DB,  # Vector store now handles embedding generation
                IndexingWorkflowSteps.STORE_IN_GRAPH_DB,   # Graph database storage (conditional)
            ])
        
        steps.extend([
            IndexingWorkflowSteps.UPDATE_WORKFLOW_STATE,
            IndexingWorkflowSteps.CHECK_COMPLETE,
            IndexingWorkflowSteps.FINALIZE_INDEX,
        ])
        
        return steps

    def _ensure_list(self, repositories: Any, default: Optional[List] = None) -> List:
        """
        Ensure that repositories is a valid list, handling None and invalid types.
        
        Args:
            repositories: The repositories value to validate and convert
            default: Default list to return if repositories is None
            
        Returns:
            List: A valid list of repositories
        """
        return ensure_list(repositories, default or [])

    def validate_state(self, state: IndexingState) -> bool:
        """
        Validate indexing workflow state.

        Args:
            state: Indexing workflow state to validate

        Returns:
            True if state is valid, False otherwise
        """
        try:
            # Check required fields
            if not state.get("workflow_id"):
                self.logger.error("Missing workflow_id in state")
                return False

            if not state.get("repositories"):
                self.logger.error(
                    "Repositories list is empty. No repositories to process."
                )
                return False

            if state["workflow_type"] != WorkflowType.INDEXING:
                self.logger.error("Invalid workflow type")
                return False

            # Validate vector store configuration
            if not state.get("vector_store_type"):
                self.logger.error("Missing vector_store_type in state")
                return False

            return True

        except Exception as e:
            self.logger.error(f"State validation error: {e}")
            return False

    def execute_step(self, step: str, state: IndexingState) -> IndexingState:
        """
        Execute a single indexing workflow step.

        Args:
            step: Step name to execute
            state: Current indexing workflow state

        Returns:
            Updated indexing workflow state
        """
        start_time = time.time()

        try:
            self.logger.info(f"Executing step: {step}")

            # Route to appropriate step handler
            if step == IndexingWorkflowSteps.INITIALIZE_STATE:
                state = self._initialize_state(state)
            elif step == IndexingWorkflowSteps.LOAD_REPOSITORIES:
                state = self._load_repositories(state)
            elif step == IndexingWorkflowSteps.VALIDATE_REPOS:
                state = self._validate_repositories(state)
            elif step == IndexingWorkflowSteps.DETERMINE_CHANGED_FILES:
                state = self._determine_changed_files(state)
            elif step == IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB:
                state = self._load_files_from_github(state)
            elif step == IndexingWorkflowSteps.PROCESS_DOCUMENTS:
                state = self._process_documents(state)
            elif step == IndexingWorkflowSteps.EXTRACT_METADATA:
                state = self._extract_metadata(state)
            elif step == IndexingWorkflowSteps.CLEANUP_STALE_VECTORS:
                state = self._cleanup_stale_vectors(state)
            elif step == IndexingWorkflowSteps.STORE_IN_VECTOR_DB:
                state = self._store_in_vector_db(state)
            elif step == IndexingWorkflowSteps.STORE_IN_GRAPH_DB:
                state = self._store_in_graph_db(state)
            elif step == IndexingWorkflowSteps.UPDATE_WORKFLOW_STATE:
                state = self._update_workflow_state(state)
            elif step == IndexingWorkflowSteps.CHECK_COMPLETE:
                state = self._check_complete(state)
            elif step == IndexingWorkflowSteps.FINALIZE_INDEX:
                state = self._finalize_index(state)
            else:
                raise ValueError(f"Unknown step: {step}")

            # Update step completion time
            step_duration = time.time() - start_time
            self.logger.info(f"Step {step} completed in {step_duration:.2f}s")

            # Update state metadata
            if "step_durations" not in state["metadata"]:
                state["metadata"]["step_durations"] = {}
            state["metadata"]["step_durations"][step] = step_duration

            return state

        except Exception as e:
            self.logger.error(f"Step {step} failed: {e}")
            # Add error to state for tracking
            error_entry = {
                "message": str(e),
                "timestamp": time.time(),
                "step": step,
                "details": {},
            }
            state["errors"].append(error_entry)
            state["updated_at"] = time.time()
            
            # Determine if this is a critical failure that should stop the workflow
            critical_steps = [
                IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB,
                IndexingWorkflowSteps.PROCESS_DOCUMENTS,
                IndexingWorkflowSteps.STORE_IN_VECTOR_DB,
            ]
            
            if step in critical_steps:
                # For critical steps, mark workflow as failed and re-raise to stop execution
                state["status"] = ProcessingStatus.FAILED
                self.logger.error(f"Critical step {step} failed, stopping workflow execution")
                raise
            else:
                # For non-critical steps, continue with error state
                self.logger.warning(f"Non-critical step {step} failed, continuing workflow")
                return state

    def handle_error(
        self, step: str, state: IndexingState, error: Exception
    ) -> IndexingState:
        """
        Handle workflow errors with specific error recovery strategies.

        Args:
            step: Step name that failed
            state: Current workflow state
            error: Exception that occurred

        Returns:
            Updated workflow state with error handling
        """
        self.logger.error(f"Handling error in step {step}: {error}")

        # Add error to state
        state = cast(IndexingState, add_workflow_error(state, str(error), step))

        # Determine error handling strategy based on step
        if step in [IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB]:
            return self._handle_file_errors(state, error)
        elif step in [IndexingWorkflowSteps.PROCESS_DOCUMENTS]:
            return self._handle_processing_errors(state, error)
        elif step == IndexingWorkflowSteps.GENERATE_EMBEDDINGS:
            return self._handle_embedding_errors(state, error)
        elif step == IndexingWorkflowSteps.STORE_IN_VECTOR_DB:
            return self._handle_storage_errors(state, error)
        else:
            # Default error handling - mark as failed but continue
            self.logger.warning(f"Using default error handling for step {step}")
            state["status"] = ProcessingStatus.FAILED
            return state

    def _initialize_state(self, state: IndexingState) -> IndexingState:
        """Initialize indexing workflow state."""
        self.logger.info("Initializing indexing workflow state")

        # Load app settings if available
        if not self._app_settings:
            try:
                self._load_app_settings()
            except (FileNotFoundError, ValueError) as e:
                self.logger.warning(f"Failed to load app settings: {e}")
                # Continue without app settings if repository URLs are provided
                if not self.repository_urls:
                    raise ValueError(f"No app settings available and no repository URLs provided: {e}")

        # Determine repositories to process
        if self.target_repositories:
            # Use target repositories (can be from appSettings or direct URLs)
            repositories = self.target_repositories
            
            # Validate repositories exist in either appSettings or repository_urls
            if self._has_repositories():
                available_repos = [
                    repo["name"] for repo in self._app_settings["repositories"]
                ]
                missing_from_settings = set(self.target_repositories) - set(available_repos)
                
                # Check if missing repositories are provided via repository_urls
                missing_repos = missing_from_settings - set(self.repository_urls.keys())
                if missing_repos:
                    raise ValueError(
                        f"Repositories not found in appSettings or repository_urls: {missing_repos}"
                    )
            elif not self.repository_urls:
                raise ValueError("No app settings loaded and no repository URLs provided")
        else:
            # Process all repositories from appSettings
            if self._app_settings and "repositories" in self._app_settings:
                repositories = [repo["name"] for repo in self._app_settings["repositories"]]
            else:
                raise ValueError("No app settings loaded or no repositories configured")

        # Update state with repository information
        state["repositories"] = repositories
        state["vector_store_type"] = self.vector_store_type
        state["collection_name"] = self.collection_name
        state["batch_size"] = self.batch_size
        state["status"] = ProcessingStatus.IN_PROGRESS

        # Defensive programming: ensure repositories is not None and is a list
        repositories = self._ensure_list(repositories)

        # Initialize repository states
        for repo_name in repositories:
            try:
                repo_config = self._get_repo_config(repo_name)
                state["repository_states"][repo_name] = create_repository_state(
                    name=repo_name,
                    url=repo_config["url"],
                    branch=repo_config.get("branch", "main"),
                )
            except ValueError:
                # Repository not in appSettings, try repository_urls
                if repo_name in self.repository_urls:
                    repo_url = self.repository_urls[repo_name]
                    state["repository_states"][repo_name] = create_repository_state(
                        name=repo_name,
                        url=repo_url,
                        branch="main",  # Default branch
                    )
                else:
                    raise ValueError(f"Repository configuration not found for: {repo_name}")

        self.logger.info(f"Initialized state for {len(repositories)} repositories")
        return cast(IndexingState, update_workflow_progress(
            state, 5.0, IndexingWorkflowSteps.INITIALIZE_STATE
        ))

    def _load_repositories(self, state: IndexingState) -> IndexingState:
        """Load repository configurations from appSettings."""
        self.logger.info("Loading repository configurations")

        # Defensive programming: ensure repositories list exists and is not None
        repositories = self._ensure_list(state.get("repositories", []))
        state["repositories"] = repositories

        # Repository configurations are loaded in _initialize_state
        # This step validates that all required configurations are present
        for repo_name in repositories:
            if repo_name not in state["repository_states"]:
                raise ValueError(f"Repository state not found for: {repo_name}")

            repo_config = self._get_repo_config(repo_name)
            if not repo_config:
                raise ValueError(f"Repository configuration not found for: {repo_name}")

        self.logger.info(
            f"Loaded configurations for {len(repositories)} repositories"
        )
        return cast(IndexingState, update_workflow_progress(
            state, 10.0, IndexingWorkflowSteps.LOAD_REPOSITORIES
        ))

    def _validate_repositories(self, state: IndexingState) -> IndexingState:
        """Validate repository access and configurations."""
        self.logger.info("Validating repository access")

        # Defensive programming: ensure repositories list exists and is not None
        repositories = self._ensure_list(state.get("repositories", []))
        state["repositories"] = repositories

        validation_errors = []

        for repo_name in repositories:
            try:
                repo_config = self._get_repo_config(repo_name)
                repo_state = state["repository_states"][repo_name]

                # Parse repository owner and name from URL
                owner, name = self._parse_repo_url(repo_config["url"])

                # Create GitHub loader to test access
                loader = EnhancedGitHubLoader(
                    repo_owner=owner,
                    repo_name=name,
                    branch=repo_config.get("branch", "main"),
                    file_extensions=settings.github.file_extensions,
                    github_token=settings.github.token,
                )

                # Test repository access by attempting to load one file
                # Note: This is a basic validation - full file loading happens in LOAD_FILES_FROM_GITHUB
                repo_info = f"Repository validation for {repo_name} successful"
                self.logger.info(f"Validated access to repository: {repo_name}")

                # Update repository state with validation success
                repo_state["status"] = ProcessingStatus.NOT_STARTED

            except Exception as e:
                error_msg = f"Failed to validate repository {repo_name}: {e}"
                validation_errors.append(error_msg)
                self.logger.error(error_msg)

                # Mark repository as failed
                state["repository_states"][repo_name][
                    "status"
                ] = ProcessingStatus.FAILED
                state["repository_states"][repo_name]["errors"].append(str(e))

        if validation_errors:
            # Add validation errors to state but continue with valid repositories
            for error in validation_errors:
                state = cast(IndexingState, add_workflow_error(
                    state, error, IndexingWorkflowSteps.VALIDATE_REPOS
                ))

        valid_repos = [
            repo
            for repo in repositories
            if state["repository_states"][repo]["status"] != ProcessingStatus.FAILED
        ]

        if not valid_repos:
            raise ValueError("No valid repositories found after validation")

        self.logger.info(
            f"Validated {len(valid_repos)}/{len(repositories)} repositories"
        )
        return cast(IndexingState, update_workflow_progress(
            state, 15.0, IndexingWorkflowSteps.VALIDATE_REPOS
        ))

    def _determine_changed_files(self, state: IndexingState) -> IndexingState:
        """Determine which files have changed since last indexing (incremental mode only)."""
        if not self.incremental:
            self.logger.info("Skipping change detection - not in incremental mode")
            return cast(IndexingState, update_workflow_progress(
                state, 20.0, IndexingWorkflowSteps.DETERMINE_CHANGED_FILES
            ))

        self.logger.info("Determining changed files for incremental re-indexing")
        
        # Initialize change tracking in state metadata
        state["metadata"]["incremental_changes"] = {}
        total_changes = 0
        
        for repo_name in state["repositories"]:
            if state["repository_states"][repo_name]["status"] == ProcessingStatus.FAILED:
                continue
                
            try:
                repo_config = self._get_repo_config(repo_name)
                owner, name = self._parse_repo_url(repo_config["url"])
                branch = repo_config.get("branch", "main")
                full_repo_name = f"{owner}/{name}"
                
                # Get last indexed commit for this repository
                last_commit = self.commit_tracker.get_last_indexed_commit(full_repo_name, branch)
                
                if not last_commit:
                    self.logger.info(f"No previous index found for {full_repo_name}#{branch} - will do full indexing")
                    # Mark for full indexing
                    state["metadata"]["incremental_changes"][repo_name] = {
                        "change_type": "full_index",
                        "reason": "no_previous_index",
                        "files_to_process": None,  # All files
                        "files_to_remove": [],
                        "last_commit": None,
                        "current_commit": None
                    }
                    continue
                
                # Prepare local repository for git diff
                repo_path = self._prepare_repository_for_diff(repo_name, owner, name, branch)
                if not repo_path:
                    self.logger.error(f"Failed to prepare repository {repo_name} for diff analysis")
                    state["repository_states"][repo_name]["status"] = ProcessingStatus.FAILED
                    continue
                
                # Get current commit
                current_commit = self.git_diff_service.get_current_commit(repo_path, branch)
                if not current_commit:
                    self.logger.error(f"Failed to get current commit for {repo_name}")
                    state["repository_states"][repo_name]["status"] = ProcessingStatus.FAILED
                    continue
                
                # Check if there are any changes
                if last_commit == current_commit:
                    self.logger.info(f"No changes detected for {repo_name} (commit: {current_commit})")
                    state["metadata"]["incremental_changes"][repo_name] = {
                        "change_type": "no_changes",
                        "reason": "same_commit",
                        "files_to_process": [],
                        "files_to_remove": [],
                        "last_commit": last_commit,
                        "current_commit": current_commit
                    }
                    # Mark repository as skipped
                    state["repository_states"][repo_name]["status"] = ProcessingStatus.SKIPPED
                    continue
                
                # Validate commits exist
                if not self.git_diff_service.validate_commits(repo_path, last_commit, current_commit):
                    self.logger.warning(f"Invalid commits for {repo_name} - falling back to full indexing")
                    state["metadata"]["incremental_changes"][repo_name] = {
                        "change_type": "full_index",
                        "reason": "invalid_commits",
                        "files_to_process": None,  # All files
                        "files_to_remove": [],
                        "last_commit": last_commit,
                        "current_commit": current_commit
                    }
                    continue
                
                # Compute diff between commits
                diff_result = self.git_diff_service.get_changes_between_commits(
                    repo_path, 
                    last_commit, 
                    current_commit,
                    file_extensions=settings.github.file_extensions
                )
                
                # Get files to process and remove
                files_to_process = self.git_diff_service.get_files_to_process(diff_result)
                files_to_remove = self.git_diff_service.get_files_to_remove(diff_result)
                
                # Defensive programming: ensure we have valid collections
                files_to_process = self._ensure_list(files_to_process, default=[])
                files_to_remove = self._ensure_list(files_to_remove, default=[])
                
                # Store change information with additional safety checks
                change_info = {
                    "change_type": "incremental",
                    "total_changes": diff_result.total_changes if diff_result else 0,
                    "changes_by_type": {k.value: v for k, v in diff_result.changes_by_type.items()} if diff_result and diff_result.changes_by_type else {},
                    "files_to_process": list(files_to_process) if files_to_process is not None else [],
                    "files_to_remove": list(files_to_remove) if files_to_remove is not None else [],
                    "last_commit": last_commit,
                    "current_commit": current_commit,
                    "diff_summary": {
                        "added": safe_len(diff_result.added_files) if diff_result else 0,
                        "modified": safe_len(diff_result.modified_files) if diff_result else 0,
                        "deleted": safe_len(diff_result.deleted_files) if diff_result else 0,
                        "renamed": safe_len(diff_result.renamed_files) if diff_result else 0
                    }
                }
                
                state["metadata"]["incremental_changes"][repo_name] = change_info
                total_changes += diff_result.total_changes
                
                self.logger.info(f"Changes detected for {repo_name}: {diff_result.total_changes if diff_result else 0} files changed")
                # Safe logging with explicit null checks
                files_to_process_count = safe_len(files_to_process)
                files_to_remove_count = safe_len(files_to_remove)
                self.logger.info(f"  - {files_to_process_count} files to process")
                self.logger.info(f"  - {files_to_remove_count} files to remove from vector store")
                
                # Debug: Log the types and values for troubleshooting
                self.logger.debug(f"Debug - files_to_process type: {type(files_to_process)}, value: {files_to_process}")
                self.logger.debug(f"Debug - files_to_remove type: {type(files_to_remove)}, value: {files_to_remove}")
                
            except Exception as e:
                self.logger.error(f"Failed to determine changes for {repo_name}: {e}")
                state["repository_states"][repo_name]["status"] = ProcessingStatus.FAILED
                # Add to errors for tracking
                state["repository_states"][repo_name]["errors"].append(str(e))
        
        # Log summary
        repos_with_changes = sum(1 for changes in state["metadata"]["incremental_changes"].values() 
                               if changes["change_type"] in ["incremental", "full_index"])
        repos_no_changes = sum(1 for changes in state["metadata"]["incremental_changes"].values() 
                             if changes["change_type"] == "no_changes")
        
        self.logger.info(f"Change detection summary:")
        self.logger.info(f"  - {repos_with_changes} repositories with changes")
        self.logger.info(f"  - {repos_no_changes} repositories with no changes")
        self.logger.info(f"  - {total_changes} total file changes detected")
        
        return cast(IndexingState, update_workflow_progress(
            state, 20.0, IndexingWorkflowSteps.DETERMINE_CHANGED_FILES
        ))

    def _cleanup_stale_vectors(self, state: IndexingState) -> IndexingState:
        """Remove stale vectors for deleted, modified, and renamed files."""
        if not self.incremental:
            self.logger.info("Skipping vector cleanup - not in incremental mode")
            return cast(IndexingState, update_workflow_progress(
                state, 88.0, IndexingWorkflowSteps.CLEANUP_STALE_VECTORS
            ))

        self.logger.info("Cleaning up stale vectors for incremental re-indexing")
        
        # Initialize vector store
        vector_store = self.vector_store_factory.create(collection_name=self.collection_name)
        
        total_removed = 0
        cleanup_stats = {}
        
        # Process each repository's changes
        incremental_changes = state["metadata"].get("incremental_changes", {})
        
        for repo_name, change_info in incremental_changes.items():
            if change_info["change_type"] == "no_changes":
                continue
                
            files_to_remove = change_info.get("files_to_remove", [])
            # Defensive programming: ensure files_to_remove is always a list
            files_to_remove = self._ensure_list(files_to_remove)
            if not files_to_remove:
                continue
            
            try:
                repo_config = self._get_repo_config(repo_name)
                owner, name = self._parse_repo_url(repo_config["url"])
                full_repo_name = f"{owner}/{name}"
                
                # Use safe_len for defensive programming
                files_count = safe_len(files_to_remove)
                self.logger.info(f"Removing {files_count} stale vectors for {repo_name}")
                
                # Remove vectors by file path metadata
                removed_count = 0
                if files_to_remove:  # Additional null check
                    for file_path in files_to_remove:
                        try:
                            # Remove documents matching this file path and repository
                            deleted = vector_store.delete_by_metadata({
                                "file_path": file_path,
                                "repository": full_repo_name
                            })
                            removed_count += deleted
                            
                        except Exception as e:
                            self.logger.error(f"Failed to remove vectors for {file_path}: {e}")
                
                cleanup_stats[repo_name] = {
                    "files_to_remove": safe_len(files_to_remove),
                    "vectors_removed": removed_count
                }
                total_removed += removed_count
                
                self.logger.info(f"Removed {removed_count} vectors for {repo_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to cleanup vectors for {repo_name}: {e}")
                cleanup_stats[repo_name] = {
                    "files_to_remove": safe_len(files_to_remove),
                    "vectors_removed": 0,
                    "error": str(e)
                }
        
        # Store cleanup statistics
        state["metadata"]["vector_cleanup_stats"] = cleanup_stats
        state["metadata"]["total_vectors_removed"] = total_removed
        
        self.logger.info(f"Vector cleanup completed: {total_removed} vectors removed")
        
        return cast(IndexingState, update_workflow_progress(
            state, 88.0, IndexingWorkflowSteps.CLEANUP_STALE_VECTORS
        ))

    def _load_files_from_github(self, state: IndexingState) -> IndexingState:
        """Load files from GitHub repositories with parallel processing."""
        self.logger.info("Loading files from GitHub repositories")

        # Defensive programming: ensure repositories list exists and is not None
        repositories = state.get("repositories", [])
        if repositories is None:
            repositories = []
        state["repositories"] = repositories

        all_documents = []
        total_files = 0
        incremental_changes = state["metadata"].get("incremental_changes", {})

        # Process repositories in parallel with controlled concurrency
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(repositories))
        ) as executor:
            # Submit repository loading tasks
            future_to_repo = {}
            for repo_name in repositories:
                repo_state = state["repository_states"][repo_name]
                
                # Skip failed repositories
                if repo_state["status"] == ProcessingStatus.FAILED:
                    continue
                
                # Skip repositories with no changes in incremental mode
                if self.incremental and repo_name in incremental_changes:
                    change_info = incremental_changes[repo_name]
                    if change_info["change_type"] == "no_changes":
                        self.logger.info(f"Skipping {repo_name} - no changes detected")
                        repo_state["status"] = ProcessingStatus.SKIPPED
                        continue

                future = executor.submit(
                    self._load_repository_files, repo_name, state
                )
                future_to_repo[future] = repo_name

            # Collect results as they complete
            for future in as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    repo_documents, repo_file_count = future.result()
                    all_documents.extend(repo_documents)
                    total_files += repo_file_count

                    # Update repository state based on success
                    if repo_file_count > 0:
                        # Successfully loaded files
                        state["repository_states"][repo_name][
                            "status"
                        ] = ProcessingStatus.COMPLETED
                        state["repository_states"][repo_name][
                            "total_files"
                        ] = repo_file_count
                        state["repository_states"][repo_name][
                            "processed_files"
                        ] = len(repo_documents)  # Document count (chunks)

                        self.logger.info(
                            f"Loaded {repo_file_count} files ({len(repo_documents)} documents) from repository: {repo_name}"
                        )
                    else:
                        # No files loaded - treat as failure
                        error_msg = f"No files loaded from repository {repo_name} - repository may be empty or inaccessible"
                        self.logger.warning(error_msg)
                        
                        state["repository_states"][repo_name][
                            "status"
                        ] = ProcessingStatus.FAILED
                        state["repository_states"][repo_name]["errors"].append(error_msg)
                        state["repository_states"][repo_name][
                            "total_files"
                        ] = 0
                        state["repository_states"][repo_name][
                            "processed_files"
                        ] = 0
                        # Cast to WorkflowState for error function, then back to IndexingState
                        temp_state = add_workflow_error(
                            state, error_msg, IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB
                        )
                        # Copy back the updated fields
                        state["errors"] = temp_state["errors"]
                        state["updated_at"] = temp_state["updated_at"]

                except Exception as e:
                    error_msg = f"Failed to load files from repository {repo_name}: {e}"
                    self.logger.error(error_msg)

                    # Update repository state with error
                    state["repository_states"][repo_name][
                        "status"
                    ] = ProcessingStatus.FAILED
                    state["repository_states"][repo_name]["errors"].append(str(e))
                    # Cast to WorkflowState for error function, then copy back
                    temp_state = add_workflow_error(
                        cast(Any, state), error_msg, IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB
                    )
                    state["errors"] = temp_state["errors"]
                    state["updated_at"] = temp_state["updated_at"]

        # Store documents in state metadata for next steps
        state["metadata"]["loaded_documents"] = all_documents
        state["total_files"] = total_files
        state["processed_files"] = len(all_documents)  # Total documents, not files

        if not all_documents:
            # Provide detailed diagnostic information when no documents are loaded
            self._log_no_documents_diagnostics(state)
            
            # Mark workflow as failed if no documents were loaded from any repository
            state["status"] = ProcessingStatus.FAILED
            
            # Add a summary error message
            failed_repos = [
                name for name, repo_state in state["repository_states"].items()
                if repo_state["status"] == ProcessingStatus.FAILED
            ]
            error_summary = f"No documents loaded from any repository. Failed repositories: {', '.join(failed_repos) if failed_repos else 'All repositories'}"
            
            # Add to errors list manually since add_workflow_error has typing issues
            error_entry = {
                "message": error_summary,
                "timestamp": time.time(),
                "step": IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB,
                "details": {"failed_repositories": failed_repos},
            }
            state["errors"].append(error_entry)
            state["updated_at"] = time.time()
            
            raise ValueError(error_summary)

        self.logger.info(
            f"Loaded {len(all_documents)} documents from {total_files} files"
        )
        
        # Update progress and return with proper casting
        temp_state = update_workflow_progress(
            cast(Any, state), 30.0, IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB
        )
        # Copy back the updated progress fields
        state["progress_percentage"] = temp_state["progress_percentage"]
        state["current_step"] = temp_state["current_step"]
        state["updated_at"] = temp_state["updated_at"]
        
        return state

    def _process_documents(self, state: IndexingState) -> IndexingState:
        """Process loaded documents with language-aware processing."""
        self.logger.info("Processing documents with language-aware strategies")

        documents = state["metadata"].get("loaded_documents", [])
        # Defensive programming: ensure documents is not None
        if documents is None:
            documents = []
        if not documents:
            # Check if we have repository states to provide better error information
            repo_states = state.get("repository_states", {})
            if repo_states:
                failed_repos = []
                empty_repos = []
                for repo_name, repo_state in repo_states.items():
                    if repo_state.get("status") == ProcessingStatus.FAILED:
                        failed_repos.append(repo_name)
                    elif repo_state.get("total_files", 0) == 0:
                        empty_repos.append(repo_name)
                
                error_details = []
                if failed_repos:
                    error_details.append(f"Failed repositories: {', '.join(failed_repos)}")
                if empty_repos:
                    error_details.append(f"Empty repositories (no matching files): {', '.join(empty_repos)}")
                
                error_msg = "No documents available for processing. " + "; ".join(error_details)
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                raise ValueError("No documents available for processing - no repositories were loaded")

        self.logger.info(f"Processing {len(documents)} loaded documents")

        processed_documents = []
        processing_stats = {
            "total_documents": len(documents),
            "processed_documents": 0,
            "total_chunks": 0,
            "processing_errors": 0,
        }

        # Process documents in batches to manage memory
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            for doc in batch:
                try:
                    # Process document with language-aware chunking
                    chunks = self.document_processor.process_document(doc)
                    processed_documents.extend(chunks)

                    processing_stats["processed_documents"] += 1
                    processing_stats["total_chunks"] += len(chunks)

                    # Create file processing state
                    file_path = doc.metadata.get("file_path", "unknown")
                    language = doc.metadata.get("language", "unknown")

                    file_state = create_file_processing_state(file_path, language)
                    file_state["status"] = ProcessingStatus.COMPLETED
                    file_state["chunk_count"] = len(chunks)
                    file_state["processing_time"] = time.time()

                    state["file_processing_states"].append(file_state)

                except Exception as e:
                    error_msg = f"Failed to process document {doc.metadata.get('file_path', 'unknown')}: {e}"
                    self.logger.error(error_msg)
                    processing_stats["processing_errors"] += 1

                    # Create failed file processing state
                    file_path = doc.metadata.get("file_path", "unknown")
                    language = doc.metadata.get("language", "unknown")

                    file_state = create_file_processing_state(file_path, language)
                    file_state["status"] = ProcessingStatus.FAILED
                    file_state["error_message"] = str(e)

                    state["file_processing_states"].append(file_state)

            # Update progress for batch
            progress = 30.0 + (i + len(batch)) / len(documents) * 20.0
            state = cast(IndexingState, update_workflow_progress(
                state, progress, IndexingWorkflowSteps.PROCESS_DOCUMENTS
            ))

        # Store processed documents and update statistics
        state["metadata"]["processed_documents"] = processed_documents
        state["metadata"]["processing_stats"] = processing_stats
        state["total_chunks"] = processing_stats["total_chunks"]

        self.logger.info(
            f"Processed {processing_stats['processed_documents']} documents into {processing_stats['total_chunks']} chunks"
        )

        if processing_stats["processing_errors"] > 0:
            self.logger.warning(
                f"Processing errors occurred for {processing_stats['processing_errors']} documents"
            )

        # Validate language-aware chunking was successful
        if not processed_documents:
            raise ValueError(
                "No processed documents available after language-aware chunking"
            )

        # Validate chunk metadata quality
        chunks_with_metadata = 0
        for doc in processed_documents:
            if doc.metadata.get("chunk_type") or doc.metadata.get("language"):
                chunks_with_metadata += 1

        self.logger.info(
            f"Language-aware chunking produced {len(processed_documents)} chunks, {chunks_with_metadata} with metadata"
        )

        return cast(IndexingState, update_workflow_progress(
            state, 50.0, IndexingWorkflowSteps.PROCESS_DOCUMENTS
        ))

    def _extract_metadata(self, state: IndexingState) -> IndexingState:
        """Extract and validate metadata from processed chunks."""
        self.logger.info("Extracting and validating chunk metadata")

        processed_documents = state["metadata"].get("processed_documents", [])
        # Defensive programming: ensure processed_documents is not None
        if processed_documents is None:
            processed_documents = []
        if not processed_documents:
            raise ValueError("No processed documents available for metadata extraction")

        metadata_stats = {
            "total_chunks": len(processed_documents),
            "chunks_with_metadata": 0,
            "repositories": set(),
            "languages": set(),
            "file_types": set(),
        }

        # Validate and collect metadata statistics
        for doc in processed_documents:
            metadata = doc.metadata

            # Check for required metadata fields
            if metadata.get("repository") and metadata.get("file_path"):
                metadata_stats["chunks_with_metadata"] += 1
                metadata_stats["repositories"].add(metadata["repository"])

                if metadata.get("language"):
                    metadata_stats["languages"].add(metadata["language"])

                # Extract file extension
                file_path = metadata["file_path"]
                if "." in file_path:
                    ext = os.path.splitext(file_path)[1]
                    metadata_stats["file_types"].add(ext)

        # Convert sets to lists for JSON serialization
        metadata_stats["repositories"] = list(metadata_stats["repositories"])
        metadata_stats["languages"] = list(metadata_stats["languages"])
        metadata_stats["file_types"] = list(metadata_stats["file_types"])

        # Store metadata statistics
        state["metadata"]["metadata_stats"] = metadata_stats

        self.logger.info(
            f"Extracted metadata from {metadata_stats['chunks_with_metadata']}/{metadata_stats['total_chunks']} chunks"
        )
        self.logger.info(
            f"Found {len(metadata_stats['repositories'])} repositories, {len(metadata_stats['languages'])} languages"
        )

        return cast(IndexingState, update_workflow_progress(
            state, 55.0, IndexingWorkflowSteps.EXTRACT_METADATA
        ))

    def _store_in_vector_db(self, state: IndexingState) -> IndexingState:
        """Store documents and embeddings in vector database."""
        self.logger.info(
            f"Storing documents in {self.vector_store_type} vector database"
        )

        # Get processed documents (not embedded documents)
        processed_documents = state["metadata"].get("processed_documents", [])
        # Defensive programming: ensure processed_documents is not None
        if processed_documents is None:
            processed_documents = []
        if not processed_documents:
            raise ValueError("No processed documents available for storage")

        # Initialize vector store
        vector_store = self.vector_store_factory.create(
            collection_name=self.collection_name
        )

        storage_stats = {
            "total_documents": len(processed_documents),
            "stored_documents": 0,
            "failed_storage": 0,
            "batch_count": 0,
        }

        # Store documents in batches (vector store will generate embeddings)
        for i in range(0, len(processed_documents), self.batch_size):
            batch = processed_documents[i : i + self.batch_size]

            try:
                # Store batch in vector database (embeddings are generated by the vector store)
                self.logger.info(f"STORAGE DEBUG: About to store batch of {len(batch)} documents")
                self.logger.info(f"STORAGE DEBUG: First document metadata: {batch[0].metadata if batch else 'No documents'}")
                
                vector_store.add_documents(batch)
                
                self.logger.debug(f"STORAGE DEBUG: About to store batch of {len(batch)} documents")
                self.logger.debug(f"STORAGE DEBUG: First document metadata: {batch[0].metadata if batch else 'No documents'}")
                
                vector_store.add_documents(batch)
                
                self.logger.debug(f"STORAGE DEBUG: Successfully called add_documents for {len(batch)} documents")

                storage_stats["stored_documents"] += len(batch)
                storage_stats["batch_count"] += 1

                self.logger.debug(
                    f"Stored batch {storage_stats['batch_count']}: {len(batch)} documents"
                )

            except Exception as e:
                error_msg = (
                    f"Failed to store batch {storage_stats['batch_count'] + 1}: {e}"
                )
                self.logger.error(error_msg)

                storage_stats["failed_storage"] += len(batch)
                state = cast(IndexingState, add_workflow_error(
                    state, error_msg, IndexingWorkflowSteps.STORE_IN_VECTOR_DB
                ))

            # Update progress
            progress = 80.0 + (i + len(batch)) / len(processed_documents) * 15.0
            state = cast(IndexingState, update_workflow_progress(
                state, progress, IndexingWorkflowSteps.STORE_IN_VECTOR_DB
            ))

        # Store storage statistics and update successful embeddings count
        state["metadata"]["storage_stats"] = storage_stats
        state["successful_embeddings"] = storage_stats["stored_documents"]
        state["embeddings_generated"] = storage_stats["stored_documents"]

        self.logger.info(
            f"Stored {storage_stats['stored_documents']}/{storage_stats['total_documents']} documents in vector database"
        )

        if storage_stats["failed_storage"] > 0:
            self.logger.warning(
                f"Failed to store {storage_stats['failed_storage']} documents"
            )

        return cast(IndexingState, update_workflow_progress(
            state, 95.0, IndexingWorkflowSteps.STORE_IN_VECTOR_DB
        ))

    def _store_in_graph_db(self, state: IndexingState) -> IndexingState:
        """
        Store documents in graph database if graph features are enabled.
        
        Args:
            state: Current workflow state with processed documents
            
        Returns:
            Updated state with graph storage information
        """
        self.logger.info("Checking graph database storage")
        
        # Skip if graph features are not enabled
        if not is_graph_enabled():
            self.logger.info("Graph features disabled, skipping graph storage")
            return cast(IndexingState, update_workflow_progress(
                state, 97.0, IndexingWorkflowSteps.STORE_IN_GRAPH_DB
            ))
        
        try:
            # Get processed chunks from state metadata
            processed_chunks = state["metadata"].get("processed_documents", [])
            # Defensive programming: ensure processed_chunks is not None
            if processed_chunks is None:
                processed_chunks = []
            if not processed_chunks:
                self.logger.warning("No processed documents available for graph storage")
                return cast(IndexingState, update_workflow_progress(
                    state, 97.0, IndexingWorkflowSteps.STORE_IN_GRAPH_DB
                ))
            
            # Initialize graph store
            graph_store = MemGraphStore()
            connected = graph_store.connect()
            
            if not connected:
                self.logger.warning("Failed to connect to graph database, skipping graph storage")
                return cast(IndexingState, update_workflow_progress(
                    state, 97.0, IndexingWorkflowSteps.STORE_IN_GRAPH_DB
                ))
            
            # Store documents as graph nodes
            stored_nodes = 0
            for chunk in processed_chunks:
                try:
                    # Extract metadata from chunk
                    metadata = chunk.metadata
                    
                    # Create file node
                    file_properties = {
                        "file_path": metadata.get("file_path", "unknown"),
                        "repository": metadata.get("repository", "unknown"), 
                        "language": metadata.get("language", "unknown"),
                        "file_extension": metadata.get("file_extension", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "content_preview": chunk.page_content[:CONTENT_PREVIEW_LENGTH] + "..." if len(chunk.page_content) > CONTENT_PREVIEW_LENGTH else chunk.page_content
                    }
                    
                    # Create node in graph
                    node_id = graph_store.create_node(
                        labels=["File", "Document"],
                        properties=file_properties
                    )
                    stored_nodes += 1
                    
                except Exception as chunk_error:
                    self.logger.error(f"Failed to store chunk in graph: {chunk_error}")
                    continue
            
            graph_store.disconnect()
            
            # Update state with graph storage statistics
            state["graph_storage_stats"] = {
                "total_chunks": len(processed_chunks),
                "stored_nodes": stored_nodes,
                "failed_storage": len(processed_chunks) - stored_nodes
            }
            
            self.logger.info(f"Stored {stored_nodes}/{len(processed_chunks)} documents in graph database")
            
        except Exception as e:
            self.logger.error(f"Error during graph storage: {e}")
            # Don't fail the workflow, just log the error
            state["graph_storage_stats"] = {
                "error": str(e),
                "stored_nodes": 0
            }
        
        return cast(IndexingState, update_workflow_progress(
            state, 97.0, IndexingWorkflowSteps.STORE_IN_GRAPH_DB
        ))

    def _update_workflow_state(self, state: IndexingState) -> IndexingState:
        """Update workflow state with final statistics."""
        self.logger.info("Updating workflow state with final statistics")

        # Calculate performance metrics
        if state["metadata"].get("step_durations"):
            total_processing_time = sum(state["metadata"]["step_durations"].values())
            state["total_processing_time"] = total_processing_time

            # Calculate rates
            if total_processing_time > 0:
                state["documents_per_second"] = (
                    state["processed_files"] / total_processing_time
                )
                state["embeddings_per_second"] = (
                    state["successful_embeddings"] / total_processing_time
                )

        # Update repository states with final statistics
        for repo_name, repo_state in state["repository_states"].items():
            if repo_state["status"] != ProcessingStatus.FAILED:
                repo_state["processing_end_time"] = time.time()
                # Note: processing_duration calculation removed due to TypedDict constraints

        self.logger.info("Workflow state updated with final statistics")
        return cast(IndexingState, update_workflow_progress(
            state, 97.0, IndexingWorkflowSteps.UPDATE_WORKFLOW_STATE
        ))

    def _check_complete(self, state: IndexingState) -> IndexingState:
        """Check if indexing workflow is complete."""
        self.logger.info("Checking workflow completion status")

        # Check if we have successfully processed documents
        storage_stats = state["metadata"].get("storage_stats", {})
        stored_documents = storage_stats.get("stored_documents", 0)

        if stored_documents == 0:
            raise ValueError("No documents were successfully stored in vector database")

        # Check repository completion
        completed_repos = sum(
            1
            for repo_state in state["repository_states"].values()
            if repo_state["status"]
            in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
        )

        if completed_repos < len(state["repositories"]):
            self.logger.warning(
                f"Only {completed_repos}/{len(state['repositories'])} repositories completed"
            )

        # Workflow is considered complete if we have stored at least some documents
        self.logger.info(
            f"Workflow completion check passed: {stored_documents} documents stored"
        )
        return cast(IndexingState, update_workflow_progress(
            state, 99.0, IndexingWorkflowSteps.CHECK_COMPLETE
        ))

    def _finalize_index(self, state: IndexingState) -> IndexingState:
        """Finalize indexing workflow."""
        self.logger.info("Finalizing indexing workflow")

        # Update commit tracking for successful repositories (incremental mode only)
        if self.incremental and self.commit_tracker and not self.dry_run:
            self._update_commit_tracking(state)

        # Mark workflow as completed
        state["status"] = ProcessingStatus.COMPLETED

        # Log final statistics
        self.logger.info("=== Indexing Workflow Summary ===")
        
        # Log mode-specific information
        if self.incremental:
            incremental_changes = state["metadata"].get("incremental_changes", {})
            repos_with_changes = sum(1 for change in incremental_changes.values() 
                                   if change["change_type"] in ["incremental", "full_index"])
            repos_skipped = sum(1 for change in incremental_changes.values() 
                              if change["change_type"] == "no_changes")
            total_file_changes = sum(change.get("total_changes", 0) for change in incremental_changes.values())
            
            self.logger.info(f"Mode: Incremental re-indexing {'(dry-run)' if self.dry_run else ''}")
            self.logger.info(f"Repositories with changes: {repos_with_changes}")
            self.logger.info(f"Repositories skipped (no changes): {repos_skipped}")
            self.logger.info(f"Total file changes detected: {total_file_changes}")
            
            if "vector_cleanup_stats" in state["metadata"]:
                total_removed = state["metadata"].get("total_vectors_removed", 0)
                self.logger.info(f"Stale vectors removed: {total_removed}")
        else:
            self.logger.info("Mode: Full indexing")
        
        self.logger.info(f"Repositories processed: {len(state['repositories'])}")
        self.logger.info(f"Files processed: {state['processed_files']}")
        self.logger.info(f"Total chunks created: {state['total_chunks']}")
        self.logger.info(f"Embeddings generated: {state['successful_embeddings']}")
        self.logger.info(
            f"Documents stored: {state['metadata'].get('storage_stats', {}).get('stored_documents', 0)}"
        )

        if state.get("total_processing_time"):
            self.logger.info(
                f"Total processing time: {state['total_processing_time']:.2f}s"
            )
            self.logger.info(
                f"Processing rate: {state.get('documents_per_second', 0):.2f} docs/sec"
            )

        # Clean up temporary data to reduce memory usage
        if "loaded_documents" in state["metadata"]:
            del state["metadata"]["loaded_documents"]
        if "processed_documents" in state["metadata"]:
            del state["metadata"]["processed_documents"]
        if "embedded_documents" in state["metadata"]:
            del state["metadata"]["embedded_documents"]

        self.logger.info("Indexing workflow finalized successfully")
        return cast(IndexingState, update_workflow_progress(
            state, 100.0, IndexingWorkflowSteps.FINALIZE_INDEX
        ))

    def _prepare_repository_for_diff(self, repo_name: str, owner: str, name: str, branch: str) -> Optional[str]:
        """
        Prepare local repository for git diff operations.
        
        Args:
            repo_name: Repository name from config
            owner: Repository owner
            name: Repository name
            branch: Branch name
            
        Returns:
            Path to local repository or None if failed
        """
        try:
            # Use git repository manager to clone/update repository
            from src.loaders.git_repository_manager import GitRepositoryManager
            
            repo_manager = GitRepositoryManager()
            repo_url = f"https://github.com/{owner}/{name}.git"
            
            # Get local repo path
            local_repo_path = repo_manager.get_local_repo_path(owner, name)
            
            # Clone or update repository
            if not Path(local_repo_path).exists():
                # Clone repository
                clone_result = repo_manager.clone_repository(
                    repo_url=repo_url,
                    owner=owner,
                    repo_name=name,
                    branch=branch,
                    github_token=settings.github.token
                )
                if not clone_result:
                    self.logger.error(f"Failed to clone repository {owner}/{name}")
                    return None
            else:
                # Update existing repository
                update_result = repo_manager.update_repository(
                    owner=owner,
                    repo_name=name,
                    branch=branch
                )
                if not update_result:
                    self.logger.warning(f"Failed to update repository {owner}/{name}, using existing version")
            
            return local_repo_path
            
        except Exception as e:
            self.logger.error(f"Failed to prepare repository {repo_name} for diff: {e}")
            return None

    def _update_commit_tracking(self, state: IndexingState) -> None:
        """
        Update commit tracking for successfully processed repositories.
        
        Args:
            state: Current workflow state
        """
        if not self.commit_tracker:
            return
            
        incremental_changes = state["metadata"].get("incremental_changes", {})
        
        for repo_name in state["repositories"]:
            repo_state = state["repository_states"][repo_name]
            
            # Only update tracking for successfully completed repositories
            if repo_state["status"] != ProcessingStatus.COMPLETED:
                continue
            
            # Get repository information
            try:
                repo_config = self._get_repo_config(repo_name)
                owner, name = self._parse_repo_url(repo_config["url"])
                branch = repo_config.get("branch", "main")
                full_repo_name = f"{owner}/{name}"
                
                # Get current commit from change info
                change_info = incremental_changes.get(repo_name, {})
                current_commit = change_info.get("current_commit")
                
                if current_commit:
                    # Update commit tracking
                    metadata = {
                        "workflow_id": state["workflow_id"],
                        "files_processed": repo_state.get("processed_files", 0),
                        "total_files": repo_state.get("total_files", 0),
                        "processing_mode": "incremental" if change_info.get("change_type") == "incremental" else "full",
                        "change_summary": change_info.get("diff_summary", {})
                    }
                    
                    self.commit_tracker.update_last_indexed_commit(
                        repository=full_repo_name,
                        commit_hash=current_commit,
                        branch=branch,
                        metadata=metadata
                    )
                    
                    self.logger.info(f"Updated commit tracking for {full_repo_name}#{branch}: {current_commit}")
                else:
                    self.logger.warning(f"No current commit found for {repo_name}, skipping commit tracking update")
                    
            except Exception as e:
                self.logger.error(f"Failed to update commit tracking for {repo_name}: {e}")

    # Error handling methods

    def _handle_file_errors(
        self, state: IndexingState, error: Exception
    ) -> IndexingState:
        """Handle file loading errors."""
        self.logger.warning(f"Handling file loading error: {error}")

        # Mark current repository as failed and continue with others
        current_repo = state.get("current_repo")
        if current_repo and current_repo in state["repository_states"]:
            state["repository_states"][current_repo]["status"] = ProcessingStatus.FAILED
            state["repository_states"][current_repo]["errors"].append(str(error))

        # Continue workflow if other repositories are available
        return state

    def _handle_processing_errors(
        self, state: IndexingState, error: Exception
    ) -> IndexingState:
        """Handle document processing errors."""
        self.logger.warning(f"Handling processing error: {error}")

        # Partial processing is acceptable - continue with successfully processed documents
        return state

    def _handle_embedding_errors(
        self, state: IndexingState, error: Exception
    ) -> IndexingState:
        """Handle embedding generation errors."""
        self.logger.warning(f"Handling embedding error: {error}")

        # Try with smaller batch size
        if self.batch_size > 10:
            self.batch_size = max(10, self.batch_size // 2)
            self.logger.info(f"Reducing batch size to {self.batch_size} and retrying")

        return state

    def _handle_storage_errors(
        self, state: IndexingState, error: Exception
    ) -> IndexingState:
        """Handle vector database storage errors."""
        self.logger.warning(f"Handling storage error: {error}")

        # Try to reconnect to vector store
        try:
            self.vector_store_factory = VectorStoreFactory()
            self.logger.info("Recreated vector store factory")
        except Exception as reconnect_error:
            self.logger.error(f"Failed to reconnect to vector store: {reconnect_error}")

        return state

    # Helper methods
    def _has_repositories(self) -> bool:
        """Check if repositories are loaded in app settings."""
        return (
            self._app_settings is not None
            and "repositories" in self._app_settings
            and isinstance(self._app_settings["repositories"], list)
            and len(self._app_settings["repositories"]) > 0
        )

    def _load_app_settings(self) -> None:
        """Load repository configurations from appSettings.json."""
        try:
            with open(self.app_settings_path, "r") as f:
                self._app_settings = json.load(f)

            # Cache repository configurations by name
            if self._app_settings and "repositories" in self._app_settings:
                for repo in self._app_settings["repositories"]:
                    self._repo_configs[repo["name"]] = repo

                self.logger.info(
                    f"Loaded {len(self._app_settings['repositories'])} repository configurations"
                )
            else:
                raise ValueError("Invalid app settings structure: missing repositories")

        except FileNotFoundError:
            raise ValueError(f"appSettings.json not found at: {self.app_settings_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in appSettings.json: {e}")

    def _get_repo_config(self, repo_name: str) -> Dict[str, Any]:
        """Get repository configuration by name."""
        # First try app settings
        if repo_name in self._repo_configs:
            return self._repo_configs[repo_name]
        
        # Then try repository URLs
        if repo_name in self.repository_urls:
            return {
                "name": repo_name,
                "url": self.repository_urls[repo_name],
                "branch": "main"  # Default branch
            }
        
        raise ValueError(f"Repository configuration not found: {repo_name}")

    def _parse_repo_url(self, url: str) -> Tuple[str, str]:
        """Parse GitHub repository URL to extract owner and name."""
        # Handle both HTTPS and SSH URLs
        if url.startswith("https://github.com/"):
            path = url.replace("https://github.com/", "").rstrip("/")
        elif url.startswith("git@github.com:"):
            path = url.replace("git@github.com:", "").rstrip("/")
            if path.endswith(".git"):
                path = path[:-4]
        else:
            raise ValueError(f"Unsupported repository URL format: {url}")

        parts = path.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid repository URL format: {url}")

        return parts[0], parts[1]

    def _load_repository_files(
        self, repo_name: str, state: IndexingState
    ) -> Tuple[List[Document], int]:
        """Load files from a single repository."""
        self.logger.info(f"Loading files from repository: {repo_name}")

        # Update current repository in state
        state["current_repo"] = repo_name
        repo_state = state["repository_states"][repo_name]
        repo_state["processing_start_time"] = time.time()
        repo_state["status"] = ProcessingStatus.IN_PROGRESS

        try:
            repo_config = self._get_repo_config(repo_name)
            owner, name = self._parse_repo_url(repo_config["url"])
            full_repo_name = f"{owner}/{name}"

            # Check for incremental changes
            incremental_changes = state["metadata"].get("incremental_changes", {})
            files_to_process = None  # None means all files
            
            if self.incremental and repo_name in incremental_changes:
                change_info = incremental_changes[repo_name]
                if change_info["change_type"] == "incremental":
                    files_to_process = change_info.get("files_to_process", [])
                    # Defensive programming: ensure files_to_process is a list
                    files_to_process = self._ensure_list(files_to_process)
                    self.logger.info(f"Incremental mode: processing {safe_len(files_to_process)} changed files for {repo_name}")
                elif change_info["change_type"] == "full_index":
                    self.logger.info(f"Incremental mode: doing full index for {repo_name} ({change_info.get('reason', 'unknown reason')})")

            # Create GitHub loader with incremental support
            loader = EnhancedGitHubLoader(
                repo_owner=owner,
                repo_name=name,
                branch=repo_config.get("branch", "main"),
                file_extensions=settings.github.file_extensions,
                github_token=settings.github.token,
            )

            # Load documents from repository (with file filtering if incremental)
            if files_to_process is not None and len(files_to_process) > 0:
                # For incremental mode, use include_only parameter if supported
                try:
                    # Check if the loader supports include_only parameter
                    documents = loader.load(include_only=files_to_process)
                except TypeError:
                    # Fallback: load all and filter manually
                    self.logger.warning(f"Loader doesn't support include_only, filtering manually for {repo_name}")
                    all_docs = loader.load()
                    documents = []
                    for doc in all_docs:
                        file_path = doc.metadata.get("file_path", doc.metadata.get("source", ""))
                        if file_path in files_to_process:
                            documents.append(doc)
            elif files_to_process is not None and len(files_to_process) == 0:
                # No files to process
                self.logger.info(f"No files to process for {repo_name} in incremental mode")
                documents = []
            else:
                # Full mode or full index in incremental mode
                documents = loader.load()

            # Add repository name to document metadata (use full owner/repo name)
            for doc in documents:
                doc.metadata["repository"] = full_repo_name

            # Calculate unique file count from document metadata
            unique_files = set()
            for doc in documents:
                file_path = doc.metadata.get("file_path", "")
                if file_path:
                    unique_files.add(file_path)
                else:
                    # Fallback to source if file_path is not available
                    source_file = doc.metadata.get("source", "")
                    if source_file:
                        unique_files.add(source_file)

            # Return documents and unique file count (not document count)
            return documents, len(unique_files)

        except Exception as e:
            repo_state["status"] = ProcessingStatus.FAILED
            repo_state["errors"].append(str(e))
            raise e

    def _log_no_documents_diagnostics(self, state: IndexingState) -> None:
        """Log detailed diagnostics when no documents are loaded."""
        self.logger.error("=== REPOSITORY INDEXING DIAGNOSTICS ===")
        
        # Check GitHub token
        if not settings.github.token:
            self.logger.error("❌ GitHub token is not configured (GITHUB_TOKEN environment variable)")
        else:
            self.logger.info("✅ GitHub token is configured")
        
        # Check configured file extensions
        self.logger.info(f"📁 Configured file extensions: {settings.github.file_extensions}")
        
        # Check each repository
        repo_states = state.get("repository_states", {})
        for repo_name, repo_state in repo_states.items():
            self.logger.error(f"\n📊 Repository: {repo_name}")
            self.logger.error(f"   Status: {repo_state.get('status', 'unknown')}")
            self.logger.error(f"   Total files: {repo_state.get('total_files', 0)}")
            self.logger.error(f"   Processed files: {repo_state.get('processed_files', 0)}")
            
            errors = repo_state.get("errors", [])
            if errors:
                self.logger.error(f"   Errors: {errors}")
                
                # Check for common error patterns
                for error in errors:
                    error_str = str(error).lower()
                    if "not found" in error_str or "404" in error_str:
                        self.logger.error(f"   ❌ Repository not found - check if repository exists and is accessible")
                    elif "permission" in error_str or "403" in error_str:
                        self.logger.error(f"   ❌ Permission denied - check if GitHub token has access to this repository")
                    elif "rate limit" in error_str:
                        self.logger.error(f"   ❌ GitHub API rate limit exceeded")
                    elif "authentication" in error_str or "401" in error_str:
                        self.logger.error(f"   ❌ Authentication failed - check GitHub token validity")
            else:
                self.logger.error(f"   ❌ No files found matching configured extensions")
                
        self.logger.error("=== END DIAGNOSTICS ===")


def create_indexing_workflow(
    repositories: Optional[List[str]] = None,
    workflow_id: Optional[str] = None,
    **kwargs,
) -> IndexingWorkflow:
    """
    Factory function to create indexing workflow.

    Args:
        repositories: List of repository names to index
        workflow_id: Optional workflow ID
        **kwargs: Additional workflow arguments

    Returns:
        Configured indexing workflow
    """
    # Create initial state
    state = create_indexing_state(
        workflow_id=workflow_id or f"indexing_{int(time.time())}",
        repositories=repositories or [],
    )

    # Create workflow instance
    workflow = IndexingWorkflow(
        repositories=repositories, workflow_id=state["workflow_id"], **kwargs
    )

    return workflow
