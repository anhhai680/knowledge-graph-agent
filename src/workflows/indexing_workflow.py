"""
LangGraph Indexing Workflow Implementation.

This module implements a complete stateful indexing workflow using LangGraph
for processing multiple GitHub repositories, document chunking, and vector storage.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import Document

from src.config.settings import settings
from src.loaders.github_loader import GitHubLoader
from src.processors.document_processor import DocumentProcessor
from src.llm.embedding_factory import EmbeddingFactory
from src.vectorstores.store_factory import VectorStoreFactory
from src.workflows.base_workflow import BaseWorkflow
from src.workflows.workflow_states import (
    IndexingState,
    ProcessingStatus,
    WorkflowType,
    create_indexing_state,
    create_repository_state,
    create_file_processing_state,
    update_workflow_progress,
    add_workflow_error
)


class IndexingWorkflowSteps(str):
    """Indexing workflow step enumeration."""
    INITIALIZE_STATE = "initialize_state"
    LOAD_REPOSITORIES = "load_repositories"
    VALIDATE_REPOS = "validate_repos"
    LOAD_FILES_FROM_GITHUB = "load_files_from_github"
    PROCESS_DOCUMENTS = "process_documents"
    LANGUAGE_AWARE_CHUNKING = "language_aware_chunking"
    EXTRACT_METADATA = "extract_metadata"
    GENERATE_EMBEDDINGS = "generate_embeddings"
    STORE_IN_VECTOR_DB = "store_in_vector_db"
    UPDATE_WORKFLOW_STATE = "update_workflow_state"
    CHECK_COMPLETE = "check_complete"
    FINALIZE_INDEX = "finalize_index"
    
    # Error handling states
    HANDLE_FILE_ERRORS = "handle_file_errors"
    HANDLE_PROCESSING_ERRORS = "handle_processing_errors"
    HANDLE_EMBEDDING_ERRORS = "handle_embedding_errors"
    HANDLE_STORAGE_ERRORS = "handle_storage_errors"


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
        **kwargs
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
            **kwargs: Additional arguments passed to BaseWorkflow
        """
        super().__init__(**kwargs)
        
        self.app_settings_path = app_settings_path
        self.target_repositories = repositories
        self.vector_store_type = vector_store_type or settings.database_type.value
        self.collection_name = collection_name or (
            settings.pinecone.collection_name
            if self.vector_store_type == "pinecone" and settings.pinecone
            else settings.chroma.collection_name
        )
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_factory = EmbeddingFactory()
        self.vector_store_factory = VectorStoreFactory()
        
        # Repository configuration cache
        self._repo_configs: Dict[str, Dict[str, Any]] = {}
        self._app_settings: Optional[Dict[str, Any]] = None
        
        self.logger.info(f"Initialized indexing workflow with vector store: {self.vector_store_type}")
    
    def define_steps(self) -> List[str]:
        """
        Define the indexing workflow steps.
        
        Returns:
            List of step names in execution order
        """
        return [
            IndexingWorkflowSteps.INITIALIZE_STATE,
            IndexingWorkflowSteps.LOAD_REPOSITORIES,
            IndexingWorkflowSteps.VALIDATE_REPOS,
            IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB,
            IndexingWorkflowSteps.PROCESS_DOCUMENTS,
            IndexingWorkflowSteps.EXTRACT_METADATA,
            IndexingWorkflowSteps.GENERATE_EMBEDDINGS,
            IndexingWorkflowSteps.STORE_IN_VECTOR_DB,
            IndexingWorkflowSteps.UPDATE_WORKFLOW_STATE,
            IndexingWorkflowSteps.CHECK_COMPLETE,
            IndexingWorkflowSteps.FINALIZE_INDEX
        ]
    
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
                self.logger.error("Repositories list is empty. No repositories to process.")
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
            elif step == IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB:
                state = self._load_files_from_github(state)
            elif step == IndexingWorkflowSteps.PROCESS_DOCUMENTS:
                state = self._process_documents(state)
            elif step == IndexingWorkflowSteps.EXTRACT_METADATA:
                state = self._extract_metadata(state)
            elif step == IndexingWorkflowSteps.GENERATE_EMBEDDINGS:
                state = self._generate_embeddings(state)
            elif step == IndexingWorkflowSteps.STORE_IN_VECTOR_DB:
                state = self._store_in_vector_db(state)
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
            return add_workflow_error(state, str(e), step)
    
    def handle_error(
        self,
        step: str,
        state: IndexingState,
        error: Exception
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
        state = add_workflow_error(state, str(error), step)
        
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
        
        # Load app settings if not already loaded
        if not self._app_settings:
            self._load_app_settings()
        
        # Determine repositories to process
        if self.target_repositories:
            # Filter to only requested repositories
            available_repos = [repo["name"] for repo in self._app_settings["repositories"]]
            missing_repos = set(self.target_repositories) - set(available_repos)
            if missing_repos:
                raise ValueError(f"Repositories not found in appSettings: {missing_repos}")
            repositories = self.target_repositories
        else:
            # Process all repositories from appSettings
            repositories = [repo["name"] for repo in self._app_settings["repositories"]]
        
        # Update state with repository information
        state["repositories"] = repositories
        state["vector_store_type"] = self.vector_store_type
        state["collection_name"] = self.collection_name
        state["batch_size"] = self.batch_size
        state["status"] = ProcessingStatus.IN_PROGRESS
        
        # Initialize repository states
        for repo_name in repositories:
            repo_config = self._get_repo_config(repo_name)
            state["repository_states"][repo_name] = create_repository_state(
                name=repo_name,
                url=repo_config["url"],
                branch=repo_config.get("branch", "main")
            )
        
        self.logger.info(f"Initialized state for {len(repositories)} repositories")
        return update_workflow_progress(state, 5.0, IndexingWorkflowSteps.INITIALIZE_STATE)
    
    def _load_repositories(self, state: IndexingState) -> IndexingState:
        """Load repository configurations from appSettings."""
        self.logger.info("Loading repository configurations")
        
        # Repository configurations are loaded in _initialize_state
        # This step validates that all required configurations are present
        for repo_name in state["repositories"]:
            if repo_name not in state["repository_states"]:
                raise ValueError(f"Repository state not found for: {repo_name}")
            
            repo_config = self._get_repo_config(repo_name)
            if not repo_config:
                raise ValueError(f"Repository configuration not found for: {repo_name}")
        
        self.logger.info(f"Loaded configurations for {len(state['repositories'])} repositories")
        return update_workflow_progress(state, 10.0, IndexingWorkflowSteps.LOAD_REPOSITORIES)
    
    def _validate_repositories(self, state: IndexingState) -> IndexingState:
        """Validate repository access and configurations."""
        self.logger.info("Validating repository access")
        
        validation_errors = []
        
        for repo_name in state["repositories"]:
            try:
                repo_config = self._get_repo_config(repo_name)
                repo_state = state["repository_states"][repo_name]
                
                # Parse repository owner and name from URL
                owner, name = self._parse_repo_url(repo_config["url"])
                
                # Create GitHub loader to test access
                loader = GitHubLoader(
                    repo_owner=owner,
                    repo_name=name,
                    branch=repo_config.get("branch", "main"),
                    file_extensions=settings.github.file_extensions,
                    github_token=settings.github.token
                )
                
                # Test repository access by getting basic info
                repo_info = loader.get_repository_info()
                self.logger.info(f"Validated access to repository: {repo_name}")
                
                # Update repository state with validation success
                repo_state["status"] = ProcessingStatus.NOT_STARTED
                
            except Exception as e:
                error_msg = f"Failed to validate repository {repo_name}: {e}"
                validation_errors.append(error_msg)
                self.logger.error(error_msg)
                
                # Mark repository as failed
                state["repository_states"][repo_name]["status"] = ProcessingStatus.FAILED
                state["repository_states"][repo_name]["errors"].append(str(e))
        
        if validation_errors:
            # Add validation errors to state but continue with valid repositories
            for error in validation_errors:
                state = add_workflow_error(state, error, IndexingWorkflowSteps.VALIDATE_REPOS)
        
        valid_repos = [
            repo for repo in state["repositories"]
            if state["repository_states"][repo]["status"] != ProcessingStatus.FAILED
        ]
        
        if not valid_repos:
            raise ValueError("No valid repositories found after validation")
        
        self.logger.info(f"Validated {len(valid_repos)}/{len(state['repositories'])} repositories")
        return update_workflow_progress(state, 15.0, IndexingWorkflowSteps.VALIDATE_REPOS)
    
    def _load_files_from_github(self, state: IndexingState) -> IndexingState:
        """Load files from GitHub repositories with parallel processing."""
        self.logger.info("Loading files from GitHub repositories")
        
        all_documents = []
        total_files = 0
        
        # Process repositories in parallel with controlled concurrency
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(state["repositories"]))) as executor:
            # Submit repository loading tasks
            future_to_repo = {}
            for repo_name in state["repositories"]:
                if state["repository_states"][repo_name]["status"] != ProcessingStatus.FAILED:
                    future = executor.submit(self._load_repository_files, repo_name, state)
                    future_to_repo[future] = repo_name
            
            # Collect results as they complete
            for future in as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    repo_documents, repo_file_count = future.result()
                    all_documents.extend(repo_documents)
                    total_files += repo_file_count
                    
                    # Update repository state
                    state["repository_states"][repo_name]["status"] = ProcessingStatus.COMPLETED
                    state["repository_states"][repo_name]["total_files"] = repo_file_count
                    state["repository_states"][repo_name]["processed_files"] = repo_file_count
                    
                    self.logger.info(f"Loaded {repo_file_count} files from repository: {repo_name}")
                    
                except Exception as e:
                    error_msg = f"Failed to load files from repository {repo_name}: {e}"
                    self.logger.error(error_msg)
                    
                    # Update repository state with error
                    state["repository_states"][repo_name]["status"] = ProcessingStatus.FAILED
                    state["repository_states"][repo_name]["errors"].append(str(e))
                    state = add_workflow_error(state, error_msg, IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB)
        
        # Store documents in state metadata for next steps
        state["metadata"]["loaded_documents"] = all_documents
        state["total_files"] = total_files
        state["processed_files"] = total_files
        
        if not all_documents:
            raise ValueError("No documents loaded from any repository")
        
        self.logger.info(f"Loaded {len(all_documents)} documents from {total_files} files")
        return update_workflow_progress(state, 30.0, IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB)
    
    def _process_documents(self, state: IndexingState) -> IndexingState:
        """Process loaded documents with language-aware processing."""
        self.logger.info("Processing documents with language-aware strategies")
        
        documents = state["metadata"].get("loaded_documents", [])
        if not documents:
            raise ValueError("No documents available for processing")
        
        processed_documents = []
        processing_stats = {
            "total_documents": len(documents),
            "processed_documents": 0,
            "total_chunks": 0,
            "processing_errors": 0
        }
        
        # Process documents in batches to manage memory
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
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
            state = update_workflow_progress(state, progress, IndexingWorkflowSteps.PROCESS_DOCUMENTS)
        
        # Store processed documents and update statistics
        state["metadata"]["processed_documents"] = processed_documents
        state["metadata"]["processing_stats"] = processing_stats
        state["total_chunks"] = processing_stats["total_chunks"]
        
        self.logger.info(f"Processed {processing_stats['processed_documents']} documents into {processing_stats['total_chunks']} chunks")
        
        if processing_stats["processing_errors"] > 0:
            self.logger.warning(f"Processing errors occurred for {processing_stats['processing_errors']} documents")
        
        # Validate language-aware chunking was successful
        if not processed_documents:
            raise ValueError("No processed documents available after language-aware chunking")
        
        # Validate chunk metadata quality
        chunks_with_metadata = 0
        for doc in processed_documents:
            if doc.metadata.get("chunk_type") or doc.metadata.get("language"):
                chunks_with_metadata += 1
        
        self.logger.info(f"Language-aware chunking produced {len(processed_documents)} chunks, {chunks_with_metadata} with metadata")
        
        return update_workflow_progress(state, 50.0, IndexingWorkflowSteps.PROCESS_DOCUMENTS)
    
    def _extract_metadata(self, state: IndexingState) -> IndexingState:
        """Extract and validate metadata from processed chunks."""
        self.logger.info("Extracting and validating chunk metadata")
        
        processed_documents = state["metadata"].get("processed_documents", [])
        if not processed_documents:
            raise ValueError("No processed documents available for metadata extraction")
        
        metadata_stats = {
            "total_chunks": len(processed_documents),
            "chunks_with_metadata": 0,
            "repositories": set(),
            "languages": set(),
            "file_types": set()
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
        
        self.logger.info(f"Extracted metadata from {metadata_stats['chunks_with_metadata']}/{metadata_stats['total_chunks']} chunks")
        self.logger.info(f"Found {len(metadata_stats['repositories'])} repositories, {len(metadata_stats['languages'])} languages")
        
        return update_workflow_progress(state, 55.0, IndexingWorkflowSteps.EXTRACT_METADATA)
    
    def _generate_embeddings(self, state: IndexingState) -> IndexingState:
        """Generate embeddings for processed documents."""
        self.logger.info("Generating embeddings for processed documents")
        
        processed_documents = state["metadata"].get("processed_documents", [])
        if not processed_documents:
            raise ValueError("No processed documents available for embedding generation")
        
        # Initialize embedding provider
        embedding_provider = self.embedding_factory.create()
        
        embeddings_stats = {
            "total_documents": len(processed_documents),
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "batch_count": 0,
            "total_tokens": 0
        }
        
        embedded_documents = []
        
        # Process documents in batches
        for i in range(0, len(processed_documents), self.batch_size):
            batch = processed_documents[i:i + self.batch_size]
            batch_texts = [doc.page_content for doc in batch]
            
            try:
                # Generate embeddings for batch
                batch_embeddings = embedding_provider.embed_documents(batch_texts)
                
                # Attach embeddings to documents
                for doc, embedding in zip(batch, batch_embeddings):
                    doc.metadata["embedding"] = embedding
                    embedded_documents.append(doc)
                
                embeddings_stats["successful_embeddings"] += len(batch)
                embeddings_stats["batch_count"] += 1
                
                # Estimate token usage (rough approximation)
                batch_tokens = sum(len(text.split()) * 1.3 for text in batch_texts)  # ~1.3 tokens per word
                embeddings_stats["total_tokens"] += int(batch_tokens)
                
                self.logger.debug(f"Generated embeddings for batch {embeddings_stats['batch_count']}: {len(batch)} documents")
                
            except Exception as e:
                error_msg = f"Failed to generate embeddings for batch {embeddings_stats['batch_count'] + 1}: {e}"
                self.logger.error(error_msg)
                
                # Add failed documents without embeddings
                for doc in batch:
                    embedded_documents.append(doc)
                
                embeddings_stats["failed_embeddings"] += len(batch)
                state = add_workflow_error(state, error_msg, IndexingWorkflowSteps.GENERATE_EMBEDDINGS)
            
            # Update progress
            progress = 55.0 + (i + len(batch)) / len(processed_documents) * 25.0
            state = update_workflow_progress(state, progress, IndexingWorkflowSteps.GENERATE_EMBEDDINGS)
        
        # Store embedded documents and statistics
        state["metadata"]["embedded_documents"] = embedded_documents
        state["metadata"]["embeddings_stats"] = embeddings_stats
        state["embeddings_generated"] = embeddings_stats["successful_embeddings"]
        state["successful_embeddings"] = embeddings_stats["successful_embeddings"]
        state["failed_embeddings"] = embeddings_stats["failed_embeddings"]
        
        self.logger.info(f"Generated embeddings for {embeddings_stats['successful_embeddings']}/{embeddings_stats['total_documents']} documents")
        
        if embeddings_stats["failed_embeddings"] > 0:
            self.logger.warning(f"Failed to generate embeddings for {embeddings_stats['failed_embeddings']} documents")
        
        return update_workflow_progress(state, 80.0, IndexingWorkflowSteps.GENERATE_EMBEDDINGS)
    
    def _store_in_vector_db(self, state: IndexingState) -> IndexingState:
        """Store documents and embeddings in vector database."""
        self.logger.info(f"Storing documents in {self.vector_store_type} vector database")
        
        embedded_documents = state["metadata"].get("embedded_documents", [])
        if not embedded_documents:
            raise ValueError("No embedded documents available for storage")
        
        # Initialize vector store
        vector_store = self.vector_store_factory.create(collection_name=self.collection_name)
        
        storage_stats = {
            "total_documents": len(embedded_documents),
            "stored_documents": 0,
            "failed_storage": 0,
            "batch_count": 0
        }
        
        # Filter documents with embeddings
        documents_with_embeddings = [
            doc for doc in embedded_documents 
            if doc.metadata.get("embedding") is not None
        ]
        
        documents_without_embeddings = len(embedded_documents) - len(documents_with_embeddings)
        if documents_without_embeddings > 0:
            self.logger.warning(f"Skipping {documents_without_embeddings} documents without embeddings")
        
        # Store documents in batches
        for i in range(0, len(documents_with_embeddings), self.batch_size):
            batch = documents_with_embeddings[i:i + self.batch_size]
            
            try:
                # Prepare documents for storage (remove embedding from metadata to avoid duplication)
                storage_docs = []
                embeddings = []
                
                for doc in batch:
                    # Create clean document without embedding in metadata
                    clean_metadata = {k: v for k, v in doc.metadata.items() if k != "embedding"}
                    storage_doc = Document(
                        page_content=doc.page_content,
                        metadata=clean_metadata
                    )
                    storage_docs.append(storage_doc)
                    embeddings.append(doc.metadata["embedding"])
                
                # Store batch in vector database
                vector_store.add_documents(storage_docs, embeddings=embeddings)
                
                storage_stats["stored_documents"] += len(batch)
                storage_stats["batch_count"] += 1
                
                self.logger.debug(f"Stored batch {storage_stats['batch_count']}: {len(batch)} documents")
                
            except Exception as e:
                error_msg = f"Failed to store batch {storage_stats['batch_count'] + 1}: {e}"
                self.logger.error(error_msg)
                
                storage_stats["failed_storage"] += len(batch)
                state = add_workflow_error(state, error_msg, IndexingWorkflowSteps.STORE_IN_VECTOR_DB)
            
            # Update progress
            progress = 80.0 + (i + len(batch)) / len(documents_with_embeddings) * 15.0
            state = update_workflow_progress(state, progress, IndexingWorkflowSteps.STORE_IN_VECTOR_DB)
        
        # Store storage statistics
        state["metadata"]["storage_stats"] = storage_stats
        
        self.logger.info(f"Stored {storage_stats['stored_documents']}/{storage_stats['total_documents']} documents in vector database")
        
        if storage_stats["failed_storage"] > 0:
            self.logger.warning(f"Failed to store {storage_stats['failed_storage']} documents")
        
        return update_workflow_progress(state, 95.0, IndexingWorkflowSteps.STORE_IN_VECTOR_DB)
    
    def _update_workflow_state(self, state: IndexingState) -> IndexingState:
        """Update workflow state with final statistics."""
        self.logger.info("Updating workflow state with final statistics")
        
        # Calculate performance metrics
        if state["metadata"].get("step_durations"):
            total_processing_time = sum(state["metadata"]["step_durations"].values())
            state["total_processing_time"] = total_processing_time
            
            # Calculate rates
            if total_processing_time > 0:
                state["documents_per_second"] = state["processed_files"] / total_processing_time
                state["embeddings_per_second"] = state["successful_embeddings"] / total_processing_time
        
        # Update repository states with final statistics
        for repo_name, repo_state in state["repository_states"].items():
            if repo_state["status"] != ProcessingStatus.FAILED:
                repo_state["processing_end_time"] = time.time()
                if repo_state.get("processing_start_time"):
                    repo_state["processing_duration"] = (
                        repo_state["processing_end_time"] - 
                        repo_state["processing_start_time"]
                    )
        
        self.logger.info("Workflow state updated with final statistics")
        return update_workflow_progress(state, 97.0, IndexingWorkflowSteps.UPDATE_WORKFLOW_STATE)
    
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
            1 for repo_state in state["repository_states"].values()
            if repo_state["status"] in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
        )
        
        if completed_repos < len(state["repositories"]):
            self.logger.warning(f"Only {completed_repos}/{len(state['repositories'])} repositories completed")
        
        # Workflow is considered complete if we have stored at least some documents
        self.logger.info(f"Workflow completion check passed: {stored_documents} documents stored")
        return update_workflow_progress(state, 99.0, IndexingWorkflowSteps.CHECK_COMPLETE)
    
    def _finalize_index(self, state: IndexingState) -> IndexingState:
        """Finalize indexing workflow."""
        self.logger.info("Finalizing indexing workflow")
        
        # Mark workflow as completed
        state["status"] = ProcessingStatus.COMPLETED
        
        # Log final statistics
        self.logger.info("=== Indexing Workflow Summary ===")
        self.logger.info(f"Repositories processed: {len(state['repositories'])}")
        self.logger.info(f"Files processed: {state['processed_files']}")
        self.logger.info(f"Total chunks created: {state['total_chunks']}")
        self.logger.info(f"Embeddings generated: {state['successful_embeddings']}")
        self.logger.info(f"Documents stored: {state['metadata'].get('storage_stats', {}).get('stored_documents', 0)}")
        
        if state.get("total_processing_time"):
            self.logger.info(f"Total processing time: {state['total_processing_time']:.2f}s")
            self.logger.info(f"Processing rate: {state.get('documents_per_second', 0):.2f} docs/sec")
        
        # Clean up temporary data to reduce memory usage
        if "loaded_documents" in state["metadata"]:
            del state["metadata"]["loaded_documents"]
        if "processed_documents" in state["metadata"]:
            del state["metadata"]["processed_documents"]
        if "embedded_documents" in state["metadata"]:
            del state["metadata"]["embedded_documents"]
        
        self.logger.info("Indexing workflow finalized successfully")
        return update_workflow_progress(state, 100.0, IndexingWorkflowSteps.FINALIZE_INDEX)
    
    # Error handling methods
    
    def _handle_file_errors(self, state: IndexingState, error: Exception) -> IndexingState:
        """Handle file loading errors."""
        self.logger.warning(f"Handling file loading error: {error}")
        
        # Mark current repository as failed and continue with others
        current_repo = state.get("current_repo")
        if current_repo and current_repo in state["repository_states"]:
            state["repository_states"][current_repo]["status"] = ProcessingStatus.FAILED
            state["repository_states"][current_repo]["errors"].append(str(error))
        
        # Continue workflow if other repositories are available
        return state
    
    def _handle_processing_errors(self, state: IndexingState, error: Exception) -> IndexingState:
        """Handle document processing errors."""
        self.logger.warning(f"Handling processing error: {error}")
        
        # Partial processing is acceptable - continue with successfully processed documents
        return state
    
    def _handle_embedding_errors(self, state: IndexingState, error: Exception) -> IndexingState:
        """Handle embedding generation errors."""
        self.logger.warning(f"Handling embedding error: {error}")
        
        # Try with smaller batch size
        if self.batch_size > 10:
            self.batch_size = max(10, self.batch_size // 2)
            self.logger.info(f"Reducing batch size to {self.batch_size} and retrying")
        
        return state
    
    def _handle_storage_errors(self, state: IndexingState, error: Exception) -> IndexingState:
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
    
    def _load_app_settings(self) -> None:
        """Load repository configurations from appSettings.json."""
        try:
            with open(self.app_settings_path, 'r') as f:
                self._app_settings = json.load(f)
            
            # Cache repository configurations by name
            for repo in self._app_settings["repositories"]:
                self._repo_configs[repo["name"]] = repo
                
            self.logger.info(f"Loaded {len(self._app_settings['repositories'])} repository configurations")
            
        except FileNotFoundError:
            raise ValueError(f"appSettings.json not found at: {self.app_settings_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in appSettings.json: {e}")
    
    def _get_repo_config(self, repo_name: str) -> Dict[str, Any]:
        """Get repository configuration by name."""
        if repo_name not in self._repo_configs:
            raise ValueError(f"Repository configuration not found: {repo_name}")
        return self._repo_configs[repo_name]
    
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
    
    def _load_repository_files(self, repo_name: str, state: IndexingState) -> Tuple[List[Document], int]:
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
            
            # Create GitHub loader
            loader = GitHubLoader(
                repo_owner=owner,
                repo_name=name,
                branch=repo_config.get("branch", "main"),
                file_extensions=settings.github.file_extensions,
                github_token=settings.github.token
            )
            
            # Load documents from repository
            documents = loader.load()
            
            # Add repository name to document metadata
            for doc in documents:
                doc.metadata["repository"] = repo_name
            
            return documents, len(documents)
            
        except Exception as e:
            repo_state["status"] = ProcessingStatus.FAILED
            repo_state["errors"].append(str(e))
            raise e


def create_indexing_workflow(
    repositories: Optional[List[str]] = None,
    workflow_id: Optional[str] = None,
    **kwargs
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
        repositories=repositories or []
    )
    
    # Create workflow instance
    workflow = IndexingWorkflow(
        repositories=repositories,
        workflow_id=state["workflow_id"],
        **kwargs
    )
    
    return workflow
