"""
Unit tests for IndexingWorkflow.

This module contains comprehensive unit tests for the LangGraph indexing workflow
implementation including state management, error handling, and component integration.
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Any

import pytest

from src.workflows.indexing_workflow import (
    IndexingWorkflow,
    IndexingWorkflowSteps,
    create_indexing_workflow
)
from src.workflows.workflow_states import (
    IndexingState,
    ProcessingStatus,
    WorkflowType,
    create_indexing_state,
    create_repository_state
)
from langchain.schema import Document


class TestIndexingWorkflow(unittest.TestCase):
    """Test cases for IndexingWorkflow class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary appSettings.json
        self.temp_dir = tempfile.mkdtemp()
        self.app_settings_path = os.path.join(self.temp_dir, "appSettings.json")
        
        self.test_app_settings = {
            "repositories": [
                {
                    "name": "test-repo-1",
                    "url": "https://github.com/test/repo1",
                    "branch": "main",
                    "description": "Test repository 1"
                },
                {
                    "name": "test-repo-2", 
                    "url": "https://github.com/test/repo2",
                    "branch": "master",
                    "description": "Test repository 2"
                }
            ]
        }
        
        with open(self.app_settings_path, 'w') as f:
            json.dump(self.test_app_settings, f)
        
        # Create test workflow
        self.workflow = IndexingWorkflow(
            app_settings_path=self.app_settings_path,
            max_retries=1,  # Reduce retries for faster tests
            retry_delay=0.1
        )
        
        # Create test state
        self.test_state = create_indexing_state(
            workflow_id="test-workflow-123",
            repositories=["test-repo-1", "test-repo-2"]
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_define_steps(self):
        """Test workflow step definition."""
        steps = self.workflow.define_steps()
        
        expected_steps = [
            IndexingWorkflowSteps.INITIALIZE_STATE,
            IndexingWorkflowSteps.LOAD_REPOSITORIES,
            IndexingWorkflowSteps.VALIDATE_REPOS,
            IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB,
            IndexingWorkflowSteps.PROCESS_DOCUMENTS,
            IndexingWorkflowSteps.LANGUAGE_AWARE_CHUNKING,
            IndexingWorkflowSteps.EXTRACT_METADATA,
            IndexingWorkflowSteps.GENERATE_EMBEDDINGS,
            IndexingWorkflowSteps.STORE_IN_VECTOR_DB,
            IndexingWorkflowSteps.UPDATE_WORKFLOW_STATE,
            IndexingWorkflowSteps.CHECK_COMPLETE,
            IndexingWorkflowSteps.FINALIZE_INDEX
        ]
        
        self.assertEqual(steps, expected_steps)
        self.assertEqual(len(steps), 12)
    
    def test_validate_state_valid(self):
        """Test state validation with valid state."""
        self.assertTrue(self.workflow.validate_state(self.test_state))
    
    def test_validate_state_missing_workflow_id(self):
        """Test state validation with missing workflow_id."""
        invalid_state = self.test_state.copy()
        del invalid_state["workflow_id"]
        
        self.assertFalse(self.workflow.validate_state(invalid_state))
    
    def test_validate_state_missing_repositories(self):
        """Test state validation with missing repositories."""
        invalid_state = self.test_state.copy()
        invalid_state["repositories"] = []
        
        self.assertFalse(self.workflow.validate_state(invalid_state))
    
    def test_validate_state_wrong_workflow_type(self):
        """Test state validation with wrong workflow type."""
        invalid_state = self.test_state.copy()
        invalid_state["workflow_type"] = "query"
        
        self.assertFalse(self.workflow.validate_state(invalid_state))
    
    def test_initialize_state(self):
        """Test state initialization step."""
        # Mock app settings loading
        self.workflow._app_settings = self.test_app_settings
        self.workflow._repo_configs = {
            repo["name"]: repo for repo in self.test_app_settings["repositories"]
        }
        
        initial_state = create_indexing_state("test-id", [])
        result_state = self.workflow._initialize_state(initial_state)
        
        # Verify state updates
        self.assertEqual(result_state["status"], ProcessingStatus.IN_PROGRESS)
        self.assertEqual(len(result_state["repositories"]), 2)
        self.assertEqual(len(result_state["repository_states"]), 2)
        self.assertEqual(result_state["progress_percentage"], 5.0)
        
        # Verify repository states
        for repo_name in ["test-repo-1", "test-repo-2"]:
            self.assertIn(repo_name, result_state["repository_states"])
            repo_state = result_state["repository_states"][repo_name]
            self.assertEqual(repo_state["name"], repo_name)
            self.assertEqual(repo_state["status"], ProcessingStatus.NOT_STARTED)
    
    def test_initialize_state_with_target_repositories(self):
        """Test state initialization with specific target repositories."""
        # Set target repositories
        self.workflow.target_repositories = ["test-repo-1"]
        self.workflow._app_settings = self.test_app_settings
        self.workflow._repo_configs = {
            repo["name"]: repo for repo in self.test_app_settings["repositories"]
        }
        
        initial_state = create_indexing_state("test-id", [])
        result_state = self.workflow._initialize_state(initial_state)
        
        # Should only process target repository
        self.assertEqual(len(result_state["repositories"]), 1)
        self.assertEqual(result_state["repositories"][0], "test-repo-1")
    
    def test_initialize_state_missing_repository(self):
        """Test state initialization with missing target repository."""
        # Set non-existent target repository
        self.workflow.target_repositories = ["non-existent-repo"]
        self.workflow._app_settings = self.test_app_settings
        self.workflow._repo_configs = {
            repo["name"]: repo for repo in self.test_app_settings["repositories"]
        }
        
        initial_state = create_indexing_state("test-id", [])
        
        with self.assertRaises(ValueError) as context:
            self.workflow._initialize_state(initial_state)
        
        self.assertIn("Repositories not found", str(context.exception))
    
    def test_load_repositories(self):
        """Test repository loading step."""
        # Setup state with repository configurations
        state = self.test_state.copy()
        state["repository_states"] = {
            "test-repo-1": create_repository_state("test-repo-1", "https://github.com/test/repo1"),
            "test-repo-2": create_repository_state("test-repo-2", "https://github.com/test/repo2")
        }
        
        self.workflow._repo_configs = {
            repo["name"]: repo for repo in self.test_app_settings["repositories"]
        }
        
        result_state = self.workflow._load_repositories(state)
        
        self.assertEqual(result_state["progress_percentage"], 10.0)
        self.assertEqual(result_state["current_step"], IndexingWorkflowSteps.LOAD_REPOSITORIES)
    
    def test_load_repositories_missing_config(self):
        """Test repository loading with missing configuration."""
        state = self.test_state.copy()
        state["repositories"] = ["missing-repo"]
        
        with self.assertRaises(ValueError) as context:
            self.workflow._load_repositories(state)
        
        self.assertIn("Repository configuration not found", str(context.exception))
    
    @patch('src.workflows.indexing_workflow.GitHubLoader')
    def test_validate_repositories_success(self, mock_loader_class):
        """Test repository validation with successful access."""
        # Setup mock loader
        mock_loader = Mock()
        mock_loader.get_repository_info.return_value = {"name": "test-repo"}
        mock_loader_class.return_value = mock_loader
        
        # Setup state
        state = self.test_state.copy()
        state["repository_states"] = {
            "test-repo-1": create_repository_state("test-repo-1", "https://github.com/test/repo1"),
            "test-repo-2": create_repository_state("test-repo-2", "https://github.com/test/repo2")
        }
        
        self.workflow._repo_configs = {
            repo["name"]: repo for repo in self.test_app_settings["repositories"]
        }
        
        result_state = self.workflow._validate_repositories(state)
        
        # Verify loader was called for both repositories
        self.assertEqual(mock_loader_class.call_count, 2)
        self.assertEqual(mock_loader.get_repository_info.call_count, 2)
        
        # Verify state updates
        self.assertEqual(result_state["progress_percentage"], 15.0)
        for repo_state in result_state["repository_states"].values():
            self.assertEqual(repo_state["status"], ProcessingStatus.NOT_STARTED)
    
    @patch('src.workflows.indexing_workflow.GitHubLoader')
    def test_validate_repositories_failure(self, mock_loader_class):
        """Test repository validation with access failure."""
        # Setup mock loader to raise exception
        mock_loader_class.side_effect = Exception("Repository not found")
        
        # Setup state
        state = self.test_state.copy()
        state["repository_states"] = {
            "test-repo-1": create_repository_state("test-repo-1", "https://github.com/test/repo1")
        }
        state["repositories"] = ["test-repo-1"]
        
        self.workflow._repo_configs = {
            "test-repo-1": self.test_app_settings["repositories"][0]
        }
        
        with self.assertRaises(ValueError) as context:
            self.workflow._validate_repositories(state)
        
        self.assertIn("No valid repositories found", str(context.exception))
    
    @patch('src.workflows.indexing_workflow.IndexingWorkflow._load_repository_files')
    def test_load_files_from_github(self, mock_load_files):
        """Test loading files from GitHub repositories."""
        # Setup mock return values
        mock_documents = [
            Document(page_content="test content 1", metadata={"file_path": "test1.py"}),
            Document(page_content="test content 2", metadata={"file_path": "test2.py"})
        ]
        mock_load_files.return_value = (mock_documents, len(mock_documents))
        
        # Setup state
        state = self.test_state.copy()
        state["repository_states"] = {
            "test-repo-1": create_repository_state("test-repo-1", "https://github.com/test/repo1")
        }
        state["repositories"] = ["test-repo-1"]
        
        result_state = self.workflow._load_files_from_github(state)
        
        # Verify mock was called
        mock_load_files.assert_called_once_with("test-repo-1", state)
        
        # Verify state updates
        self.assertEqual(result_state["total_files"], 2)
        self.assertEqual(result_state["processed_files"], 2)
        self.assertEqual(len(result_state["metadata"]["loaded_documents"]), 2)
        self.assertEqual(result_state["progress_percentage"], 30.0)
        
        # Verify repository state
        repo_state = result_state["repository_states"]["test-repo-1"]
        self.assertEqual(repo_state["status"], ProcessingStatus.COMPLETED)
        self.assertEqual(repo_state["total_files"], 2)
    
    @patch('src.workflows.indexing_workflow.DocumentProcessor')
    def test_process_documents(self, mock_processor_class):
        """Test document processing step."""
        # Setup mock processor
        mock_processor = Mock()
        mock_chunks = [
            Document(page_content="chunk 1", metadata={"file_path": "test.py", "chunk_type": "function"}),
            Document(page_content="chunk 2", metadata={"file_path": "test.py", "chunk_type": "class"})
        ]
        mock_processor.process_document.return_value = mock_chunks
        mock_processor_class.return_value = mock_processor
        
        # Setup state with loaded documents
        state = self.test_state.copy()
        test_documents = [
            Document(page_content="test content", metadata={"file_path": "test.py", "language": "python"})
        ]
        state["metadata"]["loaded_documents"] = test_documents
        
        result_state = self.workflow._process_documents(state)
        
        # Verify processor was called
        mock_processor.process_document.assert_called_once_with(test_documents[0])
        
        # Verify state updates
        self.assertEqual(len(result_state["metadata"]["processed_documents"]), 2)
        self.assertEqual(result_state["total_chunks"], 2)
        self.assertEqual(len(result_state["file_processing_states"]), 1)
        
        # Verify file processing state
        file_state = result_state["file_processing_states"][0]
        self.assertEqual(file_state["file_path"], "test.py")
        self.assertEqual(file_state["language"], "python")
        self.assertEqual(file_state["status"], ProcessingStatus.COMPLETED)
        self.assertEqual(file_state["chunk_count"], 2)
    
    def test_language_aware_chunking(self):
        """Test language-aware chunking step (validation only)."""
        # Setup state with processed documents
        state = self.test_state.copy()
        processed_docs = [
            Document(page_content="chunk", metadata={"chunk_type": "function", "language": "python"})
        ]
        state["metadata"]["processed_documents"] = processed_docs
        
        result_state = self.workflow._language_aware_chunking(state)
        
        self.assertEqual(result_state["progress_percentage"], 55.0)
        self.assertEqual(result_state["current_step"], IndexingWorkflowSteps.LANGUAGE_AWARE_CHUNKING)
    
    def test_extract_metadata(self):
        """Test metadata extraction step."""
        # Setup state with processed documents
        state = self.test_state.copy()
        processed_docs = [
            Document(
                page_content="chunk 1",
                metadata={
                    "repository": "test-repo-1",
                    "file_path": "test.py",
                    "language": "python",
                    "chunk_type": "function"
                }
            ),
            Document(
                page_content="chunk 2",
                metadata={
                    "repository": "test-repo-2",
                    "file_path": "test.js",
                    "language": "javascript",
                    "chunk_type": "class"
                }
            )
        ]
        state["metadata"]["processed_documents"] = processed_docs
        
        result_state = self.workflow._extract_metadata(state)
        
        # Verify metadata statistics
        metadata_stats = result_state["metadata"]["metadata_stats"]
        self.assertEqual(metadata_stats["total_chunks"], 2)
        self.assertEqual(metadata_stats["chunks_with_metadata"], 2)
        self.assertEqual(set(metadata_stats["repositories"]), {"test-repo-1", "test-repo-2"})
        self.assertEqual(set(metadata_stats["languages"]), {"python", "javascript"})
        self.assertEqual(set(metadata_stats["file_types"]), {".py", ".js"})
        
        self.assertEqual(result_state["progress_percentage"], 60.0)
    
    @patch('src.workflows.indexing_workflow.EmbeddingFactory')
    def test_generate_embeddings(self, mock_embedding_factory_class):
        """Test embedding generation step."""
        # Setup mock embedding provider
        mock_factory = Mock()
        mock_provider = Mock()
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_provider.embed_documents.return_value = mock_embeddings
        mock_factory.create.return_value = mock_provider
        mock_embedding_factory_class.return_value = mock_factory
        
        # Setup state with processed documents
        state = self.test_state.copy()
        processed_docs = [
            Document(page_content="chunk 1", metadata={"file_path": "test1.py"}),
            Document(page_content="chunk 2", metadata={"file_path": "test2.py"})
        ]
        state["metadata"]["processed_documents"] = processed_docs
        
        result_state = self.workflow._generate_embeddings(state)
        
        # Verify embedding provider was called
        mock_provider.embed_documents.assert_called_once_with(["chunk 1", "chunk 2"])
        
        # Verify state updates
        embedded_docs = result_state["metadata"]["embedded_documents"]
        self.assertEqual(len(embedded_docs), 2)
        self.assertEqual(embedded_docs[0].metadata["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(embedded_docs[1].metadata["embedding"], [0.4, 0.5, 0.6])
        
        self.assertEqual(result_state["embeddings_generated"], 2)
        self.assertEqual(result_state["successful_embeddings"], 2)
        self.assertEqual(result_state["failed_embeddings"], 0)
    
    @patch('src.workflows.indexing_workflow.VectorStoreFactory')
    def test_store_in_vector_db(self, mock_vector_store_factory_class):
        """Test vector database storage step."""
        # Setup mock vector store
        mock_factory = Mock()
        mock_store = Mock()
        mock_factory.create.return_value = mock_store
        mock_vector_store_factory_class.return_value = mock_factory
        
        # Setup state with embedded documents
        state = self.test_state.copy()
        embedded_docs = [
            Document(
                page_content="chunk 1",
                metadata={"file_path": "test1.py", "embedding": [0.1, 0.2, 0.3]}
            ),
            Document(
                page_content="chunk 2", 
                metadata={"file_path": "test2.py", "embedding": [0.4, 0.5, 0.6]}
            )
        ]
        state["metadata"]["embedded_documents"] = embedded_docs
        
        result_state = self.workflow._store_in_vector_db(state)
        
        # Verify vector store was called
        mock_store.add_documents.assert_called_once()
        call_args = mock_store.add_documents.call_args
        
        # Check that documents were cleaned (no embedding in metadata)
        stored_docs = call_args[0][0]
        self.assertEqual(len(stored_docs), 2)
        for doc in stored_docs:
            self.assertNotIn("embedding", doc.metadata)
        
        # Check that embeddings were passed separately
        embeddings = call_args[1]["embeddings"]
        self.assertEqual(embeddings, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        # Verify state updates
        storage_stats = result_state["metadata"]["storage_stats"]
        self.assertEqual(storage_stats["stored_documents"], 2)
        self.assertEqual(storage_stats["failed_storage"], 0)
    
    def test_update_workflow_state(self):
        """Test workflow state update step."""
        # Setup state with step durations
        state = self.test_state.copy()
        state["metadata"]["step_durations"] = {
            "step1": 1.0,
            "step2": 2.0,
            "step3": 1.5
        }
        state["processed_files"] = 10
        state["successful_embeddings"] = 50
        
        # Add repository states with start times
        for repo_name in state["repositories"]:
            state["repository_states"][repo_name]["processing_start_time"] = time.time() - 10
        
        result_state = self.workflow._update_workflow_state(state)
        
        # Verify performance metrics
        self.assertEqual(result_state["total_processing_time"], 4.5)
        self.assertAlmostEqual(result_state["documents_per_second"], 10 / 4.5, places=2)
        self.assertAlmostEqual(result_state["embeddings_per_second"], 50 / 4.5, places=2)
        
        # Verify repository states have end times
        for repo_state in result_state["repository_states"].values():
            self.assertIsNotNone(repo_state["processing_end_time"])
            if repo_state.get("processing_start_time"):
                self.assertIsNotNone(repo_state.get("processing_duration"))
    
    def test_check_complete_success(self):
        """Test workflow completion check with successful storage."""
        # Setup state with storage statistics
        state = self.test_state.copy()
        state["metadata"]["storage_stats"] = {
            "stored_documents": 10,
            "failed_storage": 0
        }
        
        result_state = self.workflow._check_complete(state)
        
        self.assertEqual(result_state["progress_percentage"], 99.0)
        self.assertEqual(result_state["current_step"], IndexingWorkflowSteps.CHECK_COMPLETE)
    
    def test_check_complete_no_documents(self):
        """Test workflow completion check with no stored documents."""
        # Setup state with no stored documents
        state = self.test_state.copy()
        state["metadata"]["storage_stats"] = {
            "stored_documents": 0,
            "failed_storage": 5
        }
        
        with self.assertRaises(ValueError) as context:
            self.workflow._check_complete(state)
        
        self.assertIn("No documents were successfully stored", str(context.exception))
    
    def test_finalize_index(self):
        """Test index finalization step."""
        # Setup state with complete statistics
        state = self.test_state.copy()
        state["metadata"]["loaded_documents"] = [Document(page_content="test")]
        state["metadata"]["processed_documents"] = [Document(page_content="test")]
        state["metadata"]["embedded_documents"] = [Document(page_content="test")]
        state["metadata"]["storage_stats"] = {"stored_documents": 5}
        state["processed_files"] = 10
        state["total_chunks"] = 20
        state["successful_embeddings"] = 15
        state["total_processing_time"] = 30.0
        
        result_state = self.workflow._finalize_index(state)
        
        # Verify workflow marked as completed
        self.assertEqual(result_state["status"], ProcessingStatus.COMPLETED)
        self.assertEqual(result_state["progress_percentage"], 100.0)
        
        # Verify temporary data was cleaned up
        self.assertNotIn("loaded_documents", result_state["metadata"])
        self.assertNotIn("processed_documents", result_state["metadata"])
        self.assertNotIn("embedded_documents", result_state["metadata"])
    
    def test_parse_repo_url_https(self):
        """Test parsing HTTPS repository URL."""
        url = "https://github.com/owner/repo"
        owner, name = self.workflow._parse_repo_url(url)
        
        self.assertEqual(owner, "owner")
        self.assertEqual(name, "repo")
    
    def test_parse_repo_url_ssh(self):
        """Test parsing SSH repository URL."""
        url = "git@github.com:owner/repo.git"
        owner, name = self.workflow._parse_repo_url(url)
        
        self.assertEqual(owner, "owner")
        self.assertEqual(name, "repo")
    
    def test_parse_repo_url_invalid(self):
        """Test parsing invalid repository URL."""
        invalid_urls = [
            "https://gitlab.com/owner/repo",
            "https://github.com/owner",
            "invalid-url"
        ]
        
        for url in invalid_urls:
            with self.assertRaises(ValueError):
                self.workflow._parse_repo_url(url)
    
    def test_error_handling_file_errors(self):
        """Test error handling for file loading errors."""
        state = self.test_state.copy()
        state["current_repo"] = "test-repo-1"
        state["repository_states"]["test-repo-1"] = create_repository_state(
            "test-repo-1", "https://github.com/test/repo1"
        )
        
        error = Exception("File not found")
        result_state = self.workflow._handle_file_errors(state, error)
        
        # Verify repository marked as failed
        repo_state = result_state["repository_states"]["test-repo-1"]
        self.assertEqual(repo_state["status"], ProcessingStatus.FAILED)
        self.assertIn("File not found", repo_state["errors"])
    
    def test_error_handling_embedding_errors(self):
        """Test error handling for embedding generation errors."""
        original_batch_size = self.workflow.batch_size
        self.workflow.batch_size = 50
        
        error = Exception("Embedding API error")
        result_state = self.workflow._handle_embedding_errors(self.test_state, error)
        
        # Should reduce batch size
        self.assertEqual(self.workflow.batch_size, 25)
    
    def test_create_indexing_workflow_factory(self):
        """Test workflow factory function."""
        repositories = ["test-repo"]
        workflow_id = "test-workflow-123"
        
        workflow = create_indexing_workflow(
            repositories=repositories,
            workflow_id=workflow_id,
            app_settings_path=self.app_settings_path
        )
        
        self.assertIsInstance(workflow, IndexingWorkflow)
        self.assertEqual(workflow.target_repositories, repositories)
        self.assertEqual(workflow.workflow_id, workflow_id)


class TestIndexingWorkflowIntegration(unittest.TestCase):
    """Integration tests for IndexingWorkflow."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create temporary appSettings.json
        self.temp_dir = tempfile.mkdtemp()
        self.app_settings_path = os.path.join(self.temp_dir, "appSettings.json")
        
        self.test_app_settings = {
            "repositories": [
                {
                    "name": "test-repo",
                    "url": "https://github.com/test/repo",
                    "branch": "main"
                }
            ]
        }
        
        with open(self.app_settings_path, 'w') as f:
            json.dump(self.test_app_settings, f)
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('src.workflows.indexing_workflow.GitHubLoader')
    @patch('src.workflows.indexing_workflow.DocumentProcessor')
    @patch('src.workflows.indexing_workflow.EmbeddingFactory')
    @patch('src.workflows.indexing_workflow.VectorStoreFactory')
    def test_full_workflow_execution(self, mock_vector_factory, mock_embed_factory, 
                                   mock_doc_processor, mock_loader_class):
        """Test complete workflow execution end-to-end."""
        # Setup mocks
        # GitHub loader
        mock_loader = Mock()
        mock_documents = [
            Document(
                page_content="def test(): pass",
                metadata={"file_path": "test.py", "language": "python"}
            )
        ]
        mock_loader.load.return_value = mock_documents
        mock_loader.get_repository_info.return_value = {"name": "test-repo"}
        mock_loader_class.return_value = mock_loader
        
        # Document processor
        mock_processor = Mock()
        mock_chunks = [
            Document(
                page_content="def test(): pass",
                metadata={
                    "file_path": "test.py",
                    "language": "python",
                    "chunk_type": "function",
                    "repository": "test-repo"
                }
            )
        ]
        mock_processor.process_document.return_value = mock_chunks
        mock_doc_processor.return_value = mock_processor
        
        # Embedding factory
        mock_embedding_provider = Mock()
        mock_embedding_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_embed_factory_instance = Mock()
        mock_embed_factory_instance.create.return_value = mock_embedding_provider
        mock_embed_factory.return_value = mock_embed_factory_instance
        
        # Vector store factory
        mock_vector_store = Mock()
        mock_vector_factory_instance = Mock()
        mock_vector_factory_instance.create.return_value = mock_vector_store
        mock_vector_factory.return_value = mock_vector_factory_instance
        
        # Create and execute workflow
        workflow = IndexingWorkflow(
            repositories=["test-repo"],
            app_settings_path=self.app_settings_path,
            max_retries=1,
            retry_delay=0.1
        )
        
        initial_state = create_indexing_state(
            workflow_id="test-integration",
            repositories=["test-repo"]
        )
        
        final_state = workflow.invoke(initial_state)
        
        # Verify workflow completed successfully
        self.assertEqual(final_state["status"], ProcessingStatus.COMPLETED)
        self.assertEqual(final_state["progress_percentage"], 100.0)
        
        # Verify all components were called
        mock_loader.get_repository_info.assert_called_once()
        mock_loader.load.assert_called_once()
        mock_processor.process_document.assert_called_once()
        mock_embedding_provider.embed_documents.assert_called_once()
        mock_vector_store.add_documents.assert_called_once()
        
        # Verify final statistics
        self.assertEqual(final_state["processed_files"], 1)
        self.assertEqual(final_state["total_chunks"], 1)
        self.assertEqual(final_state["successful_embeddings"], 1)


if __name__ == "__main__":
    unittest.main()
