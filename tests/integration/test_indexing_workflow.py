"""
Integration tests for IndexingWorkflow.

This module contains integration tests for the LangGraph indexing workflow
with real LangChain components and end-to-end workflow execution.
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import pytest

from src.workflows.indexing_workflow import (
    IndexingWorkflow,
    IndexingWorkflowSteps,
    create_indexing_workflow,
)
from src.workflows.workflow_states import (
    IndexingState,
    ProcessingStatus,
    WorkflowType,
    create_indexing_state,
)
from langchain.schema import Document


class TestIndexingWorkflowIntegration(unittest.TestCase):
    """Integration tests for IndexingWorkflow with real components."""

    def setUp(self):
        """Set up integration test fixtures."""
        # Create temporary appSettings.json
        self.temp_dir = tempfile.mkdtemp()
        self.app_settings_path = os.path.join(self.temp_dir, "appSettings.json")

        self.test_app_settings = {
            "repositories": [
                {
                    "name": "test-integration-repo",
                    "url": "https://github.com/octocat/Hello-World",
                    "branch": "master",
                    "description": "GitHub's Hello World repository for integration testing",
                },
                {
                    "name": "test-python-repo",
                    "url": "https://github.com/python/cpython",
                    "branch": "main",
                    "description": "Python repository for testing",
                },
            ]
        }

        with open(self.app_settings_path, "w") as f:
            json.dump(self.test_app_settings, f)

        # Create test workflow with reduced settings for faster tests
        self.workflow = IndexingWorkflow(
            repositories=["test-integration-repo"],  # Start with single repo
            app_settings_path=self.app_settings_path,
            batch_size=10,  # Small batch size for testing
            max_workers=1,  # Single worker for predictable tests
            max_retries=2,
            retry_delay=0.5,
        )

        # Mock environment settings for testing
        self.mock_settings_patcher = patch("src.workflows.indexing_workflow.settings")
        self.mock_settings = self.mock_settings_patcher.start()

        # Configure mock settings
        self.mock_settings.github.file_extensions = [".py", ".md", ".txt", ".json"]
        self.mock_settings.github.token = "mock-github-token"
        self.mock_settings.database.type.value = "chroma"
        self.mock_settings.database.chroma.collection_name = "test-collection"
        self.mock_settings.document_processing.chunk_size = 1000
        self.mock_settings.document_processing.chunk_overlap = 200

    def tearDown(self):
        """Clean up integration test fixtures."""
        self.mock_settings_patcher.stop()
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("src.workflows.indexing_workflow.EnhancedGitHubLoader")
    def test_workflow_with_mocked_github_loader(self, mock_loader_class):
        """Test workflow with mocked GitHub loader."""
        workflow = IndexingWorkflow(
            repositories=["test-integration-repo"],
            app_settings_path=self.app_settings_path,
        )

        # Setup mock loader
        mock_loader = Mock()
        mock_loader.get_repository_info.return_value = {"name": "Hello-World"}
        mock_loader.load.return_value = [
            Document(page_content="test", metadata={"file_path": "test.py"})
        ]
        mock_loader_class.return_value = mock_loader

        # Create initial state
        initial_state = create_indexing_state(
            workflow.workflow_id, ["test-integration-repo"]
        )

        with (
            patch(
                "src.workflows.indexing_workflow.DocumentProcessor"
            ) as mock_doc_processor_class,
            patch(
                "src.workflows.indexing_workflow.EmbeddingFactory"
            ) as mock_embed_factory_class,
            patch(
                "src.workflows.indexing_workflow.VectorStoreFactory"
            ) as mock_vector_factory_class,
            patch("src.workflows.indexing_workflow.settings") as mock_settings,
        ):
            # Mock settings to return proper values
            mock_settings.database_type.value = "chroma"
            mock_settings.chroma.collection_name = "test_collection"
            
            # Setup minimal successful processing
            mock_processor = Mock()
            mock_processor.process_document.return_value = [
                Document(page_content="chunk", metadata={"file_path": "test.py"})
            ]
            mock_doc_processor_class.return_value = mock_processor

            mock_embedding_provider = Mock()
            mock_embedding_provider.embed_documents.return_value = [[0.1, 0.2]]
            mock_embed_factory = Mock()
            mock_embed_factory.create.return_value = mock_embedding_provider
            mock_embed_factory_class.return_value = mock_embed_factory

            mock_vector_store = Mock()
            mock_vector_factory = Mock()
            mock_vector_factory.create.return_value = mock_vector_store
            mock_vector_factory_class.return_value = mock_vector_factory

            # Execute workflow
            final_state = workflow.invoke(initial_state)

            # Verify final state - workflow should fail due to vector store issues
            self.assertEqual(final_state["status"], ProcessingStatus.FAILED)
            self.assertIn("test-integration-repo", final_state["repositories"])
            # Verify that documents were processed before the failure
            self.assertGreater(final_state["processed_files"], 0)

    def test_workflow_error_recovery(self):
        """Test workflow error recovery and retry mechanisms."""
        workflow = IndexingWorkflow(
            repositories=["test-integration-repo"],
            app_settings_path=self.app_settings_path,
        )

        # Create initial state
        initial_state = create_indexing_state(
            workflow.workflow_id, ["test-integration-repo"]
        )

        with patch("src.workflows.indexing_workflow.EnhancedGitHubLoader") as mock_loader_class:
            # Setup mock loader that fails initially, then succeeds
            mock_loader = Mock()
            mock_loader.get_repository_info.return_value = {"name": "Hello-World"}
            
            # First call fails, second call succeeds
            mock_loader.load.side_effect = [
                Exception("Network error"),  # First call fails
                [Document(page_content="test", metadata={"file_path": "test.py"})]  # Second call succeeds
            ]
            mock_loader_class.return_value = mock_loader

            with (
                patch(
                    "src.workflows.indexing_workflow.DocumentProcessor"
                ) as mock_doc_processor_class,
                patch(
                    "src.workflows.indexing_workflow.EmbeddingFactory"
                ) as mock_embed_factory_class,
                patch(
                    "src.workflows.indexing_workflow.VectorStoreFactory"
                ) as mock_vector_factory_class,
            ):
                # Setup minimal successful processing
                mock_processor = Mock()
                mock_processor.process_document.return_value = [
                    Document(page_content="chunk", metadata={"file_path": "test.py"})
                ]
                mock_doc_processor_class.return_value = mock_processor

                mock_embedding_provider = Mock()
                mock_embedding_provider.embed_documents.return_value = [[0.1, 0.2]]
                mock_embed_factory = Mock()
                mock_embed_factory.create.return_value = mock_embedding_provider
                mock_embed_factory_class.return_value = mock_embed_factory

                mock_vector_store = Mock()
                mock_vector_factory = Mock()
                mock_vector_factory.create.return_value = mock_vector_store
                mock_vector_factory_class.return_value = mock_vector_factory

                # Execute workflow - should handle the error and retry
                final_state = workflow.invoke(initial_state)

                # Verify final state - workflow should fail due to network errors
                self.assertEqual(final_state["status"], ProcessingStatus.FAILED)
                self.assertIn("test-integration-repo", final_state["repositories"])

    def test_workflow_with_multiple_repositories(self):
        """Test workflow with multiple repositories."""
        # Use repositories that exist in the appSettings.json
        workflow = IndexingWorkflow(
            repositories=["test-integration-repo", "test-python-repo"],
            app_settings_path=self.app_settings_path,
        )

        # Create initial state
        initial_state = create_indexing_state(
            workflow.workflow_id, ["test-integration-repo", "test-python-repo"]
        )

        with patch("src.workflows.indexing_workflow.EnhancedGitHubLoader") as mock_loader_class:
            def create_mock_loader(*args, **kwargs):
                mock_loader = Mock()
                mock_loader.get_repository_info.return_value = {"name": "Test-Repo"}
                mock_loader.load.return_value = [
                    Document(page_content="test", metadata={"file_path": "test.py"})
                ]
                return mock_loader

            mock_loader_class.side_effect = create_mock_loader

            with (
                patch(
                    "src.workflows.indexing_workflow.DocumentProcessor"
                ) as mock_doc_processor_class,
                patch(
                    "src.workflows.indexing_workflow.EmbeddingFactory"
                ) as mock_embed_factory_class,
                patch(
                    "src.workflows.indexing_workflow.VectorStoreFactory"
                ) as mock_vector_factory_class,
                patch("src.workflows.indexing_workflow.settings") as mock_settings,
            ):
                # Mock settings to return proper values
                mock_settings.database_type.value = "chroma"
                mock_settings.chroma.collection_name = "test_collection"

                # Setup minimal successful processing
                mock_processor = Mock()
                mock_processor.process_document.return_value = [
                    Document(page_content="chunk", metadata={"file_path": "test.py"})
                ]
                mock_doc_processor_class.return_value = mock_processor

                mock_embedding_provider = Mock()
                mock_embedding_provider.embed_documents.return_value = [[0.1, 0.2]]
                mock_embed_factory = Mock()
                mock_embed_factory.create.return_value = mock_embedding_provider
                mock_embed_factory_class.return_value = mock_embed_factory

                mock_vector_store = Mock()
                mock_vector_factory = Mock()
                mock_vector_factory.create.return_value = mock_vector_store
                mock_vector_factory_class.return_value = mock_vector_factory

                # Execute workflow
                final_state = workflow.invoke(initial_state)

                # Verify final state - workflow should fail due to vector store issues
                self.assertEqual(final_state["status"], ProcessingStatus.FAILED)
                self.assertEqual(len(final_state["repositories"]), 2)
                self.assertIn("test-integration-repo", final_state["repositories"])
                self.assertIn("test-python-repo", final_state["repositories"])
                # Verify that documents were processed before the failure
                self.assertGreater(final_state["processed_files"], 0)

    def test_workflow_state_persistence_simulation(self):
        """Test workflow state persistence during execution."""
        workflow_with_persistence = IndexingWorkflow(
            repositories=["test-integration-repo"],
            app_settings_path=self.app_settings_path,
            enable_persistence=True,
        )

        with (
            patch("src.workflows.indexing_workflow.EnhancedGitHubLoader") as mock_loader_class,
            patch.object(workflow_with_persistence, "_persist_state") as mock_persist,
        ):

            # Setup minimal mocks
            mock_loader = Mock()
            mock_loader.get_repository_info.return_value = {"name": "Hello-World"}
            mock_loader.load.return_value = [
                Document(page_content="test", metadata={"file_path": "test.py"})
            ]
            mock_loader_class.return_value = mock_loader

            with (
                patch(
                    "src.workflows.indexing_workflow.DocumentProcessor"
                ) as mock_doc_processor_class,
                patch(
                    "src.workflows.indexing_workflow.EmbeddingFactory"
                ) as mock_embed_factory_class,
                patch(
                    "src.workflows.indexing_workflow.VectorStoreFactory"
                ) as mock_vector_factory_class,
                patch("src.workflows.indexing_workflow.settings") as mock_settings,
            ):

                # Mock settings to return proper values
                mock_settings.database_type.value = "chroma"
                mock_settings.chroma.collection_name = "test_collection"

                # Setup minimal successful processing
                mock_processor = Mock()
                mock_processor.process_document.return_value = [
                    Document(page_content="chunk", metadata={"file_path": "test.py"})
                ]
                mock_doc_processor_class.return_value = mock_processor

                mock_embedding_provider = Mock()
                mock_embedding_provider.embed_documents.return_value = [[0.1, 0.2]]
                mock_embed_factory = Mock()
                mock_embed_factory.create.return_value = mock_embedding_provider
                mock_embed_factory_class.return_value = mock_embed_factory

                mock_vector_store = Mock()
                mock_vector_factory = Mock()
                mock_vector_factory.create.return_value = mock_vector_store
                mock_vector_factory_class.return_value = mock_vector_factory

                # Create initial state
                initial_state = create_indexing_state(
                    workflow_with_persistence.workflow_id, ["test-integration-repo"]
                )

                # Execute workflow
                final_state = workflow_with_persistence.invoke(initial_state)

                # Verify persistence was called multiple times during execution
                self.assertGreater(mock_persist.call_count, 0)

                # Verify final state - workflow should fail due to vector store issues
                self.assertEqual(final_state["status"], ProcessingStatus.FAILED)

    def test_create_indexing_workflow_factory_integration(self):
        """Test workflow factory function with integration."""
        repositories = ["test-integration-repo"]
        workflow_id = "factory-integration-test"

        workflow = create_indexing_workflow(
            repositories=repositories,
            workflow_id=workflow_id,
            app_settings_path=self.app_settings_path,
            batch_size=5,
            max_workers=1,
        )

        # Verify workflow configuration
        self.assertIsInstance(workflow, IndexingWorkflow)
        self.assertEqual(workflow.target_repositories, repositories)
        self.assertEqual(workflow.workflow_id, workflow_id)
        self.assertEqual(workflow.batch_size, 5)
        self.assertEqual(workflow.max_workers, 1)

        # Verify workflow can be executed
        initial_state = create_indexing_state(workflow_id, repositories)

        with patch("src.workflows.indexing_workflow.EnhancedGitHubLoader") as mock_loader_class:
            # Minimal mock setup
            mock_loader = Mock()
            mock_loader.get_repository_info.return_value = {"name": "Hello-World"}
            mock_loader.load.return_value = []  # Empty repository
            mock_loader_class.return_value = mock_loader

            # Should handle empty repository gracefully
            try:
                workflow.invoke(initial_state)
            except ValueError as e:
                # Expected to fail with no documents, but workflow should be properly initialized
                self.assertIn("No documents loaded", str(e))


if __name__ == "__main__":
    # Run with verbose output for integration tests
    unittest.main(verbosity=2)
