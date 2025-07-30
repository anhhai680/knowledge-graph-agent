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
    create_indexing_workflow
)
from src.workflows.workflow_states import (
    IndexingState,
    ProcessingStatus,
    WorkflowType,
    create_indexing_state
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
                    "description": "GitHub's Hello World repository for integration testing"
                },
                {
                    "name": "test-python-repo",
                    "url": "https://github.com/python/cpython",
                    "branch": "main", 
                    "description": "Python repository for testing"
                }
            ]
        }
        
        with open(self.app_settings_path, 'w') as f:
            json.dump(self.test_app_settings, f)
        
        # Create test workflow with reduced settings for faster tests
        self.workflow = IndexingWorkflow(
            repositories=["test-integration-repo"],  # Start with single repo
            app_settings_path=self.app_settings_path,
            batch_size=10,  # Small batch size for testing
            max_workers=1,  # Single worker for predictable tests
            max_retries=2,
            retry_delay=0.5
        )
        
        # Mock environment settings for testing
        self.mock_settings_patcher = patch('src.workflows.indexing_workflow.settings')
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
    
    @patch('src.workflows.indexing_workflow.GitHubLoader')
    def test_workflow_with_mocked_github_loader(self, mock_loader_class):
        """Test workflow execution with mocked GitHub loader."""
        # Setup comprehensive mock GitHub loader
        mock_loader = Mock()
        
        # Mock repository info for validation
        mock_loader.get_repository_info.return_value = {
            "name": "Hello-World",
            "full_name": "octocat/Hello-World",
            "description": "My first repository on GitHub!"
        }
        
        # Mock document loading with realistic documents
        mock_documents = [
            Document(
                page_content="""
def hello_world():
    \"\"\"Print hello world message.\"\"\"
    print("Hello, World!")
    return "success"

if __name__ == "__main__":
    hello_world()
                """.strip(),
                metadata={
                    "file_path": "hello.py",
                    "language": "python",
                    "repository": "test-integration-repo",
                    "url": "https://github.com/octocat/Hello-World/blob/master/hello.py",
                    "commit_sha": "abc123",
                    "size": 150
                }
            ),
            Document(
                page_content="""
# Hello World

This is a simple Hello World repository for testing purposes.

## Features

- Simple Python script
- Basic documentation
- Example for integration testing

## Usage

```python
python hello.py
```
                """.strip(),
                metadata={
                    "file_path": "README.md",
                    "language": "markdown",
                    "repository": "test-integration-repo",
                    "url": "https://github.com/octocat/Hello-World/blob/master/README.md",
                    "commit_sha": "abc123",
                    "size": 300
                }
            ),
            Document(
                page_content="""
{
    "name": "hello-world",
    "version": "1.0.0",
    "description": "A simple hello world project",
    "main": "hello.py",
    "scripts": {
        "start": "python hello.py"
    },
    "author": "octocat",
    "license": "MIT"
}
                """.strip(),
                metadata={
                    "file_path": "package.json",
                    "language": "json",
                    "repository": "test-integration-repo",
                    "url": "https://github.com/octocat/Hello-World/blob/master/package.json",
                    "commit_sha": "abc123",
                    "size": 200
                }
            )
        ]
        
        mock_loader.load.return_value = mock_documents
        mock_loader_class.return_value = mock_loader
        
        # Create initial state
        initial_state = create_indexing_state(
            workflow_id="integration-test-001",
            repositories=["test-integration-repo"],
            vector_store_type="chroma",
            collection_name="test-integration-collection"
        )
        
        # Execute workflow
        with patch('src.workflows.indexing_workflow.DocumentProcessor') as mock_doc_processor_class, \
             patch('src.workflows.indexing_workflow.EmbeddingFactory') as mock_embed_factory_class, \
             patch('src.workflows.indexing_workflow.VectorStoreFactory') as mock_vector_factory_class:
            
            # Setup document processor mock
            mock_processor = Mock()
            mock_chunks = []
            for i, doc in enumerate(mock_documents):
                # Simulate chunking each document into 2 chunks
                for j in range(2):
                    chunk = Document(
                        page_content=f"Chunk {j+1} of {doc.page_content[:50]}...",
                        metadata={
                            **doc.metadata,
                            "chunk_index": j,
                            "chunk_type": "content" if j == 0 else "continuation",
                            "line_start": j * 10 + 1,
                            "line_end": (j + 1) * 10,
                            "tokens": 25 + j * 5
                        }
                    )
                    mock_chunks.append(chunk)
            
            mock_processor.process_document.side_effect = lambda doc: [
                chunk for chunk in mock_chunks 
                if chunk.metadata["file_path"] == doc.metadata["file_path"]
            ]
            mock_doc_processor_class.return_value = mock_processor
            
            # Setup embedding factory mock
            mock_embedding_provider = Mock()
            # Generate realistic embeddings (384-dimensional for testing)
            embedding_dim = 384
            mock_embeddings = [
                [0.1 + i * 0.01 + j * 0.001 for j in range(embedding_dim)]
                for i in range(len(mock_chunks))
            ]
            mock_embedding_provider.embed_documents.return_value = mock_embeddings
            
            mock_embed_factory = Mock()
            mock_embed_factory.create.return_value = mock_embedding_provider
            mock_embed_factory_class.return_value = mock_embed_factory
            
            # Setup vector store factory mock
            mock_vector_store = Mock()
            mock_vector_store.add_documents.return_value = None  # Successful storage
            
            mock_vector_factory = Mock()
            mock_vector_factory.create.return_value = mock_vector_store
            mock_vector_factor_class = mock_vector_factory_class
            mock_vector_factor_class.return_value = mock_vector_factory
            
            # Execute workflow
            start_time = time.time()
            final_state = self.workflow.invoke(initial_state)
            execution_time = time.time() - start_time
            
            # Verify workflow completed successfully
            self.assertEqual(final_state["status"], ProcessingStatus.COMPLETED)
            self.assertEqual(final_state["progress_percentage"], 100.0)
            self.assertIsNone(final_state.get("current_step"))  # Should be None when completed
            
            # Verify repository processing
            self.assertEqual(len(final_state["repositories"]), 1)
            self.assertEqual(final_state["repositories"][0], "test-integration-repo")
            
            repo_state = final_state["repository_states"]["test-integration-repo"]
            self.assertEqual(repo_state["status"], ProcessingStatus.COMPLETED)
            self.assertEqual(repo_state["total_files"], 3)
            self.assertEqual(repo_state["processed_files"], 3)
            
            # Verify document processing statistics
            self.assertEqual(final_state["processed_files"], 3)
            self.assertEqual(final_state["total_chunks"], 6)  # 3 docs * 2 chunks each
            self.assertEqual(final_state["successful_embeddings"], 6)
            self.assertEqual(final_state["failed_embeddings"], 0)
            
            # Verify file processing states
            self.assertEqual(len(final_state["file_processing_states"]), 3)
            
            for file_state in final_state["file_processing_states"]:
                self.assertEqual(file_state["status"], ProcessingStatus.COMPLETED)
                self.assertEqual(file_state["chunk_count"], 2)
                self.assertIsNotNone(file_state["processing_time"])
                self.assertIn(file_state["file_path"], ["hello.py", "README.md", "package.json"])
                self.assertIn(file_state["language"], ["python", "markdown", "json"])
            
            # Verify metadata statistics
            metadata_stats = final_state["metadata"]["metadata_stats"]
            self.assertEqual(metadata_stats["total_chunks"], 6)
            self.assertEqual(metadata_stats["chunks_with_metadata"], 6)
            self.assertEqual(metadata_stats["repositories"], ["test-integration-repo"])
            self.assertIn("python", metadata_stats["languages"])
            self.assertIn("markdown", metadata_stats["languages"])
            self.assertIn("json", metadata_stats["languages"])
            
            # Verify embeddings statistics
            embeddings_stats = final_state["metadata"]["embeddings_stats"]
            self.assertEqual(embeddings_stats["total_documents"], 6)
            self.assertEqual(embeddings_stats["successful_embeddings"], 6)
            self.assertEqual(embeddings_stats["failed_embeddings"], 0)
            self.assertEqual(embeddings_stats["batch_count"], 1)  # All fit in one batch
            
            # Verify storage statistics
            storage_stats = final_state["metadata"]["storage_stats"]
            self.assertEqual(storage_stats["total_documents"], 6)
            self.assertEqual(storage_stats["stored_documents"], 6)
            self.assertEqual(storage_stats["failed_storage"], 0)
            self.assertEqual(storage_stats["batch_count"], 1)
            
            # Verify performance metrics
            self.assertIsNotNone(final_state["total_processing_time"])
            if final_state["total_processing_time"] is not None:
                self.assertGreater(final_state["total_processing_time"], 0.0)
            self.assertIsNotNone(final_state.get("documents_per_second"))
            self.assertIsNotNone(final_state.get("embeddings_per_second"))
            
            # Verify workflow metadata
            workflow_metadata = self.workflow.get_metadata()
            self.assertEqual(workflow_metadata["status"], "completed")
            self.assertEqual(workflow_metadata["progress_percentage"], 100.0)
            self.assertGreater(len(workflow_metadata["executed_steps"]), 0)
            self.assertIsNotNone(workflow_metadata["duration"])
            
            # Verify component calls
            mock_loader.get_repository_info.assert_called_once()
            mock_loader.load.assert_called_once()
            self.assertEqual(mock_processor.process_document.call_count, 3)
            mock_embedding_provider.embed_documents.assert_called_once()
            mock_vector_store.add_documents.assert_called_once()
            
            # Log performance information
            print(f"\\nIntegration Test Performance:")
            print(f"Execution time: {execution_time:.2f}s")
            print(f"Files processed: {final_state['processed_files']}")
            print(f"Chunks created: {final_state['total_chunks']}")
            print(f"Embeddings generated: {final_state['successful_embeddings']}")
            print(f"Processing rate: {final_state.get('documents_per_second', 0):.2f} docs/sec")
            print(f"Embedding rate: {final_state.get('embeddings_per_second', 0):.2f} embeddings/sec")
    
    def test_workflow_error_recovery(self):
        """Test workflow error recovery mechanisms."""
        initial_state = create_indexing_state(
            workflow_id="error-recovery-test",
            repositories=["test-integration-repo"]
        )
        
        with patch('src.workflows.indexing_workflow.GitHubLoader') as mock_loader_class:
            # Setup loader that fails on first call, succeeds on retry
            mock_loader = Mock()
            mock_loader.get_repository_info.side_effect = [
                ConnectionError("Network timeout"),  # First call fails
                {"name": "Hello-World"}  # Second call succeeds
            ]
            
            mock_documents = [
                Document(
                    page_content="test content",
                    metadata={"file_path": "test.py", "repository": "test-integration-repo"}
                )
            ]
            mock_loader.load.return_value = mock_documents
            mock_loader_class.return_value = mock_loader
            
            # Mock other components for successful processing
            with patch('src.workflows.indexing_workflow.DocumentProcessor') as mock_doc_processor_class, \
                 patch('src.workflows.indexing_workflow.EmbeddingFactory') as mock_embed_factory_class, \
                 patch('src.workflows.indexing_workflow.VectorStoreFactory') as mock_vector_factory_class:
                
                # Setup mocks for successful processing after retry
                mock_processor = Mock()
                mock_processor.process_document.return_value = [
                    Document(page_content="chunk", metadata={"file_path": "test.py", "chunk_type": "content"})
                ]
                mock_doc_processor_class.return_value = mock_processor
                
                mock_embedding_provider = Mock()
                mock_embedding_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]
                mock_embed_factory = Mock()
                mock_embed_factory.create.return_value = mock_embedding_provider
                mock_embed_factory_class.return_value = mock_embed_factory
                
                mock_vector_store = Mock()
                mock_vector_factory = Mock()
                mock_vector_factory.create.return_value = mock_vector_store
                mock_vector_factory_class.return_value = mock_vector_factory
                
                # Execute workflow - should succeed after retry
                final_state = self.workflow.invoke(initial_state)
                
                # Verify workflow completed despite initial error
                self.assertEqual(final_state["status"], ProcessingStatus.COMPLETED)
                
                # Verify retry occurred
                self.assertEqual(mock_loader.get_repository_info.call_count, 2)
                
                # Verify error was logged but workflow succeeded
                self.assertGreater(len(final_state["errors"]), 0)
                
                # Verify workflow metadata shows retry
                workflow_metadata = self.workflow.get_metadata()
                self.assertGreater(workflow_metadata["retry_count"], 0)
    
    def test_workflow_with_multiple_repositories(self):
        """Test workflow execution with multiple repositories."""
        # Update workflow for multiple repositories
        multi_repo_workflow = IndexingWorkflow(
            repositories=["test-integration-repo", "test-python-repo"],
            app_settings_path=self.app_settings_path,
            batch_size=5,
            max_workers=2  # Test parallel processing
        )
        
        initial_state = create_indexing_state(
            workflow_id="multi-repo-test",
            repositories=["test-integration-repo", "test-python-repo"]
        )
        
        with patch('src.workflows.indexing_workflow.GitHubLoader') as mock_loader_class:
            # Setup different responses for different repositories
            def create_mock_loader(*args, **kwargs):
                mock_loader = Mock()
                repo_name = kwargs.get('repo_name', 'unknown')
                
                if repo_name == 'Hello-World':
                    mock_loader.get_repository_info.return_value = {"name": "Hello-World"}
                    mock_loader.load.return_value = [
                        Document(
                            page_content="hello world code",
                            metadata={"file_path": "hello.py", "language": "python"}
                        )
                    ]
                elif repo_name == 'cpython':
                    mock_loader.get_repository_info.return_value = {"name": "cpython"}
                    mock_loader.load.return_value = [
                        Document(
                            page_content="python source code",
                            metadata={"file_path": "main.c", "language": "c"}
                        ),
                        Document(
                            page_content="python documentation", 
                            metadata={"file_path": "README.rst", "language": "rst"}
                        )
                    ]
                
                return mock_loader
            
            mock_loader_class.side_effect = create_mock_loader
            
            # Mock other components
            with patch('src.workflows.indexing_workflow.DocumentProcessor') as mock_doc_processor_class, \
                 patch('src.workflows.indexing_workflow.EmbeddingFactory') as mock_embed_factory_class, \
                 patch('src.workflows.indexing_workflow.VectorStoreFactory') as mock_vector_factory_class:
                
                # Setup mocks
                mock_processor = Mock()
                mock_processor.process_document.side_effect = lambda doc: [
                    Document(
                        page_content=f"chunk of {doc.page_content[:20]}",
                        metadata={**doc.metadata, "chunk_type": "content"}
                    )
                ]
                mock_doc_processor_class.return_value = mock_processor
                
                mock_embedding_provider = Mock()
                mock_embedding_provider.embed_documents.side_effect = lambda texts: [
                    [i * 0.1 + j * 0.01 for j in range(10)] for i in range(len(texts))
                ]
                mock_embed_factory = Mock()
                mock_embed_factory.create.return_value = mock_embedding_provider
                mock_embed_factory_class.return_value = mock_embed_factory
                
                mock_vector_store = Mock()
                mock_vector_factory = Mock()
                mock_vector_factory.create.return_value = mock_vector_store
                mock_vector_factory_class.return_value = mock_vector_factory
                
                # Execute workflow
                final_state = multi_repo_workflow.invoke(initial_state)
                
                # Verify both repositories processed
                self.assertEqual(final_state["status"], ProcessingStatus.COMPLETED)
                self.assertEqual(len(final_state["repositories"]), 2)
                
                # Verify repository states
                for repo_name in ["test-integration-repo", "test-python-repo"]:
                    self.assertIn(repo_name, final_state["repository_states"])
                    repo_state = final_state["repository_states"][repo_name]
                    self.assertEqual(repo_state["status"], ProcessingStatus.COMPLETED)
                    self.assertGreater(repo_state["total_files"], 0)
                
                # Verify total statistics
                self.assertEqual(final_state["processed_files"], 3)  # 1 + 2 files
                self.assertEqual(final_state["total_chunks"], 3)  # 1 chunk per file
                self.assertEqual(final_state["successful_embeddings"], 3)
                
                # Verify parallel processing occurred (both loaders called)
                self.assertEqual(mock_loader_class.call_count, 2)
    
    def test_workflow_state_persistence_simulation(self):
        """Test workflow state persistence capabilities."""
        initial_state = create_indexing_state(
            workflow_id="persistence-test",
            repositories=["test-integration-repo"]
        )
        
        # Enable persistence
        workflow_with_persistence = IndexingWorkflow(
            repositories=["test-integration-repo"],
            app_settings_path=self.app_settings_path,
            enable_persistence=True
        )
        
        with patch('src.workflows.indexing_workflow.GitHubLoader') as mock_loader_class, \
             patch.object(workflow_with_persistence, '_persist_state') as mock_persist:
            
            # Setup minimal mocks
            mock_loader = Mock()
            mock_loader.get_repository_info.return_value = {"name": "Hello-World"}
            mock_loader.load.return_value = [
                Document(page_content="test", metadata={"file_path": "test.py"})
            ]
            mock_loader_class.return_value = mock_loader
            
            with patch('src.workflows.indexing_workflow.DocumentProcessor') as mock_doc_processor_class, \
                 patch('src.workflows.indexing_workflow.EmbeddingFactory') as mock_embed_factory_class, \
                 patch('src.workflows.indexing_workflow.VectorStoreFactory') as mock_vector_factory_class:
                
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
                final_state = workflow_with_persistence.invoke(initial_state)
                
                # Verify persistence was called multiple times during execution
                self.assertGreater(mock_persist.call_count, 0)
                
                # Verify final state
                self.assertEqual(final_state["status"], ProcessingStatus.COMPLETED)
    
    def test_create_indexing_workflow_factory_integration(self):
        """Test workflow factory function with integration."""
        repositories = ["test-integration-repo"]
        workflow_id = "factory-integration-test"
        
        workflow = create_indexing_workflow(
            repositories=repositories,
            workflow_id=workflow_id,
            app_settings_path=self.app_settings_path,
            batch_size=5,
            max_workers=1
        )
        
        # Verify workflow configuration
        self.assertIsInstance(workflow, IndexingWorkflow)
        self.assertEqual(workflow.target_repositories, repositories)
        self.assertEqual(workflow.workflow_id, workflow_id)
        self.assertEqual(workflow.batch_size, 5)
        self.assertEqual(workflow.max_workers, 1)
        
        # Verify workflow can be executed
        initial_state = create_indexing_state(workflow_id, repositories)
        
        with patch('src.workflows.indexing_workflow.GitHubLoader') as mock_loader_class:
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
