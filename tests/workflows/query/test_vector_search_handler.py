"""
Unit tests for VectorSearchHandler.

Tests the vector search functionality of the modular vector search handler.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from src.workflows.query.handlers.vector_search_handler import VectorSearchHandler
from src.workflows.workflow_states import (
    QueryState, 
    SearchStrategy, 
    create_query_state,
    ProcessingStatus
)


class TestVectorSearchHandler:
    """Test suite for VectorSearchHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = VectorSearchHandler(collection_name="test-collection")

    def test_define_steps(self):
        """Test that handler defines correct steps."""
        steps = self.handler.define_steps()
        expected_steps = ["extract_filters", "perform_search", "process_results"]
        assert steps == expected_steps

    def test_validate_state_valid(self):
        """Test state validation with valid state."""
        state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        state["processed_query"] = "test query"
        assert self.handler.validate_state(state) is True

    def test_validate_state_invalid(self):
        """Test state validation with invalid state."""
        state = create_query_state(workflow_id="test-123", original_query="")
        assert self.handler.validate_state(state) is False

    def test_extract_filters_python(self):
        """Test filter extraction for Python queries."""
        filters = self.handler._extract_filters_from_query("show me python functions")
        assert "language" in filters
        assert "python" in filters["language"]["$in"]

    def test_extract_filters_javascript(self):
        """Test filter extraction for JavaScript queries."""
        filters = self.handler._extract_filters_from_query("javascript code examples")
        assert "language" in filters
        assert "javascript" in filters["language"]["$in"]

    def test_extract_filters_file_types(self):
        """Test filter extraction for file types."""
        filters = self.handler._extract_filters_from_query("show me .py files")
        assert "file_extension" in filters
        assert ".py" in filters["file_extension"]["$in"]

    def test_extract_filters_multiple(self):
        """Test filter extraction with multiple criteria."""
        filters = self.handler._extract_filters_from_query("python .py typescript .ts")
        assert "language" in filters
        assert "file_extension" in filters
        assert "python" in filters["language"]["$in"]
        assert "typescript" in filters["language"]["$in"]
        assert ".py" in filters["file_extension"]["$in"]
        assert ".ts" in filters["file_extension"]["$in"]

    def test_extract_filters_step(self):
        """Test filter extraction step."""
        state = create_query_state(
            workflow_id="test-123",
            original_query="python functions"
        )
        state["processed_query"] = "python functions"
        
        result = self.handler.execute_step("extract_filters", state)
        
        assert "search_filters" in result
        assert "language" in result["search_filters"]

    @patch('src.workflows.query.handlers.vector_search_handler.VectorSearchHandler._get_vector_store')
    def test_perform_search_step(self, mock_get_vector_store):
        """Test vector search step."""
        # Mock vector store
        mock_vector_store = Mock()
        mock_documents = [
            Document(page_content="test content 1", metadata={"file_path": "test1.py"}),
            Document(page_content="test content 2", metadata={"file_path": "test2.py"})
        ]
        mock_vector_store.similarity_search.return_value = mock_documents
        mock_get_vector_store.return_value = mock_vector_store
        
        state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        state["processed_query"] = "test query"
        state["search_filters"] = {}
        state["retrieval_config"] = {"k": 4}
        state["search_strategy"] = SearchStrategy.SEMANTIC
        state["document_retrieval"] = {"retrieved_documents": []}
        
        result = self.handler.execute_step("perform_search", state)
        
        assert "context_documents" in result
        assert len(result["context_documents"]) == 2
        assert "retrieval_time" in result
        assert result["document_retrieval"]["retrieved_documents"] == result["context_documents"]

    @patch('src.workflows.query.handlers.vector_search_handler.VectorSearchHandler._get_vector_store')
    def test_perform_vector_search_semantic(self, mock_get_vector_store):
        """Test semantic vector search."""
        mock_vector_store = Mock()
        mock_documents = [Document(page_content="test", metadata={})]
        mock_vector_store.similarity_search.return_value = mock_documents
        mock_get_vector_store.return_value = mock_vector_store
        
        documents = self.handler._perform_vector_search(
            "test query", SearchStrategy.SEMANTIC, 4, {}
        )
        
        assert len(documents) == 1
        mock_vector_store.similarity_search.assert_called_once_with(
            query="test query", k=4, filter={}
        )

    @patch('src.workflows.query.handlers.vector_search_handler.VectorSearchHandler._get_vector_store')
    def test_perform_vector_search_keyword(self, mock_get_vector_store):
        """Test keyword vector search fallback."""
        mock_vector_store = Mock()
        mock_documents = [Document(page_content="test", metadata={})]
        mock_vector_store.similarity_search.return_value = mock_documents
        # No keyword_search method available
        mock_get_vector_store.return_value = mock_vector_store
        
        documents = self.handler._perform_vector_search(
            "test query", SearchStrategy.KEYWORD, 4, {}
        )
        
        assert len(documents) == 1
        # Should fall back to similarity search
        mock_vector_store.similarity_search.assert_called_once()

    @patch('src.workflows.query.handlers.vector_search_handler.VectorSearchHandler._get_vector_store')
    def test_perform_vector_search_hybrid(self, mock_get_vector_store):
        """Test hybrid vector search fallback."""
        mock_vector_store = Mock()
        mock_documents = [Document(page_content="test", metadata={})]
        mock_vector_store.similarity_search.return_value = mock_documents
        # No hybrid_search method available
        mock_get_vector_store.return_value = mock_vector_store
        
        documents = self.handler._perform_vector_search(
            "test query", SearchStrategy.HYBRID, 4, {}
        )
        
        assert len(documents) == 1
        # Should fall back to similarity search
        mock_vector_store.similarity_search.assert_called_once()

    def test_process_results_step(self):
        """Test results processing step."""
        state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        state["context_documents"] = [
            {"content": "test1", "metadata": {}, "source": "test1.py"},
            {"content": "test2", "metadata": {}, "source": "test2.py"}
        ]
        
        result = self.handler.execute_step("process_results", state)
        
        assert "workflow_progress" in result
        assert result["workflow_progress"]["percentage"] == 50.0

    @patch('src.workflows.query.handlers.vector_search_handler.VectorSearchHandler._get_vector_store')
    def test_full_workflow_execution(self, mock_get_vector_store):
        """Test complete workflow execution."""
        # Mock vector store
        mock_vector_store = Mock()
        mock_documents = [
            Document(page_content="test content", metadata={"file_path": "test.py"})
        ]
        mock_vector_store.similarity_search.return_value = mock_documents
        mock_get_vector_store.return_value = mock_vector_store
        
        state = create_query_state(
            workflow_id="test-123",
            original_query="python functions"
        )
        state["processed_query"] = "python functions"
        state["retrieval_config"] = {"k": 4}
        state["search_strategy"] = SearchStrategy.SEMANTIC
        state["document_retrieval"] = {"retrieved_documents": []}
        
        # Execute full workflow
        result = self.handler.invoke(state)
        
        # Verify all steps completed
        assert "search_filters" in result
        assert "context_documents" in result
        assert len(result["context_documents"]) == 1
        assert "retrieval_time" in result

    @patch('src.workflows.query.handlers.vector_search_handler.VectorSearchHandler._get_vector_store')
    def test_error_handling(self, mock_get_vector_store):
        """Test error handling in vector search."""
        # Mock vector store to raise exception
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.side_effect = Exception("Search failed")
        mock_get_vector_store.return_value = mock_vector_store
        
        with pytest.raises(Exception, match="Search failed"):
            self.handler._perform_vector_search(
                "test query", SearchStrategy.SEMANTIC, 4, {}
            )
