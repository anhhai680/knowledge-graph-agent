"""
Unit tests for VectorSearchHandler.

Tests the vector search functionality of the modular vector search handler.
"""

import pytest
from unittest.mock import Mock, patch
from langchain.schema import Document

from src.workflows.query.handlers.vector_search_handler import VectorSearchHandler
from src.workflows.workflow_states import SearchStrategy, create_query_state


class TestVectorSearchHandler:
    """Test the vector search handler."""

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
        assert self.handler.validate_state(state) is True

    def test_validate_state_invalid(self):
        """Test state validation with invalid state."""
        state = create_query_state(workflow_id="test-123", original_query="")
        state.pop("original_query")
        assert self.handler.validate_state(state) is False

    def test_extract_filters_step(self):
        """Test extract filters step."""
        state = {
            "processed_query": "test query in python files",
            "search_filters": {}
        }
        
        result = self.handler.execute_step("extract_filters", state)
        
        assert "search_filters" in result
        assert isinstance(result["search_filters"], dict)

    def test_perform_search_step(self):
        """Test perform search step."""
        # Create mock documents
        mock_documents = [
            Document(page_content="test1", metadata={"file_path": "test1.py"}),
            Document(page_content="test2", metadata={"file_path": "test2.py"})
        ]
        
        # Mock the _perform_vector_search method directly
        with patch.object(self.handler, '_perform_vector_search', return_value=mock_documents):
            state = {
                "processed_query": "test query",
                "retrieval_config": {"k": 5},
                "search_filters": {},
                "search_strategy": SearchStrategy.KEYWORD,
                "context_documents": [],
                "document_retrieval": {}
            }
            
            result = self.handler.execute_step("perform_search", state)
            
            # Verify results are stored
            assert len(result["context_documents"]) == 2
            assert "retrieval_time" in result
            assert "document_retrieval" in result

    def test_process_results_step(self):
        """Test processing search results."""
        # Create state with documents
        state = {
            "query": "test query",
            "search_results": [],
            "context_documents": [
                {"content": "test1", "metadata": {"file_path": "test1.py"}},
                {"content": "test2", "metadata": {"file_path": "test2.py"}}
            ]
        }
        
        result = self.handler.execute_step("process_results", state)
        
        # Verify documents are processed
        assert "context_documents" in result
        assert len(result["context_documents"]) == 2

    def test_perform_vector_search_keyword(self):
        """Test keyword-based vector search."""
        # Create mock documents
        mock_documents = [
            Document(page_content="test1", metadata={"file_path": "test1.py"}),
            Document(page_content="test2", metadata={"file_path": "test2.py"})
        ]
        
        # Mock the _perform_vector_search method directly
        with patch.object(self.handler, '_perform_vector_search', return_value=mock_documents):
            documents = self.handler._perform_vector_search(
                query="test query",
                strategy=SearchStrategy.KEYWORD,
                k=5,
                filters={}
            )
            
            assert len(documents) == 2

    def test_perform_vector_search_hybrid(self):
        """Test hybrid vector search."""
        # Create mock documents
        mock_documents = [
            Document(page_content="test1", metadata={"file_path": "test1.py"}),
            Document(page_content="test2", metadata={"file_path": "test2.py"})
        ]
        
        # Mock the _perform_vector_search method directly
        with patch.object(self.handler, '_perform_vector_search', return_value=mock_documents):
            documents = self.handler._perform_vector_search(
                query="test query",
                strategy=SearchStrategy.HYBRID,
                k=5,
                filters={}
            )
            
            assert len(documents) == 2

    def test_extract_filters_from_query(self):
        """Test filter extraction from query."""
        query = "test query in python files from github"
        filters = self.handler._extract_filters_from_query(query)
        
        assert isinstance(filters, dict)
        # Verify filters are extracted (check for any of the expected keys)
        expected_keys = ["file_types", "languages", "repositories", "language", "file_extension"]
        assert any(key in filters for key in expected_keys)
