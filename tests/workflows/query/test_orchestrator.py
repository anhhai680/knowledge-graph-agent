"""
Unit tests for QueryWorkflowOrchestrator.

Tests the main orchestrator that composes step handlers while maintaining
LangGraph integration and existing state management.
"""

import pytest
from unittest.mock import Mock, patch
import asyncio

from src.workflows.query.orchestrator.query_orchestrator import QueryWorkflowOrchestrator
from src.workflows.workflow_states import (
    QueryState, 
    QueryIntent,
    SearchStrategy,
    ProcessingStatus,
    create_query_state
)


class TestQueryWorkflowOrchestrator:
    """Test the query workflow orchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = QueryWorkflowOrchestrator()

    def test_parse_and_analyze_step(self):
        """Test parse and analyze step delegation."""
        # Create orchestrator and mock the parsing handler
        orchestrator = QueryWorkflowOrchestrator()
        orchestrator.parsing_handler = Mock()
        
        # Mock the handler's invoke method to return proper state
        mock_state = {
            "query_intent": QueryIntent.CODE_SEARCH,
            "processed_query": "test query",
            "key_terms": ["test", "query"]
        }
        orchestrator.parsing_handler.invoke.return_value = mock_state
        
        state = {"original_query": "test query"}
        result = orchestrator.execute_step("parse_and_analyze", state)
        
        # Verify handler was called
        orchestrator.parsing_handler.invoke.assert_called_once_with(state)
        
        # Verify result contains expected data
        assert result["query_intent"] == QueryIntent.CODE_SEARCH
        assert result["processed_query"] == "test query"

    def test_search_documents_step(self):
        """Test search documents step delegation."""
        # Create orchestrator and mock the search handler
        orchestrator = QueryWorkflowOrchestrator()
        orchestrator.search_handler = Mock()
        
        # Mock the handler's invoke method
        mock_state = {
            "context_documents": [
                {"content": "test1", "metadata": {}, "source": "test1.py"},
                {"content": "test2", "metadata": {}, "source": "test2.py"}
            ]
        }
        orchestrator.search_handler.invoke.return_value = mock_state
        
        state = {"processed_query": "test query"}
        result = orchestrator.execute_step("search_documents", state)
        
        # Verify handler was called
        orchestrator.search_handler.invoke.assert_called_once_with(state)
        
        # Verify result contains expected data
        assert len(result["context_documents"]) == 2

    def test_process_context_step(self):
        """Test process context step delegation."""
        # Create orchestrator and mock the context handler
        orchestrator = QueryWorkflowOrchestrator()
        orchestrator.context_handler = Mock()
        
        # Mock the handler's invoke method
        mock_state = {
            "context_documents": [
                {"content": "test1", "metadata": {}, "source": "test1.py"}
            ],
            "context_size": 1,
            "context_sufficient": True
        }
        orchestrator.context_handler.invoke.return_value = mock_state
        
        state = {
            "context_documents": [
                {"content": "test1", "metadata": {}, "source": "test1.py"}
            ]
        }
        result = orchestrator.execute_step("process_context", state)
        
        # Verify handler was called
        orchestrator.context_handler.invoke.assert_called_once_with(state)
        
        # Verify result contains expected data
        assert result["context_size"] == 1
        assert result["context_sufficient"] is True

    def test_process_context_step_insufficient(self):
        """Test process context step with insufficient context."""
        # Create orchestrator and mock the context handler
        orchestrator = QueryWorkflowOrchestrator()
        orchestrator.context_handler = Mock()
        
        # Mock the handler's invoke method to return insufficient context
        mock_state = {
            "context_documents": [],
            "context_size": 0,
            "context_sufficient": False,
            "retrieval_config": {"k": 4},  # Add required retrieval_config to mock state
            "document_retrieval": {}  # Add required document_retrieval field
        }
        orchestrator.context_handler.invoke.return_value = mock_state
        
        state = {
            "context_documents": [],
            "retrieval_config": {"k": 4}  # Add required retrieval_config
        }
        result = orchestrator.execute_step("process_context", state)
        
        # Verify handler was called (it gets called twice - once initially and once after expansion)
        assert orchestrator.context_handler.invoke.call_count == 2
        
        # Verify result contains expected data
        assert result["context_size"] == 0
        assert result["context_sufficient"] is False

    def test_generate_response_step(self):
        """Test generate response step delegation."""
        # Create orchestrator and mock the LLM handler
        orchestrator = QueryWorkflowOrchestrator()
        orchestrator.llm_handler = Mock()
        
        # Mock the handler's invoke method
        mock_state = {
            "response": "This is a test response",
            "confidence_score": 0.85
        }
        orchestrator.llm_handler.invoke.return_value = mock_state
        
        state = {
            "context_documents": [
                {"content": "test1", "metadata": {}, "source": "test1.py"}
            ]
        }
        result = orchestrator.execute_step("generate_response", state)
        
        # Verify handler was called
        orchestrator.llm_handler.invoke.assert_called_once_with(state)
        
        # Verify result contains expected data
        assert result["response"] == "This is a test response"
        assert result["confidence_score"] == 0.85

    def test_expand_search(self):
        """Test search expansion when context is insufficient."""
        # Create orchestrator and mock the search handler
        orchestrator = QueryWorkflowOrchestrator()
        orchestrator.search_handler = Mock()
        
        # Mock the search handler to return documents
        mock_documents = [
            {"content": "test1", "metadata": {}, "source": "test1.py"}
        ]
        
        state = {
            "query": "test query",
            "context_documents": [],
            "context_sufficient": False,
            "retrieval_config": {"k": 4}  # Add required retrieval_config
        }
        
        # Mock the VectorSearchHandler for expansion
        with patch('src.workflows.query.orchestrator.query_orchestrator.VectorSearchHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            mock_handler.invoke.return_value = {
                "context_documents": mock_documents,
                "document_retrieval": {"retrieved_documents": mock_documents}
            }
            
            result = orchestrator._expand_search(state)
            
            # Verify expansion was performed
            assert len(result["context_documents"]) == 1

    def test_error_handling(self):
        """Test error handling in workflow execution."""
        # Create orchestrator
        orchestrator = QueryWorkflowOrchestrator()
        
        # Mock the parsing handler to raise an exception
        orchestrator.parsing_handler = Mock()
        orchestrator.parsing_handler.invoke.side_effect = Exception("Test error")
        
        state = {"original_query": "test query"}
        
        # Execute workflow - should handle the error gracefully
        with pytest.raises(Exception):
            orchestrator.execute_step("parse_and_analyze", state)
