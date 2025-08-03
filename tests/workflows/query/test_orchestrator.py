"""
Unit tests for QueryWorkflowOrchestrator.

Tests the main orchestrator that composes step handlers while maintaining
LangGraph integration and existing state management.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
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
    """Test suite for QueryWorkflowOrchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = QueryWorkflowOrchestrator(
            collection_name="test-collection",
            default_k=4,
            max_k=20
        )

    def test_define_steps(self):
        """Test that orchestrator defines correct steps."""
        steps = self.orchestrator.define_steps()
        expected_steps = [
            "parse_and_analyze",
            "search_documents", 
            "process_context",
            "generate_response",
            "finalize_response"
        ]
        assert steps == expected_steps

    def test_validate_state_valid(self):
        """Test state validation with valid state."""
        state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        assert self.orchestrator.validate_state(state) is True

    def test_validate_state_invalid(self):
        """Test state validation with invalid state."""
        state = create_query_state(workflow_id="test-123", original_query="")
        state.pop("original_query")
        assert self.orchestrator.validate_state(state) is False

    def test_determine_search_strategy_code_search(self):
        """Test search strategy determination for code search."""
        strategy = self.orchestrator._determine_search_strategy(
            QueryIntent.CODE_SEARCH, "short query"
        )
        assert strategy == SearchStrategy.SEMANTIC

    def test_determine_search_strategy_debugging(self):
        """Test search strategy determination for debugging."""
        strategy = self.orchestrator._determine_search_strategy(
            QueryIntent.DEBUGGING, "debug this error"
        )
        assert strategy == SearchStrategy.HYBRID

    def test_determine_search_strategy_long_query(self):
        """Test search strategy determination for long queries."""
        long_query = "this is a very long query with many words that should trigger hybrid search"
        strategy = self.orchestrator._determine_search_strategy(
            QueryIntent.CODE_SEARCH, long_query
        )
        assert strategy == SearchStrategy.HYBRID

    @patch('src.workflows.query.orchestrator.query_orchestrator.QueryWorkflowOrchestrator.parsing_handler')
    def test_parse_and_analyze_step(self, mock_parsing_handler):
        """Test parse and analyze step execution."""
        # Mock parsing handler
        mock_state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        mock_state["query_intent"] = QueryIntent.CODE_SEARCH
        mock_parsing_handler.invoke.return_value = mock_state
        
        state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        
        result = self.orchestrator.execute_step("parse_and_analyze", state)
        
        assert "search_strategy" in result
        mock_parsing_handler.invoke.assert_called_once()

    @patch('src.workflows.query.orchestrator.query_orchestrator.QueryWorkflowOrchestrator.search_handler')
    def test_search_documents_step(self, mock_search_handler):
        """Test search documents step execution."""
        # Mock search handler
        mock_state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        mock_search_handler.invoke.return_value = mock_state
        
        state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        
        result = self.orchestrator.execute_step("search_documents", state)
        
        mock_search_handler.invoke.assert_called_once()

    @patch('src.workflows.query.orchestrator.query_orchestrator.QueryWorkflowOrchestrator.context_handler')
    def test_process_context_step(self, mock_context_handler):
        """Test process context step execution."""
        # Mock context handler
        mock_state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        mock_state["context_sufficient"] = True
        mock_context_handler.invoke.return_value = mock_state
        
        state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        
        result = self.orchestrator.execute_step("process_context", state)
        
        mock_context_handler.invoke.assert_called_once()

    @patch('src.workflows.query.orchestrator.query_orchestrator.QueryWorkflowOrchestrator.context_handler')
    @patch('src.workflows.query.orchestrator.query_orchestrator.QueryWorkflowOrchestrator._expand_search')
    def test_process_context_step_insufficient(self, mock_expand_search, mock_context_handler):
        """Test process context step with insufficient context."""
        # Mock context handler returning insufficient context
        mock_state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        mock_state["context_sufficient"] = False
        mock_context_handler.invoke.return_value = mock_state
        
        # Mock expand search
        expanded_state = mock_state.copy()
        expanded_state["context_sufficient"] = True
        mock_expand_search.return_value = expanded_state
        mock_context_handler.invoke.side_effect = [mock_state, expanded_state]
        
        state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        
        result = self.orchestrator.execute_step("process_context", state)
        
        mock_expand_search.assert_called_once()
        assert mock_context_handler.invoke.call_count == 2

    @patch('src.workflows.query.orchestrator.query_orchestrator.QueryWorkflowOrchestrator.llm_handler')
    def test_generate_response_step(self, mock_llm_handler):
        """Test generate response step execution."""
        # Mock LLM handler
        mock_state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        mock_llm_handler.invoke.return_value = mock_state
        
        state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        
        result = self.orchestrator.execute_step("generate_response", state)
        
        mock_llm_handler.invoke.assert_called_once()

    def test_finalize_response_step(self):
        """Test finalize response step execution."""
        state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        state["llm_generation"] = {"generated_response": "test response"}
        state["context_documents"] = [
            {
                "content": "test content",
                "metadata": {
                    "file_path": "test.py",
                    "repository": "test-repo",
                    "line_start": 1,
                    "line_end": 10
                },
                "source": "test.py"
            }
        ]
        state["start_time"] = 1234567890.0
        
        with patch('time.time', return_value=1234567891.0):
            result = self.orchestrator.execute_step("finalize_response", state)
        
        assert result["status"] == ProcessingStatus.COMPLETED
        assert "response_sources" in result
        assert len(result["response_sources"]) == 1
        assert "total_query_time" in result
        assert result["total_query_time"] == 1.0

    @pytest.mark.asyncio
    async def test_execute_workflow_complete(self):
        """Test complete workflow execution."""
        with patch.multiple(
            self.orchestrator,
            parsing_handler=Mock(),
            search_handler=Mock(),
            context_handler=Mock(),
            llm_handler=Mock()
        ):
            # Mock all handlers to return successful states
            mock_state = create_query_state(
                workflow_id="test-123",
                original_query="test query"
            )
            mock_state["query_intent"] = QueryIntent.CODE_SEARCH
            mock_state["context_sufficient"] = True
            mock_state["llm_generation"] = {"generated_response": "test response"}
            mock_state["context_documents"] = []
            
            self.orchestrator.parsing_handler.invoke.return_value = mock_state
            self.orchestrator.search_handler.invoke.return_value = mock_state
            self.orchestrator.context_handler.invoke.return_value = mock_state
            self.orchestrator.llm_handler.invoke.return_value = mock_state
            
            result = await self.orchestrator.execute_workflow(
                query="test query",
                k=4
            )
            
            assert result["original_query"] == "test query"
            assert result["status"] == ProcessingStatus.COMPLETED

    @patch('src.workflows.query.handlers.vector_search_handler.VectorSearchHandler')
    def test_expand_search(self, mock_search_handler_class):
        """Test search expansion functionality."""
        # Mock expanded search handler
        mock_handler = Mock()
        mock_expanded_state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        mock_expanded_state["context_documents"] = [
            {"content": "expanded content", "metadata": {}, "source": "test.py"}
        ]
        mock_expanded_state["document_retrieval"] = {
            "retrieved_documents": mock_expanded_state["context_documents"]
        }
        mock_handler.invoke.return_value = mock_expanded_state
        mock_search_handler_class.return_value = mock_handler
        
        state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        state["retrieval_config"] = {"k": 4}
        
        result = self.orchestrator._expand_search(state)
        
        assert len(result["context_documents"]) == 1
        assert result["context_documents"][0]["content"] == "expanded content"

    def test_error_handling(self):
        """Test error handling in orchestrator."""
        # Test with invalid state
        with pytest.raises(Exception):
            self.orchestrator.execute_step("parse_and_analyze", {})
