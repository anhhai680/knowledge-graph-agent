"""
Unit tests for QueryParsingHandler.

Tests the query parsing, validation, and intent analysis functionality
of the modular query parsing handler.
"""

import pytest
from unittest.mock import Mock, patch

from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
from src.workflows.workflow_states import QueryState, QueryIntent


def create_test_query_state(workflow_id: str, original_query: str) -> QueryState:
    """Create a test query state without dependencies."""
    return {
        "workflow_id": workflow_id,
        "workflow_type": "query",
        "status": "not_started",
        "original_query": original_query,
        "processed_query": "",
        "query_intent": None,
        "search_strategy": None,
        "search_filters": {},
        "context_documents": [],
        "retrieval_config": {},
        "document_retrieval": {"retrieved_documents": []},
        "llm_generation": {"generated_response": ""},
        "workflow_progress": {"percentage": 0.0, "current_step": ""},
        "errors": []
    }


class TestQueryParsingHandler:
    """Test suite for QueryParsingHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = QueryParsingHandler()

    def test_define_steps(self):
        """Test that handler defines correct steps."""
        steps = self.handler.define_steps()
        expected_steps = ["parse_query", "validate_query", "analyze_intent"]
        assert steps == expected_steps

    def test_validate_state_valid(self):
        """Test state validation with valid state."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        assert self.handler.validate_state(state) is True

    def test_validate_state_invalid(self):
        """Test state validation with invalid state."""
        state = create_test_query_state(workflow_id="test-123", original_query="")
        state.pop("original_query")
        assert self.handler.validate_state(state) is False

    def test_parse_query_step(self):
        """Test query parsing step."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="  test query with spaces  "
        )

        result = self.handler.execute_step("parse_query", state)

        assert result["processed_query"] == "test query with spaces"

    def test_validate_query_step_valid(self):
        """Test query validation with valid query."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="valid query"
        )
        state["processed_query"] = "valid query"

        # Should not raise exception
        result = self.handler.execute_step("validate_query", state)
        assert result["processed_query"] == "valid query"

    def test_validate_query_step_too_short(self):
        """Test query validation with too short query."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="hi"
        )
        state["processed_query"] = "hi"

        with pytest.raises(ValueError, match="Query is too short"):
            self.handler.execute_step("validate_query", state)

    def test_validate_query_step_too_long(self):
        """Test query validation with too long query."""
        long_query = "x" * 1001
        state = create_test_query_state(
            workflow_id="test-123",
            original_query=long_query
        )
        state["processed_query"] = long_query

        with pytest.raises(ValueError, match="Query is too long"):
            self.handler.execute_step("validate_query", state)

    def test_analyze_intent_code_search(self):
        """Test intent analysis for code search queries."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="show me the function implementation"
        )
        state["processed_query"] = "show me the function implementation"

        result = self.handler.execute_step("analyze_intent", state)

        assert result["query_intent"] == QueryIntent.CODE_SEARCH

    def test_analyze_intent_documentation(self):
        """Test intent analysis for documentation queries."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="show me the API documentation"
        )
        state["processed_query"] = "show me the API documentation"

        result = self.handler.execute_step("analyze_intent", state)

        assert result["query_intent"] == QueryIntent.DOCUMENTATION

    def test_analyze_intent_explanation(self):
        """Test intent analysis for explanation queries."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="explain what this code does"
        )
        state["processed_query"] = "explain what this code does"

        result = self.handler.execute_step("analyze_intent", state)

        assert result["query_intent"] == QueryIntent.EXPLANATION

    def test_analyze_intent_debugging(self):
        """Test intent analysis for debugging queries."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="fix this error in my code"
        )
        state["processed_query"] = "fix this error in my code"

        result = self.handler.execute_step("analyze_intent", state)

        assert result["query_intent"] == QueryIntent.DEBUGGING

    def test_analyze_intent_architecture(self):
        """Test intent analysis for architecture queries."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="show me the system architecture"
        )
        state["processed_query"] = "show me the system architecture"

        result = self.handler.execute_step("analyze_intent", state)

        assert result["query_intent"] == QueryIntent.ARCHITECTURE

    def test_analyze_intent_default(self):
        """Test intent analysis defaults to code search."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="random query without keywords"
        )
        state["processed_query"] = "random query without keywords"

        result = self.handler.execute_step("analyze_intent", state)

        assert result["query_intent"] == QueryIntent.CODE_SEARCH

    def test_full_workflow_execution(self):
        """Test complete workflow execution."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="  explain how this function works  "
        )

        # Execute full workflow
        result = self.handler.invoke(state)

        # Verify all steps completed
        assert result["processed_query"] == "explain how this function works"
        assert result["query_intent"] == QueryIntent.EXPLANATION
        assert "workflow_progress" in result

    def test_error_handling(self):
        """Test error handling in workflow execution."""
        # Create invalid state
        state = {}

        # The handler should handle this gracefully and return a failed state
        result = self.handler.invoke(state)
        assert result["status"] == "failed"
