"""
Unit tests for QueryParsingHandler.

Tests the query parsing, validation, and intent analysis functionality
of the modular query parsing handler including configuration-driven patterns.
"""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch
from pathlib import Path

from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
from src.workflows.workflow_states import QueryState, QueryIntent, create_query_state
from src.config.query_patterns import QueryPatternsConfig, DomainPattern, TechnicalPattern


def create_test_query_state(workflow_id: str, original_query: str) -> QueryState:
    """Create a test query state using the proper factory function."""
    return create_query_state(workflow_id, original_query)


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

        # "with" should be filtered out as it's in excluded words
        assert result["processed_query"] == "test query spaces"

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
        # "this" and "how" should be filtered out as they're in excluded words
        assert result["processed_query"] == "explain function works"
        assert result["query_intent"] == QueryIntent.EXPLANATION
        # Check that workflow metadata exists (progress tracking is in metadata)
        assert "metadata" in result

    def test_error_handling(self):
        """Test error handling in workflow execution."""
        # Create state with invalid original_query
        state = create_test_query_state("test-123", "")
        state["original_query"] = ""  # Empty query should cause validation to fail
        
        # The validate_state method should return False for empty query
        assert self.handler.validate_state(state) is False

    def test_extract_key_terms_domain_patterns(self):
        """Test key term extraction for domain-specific patterns."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="show me listing management components"
        )
        
        result = self.handler.execute_step("parse_query", state)
        
        # Should extract domain-specific terms
        assert "listing" in result["processed_query"]
        assert "catalog" in result["processed_query"]  # Domain pattern match
        assert "class" in result["processed_query"]    # Technical pattern match

    def test_extract_key_terms_technical_patterns(self):
        """Test key term extraction for technical patterns."""
        state = create_test_query_state(
            workflow_id="test-123", 
            original_query="show me the main components"
        )
        
        result = self.handler.execute_step("parse_query", state)
        
        # Should extract technical terms
        assert "class" in result["processed_query"]

    def test_extract_key_terms_programming_patterns(self):
        """Test key term extraction for programming language patterns."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="show me C# classes"
        )
        
        result = self.handler.execute_step("parse_query", state)
        
        # Should extract programming language terms
        assert "class" in result["processed_query"]

    def test_extract_key_terms_api_patterns(self):
        """Test key term extraction for API patterns."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="show me API endpoints"
        )
        
        result = self.handler.execute_step("parse_query", state)
        
        # Should extract API-related terms
        processed_terms = result["processed_query"].split()
        assert any(term in ["controller", "service", "endpoint"] for term in processed_terms)

    def test_extract_key_terms_database_patterns(self):
        """Test key term extraction for database patterns."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="show me database models"
        )
        
        result = self.handler.execute_step("parse_query", state)
        
        # Should extract database-related terms
        processed_terms = result["processed_query"].split()
        assert any(term in ["model", "entity", "class"] for term in processed_terms)

    def test_extract_key_terms_general_fallback(self):
        """Test key term extraction falls back to general terms."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="custom specific terminology here"
        )
        
        result = self.handler.execute_step("parse_query", state)
        
        # Should extract general terms, excluding common words
        processed_terms = result["processed_query"].split()
        assert "custom" in processed_terms
        assert "specific" in processed_terms
        assert "terminology" in processed_terms

    def test_extract_key_terms_excluded_words(self):
        """Test that excluded words are filtered out."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="the main components and project structure"
        )
        
        result = self.handler.execute_step("parse_query", state)
        
        # Should not contain excluded words like "the", "and"
        processed_terms = result["processed_query"].split()
        assert "the" not in processed_terms
        assert "and" not in processed_terms

    def test_extract_key_terms_max_terms_limit(self):
        """Test that max terms limit is respected.""" 
        # Create query with many potential terms
        long_query = "show me the main project architecture components structure design patterns implementation details"
        state = create_test_query_state(
            workflow_id="test-123",
            original_query=long_query
        )
        
        result = self.handler.execute_step("parse_query", state)
        
        # Should respect max_terms limit (default is 5)
        processed_terms = result["processed_query"].split()
        assert len(processed_terms) <= 5

    def test_custom_configuration_loading(self):
        """Test handler with custom configuration file."""
        # Create custom configuration
        custom_config = {
            "domain_patterns": [
                {
                    "patterns": ["test-custom-domain"],
                    "key_terms": ["custom", "test"]
                }
            ],
            "technical_patterns": [],
            "programming_patterns": [],
            "api_patterns": [],
            "database_patterns": [],
            "architecture_patterns": [],
            "excluded_words": ["ignore"],
            "max_terms": 3,
            "min_word_length": 2
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_config, f)
            temp_path = f.name
        
        try:
            # Create handler with custom config
            handler = QueryParsingHandler(query_patterns_config=temp_path)
            
            state = create_test_query_state(
                workflow_id="test-123",
                original_query="show me test-custom-service details"
            )
            
            result = handler.execute_step("parse_query", state)
            
            # Should use custom patterns
            assert "custom" in result["processed_query"]
            assert "test" in result["processed_query"]
            
        finally:
            Path(temp_path).unlink()

    def test_duplicate_term_removal(self):
        """Test that duplicate terms are removed from results."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="show me car-listing-service API controller class"
        )
        
        result = self.handler.execute_step("parse_query", state)
        
        # Should not have duplicate terms even if multiple patterns match
        processed_terms = result["processed_query"].split()
        assert len(processed_terms) == len(set(processed_terms))  # No duplicates

    def test_empty_query_handling(self):
        """Test handling of empty or whitespace-only queries."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="   "
        )
        
        result = self.handler.execute_step("parse_query", state)
        
        # Should handle empty query gracefully
        assert result["processed_query"] == ""

    def test_configuration_patterns_priority(self):
        """Test that domain patterns take priority over general extraction."""
        state = create_test_query_state(
            workflow_id="test-123",
            original_query="notification service implementation details"
        )
        
        result = self.handler.execute_step("parse_query", state)
        
        # Should prioritize domain pattern terms for notification
        processed_terms = result["processed_query"].split()
        assert "notification" in processed_terms
        assert "service" in processed_terms  # Both domain and technical pattern
