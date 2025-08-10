"""
Test module for event flow analysis functionality.

This module tests the complete event flow analysis pipeline including
event flow detection, code discovery, and sequence diagram generation.
"""

import os
import pytest
import uuid
from unittest.mock import Mock, patch

# Set up test environment
os.environ['OPENAI_API_KEY'] = 'test-key'
os.environ['GITHUB_TOKEN'] = 'test-token'

from src.analyzers.event_flow_analyzer import EventFlowAnalyzer, WorkflowPattern
from src.workflows.workflow_states import create_query_state, QueryIntent
from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
from src.workflows.query.handlers.event_flow_handler import EventFlowHandler
from src.diagrams.sequence_diagram_builder import SequenceDiagramBuilder
from src.discovery.code_discovery_engine import CodeReference


class TestEventFlowAnalyzer:
    """Test cases for EventFlowAnalyzer."""
    
    def test_is_event_flow_query_positive_cases(self):
        """Test that event flow queries are correctly identified."""
        analyzer = EventFlowAnalyzer()
        
        positive_cases = [
            "Walk me through what happens when a user places an order",
            "Step by step process when authentication occurs",  
            "What happens when payment is processed?",
            "Show me the workflow for data processing",
            "Explain the process flow",
            "Sequence of events when user logs in"
        ]
        
        for query in positive_cases:
            assert analyzer.is_event_flow_query(query), f"Failed to detect event flow in: {query}"
    
    def test_is_event_flow_query_negative_cases(self):
        """Test that non-event flow queries are correctly rejected."""
        analyzer = EventFlowAnalyzer()
        
        negative_cases = [
            "How to implement a function?",
            "What is a variable?",
            "Fix this bug in my code",
            "Show me documentation",
            "Find examples of authentication"
        ]
        
        for query in negative_cases:
            assert not analyzer.is_event_flow_query(query), f"Incorrectly detected event flow in: {query}"
    
    def test_parse_query_order_processing(self):
        """Test query parsing for order processing workflow."""
        analyzer = EventFlowAnalyzer()
        query = "Walk me through what happens when a user places an order"
        
        parsed = analyzer.parse_query(query)
        
        assert parsed.workflow == WorkflowPattern.ORDER_PROCESSING
        assert "user" in parsed.entities
        assert "order" in parsed.entities
        assert "places" in parsed.actions
        assert parsed.domain == "e-commerce"
    
    def test_detect_workflow_pattern(self):
        """Test workflow pattern detection."""
        analyzer = EventFlowAnalyzer()
        
        test_cases = [
            ("authentication login process", WorkflowPattern.USER_AUTHENTICATION),
            ("order checkout payment", WorkflowPattern.ORDER_PROCESSING),
            ("data pipeline etl", WorkflowPattern.DATA_PIPELINE),
            ("api request response", WorkflowPattern.API_REQUEST_FLOW),
            ("event message queue", WorkflowPattern.EVENT_DRIVEN),
            ("generic workflow", WorkflowPattern.GENERIC_WORKFLOW)
        ]
        
        for query, expected_pattern in test_cases:
            detected_pattern = analyzer.detect_workflow_pattern(query)
            assert detected_pattern == expected_pattern, f"Expected {expected_pattern}, got {detected_pattern} for '{query}'"


class TestQueryParsingHandler:
    """Test cases for QueryParsingHandler with event flow integration."""
    
    def test_event_flow_intent_detection(self):
        """Test that event flow queries get EVENT_FLOW intent."""
        handler = QueryParsingHandler()
        
        query = "Walk me through what happens when a user places an order"
        state = create_query_state(
            workflow_id=str(uuid.uuid4()),
            original_query=query
        )
        
        result_state = handler.invoke(state)
        
        assert result_state["query_intent"] == QueryIntent.EVENT_FLOW
        assert result_state["original_query"] == query
    
    def test_non_event_flow_intent_detection(self):
        """Test that non-event flow queries get other intents."""
        handler = QueryParsingHandler()
        
        query = "How to implement a function?"
        state = create_query_state(
            workflow_id=str(uuid.uuid4()),
            original_query=query
        )
        
        result_state = handler.invoke(state)
        
        assert result_state["query_intent"] != QueryIntent.EVENT_FLOW
        assert result_state["query_intent"] in [QueryIntent.CODE_SEARCH, QueryIntent.ARCHITECTURE, QueryIntent.EXPLANATION]


class TestSequenceDiagramBuilder:
    """Test cases for SequenceDiagramBuilder."""
    
    def test_build_from_workflow_basic(self):
        """Test basic sequence diagram generation."""
        builder = SequenceDiagramBuilder()
        
        # Create a mock workflow query
        from src.analyzers.event_flow_analyzer import EventFlowQuery, WorkflowPattern
        
        workflow = EventFlowQuery(
            entities=["user", "order", "payment"],
            actions=["place", "process", "validate"],
            workflow=WorkflowPattern.ORDER_PROCESSING,
            domain="e-commerce",
            intent="Order processing workflow"
        )
        
        # Create mock code references
        code_refs = [
            CodeReference(
                repository="test-repo",
                file_path="src/order/service.py",
                line_numbers=[10, 20],
                method_name="place_order",
                context_type="service",
                language="python",
                content_snippet="def place_order(user, order)...",
                relevance_score=0.9
            )
        ]
        
        diagram = builder.build_from_workflow(workflow, code_refs)
        
        # Verify Mermaid diagram structure
        assert "```mermaid" in diagram
        assert "sequenceDiagram" in diagram
        assert "participant" in diagram
        assert "```" in diagram.split("```mermaid")[1]  # Ensure closing


class TestEventFlowHandler:
    """Test cases for EventFlowHandler."""
    
    def test_validate_event_flow_query(self):
        """Test event flow query validation."""
        handler = EventFlowHandler()
        
        # Create valid event flow state
        state = create_query_state(
            workflow_id=str(uuid.uuid4()),
            original_query="Walk me through the order process"
        )
        state["query_intent"] = QueryIntent.EVENT_FLOW
        
        # Should validate successfully
        assert handler.validate_state(state)
        
        # Create invalid state (wrong intent)
        invalid_state = create_query_state(
            workflow_id=str(uuid.uuid4()),
            original_query="How to implement a function"
        )
        invalid_state["query_intent"] = QueryIntent.CODE_SEARCH
        
        # Should fail validation
        assert not handler.validate_state(invalid_state)
    
    @patch('src.workflows.query.handlers.event_flow_handler.EventFlowHandler.get_vector_store')
    def test_discover_relevant_code_with_mock_vector_store(self, mock_get_vector_store):
        """Test code discovery with mocked vector store."""
        # Create mock vector store
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.return_value = []
        mock_get_vector_store.return_value = mock_vector_store
        
        handler = EventFlowHandler()
        
        # Create state with parsed workflow
        state = create_query_state(
            workflow_id=str(uuid.uuid4()),
            original_query="Walk me through order processing"
        )
        state["metadata"] = {
            "parsed_workflow": Mock(
                workflow=WorkflowPattern.ORDER_PROCESSING,
                entities=["user", "order"],
                actions=["place"]
            )
        }
        
        # Execute code discovery step
        result_state = handler._discover_relevant_code(state)
        
        # Verify code references were stored (even if empty due to mock)
        assert "code_references" in result_state["metadata"]
        assert "code_discovery_time" in result_state["metadata"]


class TestEventFlowIntegration:
    """Integration tests for complete event flow analysis."""
    
    def test_end_to_end_event_flow_detection(self):
        """Test complete event flow detection pipeline."""
        # Test query
        query = "Walk me through what happens when a user places an order"
        
        # Step 1: Event flow analyzer should detect this as event flow
        analyzer = EventFlowAnalyzer()
        assert analyzer.is_event_flow_query(query)
        
        # Step 2: Query parsing handler should assign EVENT_FLOW intent
        handler = QueryParsingHandler()
        state = create_query_state(
            workflow_id=str(uuid.uuid4()),
            original_query=query
        )
        
        result_state = handler.invoke(state)
        assert result_state["query_intent"] == QueryIntent.EVENT_FLOW
        
        # Step 3: Event flow analyzer should parse workflow components
        parsed = analyzer.parse_query(query)
        assert parsed.workflow == WorkflowPattern.ORDER_PROCESSING
        assert len(parsed.entities) > 0
        assert len(parsed.actions) > 0
    
    def test_diagram_generation_with_empty_code_refs(self):
        """Test diagram generation works even without code references."""
        builder = SequenceDiagramBuilder()
        
        from src.analyzers.event_flow_analyzer import EventFlowQuery, WorkflowPattern
        
        workflow = EventFlowQuery(
            entities=["user", "system"],
            actions=["request"],
            workflow=WorkflowPattern.GENERIC_WORKFLOW,
            domain="general",
            intent="Generic workflow"
        )
        
        # Test with empty code references
        diagram = builder.build_from_workflow(workflow, [])
        
        # Should still generate a valid diagram
        assert "```mermaid" in diagram
        assert "sequenceDiagram" in diagram
        assert "participant" in diagram


if __name__ == "__main__":
    # Run tests manually if executed directly
    print("Running Event Flow Analysis Tests...")
    
    # Test analyzer
    test_analyzer = TestEventFlowAnalyzer()
    test_analyzer.test_is_event_flow_query_positive_cases()
    test_analyzer.test_is_event_flow_query_negative_cases()
    test_analyzer.test_parse_query_order_processing()
    test_analyzer.test_detect_workflow_pattern()
    print("✓ EventFlowAnalyzer tests passed")
    
    # Test query parsing
    test_parsing = TestQueryParsingHandler()
    test_parsing.test_event_flow_intent_detection()
    test_parsing.test_non_event_flow_intent_detection()
    print("✓ QueryParsingHandler tests passed")
    
    # Test diagram builder
    test_diagram = TestSequenceDiagramBuilder()
    test_diagram.test_build_from_workflow_basic()
    print("✓ SequenceDiagramBuilder tests passed")
    
    # Test event flow handler
    test_handler = TestEventFlowHandler()
    test_handler.test_validate_event_flow_query()
    print("✓ EventFlowHandler tests passed")
    
    # Test integration
    test_integration = TestEventFlowIntegration()
    test_integration.test_end_to_end_event_flow_detection()
    test_integration.test_diagram_generation_with_empty_code_refs()
    print("✓ Integration tests passed")
    
    print("\nAll Event Flow Analysis tests completed successfully! 🎉")