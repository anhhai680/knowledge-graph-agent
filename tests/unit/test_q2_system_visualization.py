#!/usr/bin/env python3
"""
Unit tests for Q2 System Relationship Visualization feature.

This test module provides comprehensive unit tests for the Q2 feature
that can be integrated into the existing test suite.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestQ2SystemVisualization(unittest.TestCase):
    """Test cases for Q2 system relationship visualization feature."""

    def setUp(self):
        """Set up test environment."""
        # Set minimal environment variables
        os.environ['OPENAI_API_KEY'] = 'test_key'
        os.environ['GITHUB_TOKEN'] = 'test_token'
        os.environ['DATABASE_TYPE'] = 'chroma'
        os.environ['APP_ENV'] = 'development'

    def test_q2_pattern_detection(self):
        """Test Q2 query pattern detection accuracy."""
        # Mock the configuration loading
        with patch('src.config.query_patterns.load_query_patterns') as mock_patterns:
            mock_config = Mock()
            mock_config.domain_patterns = []
            mock_config.technical_patterns = []
            mock_config.programming_patterns = []
            mock_config.api_patterns = []
            mock_config.database_patterns = []
            mock_config.architecture_patterns = []
            mock_config.max_terms = 10
            mock_config.min_word_length = 3
            mock_config.excluded_words = {'the', 'and', 'or', 'but', 'a', 'an'}
            mock_patterns.return_value = mock_config

            from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
            
            handler = QueryParsingHandler()
            
            # Test Q2 queries that should be detected
            q2_queries = [
                "Show me how the four services are connected and explain what I'm looking at.",
                "Show me how the four services are connected",
                "How are the services connected?",
                "Explain how the services connect",
                "How do the services work together",
                "Show me the system architecture",
                "Explain the system relationships",
                "Show service connections"
            ]
            
            for query in q2_queries:
                with self.subTest(query=query):
                    is_q2 = handler._is_q2_system_relationship_query(query)
                    self.assertTrue(is_q2, f"Query '{query}' should be detected as Q2")
            
            # Test non-Q2 queries that should NOT be detected
            non_q2_queries = [
                "How do I implement a function?",
                "What is the bug in this code?",
                "Show me the documentation",
                "Find the class definition",
                "How to debug this error?",
                "What does this method do?"
            ]
            
            for query in non_q2_queries:
                with self.subTest(query=query):
                    is_q2 = handler._is_q2_system_relationship_query(query)
                    self.assertFalse(is_q2, f"Query '{query}' should NOT be detected as Q2")

    def test_q2_intent_classification(self):
        """Test that Q2 queries are classified with ARCHITECTURE intent."""
        with patch('src.config.query_patterns.load_query_patterns') as mock_patterns:
            mock_config = Mock()
            mock_config.domain_patterns = []
            mock_config.technical_patterns = []
            mock_config.programming_patterns = []
            mock_config.api_patterns = []
            mock_config.database_patterns = []
            mock_config.architecture_patterns = []
            mock_config.max_terms = 10
            mock_config.min_word_length = 3
            mock_config.excluded_words = {'the', 'and', 'or', 'but', 'a', 'an'}
            mock_patterns.return_value = mock_config

            from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
            from src.workflows.workflow_states import QueryIntent
            
            handler = QueryParsingHandler()
            
            # Test the exact Q2 query
            query = "Show me how the four services are connected and explain what I'm looking at."
            
            intent = handler._determine_query_intent(query)
            self.assertEqual(intent, QueryIntent.ARCHITECTURE, 
                           "Q2 query should be classified as ARCHITECTURE intent")

    def test_q2_workflow_integration(self):
        """Test Q2 detection through the complete workflow step execution."""
        with patch('src.config.query_patterns.load_query_patterns') as mock_patterns:
            mock_config = Mock()
            mock_config.domain_patterns = []
            mock_config.technical_patterns = []
            mock_config.programming_patterns = []
            mock_config.api_patterns = []
            mock_config.database_patterns = []
            mock_config.architecture_patterns = []
            mock_config.max_terms = 10
            mock_config.min_word_length = 3
            mock_config.excluded_words = {'the', 'and', 'or', 'but', 'a', 'an'}
            mock_patterns.return_value = mock_config

            from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
            from src.workflows.workflow_states import create_query_state, QueryIntent
            
            handler = QueryParsingHandler()
            
            # Create initial state
            query = "Show me how the four services are connected and explain what I'm looking at."
            state = create_query_state(
                workflow_id="test-q2",
                original_query=query
            )
            
            # Execute workflow steps
            steps = ["parse_query", "validate_query", "analyze_intent"]
            
            for step in steps:
                state = handler.execute_step(step, state)
            
            # Verify Q2 detection and intent
            self.assertTrue(state.get('is_q2_system_visualization', False), 
                          "Q2 system visualization should be detected")
            self.assertEqual(state.get('query_intent'), QueryIntent.ARCHITECTURE,
                           "Query intent should be ARCHITECTURE")
            self.assertIsNotNone(state.get('processed_query'), 
                               "Processed query should be set")

    def test_q2_prompt_template_selection(self):
        """Test that Q2 queries use the specialized prompt template."""
        with patch('loguru.logger') as mock_logger:
            mock_logger.bind.return_value = mock_logger
            
            from src.utils.prompt_manager import PromptManager
            from src.workflows.workflow_states import QueryIntent
            from langchain.schema import Document
            
            pm = PromptManager()
            
            # Test Q2 prompt creation
            query = "Show me how the four services are connected and explain what I'm looking at."
            context_docs = [
                Document(
                    page_content="Sample service code",
                    metadata={"file_path": "src/service.py", "repository": "test-repo"}
                )
            ]
            
            # Test with Q2 flag enabled
            result = pm.create_query_prompt(
                query=query,
                context_documents=context_docs,
                query_intent=QueryIntent.ARCHITECTURE,
                is_q2_system_visualization=True
            )
            
            # Verify Q2 template is selected
            self.assertEqual(result.get('template_type'), 'Q2SystemVisualizationTemplate',
                           "Q2 queries should use Q2SystemVisualizationTemplate")
            self.assertEqual(result.get('confidence_score'), 1.0,
                           "Q2 queries should have maximum confidence")
            self.assertEqual(result.get('system_prompt_type'), 'q2_architecture',
                           "Q2 queries should use q2_architecture system prompt")
            self.assertTrue(result.get('metadata', {}).get('is_q2_visualization', False),
                          "Q2 visualization flag should be set in metadata")

    def test_q2_template_contains_required_components(self):
        """Test that Q2 template contains all required components."""
        with patch('loguru.logger') as mock_logger:
            mock_logger.bind.return_value = mock_logger
            
            from src.utils.prompt_manager import PromptManager
            
            pm = PromptManager()
            
            # Get the Q2 template
            template = pm.q2_system_visualization_template
            
            # Convert template to string for content checking
            template_str = str(template)
            
            # Required components from the Q2 specification
            required_components = [
                "mermaid",
                "graph TB",
                "Frontend Layer",
                "API Gateway",
                "Microservices", 
                "Data Layer",
                "Message Infrastructure",
                "car-web-client",
                "car-listing-service",
                "car-order-service",
                "car-notification-service",
                "PostgreSQL",
                "MongoDB",
                "RabbitMQ"
            ]
            
            for component in required_components:
                self.assertIn(component, template_str,
                            f"Q2 template should contain '{component}'")

    def test_non_q2_queries_use_normal_templates(self):
        """Test that non-Q2 queries don't use the Q2 template."""
        with patch('loguru.logger') as mock_logger:
            mock_logger.bind.return_value = mock_logger
            
            from src.utils.prompt_manager import PromptManager
            from src.workflows.workflow_states import QueryIntent
            from langchain.schema import Document
            
            pm = PromptManager()
            
            # Test non-Q2 query
            query = "How do I implement a function?"
            context_docs = [
                Document(
                    page_content="Sample code",
                    metadata={"file_path": "src/code.py", "repository": "test-repo"}
                )
            ]
            
            # Test with Q2 flag disabled (normal case)
            result = pm.create_query_prompt(
                query=query,
                context_documents=context_docs,
                query_intent=QueryIntent.CODE_SEARCH,
                is_q2_system_visualization=False
            )
            
            # Verify normal template is used
            self.assertNotEqual(result.get('template_type'), 'Q2SystemVisualizationTemplate',
                              "Non-Q2 queries should not use Q2SystemVisualizationTemplate")
            self.assertNotEqual(result.get('system_prompt_type'), 'q2_architecture',
                              "Non-Q2 queries should not use q2_architecture system prompt")
            self.assertFalse(result.get('metadata', {}).get('is_q2_visualization', False),
                           "Non-Q2 queries should not have Q2 visualization flag")


class TestQ2EndToEnd(unittest.TestCase):
    """End-to-end tests for Q2 feature."""

    def setUp(self):
        """Set up test environment."""
        os.environ['OPENAI_API_KEY'] = 'test_key'
        os.environ['GITHUB_TOKEN'] = 'test_token'
        os.environ['DATABASE_TYPE'] = 'chroma'
        os.environ['APP_ENV'] = 'development'

    def test_q2_complete_pipeline(self):
        """Test the complete Q2 processing pipeline."""
        with patch('src.config.query_patterns.load_query_patterns') as mock_patterns, \
             patch('loguru.logger') as mock_logger:
            
            # Mock configurations
            mock_config = Mock()
            mock_config.domain_patterns = []
            mock_config.technical_patterns = []
            mock_config.programming_patterns = []
            mock_config.api_patterns = []
            mock_config.database_patterns = []
            mock_config.architecture_patterns = []
            mock_config.max_terms = 10
            mock_config.min_word_length = 3
            mock_config.excluded_words = {'the', 'and', 'or', 'but', 'a', 'an'}
            mock_patterns.return_value = mock_config
            
            mock_logger.bind.return_value = mock_logger
            
            from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
            from src.utils.prompt_manager import PromptManager
            from src.workflows.workflow_states import create_query_state, QueryIntent
            from langchain.schema import Document
            
            # Step 1: Parse query
            query = "Show me how the four services are connected and explain what I'm looking at."
            handler = QueryParsingHandler()
            state = create_query_state(workflow_id="test-e2e", original_query=query)
            
            for step in ["parse_query", "validate_query", "analyze_intent"]:
                state = handler.execute_step(step, state)
            
            # Step 2: Generate prompt
            pm = PromptManager()
            context_docs = [
                Document(
                    page_content="Sample service implementation",
                    metadata={
                        "file_path": "car-listing-service/Services/CarService.cs",
                        "repository": "car-listing-service",
                        "language": "csharp"
                    }
                )
            ]
            
            result = pm.create_query_prompt(
                query=query,
                context_documents=context_docs,
                query_intent=state.get('query_intent'),
                is_q2_system_visualization=state.get('is_q2_system_visualization', False)
            )
            
            # Verify end-to-end pipeline
            self.assertTrue(state.get('is_q2_system_visualization', False),
                          "Q2 should be detected in parsing step")
            self.assertEqual(state.get('query_intent'), QueryIntent.ARCHITECTURE,
                           "Intent should be ARCHITECTURE")
            self.assertEqual(result.get('template_type'), 'Q2SystemVisualizationTemplate',
                           "Q2 template should be used in prompt generation")
            self.assertEqual(result.get('confidence_score'), 1.0,
                           "Q2 should have maximum confidence")


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestQ2SystemVisualization))
    suite.addTests(loader.loadTestsFromTestCase(TestQ2EndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)