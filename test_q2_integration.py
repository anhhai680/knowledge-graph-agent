#!/usr/bin/env python3
"""
Integration test for Q2 feature through the API workflow.

This test verifies the complete Q2 processing pipeline without requiring
external API keys or database connections.
"""

import sys
import os
import asyncio
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_q2_workflow_integration():
    """Test Q2 through the complete workflow."""
    print("Testing Q2 workflow integration...")
    
    try:
        # Set minimal environment
        os.environ['OPENAI_API_KEY'] = 'test_key'
        os.environ['GITHUB_TOKEN'] = 'test_token'
        os.environ['DATABASE_TYPE'] = 'chroma'
        os.environ['APP_ENV'] = 'development'
        
        # Import after setting environment
        from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
        from src.workflows.workflow_states import QueryState, QueryIntent, create_query_state
        
        # Create a query state for Q2
        query = "Show me how the four services are connected and explain what I'm looking at."
        
        # Initialize handler
        with patch('src.config.query_patterns.load_query_patterns') as mock_patterns:
            # Mock the patterns config
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
            
            handler = QueryParsingHandler()
        
        # Create initial state
        state = create_query_state(
            workflow_id="test-q2",
            original_query=query
        )
        
        # Execute parsing steps
        steps = ["parse_query", "validate_query", "analyze_intent"]
        
        for step in steps:
            print(f"Executing step: {step}")
            state = handler.execute_step(step, state)
            print(f"  Query intent: {state.get('query_intent')}")
            print(f"  Is Q2: {state.get('is_q2_system_visualization', False)}")
        
        # Verify Q2 detection
        assert state.get('is_q2_system_visualization') == True, "Q2 should be detected"
        assert state.get('query_intent') == QueryIntent.ARCHITECTURE, "Intent should be ARCHITECTURE"
        
        print("✅ Q2 workflow integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Q2 workflow integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_q2_prompt_manager_integration():
    """Test Q2 with the PromptManager."""
    print("\nTesting Q2 PromptManager integration...")
    
    try:
        # Set minimal environment
        os.environ['OPENAI_API_KEY'] = 'test_key'
        os.environ['GITHUB_TOKEN'] = 'test_token'
        os.environ['DATABASE_TYPE'] = 'chroma'
        os.environ['APP_ENV'] = 'development'
        
        # Mock loguru to avoid issues
        with patch('loguru.logger') as mock_logger:
            mock_logger.bind.return_value = mock_logger
            
            from src.utils.prompt_manager import PromptManager
            from src.workflows.workflow_states import QueryIntent
            from langchain.schema import Document
            
            # Create prompt manager
            pm = PromptManager()
            
            # Create test query and documents
            query = "Show me how the four services are connected and explain what I'm looking at."
            context_docs = [
                Document(
                    page_content="class CarService { public async Task<Car> GetCarAsync(int id) { /* implementation */ } }",
                    metadata={
                        "file_path": "car-listing-service/Services/CarService.cs",
                        "repository": "car-listing-service",
                        "language": "csharp",
                        "line_start": 15,
                        "line_end": 25
                    }
                ),
                Document(
                    page_content="export const useCars = () => { const [cars, setCars] = useState([]); /* implementation */ }",
                    metadata={
                        "file_path": "car-web-client/src/hooks/useCars.ts",
                        "repository": "car-web-client", 
                        "language": "typescript",
                        "line_start": 10,
                        "line_end": 30
                    }
                )
            ]
            
            # Test Q2 prompt creation
            result = pm.create_query_prompt(
                query=query,
                context_documents=context_docs,
                query_intent=QueryIntent.ARCHITECTURE,
                is_q2_system_visualization=True
            )
            
            # Verify Q2 template is used
            assert result.get('template_type') == 'Q2SystemVisualizationTemplate', f"Expected Q2 template, got {result.get('template_type')}"
            assert result.get('confidence_score') == 1.0, f"Expected confidence 1.0, got {result.get('confidence_score')}"
            assert result.get('system_prompt_type') == 'q2_architecture', f"Expected q2_architecture, got {result.get('system_prompt_type')}"
            assert result.get('metadata', {}).get('is_q2_visualization') == True, "Q2 visualization flag should be True"
            
            print("✅ Q2 PromptManager integration test PASSED!")
            return True
            
    except Exception as e:
        print(f"❌ Q2 PromptManager integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all integration tests."""
    print("=" * 80)
    print("Q2 System Relationship Visualization - Integration Tests")
    print("=" * 80)
    
    success = True
    
    # Test 1: Workflow integration
    if not await test_q2_workflow_integration():
        success = False
    
    # Test 2: PromptManager integration  
    if not await test_q2_prompt_manager_integration():
        success = False
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 All Q2 integration tests PASSED!")
    else:
        print("❌ Some Q2 integration tests FAILED!")
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)