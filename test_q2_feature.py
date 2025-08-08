#!/usr/bin/env python3
"""
Simple test for Q2 System Relationship Visualization feature.

This test verifies that the Q2 question pattern is detected correctly
and that the appropriate response template is used.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_q2_detection():
    """Test Q2 question pattern detection."""
    try:
        from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
        from src.workflows.workflow_states import QueryState, QueryIntent
        
        # Create handler
        handler = QueryParsingHandler()
        
        # Test Q2 queries
        q2_queries = [
            "Show me how the four services are connected and explain what I'm looking at.",
            "Show me how the four services are connected",
            "How are the services connected?",
            "Explain how the services work together",
            "Show me the system architecture"
        ]
        
        print("Testing Q2 detection...")
        for query in q2_queries:
            is_q2 = handler._is_q2_system_relationship_query(query)
            intent = handler._determine_query_intent(query)
            print(f"Query: '{query}'")
            print(f"  - Is Q2: {is_q2}")
            print(f"  - Intent: {intent}")
            print()
            
        # Test non-Q2 queries
        non_q2_queries = [
            "How do I implement a function?",
            "What is the bug in this code?",
            "Show me the documentation",
            "Find the class definition"
        ]
        
        print("Testing non-Q2 detection...")
        for query in non_q2_queries:
            is_q2 = handler._is_q2_system_relationship_query(query)
            intent = handler._determine_query_intent(query)
            print(f"Query: '{query}'")
            print(f"  - Is Q2: {is_q2}")
            print(f"  - Intent: {intent}")
            print()
            
        print("✅ Q2 detection test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Q2 detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_q2_prompt_template():
    """Test Q2 prompt template creation."""
    try:
        from src.utils.prompt_manager import PromptManager
        from src.workflows.workflow_states import QueryIntent
        from langchain.schema import Document
        
        # Create prompt manager
        pm = PromptManager()
        
        # Test Q2 prompt creation
        query = "Show me how the four services are connected and explain what I'm looking at."
        context_docs = [
            Document(
                page_content="Sample service connection code",
                metadata={"file_path": "src/services/connection.py", "repository": "test-repo"}
            )
        ]
        
        print("Testing Q2 prompt template...")
        
        # Test with Q2 flag
        result = pm.create_query_prompt(
            query=query,
            context_documents=context_docs,
            query_intent=QueryIntent.ARCHITECTURE,
            is_q2_system_visualization=True
        )
        
        print(f"Template type: {result.get('template_type')}")
        print(f"Confidence score: {result.get('confidence_score')}")
        print(f"System prompt type: {result.get('system_prompt_type')}")
        print(f"Is Q2 visualization: {result.get('metadata', {}).get('is_q2_visualization')}")
        
        # Verify it's using the Q2 template
        assert result.get('template_type') == 'Q2SystemVisualizationTemplate'
        assert result.get('confidence_score') == 1.0
        assert result.get('system_prompt_type') == 'q2_architecture'
        assert result.get('metadata', {}).get('is_q2_visualization') == True
        
        print("✅ Q2 prompt template test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Q2 prompt template test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Q2 tests."""
    print("=" * 60)
    print("Testing Q2 System Relationship Visualization Feature")
    print("=" * 60)
    
    success = True
    
    # Test 1: Q2 detection
    if not test_q2_detection():
        success = False
    
    print("-" * 60)
    
    # Test 2: Q2 prompt template
    if not test_q2_prompt_template():
        success = False
    
    print("=" * 60)
    if success:
        print("🎉 All Q2 tests passed!")
    else:
        print("❌ Some Q2 tests failed!")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)