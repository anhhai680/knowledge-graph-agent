#!/usr/bin/env python3
"""
Test what the Q2 prompt generation actually produces to see what would be sent to the LLM.
This will help us understand what the user should expect when the Q2 feature is working.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_q2_prompt_generation():
    """Test what prompt gets generated for Q2 queries."""
    print("🧪 Testing Q2 Prompt Generation")
    print("=" * 80)
    
    # Set environment
    os.environ['OPENAI_API_KEY'] = 'sk-test-key-placeholder'
    os.environ['GITHUB_TOKEN'] = 'ghp_test-token-placeholder'
    os.environ['DATABASE_TYPE'] = 'chroma'
    os.environ['APP_ENV'] = 'development'
    os.environ['CHROMA_COLLECTION_NAME'] = 'test-collection'
    
    try:
        from src.utils.prompt_manager import PromptManager
        from src.workflows.workflow_states import QueryIntent
        
        # Create prompt manager
        prompt_manager = PromptManager()
        
        # Test Q2 query
        query = "Show me how the four services are connected and explain what I'm looking at."
        
        print(f"📝 Testing query: '{query}'")
        print()
        
        # Generate Q2 prompt
        result = prompt_manager.create_query_prompt(
            query=query,
            context_documents=[],  # No documents for this test
            query_intent=QueryIntent.ARCHITECTURE,
            is_q2_system_visualization=True
        )
        
        print("📊 Prompt Generation Results:")
        print(f"  - Template type: {result.get('template_type')}")
        print(f"  - Confidence score: {result.get('confidence_score')}")
        print(f"  - System prompt type: {result.get('system_prompt_type')}")
        print()
        
        # Get the formatted prompt
        prompt = result.get('prompt')
        if hasattr(prompt, 'to_string'):
            prompt_text = prompt.to_string()
        elif hasattr(prompt, 'format'):
            prompt_text = str(prompt)
        else:
            prompt_text = str(prompt)
        
        print("🎯 Generated Prompt for LLM:")
        print("-" * 80)
        print(prompt_text)
        print("-" * 80)
        
        # Check if the prompt contains the expected Q2 elements
        has_mermaid = "mermaid" in prompt_text.lower()
        has_services = "car-" in prompt_text and "service" in prompt_text
        has_instructions = "conversational explanation" in prompt_text.lower()
        
        print()
        print("✅ Q2 Elements Check:")
        print(f"  - Contains Mermaid template: {has_mermaid}")
        print(f"  - Contains service references: {has_services}")
        print(f"  - Contains explanation instructions: {has_instructions}")
        
        if has_mermaid and has_services and has_instructions:
            print("\n🎉 Q2 prompt generation is working correctly!")
            print("   When connected to a real LLM, this should produce the expected Q2 response.")
            return True
        else:
            print("\n❌ Q2 prompt generation has issues.")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Q2 prompt generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_non_q2_query():
    """Test that non-Q2 queries don't use the Q2 template."""
    print("\n🔍 Testing Non-Q2 Query (for comparison)")
    print("=" * 50)
    
    try:
        from src.utils.prompt_manager import PromptManager
        from src.workflows.workflow_states import QueryIntent
        
        prompt_manager = PromptManager()
        
        # Test regular query
        query = "How do I implement a function?"
        
        result = prompt_manager.create_query_prompt(
            query=query,
            context_documents=[],
            query_intent=QueryIntent.CODE_SEARCH,
            is_q2_system_visualization=False  # Not Q2
        )
        
        print(f"📝 Testing query: '{query}'")
        print(f"  - Template type: {result.get('template_type')}")
        print(f"  - Confidence score: {result.get('confidence_score')}")
        print(f"  - System prompt type: {result.get('system_prompt_type')}")
        
        # Should NOT be Q2 template
        is_q2_template = result.get('template_type') == 'Q2SystemVisualizationTemplate'
        print(f"  - Is Q2 template: {is_q2_template}")
        
        return not is_q2_template  # Should be False for success
        
    except Exception as e:
        print(f"❌ Error testing non-Q2 query: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Q2 Prompt Generation Test")
    print("=" * 80)
    
    # Test Q2 prompt generation
    q2_works = test_q2_prompt_generation()
    
    # Test non-Q2 for comparison
    non_q2_works = test_non_q2_query()
    
    print("\n" + "=" * 80)
    print("📋 Test Results:")
    print(f"  Q2 Prompt Generation: {'✅ PASS' if q2_works else '❌ FAIL'}")
    print(f"  Non-Q2 Query Handling: {'✅ PASS' if non_q2_works else '❌ FAIL'}")
    
    if q2_works and non_q2_works:
        print("\n🎉 Q2 prompt system is working correctly!")
        print("   The issue might be in the LLM connection or web UI integration.")
        sys.exit(0)
    else:
        print("\n❌ Q2 prompt system has issues.")
        sys.exit(1)