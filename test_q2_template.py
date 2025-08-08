#!/usr/bin/env python3
"""
Test Q2 template selection without requiring LLM calls.
This will verify that the Q2 feature correctly selects the specialized template.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_q2_template_selection():
    """Test that Q2 queries select the correct specialized template."""
    print("🧪 Testing Q2 Template Selection")
    print("=" * 60)
    
    # Set environment
    os.environ['OPENAI_API_KEY'] = 'sk-test-key-placeholder'
    os.environ['GITHUB_TOKEN'] = 'ghp_test-token-placeholder'
    os.environ['DATABASE_TYPE'] = 'chroma'
    os.environ['APP_ENV'] = 'development'
    os.environ['CHROMA_COLLECTION_NAME'] = 'test-collection'
    
    try:
        from src.utils.prompt_manager import PromptManager
        from src.workflows.workflow_states import QueryIntent
        from langchain.schema import Document
        
        # Create prompt manager
        pm = PromptManager()
        
        # Test Q2 query
        query = "Show me how the four services are connected and explain what I'm looking at."
        
        # Create some sample context documents (empty is fine for this test)
        context_docs = []
        
        print(f"📝 Testing Query: '{query}'")
        print()
        
        # Test with Q2 flag set to True
        print("🎯 Testing with Q2 visualization flag = True")
        prompt_result = pm.create_query_prompt(
            query=query,
            context_documents=context_docs,
            query_intent=QueryIntent.ARCHITECTURE,
            is_q2_system_visualization=True,
        )
        
        print(f"Template Type: {prompt_result.get('template_type')}")
        print(f"Confidence Score: {prompt_result.get('confidence_score')}")
        print(f"System Prompt Type: {prompt_result.get('system_prompt_type')}")
        print(f"Q2 Visualization: {prompt_result.get('metadata', {}).get('is_q2_visualization')}")
        
        # Check if the correct Q2 template was selected
        is_q2_template = prompt_result.get('template_type') == 'Q2SystemVisualizationTemplate'
        has_max_confidence = prompt_result.get('confidence_score') == 1.0
        has_q2_metadata = prompt_result.get('metadata', {}).get('is_q2_visualization') == True
        
        print()
        print("✅ Results:")
        print(f"  Q2 Template Selected: {is_q2_template}")
        print(f"  Max Confidence (1.0): {has_max_confidence}")  
        print(f"  Q2 Metadata Set: {has_q2_metadata}")
        
        # Test without Q2 flag for comparison
        print()
        print("🎯 Testing with Q2 visualization flag = False (comparison)")
        prompt_result_normal = pm.create_query_prompt(
            query=query,
            context_documents=context_docs,
            query_intent=QueryIntent.ARCHITECTURE,
            is_q2_system_visualization=False,
        )
        
        print(f"Template Type: {prompt_result_normal.get('template_type')}")
        print(f"Q2 Visualization: {prompt_result_normal.get('metadata', {}).get('is_q2_visualization', False)}")
        
        # Verify the templates are different
        templates_different = (
            prompt_result.get('template_type') != prompt_result_normal.get('template_type')
        )
        
        print()
        if is_q2_template and has_max_confidence and has_q2_metadata and templates_different:
            print("🎉 Q2 Template Selection Works Correctly!")
            print("  ✓ Q2 queries use specialized Q2SystemVisualizationTemplate")
            print("  ✓ Q2 queries get maximum confidence (1.0)")
            print("  ✓ Q2 metadata is properly set")
            print("  ✓ Q2 and normal templates are different")
            return True
        else:
            print("❌ Q2 Template Selection Failed!")
            print(f"  Q2 Template: {is_q2_template}")
            print(f"  Max Confidence: {has_max_confidence}")
            print(f"  Q2 Metadata: {has_q2_metadata}")
            print(f"  Different Templates: {templates_different}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Q2 template selection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_q2_template_contains_mermaid():
    """Test that the Q2 template contains the expected Mermaid diagram."""
    print("\n🔍 Testing Q2 Template Content")
    print("=" * 60)
    
    try:
        from src.utils.prompt_manager import PromptManager
        from src.workflows.workflow_states import QueryIntent
        
        pm = PromptManager()
        
        # Get the Q2 template
        q2_template = pm.q2_system_visualization_template
        
        # Format the template to see its content
        sample_values = {
            "system_prompt": "Test system prompt",
            "query": "Test query",
            "context_documents": "Test context",
            "format_instructions": "Test format",
        }
        
        formatted_prompt = q2_template.format_prompt(**sample_values)
        prompt_text = str(formatted_prompt)
        
        print("🔍 Checking Q2 template content...")
        
        # Debug: print a snippet of the prompt to see what's there
        if "Here's how" in prompt_text:
            start_idx = prompt_text.find("Here's how")
            snippet = prompt_text[start_idx:start_idx+100]
            print(f"  Debug snippet: '{snippet}'")
        
        # Check for key components
        has_mermaid = "```mermaid" in prompt_text
        has_graph_tb = "graph TB" in prompt_text
        has_services = "car-listing-service" in prompt_text and "car-order-service" in prompt_text
        has_infrastructure = "RabbitMQ" in prompt_text and "PostgreSQL" in prompt_text
        has_connections = "-->" in prompt_text
        has_explanation_format = "Here's how these connections" in prompt_text or "how these connections are implemented" in prompt_text
        
        print(f"  ✓ Contains Mermaid diagram: {has_mermaid}")
        print(f"  ✓ Contains graph TB: {has_graph_tb}")
        print(f"  ✓ Contains service names: {has_services}")
        print(f"  ✓ Contains infrastructure: {has_infrastructure}")
        print(f"  ✓ Contains connections (-->): {has_connections}")
        print(f"  ✓ Contains explanation format: {has_explanation_format}")
        
        print()
        if all([has_mermaid, has_graph_tb, has_services, has_infrastructure, has_connections, has_explanation_format]):
            print("🎉 Q2 Template Contains All Required Components!")
            return True
        else:
            print("❌ Q2 Template Missing Required Components!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Q2 template content: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Q2 Template Testing")
    print("=" * 80)
    
    # Test template selection
    selection_works = test_q2_template_selection()
    
    # Test template content
    content_works = test_q2_template_contains_mermaid()
    
    print("\n" + "=" * 80)
    print("📋 Test Results:")
    print(f"  Q2 Template Selection: {'✅ PASS' if selection_works else '❌ FAIL'}")
    print(f"  Q2 Template Content: {'✅ PASS' if content_works else '❌ FAIL'}")
    
    if selection_works and content_works:
        print("🎉 Q2 Template System is Working Correctly!")
        print("  The Q2 feature should work once connected to a real LLM.")
        sys.exit(0)
    else:
        print("❌ Q2 Template System has issues.")
        sys.exit(1)