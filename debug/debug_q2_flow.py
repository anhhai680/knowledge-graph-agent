#!/usr/bin/env python3
"""
Debug script to trace Q2 query flow through the system.
"""

import asyncio
from src.workflows.query_workflow import QueryWorkflow
from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
from src.agents.rag_agent import RAGAgent

async def debug_q2_flow():
    """Debug the Q2 query flow to identify the issue."""
    
    query = "Show me the system architecture"
    print(f"🔍 Testing query: '{query}'")
    print("=" * 60)
    
    # 1. Test Q2 detection directly
    print("1. Direct Q2 Detection Test:")
    handler = QueryParsingHandler()
    is_q2_direct = handler._is_q2_system_relationship_query(query)
    print(f"   Q2 Detection Result: {is_q2_direct}")
    print()
    
    # 2. Test workflow processing
    print("2. Workflow Processing Test:")
    workflow = QueryWorkflow()
    try:
        result = await workflow.run(query=query, repositories=[], languages=[], k=5)
        print(f"   Query Intent: {result.get('query_intent')}")
        print(f"   Q2 Flag: {result.get('is_q2_system_visualization', False)}")
        llm_response = result.get('llm_generation', {}).get('generated_response', '')
        print(f"   LLM Response Length: {len(llm_response or '')}")
        print(f"   Processing Status: {result.get('llm_generation', {}).get('status')}")
        print()
        
        # 3. Test RAGAgent processing
        print("3. RAGAgent Processing Test:")
        rag_agent = RAGAgent(workflow=workflow)
        agent_result = await rag_agent._process_input({
            "query": query,
            "repositories": [],
            "language_filter": [],
            "top_k": 5
        })
        
        agent_answer = agent_result.get('answer', '')
        print(f"   Agent Answer Length: {len(agent_answer or '')}")
        print(f"   Template Type: {agent_result.get('prompt_metadata', {}).get('template_type')}")
        print(f"   Is Q2 Visualization: {agent_result.get('prompt_metadata', {}).get('metadata', {}).get('is_q2_visualization', False)}")
        print(f"   Answer Preview: {(agent_answer or '')[:200]}...")
        print()
        
        # 4. Check if Mermaid content exists
        answer = agent_result.get('answer', '') or ''
        has_mermaid = 'mermaid' in answer.lower() or 'graph TB' in answer
        print(f"4. Response Analysis:")
        print(f"   Contains Mermaid: {has_mermaid}")
        print(f"   Response Type: {'Q2 System Visualization' if has_mermaid else 'Standard Response'}")
        
    except Exception as e:
        print(f"   Error during workflow processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_q2_flow())
