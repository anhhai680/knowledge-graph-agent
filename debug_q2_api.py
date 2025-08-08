#!/usr/bin/env python3
"""
Debug script to test Q2 API response handling.
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_q2_detection_direct():
    """Test Q2 detection directly without dependencies."""
    
    print("Testing Q2 detection logic...")
    
    # Test the exact logic from the query parsing handler
    def is_q2_system_relationship_query(query: str) -> bool:
        """Direct implementation of Q2 detection logic."""
        query_lower = query.lower()
        
        q2_patterns = [
            "show me how the four services are connected",
            "show me how the four services are connected and explain what i'm looking at",
            "explain how the four services work together",
            "how are the four services connected", 
            "show me how services are connected",
            "explain how the services connect",
            "how do the services work together",
            "show the system architecture",
            "show me the system architecture",
            "explain the system relationships",
            "show service connections",
            "visualize the system",
            "diagram of the system"
        ]
        
        # Check for exact or close matches
        for pattern in q2_patterns:
            if pattern in query_lower:
                return True
        
        # Check for key combination patterns that indicate Q2-style queries
        has_services = any(term in query_lower for term in ["service", "services", "component", "components"])
        has_connection = any(term in query_lower for term in ["connect", "connected", "connection", "relationship", "work together"])
        has_visualization = any(term in query_lower for term in ["show", "explain", "diagram", "visualize", "architecture"])
        
        # If query has service + connection/visualization terms, it's likely Q2
        if has_services and (has_connection or has_visualization):
            return True
        
        return False
    
    # Test queries
    test_queries = [
        "Show me the system architecture",
        "show the system architecture", 
        "Show me how the four services are connected",
        "explain the system relationships",
        "How is authentication implemented?",  # Non-Q2
        "Find a bug in the code"  # Non-Q2
    ]
    
    print("\nQ2 Detection Results:")
    print("-" * 50)
    
    for query in test_queries:
        result = is_q2_system_relationship_query(query)
        print(f"'{query}' -> {result}")
    
    return True

def analyze_api_response_format():
    """Analyze the expected API response format for Q2 queries."""
    
    print("\nAnalyzing API Response Format...")
    print("-" * 50)
    
    # Expected Q2 response structure
    q2_response = {
        "query": "Show me the system architecture",
        "intent": "architecture",
        "strategy": "semantic",
        "results": [],  # Empty for Q2 queries
        "total_results": 0,
        "processing_time": 1.23,
        "confidence_score": 1.0,
        "suggestions": [],
        "generated_response": "Looking at the system architecture based on the code repositories:\n\n```mermaid\ngraph TB\n    subgraph \"Frontend Layer\"\n        WC[car-web-client<br/>React + TypeScript<br/>User Interface]\n    end\n    \n    subgraph \"API Gateway\"\n        AGW[Load Balancer<br/>Rate Limiting<br/>Authentication]\n    end\n    \n    subgraph \"Microservices\"\n        CLS[car-listing-service<br/>.NET 8 Web API<br/>Inventory Management]\n        OS[car-order-service<br/>.NET 8 Web API<br/>Order Processing]\n        NS[car-notification-service<br/>.NET 8 Web API<br/>Event Notifications]\n    end\n```\n\nHere's how these connections are implemented...",
        "response_type": "generated"
    }
    
    # Bad response (what user is seeing)
    bad_response = {
        "query": "Show me the system architecture",
        "intent": "architecture", 
        "strategy": "semantic",
        "results": [
            {
                "content": "\"license\": \"BlueOak-1.0.0\"\\n    },\\n    \"node_modules/param-case\": {...}",
                "metadata": {
                    "source": "github_git",
                    "repository": "anhhai680/car-web-client",
                    "language": "json"
                },
                "score": 0,
                "chunk_index": 575
            }
        ],
        "total_results": 5,
        "processing_time": 6.54,
        "confidence_score": 0.9,
        "suggestions": []
    }
    
    print("Expected Q2 Response:")
    print(f"- response_type: {q2_response.get('response_type')}")
    print(f"- generated_response length: {len(q2_response.get('generated_response', ''))}")
    print(f"- results count: {len(q2_response.get('results', []))}")
    print(f"- confidence: {q2_response.get('confidence_score')}")
    
    print("\nActual Bad Response:")
    print(f"- response_type: {bad_response.get('response_type', 'MISSING')}")
    print(f"- generated_response: {bad_response.get('generated_response', 'MISSING')}")
    print(f"- results count: {len(bad_response.get('results', []))}")
    print(f"- confidence: {bad_response.get('confidence_score')}")
    
    print("\nProblem Identified:")
    print("- API is not detecting Q2 query properly")
    print("- Falling back to regular search")
    print("- Returning search results instead of generated response")
    print("- Missing response_type and generated_response fields")
    
    return True

def check_web_interface_handling():
    """Check how web interface should handle Q2 responses."""
    
    print("\nAnalyzing Web Interface Response Handling...")
    print("-" * 50)
    
    # The web interface formatQueryResponse function logic
    print("Web Interface Logic:")
    print("1. Check for data.response_type === 'generated'")
    print("2. If generated, call formatGeneratedResponse(data.generated_response)")
    print("3. If not generated, display search results normally")
    
    print("\nWeb Interface Mermaid Handling:")
    print("1. Look for ```mermaid blocks in generated_response")
    print("2. Extract mermaid code")
    print("3. Display in styled code block with syntax highlighting")
    print("4. Add copy functionality")
    
    print("\nRequirements for Q2 to work:")
    print("✓ API must return response_type: 'generated'")
    print("✓ API must return generated_response with Mermaid diagram")
    print("✓ Web interface must detect response_type")
    print("✓ Web interface must format Mermaid diagrams properly")
    
    return True

def main():
    """Run all debug checks."""
    print("=" * 60)
    print("Q2 API Response Debug Analysis")
    print("=" * 60)
    
    # Test Q2 detection
    test_q2_detection_direct()
    
    # Analyze response format
    analyze_api_response_format()
    
    # Check web interface
    check_web_interface_handling()
    
    print("\n" + "=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)
    print("The issue is that the API is not detecting 'Show me the system architecture'")
    print("as a Q2 query and is falling back to regular search, which returns")
    print("irrelevant node_modules content instead of the generated Mermaid diagram.")
    print("")
    print("NEXT STEPS:")
    print("1. Fix Q2 detection in the API workflow")
    print("2. Ensure RAGAgent properly processes Q2 queries") 
    print("3. Verify API returns response_type: 'generated'")
    print("4. Test web interface Mermaid rendering")
    print("=" * 60)

if __name__ == "__main__":
    main()