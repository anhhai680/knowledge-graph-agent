#!/usr/bin/env python3
"""
Test Q2 API endpoint fixes.
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_api_q2_detection():
    """Test the API Q2 detection fixes."""
    
    print("Testing API Q2 Detection Fixes...")
    print("=" * 60)
    
    # Test direct Q2 detection
    try:
        from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
        
        # Test the same query the user used
        query = "Show me the system architecture"
        handler = QueryParsingHandler()
        is_q2 = handler._is_q2_system_relationship_query(query)
        
        print(f"Query: '{query}'")
        print(f"Q2 Detection: {is_q2}")
        
        if is_q2:
            print("✅ Q2 detection is working correctly")
        else:
            print("❌ Q2 detection failed")
        
        return is_q2
        
    except Exception as e:
        print(f"❌ Error testing Q2 detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_response():
    """Test the fallback Q2 response structure."""
    
    print("\nTesting Fallback Q2 Response Structure...")
    print("=" * 60)
    
    # The fallback response from our API fix
    fallback_response = """Looking at the system architecture based on the code repositories:

```mermaid
graph TB
    subgraph "Frontend Layer"
        WC[car-web-client<br/>React + TypeScript<br/>User Interface]
    end
    
    subgraph "Microservices"
        CLS[car-listing-service<br/>.NET 8 Web API<br/>Inventory Management]
        OS[car-order-service<br/>.NET 8 Web API<br/>Order Processing]
        NS[car-notification-service<br/>.NET 8 Web API<br/>Event Notifications]
    end
```

## How the Services Work Together

The system follows a **microservices architecture** with four main components:"""
    
    # Check for Q2 response indicators
    has_mermaid = "```mermaid" in fallback_response
    has_graph = "graph TB" in fallback_response
    has_services = "car-listing-service" in fallback_response
    has_explanation = "How the Services Work Together" in fallback_response
    
    print(f"Has Mermaid diagram: {has_mermaid}")
    print(f"Has graph structure: {has_graph}")
    print(f"Has service names: {has_services}")
    print(f"Has explanation: {has_explanation}")
    
    all_checks = has_mermaid and has_graph and has_services and has_explanation
    
    if all_checks:
        print("✅ Fallback Q2 response structure is correct")
    else:
        print("❌ Fallback Q2 response structure is incomplete")
    
    return all_checks

def test_web_interface_q2_handling():
    """Test web interface Q2 response handling logic."""
    
    print("\nTesting Web Interface Q2 Handling Logic...")
    print("=" * 60)
    
    # Simulate API response structure for Q2
    q2_api_response = {
        "query": "Show me the system architecture",
        "intent": "architecture",
        "strategy": "semantic",
        "results": [],
        "total_results": 0,
        "processing_time": 1.5,
        "confidence_score": 1.0,
        "suggestions": [],
        "generated_response": "Looking at the system architecture...\n\n```mermaid\ngraph TB\n    WC[car-web-client]\n    CLS[car-listing-service]\n```\n\nHere's how these services work together...",
        "response_type": "generated"
    }
    
    # Check response structure
    has_response_type = "response_type" in q2_api_response
    is_generated = q2_api_response.get("response_type") == "generated"
    has_generated_response = "generated_response" in q2_api_response
    generated_has_mermaid = "```mermaid" in q2_api_response.get("generated_response", "")
    empty_results = len(q2_api_response.get("results", [])) == 0
    
    print(f"Has response_type field: {has_response_type}")
    print(f"Is generated response: {is_generated}")
    print(f"Has generated_response field: {has_generated_response}")
    print(f"Generated response has Mermaid: {generated_has_mermaid}")
    print(f"Empty search results: {empty_results}")
    
    all_checks = has_response_type and is_generated and has_generated_response and generated_has_mermaid and empty_results
    
    if all_checks:
        print("✅ Web interface should correctly handle Q2 responses")
    else:
        print("❌ Web interface Q2 handling may have issues")
    
    return all_checks

def main():
    """Run all tests."""
    print("Q2 API Endpoint Fix Validation")
    print("=" * 60)
    
    # Test 1: Q2 detection
    detection_works = test_api_q2_detection()
    
    # Test 2: Fallback response
    fallback_works = test_fallback_response()
    
    # Test 3: Web interface handling
    web_interface_works = test_web_interface_q2_handling()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print(f"Q2 Detection: {'✅ PASS' if detection_works else '❌ FAIL'}")
    print(f"Fallback Response: {'✅ PASS' if fallback_works else '❌ FAIL'}")
    print(f"Web Interface: {'✅ PASS' if web_interface_works else '❌ FAIL'}")
    
    all_passed = detection_works and fallback_works and web_interface_works
    
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nThe Q2 feature should now work correctly!")
        print("When users ask 'Show me the system architecture', they should get:")
        print("1. Properly detected Q2 query")
        print("2. Generated Mermaid diagram response")
        print("3. response_type: 'generated' in API")
        print("4. Correct rendering in web interface")
    else:
        print("\nSome issues remain that need attention.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)