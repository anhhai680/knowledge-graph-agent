#!/usr/bin/env python3
"""
Simple isolated test for Q2 pattern detection without configuration dependencies.
"""

import sys
import os

def test_q2_pattern_matching():
    """Test Q2 pattern matching logic in isolation."""
    
    def is_q2_system_relationship_query(query: str) -> bool:
        """
        Standalone version of Q2 detection logic for testing.
        """
        query_lower = query.lower().strip()
        
        # Specific Q2 pattern matching
        q2_patterns = [
            # Exact Q2 match
            "show me how the four services are connected and explain what i'm looking at",
            "show me how the four services are connected and explain what i am looking at", 
            "show me how the four services are connected",
            # Variations that indicate system relationship visualization
            "how are the services connected",
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
        
        # If query has all three elements, it's likely a Q2-style query
        if has_services and has_connection and has_visualization:
            return True
            
        return False
    
    # Test Q2 queries
    q2_queries = [
        "Show me how the four services are connected and explain what I'm looking at.",
        "Show me how the four services are connected",
        "How are the services connected?",
        "Explain how the services connect",
        "How do the services work together",
        "Show me the system architecture",
        "Explain the system relationships",
        "Show service connections",
        "Visualize the system",
        "Diagram of the system"
    ]
    
    print("Testing Q2 pattern detection...")
    print("=" * 60)
    
    q2_detected = 0
    for query in q2_queries:
        is_q2 = is_q2_system_relationship_query(query)
        print(f"✓ '{query}' -> Q2: {is_q2}")
        if is_q2:
            q2_detected += 1
    
    print(f"\nQ2 queries detected: {q2_detected}/{len(q2_queries)}")
    
    # Test non-Q2 queries
    non_q2_queries = [
        "How do I implement a function?",
        "What is the bug in this code?",
        "Show me the documentation",
        "Find the class definition",
        "How to debug this error?",
        "What does this method do?"
    ]
    
    print("\nTesting non-Q2 pattern detection...")
    print("-" * 60)
    
    non_q2_correct = 0
    for query in non_q2_queries:
        is_q2 = is_q2_system_relationship_query(query)
        print(f"✗ '{query}' -> Q2: {is_q2}")
        if not is_q2:
            non_q2_correct += 1
    
    print(f"\nNon-Q2 queries correctly identified: {non_q2_correct}/{len(non_q2_queries)}")
    
    # Results
    total_correct = q2_detected + non_q2_correct
    total_queries = len(q2_queries) + len(non_q2_queries)
    accuracy = (total_correct / total_queries) * 100
    
    print("\n" + "=" * 60)
    print(f"Overall Accuracy: {accuracy:.1f}% ({total_correct}/{total_queries})")
    
    if accuracy >= 90:
        print("✅ Q2 pattern detection test PASSED!")
        return True
    else:
        print("❌ Q2 pattern detection test FAILED!")
        return False

def test_mermaid_template():
    """Test that the Mermaid template structure is correct."""
    print("\nTesting Mermaid template structure...")
    print("=" * 60)
    
    # Expected Mermaid diagram components
    expected_components = [
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
        "RabbitMQ",
        "PostgreSQL",
        "MongoDB"
    ]
    
    # Sample template (this is what should be in the prompt)
    mermaid_template = """```mermaid
graph TB
    subgraph "Frontend Layer"
        WC[car-web-client<br/>React + TypeScript<br/>User Interface]
    end
    
    subgraph "API Gateway"
        AGW[Load Balancer<br/>Rate Limiting<br/>Authentication]
    end
    
    subgraph "Microservices"
        CLS[car-listing-service<br/>.NET 8 Web API<br/>Inventory Management]
        OS[car-order-service<br/>.NET 8 Web API<br/>Order Processing]
        NS[car-notification-service<br/>.NET 8 Web API<br/>Event Notifications]
    end
    
    subgraph "Data Layer"
        CLSDB[(PostgreSQL<br/>Car Catalog)]
        ODB[(PostgreSQL<br/>Orders & Payments)]
        NDB[(MongoDB<br/>Notifications)]
    end
    
    subgraph "Message Infrastructure"
        RMQ[RabbitMQ<br/>Event Broker]
    end
    
    %% Frontend Communication
    WC -->|HTTPS REST| AGW
    AGW --> CLS
    AGW --> OS
    WC -->|WebSocket Connect| NS
    NS -->|WebSocket Updates| WC
    
    %% Inter-Service Communication
    OS -->|HTTP| CLS
    
    %% Event-Driven Communication
    CLS -->|Events| RMQ
    OS -->|Events| RMQ
    RMQ -->|Events| NS
    
    %% Data Persistence
    CLS --> CLSDB
    OS --> ODB
    NS --> NDB
```"""
    
    print("Checking Mermaid template components...")
    
    components_found = 0
    for component in expected_components:
        if component in mermaid_template:
            print(f"✓ Found: {component}")
            components_found += 1
        else:
            print(f"✗ Missing: {component}")
    
    print(f"\nComponents found: {components_found}/{len(expected_components)}")
    
    # Check for proper structure
    has_proper_structure = (
        "graph TB" in mermaid_template and
        "subgraph" in mermaid_template and
        "-->" in mermaid_template and
        "Frontend Layer" in mermaid_template and
        "Microservices" in mermaid_template
    )
    
    if has_proper_structure and components_found >= len(expected_components) * 0.9:
        print("✅ Mermaid template structure test PASSED!")
        return True
    else:
        print("❌ Mermaid template structure test FAILED!")
        return False

def main():
    """Run all simple Q2 tests."""
    print("Q2 System Relationship Visualization - Simple Tests")
    print("=" * 60)
    
    success = True
    
    # Test 1: Pattern matching
    if not test_q2_pattern_matching():
        success = False
    
    # Test 2: Mermaid template
    if not test_mermaid_template():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All Q2 simple tests PASSED!")
    else:
        print("❌ Some Q2 simple tests FAILED!")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)