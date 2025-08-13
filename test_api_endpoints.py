#!/usr/bin/env python3
"""
Simple API endpoint test for Generic Q&A Agent.

This script tests the API endpoints to demonstrate the Generic Q&A Agent
working through the REST API interface.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from src.api.main import create_app


def test_generic_qa_endpoints():
    """Test Generic Q&A API endpoints."""
    print("🔧 Testing Generic Q&A API Endpoints...")
    
    # Create test client
    app = create_app()
    client = TestClient(app)
    
    # Test 1: Get agent status
    print("  📊 Testing agent status endpoint...")
    try:
        response = client.get("/api/v1/generic-qa/status")
        if response.status_code == 200:
            data = response.json()
            print(f"    ✅ Agent Status: {data['agent_name']} - {data['status']}")
            print(f"    ✅ Supported Categories: {len(data['supported_categories'])}")
            print(f"    ✅ Available Templates: {len(data['available_templates'])}")
        else:
            print(f"    ❌ Status endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"    ❌ Status endpoint error: {e}")
    
    # Test 2: Get supported categories
    print("  📋 Testing categories endpoint...")
    try:
        response = client.get("/api/v1/generic-qa/categories")
        if response.status_code == 200:
            categories = response.json()
            print(f"    ✅ Retrieved {len(categories)} categories:")
            for cat in categories[:3]:  # Show first 3
                print(f"      - {cat['category']}: {cat['description'][:60]}...")
        else:
            print(f"    ❌ Categories endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"    ❌ Categories endpoint error: {e}")
    
    # Test 3: Analyze question only
    print("  🔍 Testing question analysis endpoint...")
    try:
        response = client.post(
            "/api/v1/generic-qa/analyze-question",
            params={"question": "What is the architecture pattern?"}
        )
        if response.status_code == 200:
            analysis = response.json()
            print(f"    ✅ Question Category: {analysis['category']}")
            print(f"    ✅ Confidence: {analysis['confidence']:.2f}")
            print(f"    ✅ Is Reliable: {analysis['is_reliable']}")
        else:
            print(f"    ❌ Analysis endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"    ❌ Analysis endpoint error: {e}")
    
    # Test 4: Full question processing
    print("  🤖 Testing full question processing endpoint...")
    try:
        request_data = {
            "question": "What is the business capability of this system?",
            "repository_identifier": "test/repository",
            "include_code_examples": True
        }
        response = client.post("/api/v1/generic-qa/ask", json=request_data)
        if response.status_code == 200:
            result = response.json()
            print(f"    ✅ Question processed successfully")
            print(f"    ✅ Category: {result['question_category']}")
            print(f"    ✅ Confidence: {result['confidence_score']:.2f}")
            print(f"    ✅ Processing Time: {result['processing_time_ms']}ms")
            print(f"    ✅ Template Used: {result['template_used']}")
            print(f"    ✅ Response Sections: {len(result['structured_response'].get('sections', {}))}")
        else:
            print(f"    ❌ Full processing failed: {response.status_code}")
            print(f"        Error: {response.text}")
    except Exception as e:
        print(f"    ❌ Full processing error: {e}")
    
    # Test 5: Get available templates
    print("  📋 Testing templates endpoint...")
    try:
        response = client.get("/api/v1/generic-qa/templates")
        if response.status_code == 200:
            templates = response.json()
            print(f"    ✅ Available Templates: {templates}")
        else:
            print(f"    ❌ Templates endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"    ❌ Templates endpoint error: {e}")
    
    print()


def test_health_check():
    """Test basic health check."""
    print("❤️ Testing Health Check...")
    
    app = create_app()
    client = TestClient(app)
    
    try:
        response = client.get("/health")
        if response.status_code == 200:
            health = response.json()
            print(f"  ✅ System Status: {health['status']}")
            print(f"  ✅ Version: {health['version']}")
            workflows = health.get('components', {}).get('workflows', {})
            print(f"  ✅ Workflows: indexing={workflows.get('indexing')}, query={workflows.get('query')}")
        else:
            print(f"  ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Health check error: {e}")
    
    print()


def main():
    """Run all API tests."""
    print("🚀 Generic Q&A Agent API Testing")
    print("=" * 50)
    
    test_health_check()
    test_generic_qa_endpoints()
    
    print("✅ API testing completed!")


if __name__ == "__main__":
    main()