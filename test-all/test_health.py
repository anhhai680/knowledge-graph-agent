#!/usr/bin/env python3
"""
Health Check Test for Knowledge Graph Agent.

Test script to verify the health and status of the Knowledge Graph Agent API.
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint."""
    print("🏥 Testing Health Check Endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health Check: PASSED")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Version: {data.get('version', 'unknown')}")
            print(f"   Uptime: {data.get('uptime', 0):.2f}s")
            return True
        else:
            print(f"❌ Health Check: FAILED (Status: {response.status_code})")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Health Check: FAILED (Connection Error)")
        print("   Make sure the server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Health Check: FAILED (Error: {e})")
        return False

def test_welcome_endpoint():
    """Test the welcome endpoint."""
    print("\n🏠 Testing Welcome Endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Welcome Endpoint: PASSED")
            print(f"   Message: {data.get('message', 'No message')}")
            return True
        else:
            print(f"❌ Welcome Endpoint: FAILED (Status: {response.status_code})")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Welcome Endpoint: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Welcome Endpoint: FAILED (Error: {e})")
        return False

def test_docs_endpoint():
    """Test the documentation endpoint."""
    print("\n📚 Testing Documentation Endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=10)
        
        if response.status_code == 200:
            print("✅ Documentation Endpoint: PASSED")
            print("   Interactive API docs available at http://localhost:8000/docs")
            return True
        else:
            print(f"❌ Documentation Endpoint: FAILED (Status: {response.status_code})")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Documentation Endpoint: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Documentation Endpoint: FAILED (Error: {e})")
        return False

def test_openapi_endpoint():
    """Test the OpenAPI schema endpoint."""
    print("\n🔧 Testing OpenAPI Schema Endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ OpenAPI Schema: PASSED")
            print(f"   Title: {data.get('info', {}).get('title', 'Unknown')}")
            print(f"   Version: {data.get('info', {}).get('version', 'Unknown')}")
            print(f"   Endpoints: {len(data.get('paths', {}))}")
            return True
        else:
            print(f"❌ OpenAPI Schema: FAILED (Status: {response.status_code})")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ OpenAPI Schema: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ OpenAPI Schema: FAILED (Error: {e})")
        return False

def run_health_tests():
    """Run all health-related tests."""
    print("🚀 Knowledge Graph Agent - Health Tests")
    print("=" * 50)
    
    tests = [
        test_health_endpoint,
        test_welcome_endpoint,
        test_docs_endpoint,
        test_openapi_endpoint
    ]
    
    results = []
    for test in tests:
        start_time = time.time()
        success = test()
        duration = time.time() - start_time
        results.append({
            "test": test.__name__,
            "success": success,
            "duration": duration
        })
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 HEALTH TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results if result["success"])
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print("\n❌ Failed Tests:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['test']}")
    
    print("\n💡 Next Steps:")
    print("   - If all tests passed, the server is healthy")
    print("   - If tests failed, check if the server is running")
    print("   - Run 'python main.py' to start the server")
    print("   - Check http://localhost:8000/docs for API documentation")

if __name__ == "__main__":
    run_health_tests() 