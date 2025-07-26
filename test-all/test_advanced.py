#!/usr/bin/env python3
"""
Advanced Features Test for Knowledge Graph Agent.

Test script to verify advanced features like functions, search, and code analysis.
"""

import requests
import json
import time
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_functions_endpoint():
    """Test functions endpoint."""
    print("🔧 Testing Functions API...")
    
    try:
        response = requests.get(f"{BASE_URL}/functions", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Functions API: PASSED")
            print(f"   Total Functions: {result.get('total_functions', 0)}")
            
            functions = result.get('functions', [])
            for func in functions:
                print(f"   - {func.get('name', 'Unknown')}: {func.get('description', 'No description')}")
            
            return True
        else:
            print(f"❌ Functions API: FAILED (Status: {response.status_code})")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Functions API: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Functions API: FAILED (Error: {e})")
        return False

def test_search_endpoint():
    """Test search endpoint."""
    print("\n🔍 Testing Search API...")
    
    try:
        params = {"query": "chatbot agent", "limit": 5}
        response = requests.get(f"{BASE_URL}/search", params=params, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Search API: PASSED")
            print(f"   Query: {result.get('query', 'Unknown')}")
            print(f"   Total Results: {result.get('total_results', 0)}")
            
            results = result.get('results', [])
            for i, res in enumerate(results[:3], 1):
                print(f"   Result {i}: {res.get('content', 'No content')[:50]}...")
            
            return True
        else:
            print(f"❌ Search API: FAILED (Status: {response.status_code})")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Search API: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Search API: FAILED (Error: {e})")
        return False

def test_advanced_chat_with_functions():
    """Test advanced chat with function calling."""
    print("\n🤖 Testing Advanced Chat with Functions...")
    
    try:
        payload = {
            "message": "Search for chatbot related code in the codebase",
            "user_id": "function_test_user"
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=payload, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Advanced Chat (Functions): PASSED")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Advanced Chat (Functions): FAILED (Status: {response.status_code})")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Advanced Chat (Functions): FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Advanced Chat (Functions): FAILED (Error: {e})")
        return False

def test_architecture_overview():
    """Test architecture overview function."""
    print("\n🏗️ Testing Architecture Overview...")
    
    try:
        payload = {
            "message": "Get an overview of the system architecture",
            "user_id": "arch_test_user"
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=payload, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Architecture Overview: PASSED")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Architecture Overview: FAILED (Status: {response.status_code})")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Architecture Overview: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Architecture Overview: FAILED (Error: {e})")
        return False

def test_dependency_analysis():
    """Test dependency analysis function."""
    print("\n🔗 Testing Dependency Analysis...")
    
    try:
        payload = {
            "message": "Analyze dependencies between components",
            "user_id": "dep_test_user"
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=payload, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Dependency Analysis: PASSED")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Dependency Analysis: FAILED (Status: {response.status_code})")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Dependency Analysis: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Dependency Analysis: FAILED (Error: {e})")
        return False

def test_code_search():
    """Test code search functionality."""
    print("\n🔎 Testing Code Search...")
    
    try:
        payload = {
            "message": "Find all Python files in the codebase",
            "user_id": "code_search_user"
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=payload, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Code Search: PASSED")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Code Search: FAILED (Status: {response.status_code})")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Code Search: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Code Search: FAILED (Error: {e})")
        return False

def test_multiple_search_queries():
    """Test multiple search queries."""
    print("\n📝 Testing Multiple Search Queries...")
    
    queries = [
        "chatbot agent",
        "fastapi routes",
        "openai integration",
        "vector store",
        "langchain"
    ]
    
    results = []
    for i, query in enumerate(queries, 1):
        try:
            params = {"query": query, "limit": 3}
            response = requests.get(f"{BASE_URL}/search", params=params, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Query {i} ('{query}'): {result.get('total_results', 0)} results")
                results.append(True)
            else:
                print(f"❌ Query {i} ('{query}'): FAILED")
                results.append(False)
                
        except Exception as e:
            print(f"❌ Query {i} ('{query}'): ERROR - {e}")
            results.append(False)
    
    success_count = sum(results)
    print(f"\n📊 Search Queries: {success_count}/{len(queries)} successful")
    return success_count == len(queries)

def run_advanced_tests():
    """Run all advanced feature tests."""
    print("🚀 Knowledge Graph Agent - Advanced Features Tests")
    print("=" * 60)
    
    tests = [
        test_functions_endpoint,
        test_search_endpoint,
        test_advanced_chat_with_functions,
        test_architecture_overview,
        test_dependency_analysis,
        test_code_search,
        test_multiple_search_queries
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
    print("\n" + "=" * 60)
    print("📊 ADVANCED FEATURES TEST SUMMARY")
    print("=" * 60)
    
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
    
    print("\n💡 Advanced Features:")
    print("   - Function calling for code analysis")
    print("   - Semantic search in codebase")
    print("   - Architecture overview")
    print("   - Dependency analysis")
    print("   - Multi-query search testing")

if __name__ == "__main__":
    run_advanced_tests() 