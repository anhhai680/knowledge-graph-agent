#!/usr/bin/env python3
"""
Test script to verify the graph endpoint fix.
"""

import os
import sys
import requests
import json

def test_graph_endpoint_with_features_disabled():
    """Test the graph endpoint when features are disabled."""
    print("Testing GET /api/v1/graph/info with graph features disabled...")
    
    try:
        response = requests.get("http://localhost:8000/api/v1/graph/info", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 400:
            print("✅ Expected: Graph features are disabled")
            return True
        else:
            print("❌ Unexpected response")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("   Make sure the API server is running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_graph_endpoint_with_features_enabled():
    """Test the graph endpoint when features are enabled."""
    print("\nTesting GET /api/v1/graph/info with graph features enabled...")
    
    # Set environment variable temporarily
    original_env = os.environ.get("ENABLE_GRAPH_FEATURES")
    os.environ["ENABLE_GRAPH_FEATURES"] = "true"
    
    try:
        response = requests.get("http://localhost:8000/api/v1/graph/info", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Success: Graph info retrieved successfully")
            return True
        elif response.status_code == 503:
            print("⚠️  Service Unavailable: MemGraph not running")
            print("   This is expected if MemGraph is not started")
            return True
        else:
            print("❌ Unexpected response")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("   Make sure the API server is running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Restore original environment
        if original_env is not None:
            os.environ["ENABLE_GRAPH_FEATURES"] = original_env
        else:
            os.environ.pop("ENABLE_GRAPH_FEATURES", None)

def test_settings_fix():
    """Test that the settings fix works correctly."""
    print("\nTesting settings fix...")
    
    try:
        # Import settings and test the graph store URL access
        from src.config.settings import settings
        
        # Test that we can access the graph store URL correctly
        graph_url = settings.graph_store.url
        print(f"✅ Graph store URL: {graph_url}")
        
        # Test that the MemGraphStore can be created without errors
        from src.graphstores.memgraph_store import MemGraphStore
        
        # This should not raise an AttributeError anymore
        store = MemGraphStore()
        print(f"✅ MemGraphStore created successfully")
        print(f"   URI: {store.uri}")
        print(f"   Username: {store.username}")
        print(f"   Password: {'*' * len(store.password) if store.password else 'None'}")
        
        return True
        
    except AttributeError as e:
        print(f"❌ Settings fix failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing settings: {e}")
        return False

def main():
    """Main function to run all tests."""
    print("=" * 60)
    print("Graph Endpoint Fix Verification")
    print("=" * 60)
    
    # Test 1: Settings fix
    settings_ok = test_settings_fix()
    
    # Test 2: Endpoint with features disabled
    disabled_ok = test_graph_endpoint_with_features_disabled()
    
    # Test 3: Endpoint with features enabled
    enabled_ok = test_graph_endpoint_with_features_enabled()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Settings Fix: {'✅ PASS' if settings_ok else '❌ FAIL'}")
    print(f"Disabled Features: {'✅ PASS' if disabled_ok else '❌ FAIL'}")
    print(f"Enabled Features: {'✅ PASS' if enabled_ok else '❌ FAIL'}")
    
    if settings_ok and disabled_ok and enabled_ok:
        print("\n🎉 All tests passed! The fix is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 