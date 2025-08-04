#!/usr/bin/env python3
"""
Test script for debugging MemGraph connection issues.

This script tests MemGraph connectivity and provides detailed diagnostic information.
"""

import os
import sys
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
except ImportError:
    print("Error: neo4j package not installed. Run: pip install neo4j")
    sys.exit(1)


def test_memgraph_connection(uri, username=None, password=None, max_retries=3):
    """Test MemGraph connection with retry logic."""
    print(f"Testing MemGraph connection to: {uri}")
    print(f"Authentication: {'Yes' if username and password else 'No'}")
    print("-" * 50)
    
    driver = None
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")
            
            # Create driver
            if username and password:
                driver = GraphDatabase.driver(
                    uri,
                    auth=(username, password),
                    connection_timeout=10.0,
                    max_connection_lifetime=3600,
                )
            else:
                driver = GraphDatabase.driver(
                    uri,
                    connection_timeout=10.0,
                    max_connection_lifetime=3600,
                )
            
            # Test connection
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                test_value = record["test"]
                
                print(f"✓ Connection successful!")
                print(f"✓ Test query returned: {test_value}")
                
                # Test additional queries
                try:
                    result = session.run("MATCH (n) RETURN count(n) as node_count")
                    node_count = result.single()["node_count"]
                    print(f"✓ Node count: {node_count}")
                except Exception as e:
                    print(f"! Node count query failed: {e}")
                
                return True
                
        except ServiceUnavailable as e:
            print(f"✗ Service unavailable: {e}")
            
        except AuthError as e:
            print(f"✗ Authentication failed: {e}")
            break  # Don't retry auth errors
            
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            break  # Don't retry unexpected errors
        
        finally:
            if driver:
                try:
                    driver.close()
                except Exception:
                    pass
                driver = None
        
        # Wait before retrying
        if attempt < max_retries - 1:
            delay = 2 ** attempt  # Exponential backoff
            print(f"Waiting {delay} seconds before retry...")
            time.sleep(delay)
    
    print(f"✗ Failed to connect after {max_retries} attempts")
    return False


def test_common_uris():
    """Test common MemGraph URI patterns."""
    test_cases = [
        {
            "name": "Docker Compose (Service Name)",
            "uri": "bolt://memgraph:7687",
            "description": "Use this when running inside Docker Compose"
        },
        {
            "name": "Local Development",
            "uri": "bolt://localhost:7687",
            "description": "Use this when running MemGraph locally"
        },
        {
            "name": "Docker Host (macOS/Windows)",
            "uri": "bolt://host.docker.internal:7687",
            "description": "Use this when app is in Docker but MemGraph is on host"
        }
    ]
    
    print("\\nTesting common MemGraph connection patterns:")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\\n{test_case['name']}: {test_case['description']}")
        success = test_memgraph_connection(test_case["uri"], max_retries=1)
        if success:
            print(f"✓ {test_case['name']} connection works!")
            return test_case["uri"]
        else:
            print(f"✗ {test_case['name']} connection failed")
    
    return None


def main():
    """Main test function."""
    print("MemGraph Connection Test Tool")
    print("=" * 40)
    
    # Try to load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Loaded .env file")
    except ImportError:
        print("! python-dotenv not available, using system environment")
    
    # Get configuration from environment
    uri = os.getenv("GRAPH_STORE_URL", "bolt://localhost:7687")
    username = os.getenv("GRAPH_STORE_USER")
    password = os.getenv("GRAPH_STORE_PASSWORD")
    enabled = os.getenv("ENABLE_GRAPH_FEATURES", "false").lower() == "true"
    
    print(f"Graph features enabled: {enabled}")
    
    if not enabled:
        print("\\n⚠️  Graph features are disabled in environment")
        print("Set ENABLE_GRAPH_FEATURES=true to enable")
    
    print(f"Configured URI: {uri}")
    print()
    
    # Test configured URI first
    print("Testing configured connection:")
    print("-" * 30)
    success = test_memgraph_connection(uri, username, password)
    
    if not success:
        print("\\nTrying alternative connection patterns...")
        working_uri = test_common_uris()
        
        if working_uri:
            print(f"\\n✓ Working URI found: {working_uri}")
            print(f"💡 Update your GRAPH_STORE_URL to: {working_uri}")
        else:
            print("\\n✗ No working connection found")
            print("\\nTroubleshooting tips:")
            print("1. Ensure MemGraph is running (docker-compose up memgraph)")
            print("2. Check if port 7687 is accessible")
            print("3. Verify network configuration in docker-compose.yml")
            print("4. Check MemGraph logs: docker-compose logs memgraph")


if __name__ == "__main__":
    main()
