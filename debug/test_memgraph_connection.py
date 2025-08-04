#!/usr/bin/env python3
"""
Test script for debugging MemGraph connection issues.

This script tests MemGraph connectivity using various connection parameters
and provides detailed diagnostic information.
"""

import os
import sys
import time
from typing import Optional

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
except ImportError:
    print("Error: neo4j package not installed. Run: pip install neo4j")
    sys.exit(1)


def test_memgraph_connection(
    uri: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    max_retries: int = 3
) -> bool:
    """
    Test MemGraph connection with retry logic.
    
    Args:
        uri: MemGraph connection URI
        username: Username for authentication
        password: Password for authentication
        max_retries: Maximum number of connection attempts
    
    Returns:
        bool: True if connection successful, False otherwise
    """
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
    
    print("
Testing common MemGraph connection patterns:")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"
{test_case['name']}: {test_case['description']}")
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
        print("
⚠️  Graph features are disabled in environment")
        print("Set ENABLE_GRAPH_FEATURES=true to enable")
    
    print(f"Configured URI: {uri}")
    print()
    
    # Test configured URI first
    print("Testing configured connection:")
    print("-" * 30)
    success = test_memgraph_connection(uri, username, password)
    
    if not success:
        print("
Trying alternative connection patterns...")
        working_uri = test_common_uris()
        
        if working_uri:
            print(f"
✓ Working URI found: {working_uri}")
            print(f"💡 Update your GRAPH_STORE_URL to: {working_uri}")
        else:
            print("
✗ No working connection found")
            print("
Troubleshooting tips:")
            print("1. Ensure MemGraph is running (docker-compose up memgraph)")
            print("2. Check if port 7687 is accessible")
            print("3. Verify network configuration in docker-compose.yml")
            print("4. Check MemGraph logs: docker-compose logs memgraph")


if __name__ == "__main__":
    main()
    
    # Test configured URI first
    print("Testing configured connection:")
    print("-" * 30)
    success = test_memgraph_connection(uri, username, password)
    
    if not success:
        print("\nTrying alternative connection patterns...")
        working_uri = test_common_uris()
        
        if working_uri:
            print(f"\n✓ Working URI found: {working_uri}")
            print(f"💡 Update your GRAPH_STORE_URL to: {working_uri}")
        else:
            print("\n✗ No working connection found")
            print("\nTroubleshooting tips:")
            print("1. Ensure MemGraph is running (docker-compose up memgraph)")
            print("2. Check if port 7687 is accessible")
            print("3. Verify network configuration in docker-compose.yml")
            print("4. Check MemGraph logs: docker-compose logs memgraph")


if __name__ == "__main__":
    main()
    print(f"Username: {username if username else 'None'}")
    print(f"Password: {'*' * len(password) if password else 'None'}")
    print()
    
    try:
        # Try to connect
        if username and password:
            driver = GraphDatabase.driver(uri, auth=(username, password))
        else:
            driver = GraphDatabase.driver(uri)
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                print("✅ SUCCESS: Connected to MemGraph successfully!")
                print("   The graph database is running and accessible.")
                return True
            else:
                print("❌ FAILED: Connection test query returned unexpected result")
                return False
                
    except ServiceUnavailable as e:
        print("❌ FAILED: Service unavailable")
        print(f"   Error: {e}")
        print()
        print("🔧 TROUBLESHOOTING:")
        print("   1. Check if MemGraph is running:")
        print("      docker ps | grep memgraph")
        print()
        print("   2. Start MemGraph if not running:")
        print("      docker-compose up -d memgraph")
        print()
        print("   3. Check MemGraph logs:")
        print("      docker-compose logs memgraph")
        print()
        print("   4. Verify the connection URL is correct:")
        print(f"      Current URL: {uri}")
        print("      Expected: bolt://localhost:7687 (or bolt://memgraph:7687 in Docker)")
        return False
        
    except AuthError as e:
        print("❌ FAILED: Authentication error")
        print(f"   Error: {e}")
        print()
        print("🔧 TROUBLESHOOTING:")
        print("   Check your MemGraph credentials in environment variables:")
        print("   - GRAPH_STORE_USER")
        print("   - GRAPH_STORE_PASSWORD")
        return False
        
    except Exception as e:
        print("❌ FAILED: Unexpected error")
        print(f"   Error: {e}")
        print()
        print("🔧 TROUBLESHOOTING:")
        print("   1. Check if MemGraph is running")
        print("   2. Verify network connectivity")
        print("   3. Check firewall settings")
        return False
        
    finally:
        if 'driver' in locals():
            driver.close()

def check_docker_status():
    """Check if MemGraph Docker container is running."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=memgraph", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            print("✅ Docker: MemGraph container is running")
            return True
        else:
            print("❌ Docker: MemGraph container is not running")
            return False
            
    except FileNotFoundError:
        print("⚠️  Docker: Docker command not found")
        return False
    except Exception as e:
        print(f"⚠️  Docker: Error checking container status: {e}")
        return False

def main():
    """Main function to run diagnostics."""
    print("=" * 60)
    print("MemGraph Connection Diagnostics")
    print("=" * 60)
    print()
    
    # Check Docker status first
    docker_running = check_docker_status()
    print()
    
    # Test connection
    connection_success = test_memgraph_connection()
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY:")
    if connection_success:
        print("✅ MemGraph is accessible and ready to use!")
    else:
        print("❌ MemGraph connection failed")
        if not docker_running:
            print("💡 Suggestion: Start MemGraph with 'docker-compose up -d memgraph'")
        print()
        print("To enable graph features, set the environment variable:")
        print("export ENABLE_GRAPH_FEATURES=true")
    print("=" * 60)

if __name__ == "__main__":
    main() 