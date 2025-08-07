#!/usr/bin/env python3
"""
Simple MemGraph connection test.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from graphstores.memgraph_store import MemGraphStore
    
    # Test connection
    print("🚀 Testing MemGraph Connection")
    print("=" * 40)
    
    store = MemGraphStore()
    connected = store.connect()
    
    if connected:
        print("✅ Connected to MemGraph successfully!")
        
        # Get graph info
        info = store.get_graph_info()
        print(f"📊 Node count: {info.get('node_count', 0)}")
        print(f"📊 Relationship count: {info.get('relationship_count', 0)}")
        
        # Test a simple query
        result = store.execute_query("MATCH (n) RETURN count(n) as total_nodes")
        print(f"🔍 Query result: {result.data}")
        
        store.disconnect()
    else:
        print("❌ Failed to connect to MemGraph")
        print("Make sure MemGraph is running: docker compose up -d memgraph")
        
except Exception as e:
    print(f"💥 Error: {e}")
    import traceback
    traceback.print_exc()
