#!/usr/bin/env python3
"""
Test script to verify Neo4j serialization fix.
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphstores.memgraph_store import MemGraphStore, _serialize_neo4j_object, _serialize_record
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_graph_serialization():
    """Test graph query serialization."""
    print("🚀 Testing Graph Query Serialization Fix")
    print("=" * 60)
    
    try:
        # Initialize and connect to MemGraph
        store = MemGraphStore()
        store.connect()
        
        print("✅ Connected to MemGraph")
        
        # Test a simple query that returns nodes
        query = "MATCH (n:File) RETURN n LIMIT 2"
        result = store.execute_query(query)
        
        print(f"✅ Query executed successfully")
        print(f"📊 Query result type: {type(result)}")
        print(f"📊 Data type: {type(result.data)}")
        print(f"📊 Number of results: {len(result.data)}")
        
        # Check if data can be serialized (no Neo4j objects)
        import json
        try:
            json_str = json.dumps(result.data)
            print("✅ Data is JSON serializable")
            print(f"📊 Sample data: {result.data[0] if result.data else 'No data'}")
        except TypeError as e:
            print(f"❌ Data is NOT JSON serializable: {e}")
            return False
        
        # Test the API response model
        from api.models import GraphQueryResponse
        api_response = GraphQueryResponse(
            success=True,
            result=result,
            error=None,
            processing_time=0.1,
            query=query
        )
        
        # Try to serialize the entire response
        try:
            response_dict = api_response.model_dump()
            json_str = json.dumps(response_dict)
            print("✅ Full API response is JSON serializable")
            print(f"📊 Response keys: {list(response_dict.keys())}")
        except TypeError as e:
            print(f"❌ API response is NOT JSON serializable: {e}")
            return False
        
        store.disconnect()
        print("✅ All serialization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_graph_serialization()
    if success:
        print("\n🎉 Graph serialization fix is working correctly!")
    else:
        print("\n💥 Graph serialization fix needs more work!")
        sys.exit(1)
