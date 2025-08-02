#!/usr/bin/env python3
"""
Test script to debug vector store retrieval issue.
"""
import sys
import os
sys.path.append('/Volumes/Data/Projects/AI Evelate/Workshop/knowledge-graph-agent')

import chromadb
from pathlib import Path

def test_chroma_direct():
    """Test Chroma database directly to see what's stored."""
    print("🔍 Testing Chroma database directly...")
    
    # Connect to the persistent Chroma client
    db_path = "/Volumes/Data/Projects/AI Evelate/Workshop/knowledge-graph-agent/chroma_db"
    client = chromadb.PersistentClient(path=db_path)
    
    # List collections
    collections = client.list_collections()
    print(f"📚 Found {len(collections)} collections:")
    for collection in collections:
        print(f"  - {collection.name} (count: {collection.count()})")
    
    if not collections:
        print("❌ No collections found!")
        return
    
    # Use the first collection
    collection = collections[0]
    print(f"\n🔍 Inspecting collection: {collection.name}")
    
    # Get all documents (limit to 5 for debugging)
    try:
        results = collection.get(limit=5, include=["documents", "metadatas", "ids"])
        
        print(f"📄 Found {len(results['ids'])} documents")
        
        for i, (doc_id, document, metadata) in enumerate(zip(
            results['ids'], 
            results['documents'], 
            results['metadatas']
        )):
            print(f"\n--- Document {i+1} ---")
            print(f"ID: {doc_id}")
            print(f"Content length: {len(document) if document else 0}")
            print(f"Content preview: {document[:200] if document else 'EMPTY'}...")
            print(f"Metadata keys: {list(metadata.keys()) if metadata else 'NO METADATA'}")
            if metadata:
                print(f"Repository: {metadata.get('repository', 'N/A')}")
                print(f"File path: {metadata.get('file_path', 'N/A')}")
                print(f"Language: {metadata.get('language', 'N/A')}")
                
    except Exception as e:
        print(f"❌ Error querying collection: {e}")

def test_similarity_search():
    """Test similarity search to see what gets returned."""
    print("\n🔍 Testing similarity search...")
    
    db_path = "/Volumes/Data/Projects/AI Evelate/Workshop/knowledge-graph-agent/chroma_db"
    client = chromadb.PersistentClient(path=db_path)
    
    collections = client.list_collections()
    if not collections:
        print("❌ No collections found!")
        return
        
    collection = collections[0]
    
    # Try a simple search
    try:
        results = collection.query(
            query_texts=["car listing service how it works"],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"🔍 Search results: {len(results['ids'][0])} documents found")
        
        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            results['ids'][0], 
            results['documents'][0], 
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"\n--- Search Result {i+1} ---")
            print(f"Distance: {distance:.4f}")
            print(f"Content length: {len(document) if document else 0}")
            print(f"Content preview: {document[:200] if document else 'EMPTY'}...")
            print(f"Metadata: {metadata if metadata else 'NO METADATA'}")
            
    except Exception as e:
        print(f"❌ Error in similarity search: {e}")

if __name__ == "__main__":
    test_chroma_direct()
    test_similarity_search()
