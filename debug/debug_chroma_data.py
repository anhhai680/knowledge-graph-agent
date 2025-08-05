#!/usr/bin/env python3
"""
Debug script to check what's in the Chroma vector store.

This script will directly query the Chroma collection to see if any documents
were actually stored during the indexing process.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.config.settings import settings
from src.vectorstores.store_factory import VectorStoreFactory

def check_chroma_data():
    """Check what data is in the Chroma vector store."""
    print("🔍 Checking Chroma vector store data...")
    
    try:
        # Create vector store
        vector_store_factory = VectorStoreFactory()
        vector_store = vector_store_factory.create(
            collection_name=settings.chroma.collection_name
        )
        
        print(f"Connected to collection: {settings.chroma.collection_name}")
        
        # Check collection count
        try:
            if hasattr(vector_store, 'collection'):
                count = vector_store.collection.count()
                print(f"Documents in collection: {count}")
            else:
                print("Vector store does not have collection attribute")
        except Exception as e:
            print(f"❌ Error getting count: {e}")
        
        # Try to get some documents
        try:
            if hasattr(vector_store, 'collection'):
                results = vector_store.collection.get(
                    limit=5,
                    include=["metadatas", "documents"]
                )
                
                documents = results.get("documents", [])
                metadatas = results.get("metadatas", [])
                
                print(f"Retrieved {len(documents)} sample documents")
                
                if documents and metadatas:
                    print("\nSample document metadata:")
                    for i, metadata in enumerate(metadatas[:3]):
                        print(f"\nDocument {i+1}:")
                        for key, value in metadata.items():
                            print(f"  {key}: {value}")
                            
                    print(f"\nSample document content (first 200 chars):")
                    for i, doc in enumerate(documents[:3]):
                        print(f"Document {i+1}: {doc[:200]}...")
                else:
                    print("❌ No documents found in collection")
            else:
                print("Cannot access collection directly")
        
        except Exception as e:
            print(f"❌ Error getting documents: {e}")
        
        # Try to get repository metadata
        try:
            print("\n📊 Getting repository metadata...")
            repo_metadata = vector_store.get_repository_metadata()
            print(f"Repository metadata entries: {len(repo_metadata)}")
            
            for repo in repo_metadata:
                print(f"\nRepository: {repo.get('name', 'Unknown')}")
                print(f"  URL: {repo.get('url', 'Unknown')}")
                print(f"  File count: {repo.get('file_count', 0)}")
                print(f"  Document count: {repo.get('document_count', 0)}")
                print(f"  Languages: {repo.get('languages', [])}")
                print(f"  Size: {repo.get('size_mb', 0)} MB")
        
        except Exception as e:
            print(f"❌ Error getting repository metadata: {e}")
            import traceback
            traceback.print_exc()
        
        # Check embedding dimension compatibility
        try:
            print("\n🔧 Checking embedding compatibility...")
            if hasattr(vector_store, 'check_embedding_dimension_compatibility'):
                is_compatible, msg = vector_store.check_embedding_dimension_compatibility()
                print(f"Compatible: {is_compatible}")
                print(f"Message: {msg}")
            else:
                print("Vector store does not support embedding compatibility check")
        except Exception as e:
            print(f"❌ Error checking compatibility: {e}")
            
    except Exception as e:
        print(f"❌ Error connecting to Chroma: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Debug: Chroma Vector Store Data")
    print("=" * 50)
    
    try:
        check_chroma_data()
    except Exception as e:
        print(f"❌ Error in debug script: {e}")
        import traceback
        traceback.print_exc()
