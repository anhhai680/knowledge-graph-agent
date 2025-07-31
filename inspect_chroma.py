#!/usr/bin/env python3
"""
Simple test to inspect the Chroma database directly.
"""

import sqlite3
import json
from pathlib import Path

def inspect_chroma_db():
    """Inspect the Chroma SQLite database directly."""
    db_path = Path("chroma_db/chroma.sqlite3")
    
    if not db_path.exists():
        print("❌ Chroma database not found at chroma_db/chroma.sqlite3")
        return
    
    print(f"📊 Inspecting Chroma database at: {db_path}")
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📋 Tables found: {len(tables)}")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Check if embeddings table exists
        if any('embeddings' in str(table) for table in tables):
            print("\n🔍 Checking embeddings table...")
            cursor.execute("SELECT COUNT(*) FROM embeddings;")
            count = cursor.fetchone()[0]
            print(f"📈 Total embeddings: {count}")
            
            if count > 0:
                # Get sample metadata
                cursor.execute("SELECT metadata FROM embeddings LIMIT 5;")
                samples = cursor.fetchall()
                
                print("\n📄 Sample metadata entries:")
                for i, (metadata_json,) in enumerate(samples):
                    try:
                        metadata = json.loads(metadata_json) if metadata_json else {}
                        print(f"  Sample {i+1}:")
                        print(f"    Repository: {metadata.get('repository', 'N/A')}")
                        print(f"    Source: {metadata.get('source', 'N/A')}")
                        print(f"    Language: {metadata.get('language', 'N/A')}")
                        print(f"    Size: {metadata.get('size', 'N/A')}")
                    except json.JSONDecodeError:
                        print(f"  Sample {i+1}: Invalid JSON metadata")
                
                # Aggregate repository info
                print("\n📊 Repository analysis:")
                cursor.execute("SELECT metadata FROM embeddings;")
                all_metadata = cursor.fetchall()
                
                repos = {}
                for (metadata_json,) in all_metadata:
                    try:
                        metadata = json.loads(metadata_json) if metadata_json else {}
                        repo = metadata.get('repository', 'Unknown')
                        if repo not in repos:
                            repos[repo] = {
                                'count': 0,
                                'languages': set(),
                                'sources': set()
                            }
                        repos[repo]['count'] += 1
                        if metadata.get('language'):
                            repos[repo]['languages'].add(metadata.get('language'))
                        if metadata.get('source'):
                            repos[repo]['sources'].add(metadata.get('source'))
                    except:
                        continue
                
                print(f"🏢 Repositories found: {len(repos)}")
                for repo_name, repo_data in repos.items():
                    print(f"  📁 {repo_name}:")
                    print(f"    Documents: {repo_data['count']}")
                    print(f"    Files: {len(repo_data['sources'])}")
                    print(f"    Languages: {list(repo_data['languages'])}")
        
        # Check collections table
        if any('collections' in str(table) for table in tables):
            print("\n📚 Checking collections...")
            cursor.execute("SELECT name, metadata FROM collections;")
            collections = cursor.fetchall()
            for name, metadata in collections:
                print(f"  Collection: {name}")
                if metadata:
                    try:
                        meta = json.loads(metadata)
                        print(f"    Metadata: {meta}")
                    except:
                        print(f"    Metadata: {metadata}")
        
        conn.close()
        print("\n✅ Database inspection completed!")
        
    except Exception as e:
        print(f"❌ Error inspecting database: {e}")

if __name__ == "__main__":
    print("🔍 Direct Chroma Database Inspector")
    print("=" * 50)
    inspect_chroma_db()
