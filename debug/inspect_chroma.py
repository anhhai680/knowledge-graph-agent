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
        
        # Check embedding_metadata table
        if any('embedding_metadata' in str(table) for table in tables):
            print("\n🔍 Checking embedding_metadata table...")
            cursor.execute("SELECT COUNT(*) FROM embedding_metadata;")
            count = cursor.fetchone()[0]
            print(f"📈 Total metadata entries: {count}")
            
            # Get table schema
            cursor.execute("PRAGMA table_info(embedding_metadata);")
            columns = cursor.fetchall()
            print(f"📋 Embedding_metadata table columns:")
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
            
            if count > 0:
                # Get sample data - first check what columns exist
                cursor.execute("SELECT * FROM embedding_metadata LIMIT 5;")
                sample_rows = cursor.fetchall()
                if sample_rows:
                    print(f"\n📄 Sample metadata entries:")
                    for i, row in enumerate(sample_rows):
                        print(f"  Sample {i+1}: {row}")
                
                # Try to get all metadata entries
                cursor.execute("SELECT * FROM embedding_metadata;")
                all_metadata = cursor.fetchall()
                
                # Group by embedding_id (assuming it's the first column)
                embedding_metadata = {}
                for row in all_metadata:
                    if len(row) >= 3:  # Ensure we have at least id, key, value
                        embedding_id = row[0]  # Assuming first column is id
                        key = row[1]  # Assuming second column is key
                        value = row[2]  # Assuming third column is string_value
                        
                        if embedding_id not in embedding_metadata:
                            embedding_metadata[embedding_id] = {}
                        embedding_metadata[embedding_id][key] = value
                
                # Aggregate by repository
                repos = {}
                for embedding_id, metadata in embedding_metadata.items():
                    repo_url = metadata.get('repository_url', '')
                    repo_name = metadata.get('repository', '')
                    
                    # Use repository URL as key, fallback to name
                    repo_key = repo_url or repo_name
                    if not repo_key:
                        continue
                        
                    if repo_key not in repos:
                        repos[repo_key] = {
                            'count': 0,
                            'languages': set(),
                            'sources': set(),
                            'name': repo_name,
                            'url': repo_url,
                            'branch': metadata.get('branch', 'main')
                        }
                    repos[repo_key]['count'] += 1
                    if metadata.get('language'):
                        repos[repo_key]['languages'].add(metadata.get('language'))
                    if metadata.get('source'):
                        repos[repo_key]['sources'].add(metadata.get('source'))
                
                print(f"\n🏢 Repositories found: {len(repos)}")
                for repo_key, repo_data in repos.items():
                    print(f"  📁 {repo_data['name']} ({repo_key}):")
                    print(f"    Documents: {repo_data['count']}")
                    print(f"    Files: {len(repo_data['sources'])}")
                    print(f"    Languages: {list(repo_data['languages'])}")
                    print(f"    Branch: {repo_data['branch']}")
        
        # Check embeddings table
        if any('embeddings' in str(table) for table in tables):
            print("\n🔍 Checking embeddings table...")
            cursor.execute("SELECT COUNT(*) FROM embeddings;")
            count = cursor.fetchone()[0]
            print(f"📈 Total embeddings: {count}")
        
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
