#!/usr/bin/env python3
"""
Diagnostic script to identify and fix Chroma dimension mismatch issues.

This script helps diagnose embedding dimension mismatches between the configured
embedding model and the Chroma collection, and provides options to fix them.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.vectorstores.chroma_store import ChromaStore
from src.config.settings import settings


def diagnose_chroma_issues():
    """Diagnose Chroma vector store issues."""
    print("🔍 Diagnosing Chroma vector store issues...")
    print("=" * 50)
    
    try:
        # Create ChromaStore instance
        store = ChromaStore()
        
        # Check health
        is_healthy, health_msg = store.health_check()
        print(f"Health Check: {'✅' if is_healthy else '❌'} {health_msg}")
        
        # Get collection stats
        stats = store.get_collection_stats()
        print(f"Collection Stats: {stats}")
        
        # Check embedding compatibility
        is_compatible, compatibility_msg = store.check_embedding_dimension_compatibility()
        print(f"Embedding Compatibility: {'✅' if is_compatible else '❌'} {compatibility_msg}")
        
        # Get dimension guidance
        guidance = store.get_dimension_mismatch_guidance()
        print(f"Dimension Guidance: {guidance}")
        
        return store, is_healthy, is_compatible
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {str(e)}")
        return None, False, False


def fix_dimension_mismatch(store):
    """Fix dimension mismatch by recreating the collection."""
    print("\n🔧 Attempting to fix dimension mismatch...")
    print("=" * 50)
    
    try:
        success = store.recreate_collection_with_correct_dimension()
        if success:
            print("✅ Successfully recreated collection with correct dimension")
            
            # Verify the fix
            is_healthy, health_msg = store.health_check()
            print(f"Health Check After Fix: {'✅' if is_healthy else '❌'} {health_msg}")
            
            is_compatible, compatibility_msg = store.check_embedding_dimension_compatibility()
            print(f"Embedding Compatibility After Fix: {'✅' if is_compatible else '❌'} {compatibility_msg}")
            
            return True
        else:
            print("❌ Failed to recreate collection")
            return False
            
    except Exception as e:
        print(f"❌ Error during fix: {str(e)}")
        return False


def show_manual_fix_instructions():
    """Show manual fix instructions."""
    print("\n📋 Manual Fix Instructions:")
    print("=" * 50)
    print("If the automatic fix didn't work, you can manually fix the issue:")
    print()
    print("1. Stop the application")
    print("2. Delete the Chroma collection manually:")
    print("   - If using local Chroma: Delete the chroma_db directory")
    print("   - If using Chroma server: Use the Chroma client to delete the collection")
    print("3. Restart the application - it will create a new collection with the correct dimension")
    print()
    print("Alternative solutions:")
    print("- Change the embedding model in settings to match the collection's expected dimension")
    print("- Use a different collection name in the configuration")


def main():
    """Main diagnostic function."""
    print("🚀 Chroma Dimension Mismatch Diagnostic Tool")
    print("=" * 50)
    
    # Diagnose issues
    store, is_healthy, is_compatible = diagnose_chroma_issues()
    
    if store is None:
        print("\n❌ Could not connect to Chroma. Please check your configuration.")
        return
    
    if is_healthy and is_compatible:
        print("\n✅ No issues detected! Chroma is working correctly.")
        return
    
    # Show current configuration
    print(f"\n📋 Current Configuration:")
    print(f"   Embedding Model: {settings.embedding.model}")
    print(f"   Embedding Provider: {settings.embedding.provider}")
    print(f"   Chroma Collection: {settings.chroma.collection_name}")
    print(f"   Chroma Host: {settings.chroma.host}:{settings.chroma.port}")
    
    # Ask user if they want to fix the issue
    if not is_compatible:
        print("\n❓ Dimension mismatch detected. Would you like to fix it automatically?")
        print("   This will delete the existing collection and create a new one.")
        print("   Type 'yes' to proceed, or 'no' to see manual instructions.")
        
        response = input("   Your choice (yes/no): ").strip().lower()
        
        if response == 'yes':
            success = fix_dimension_mismatch(store)
            if not success:
                show_manual_fix_instructions()
        else:
            show_manual_fix_instructions()
    else:
        print("\n⚠️  Health issues detected but no dimension mismatch.")
        print("   Please check the Chroma server and network connectivity.")


if __name__ == "__main__":
    main() 