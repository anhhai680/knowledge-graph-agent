#!/usr/bin/env python3
"""
Debug script for car-web-client repository loading issue.
"""

import os
import sys
sys.path.insert(0, 'src')

from loaders.enhanced_github_loader import EnhancedGitHubLoader
from config.settings import settings
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_car_web_client_loading():
    """Test loading car-web-client repository directly."""
    print("🚀 Testing car-web-client repository loading")
    print("=" * 60)
    
    try:
        # Create GitHub loader for car-web-client
        loader = EnhancedGitHubLoader(
            repo_owner="anhhai680",
            repo_name="car-web-client",
            branch="main",
            file_extensions=settings.github.file_extensions,
            github_token=settings.github.token,
        )
        
        print(f"✅ Created loader for anhhai680/car-web-client")
        print(f"📊 Configured file extensions: {settings.github.file_extensions}")
        
        # Load documents
        documents = loader.load()
        
        print(f"✅ Loaded {len(documents)} documents")
        
        if documents:
            print("📊 Sample documents:")
            for i, doc in enumerate(documents[:5]):
                print(f"  {i+1}. File: {doc.metadata.get('file_path', 'unknown')}")
                print(f"     Size: {len(doc.page_content)} chars")
                print(f"     Repository: {doc.metadata.get('repository', 'unknown')}")
                print()
        else:
            print("❌ No documents loaded!")
            
        # Count unique files
        unique_files = set()
        for doc in documents:
            file_path = doc.metadata.get("file_path", "")
            if file_path:
                unique_files.add(file_path)
        
        print(f"📊 Unique files loaded: {len(unique_files)}")
        if unique_files:
            print("📊 Files:")
            for file_path in sorted(unique_files):
                print(f"  - {file_path}")
        
        assert len(unique_files) > 0, "No files were loaded from the repository"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Repository loading failed: {e}"

if __name__ == "__main__":
    file_count = test_car_web_client_loading()
    print(f"\n📊 Result: {file_count} files loaded")
    if file_count > 0:
        print("✅ Repository loading works correctly!")
    else:
        print("❌ Repository loading failed!")
