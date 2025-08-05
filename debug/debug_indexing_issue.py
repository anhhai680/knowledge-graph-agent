#!/usr/bin/env python3
"""
Debug script to identify why repositories are not being indexed properly.

This script will test the git-based loader and file discovery process to identify
where the issue is occurring.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.config.settings import settings
from src.loaders.enhanced_github_loader import EnhancedGitHubLoader
from src.loaders.file_system_processor import FileSystemProcessor

def test_repository_loading():
    """Test loading a specific repository to identify issues."""
    print("🔍 Testing repository loading process...")
    
    # Test with one of the configured repositories
    repo_owner = "anhhai680"
    repo_name = "car-notification-service"
    
    print(f"Testing: {repo_owner}/{repo_name}")
    print(f"File extensions configured: {settings.github.file_extensions}")
    print(f"GitHub token configured: {'Yes' if settings.github.token else 'No'}")
    
    # Check if the repository is already cloned
    repo_path = f"temp_repo/{repo_owner}/{repo_name}"
    print(f"Repository path: {repo_path}")
    print(f"Repository exists: {Path(repo_path).exists()}")
    
    if Path(repo_path).exists():
        # Test file discovery directly
        print("\n📁 Testing file discovery...")
        file_processor = FileSystemProcessor(settings.github.file_extensions)
        
        try:
            file_paths = file_processor.scan_directory_tree(repo_path)
            print(f"Raw files found: {len(file_paths)}")
            
            # Show first few files
            for i, file_path in enumerate(file_paths[:10]):
                print(f"  {i+1}. {file_path}")
            
            # Filter by size
            filtered_paths = file_processor.filter_files_by_size(file_paths, repo_path)
            print(f"Files after size filtering: {len(filtered_paths)}")
            
            # Show filtered files
            for i, file_path in enumerate(filtered_paths[:10]):
                print(f"  {i+1}. {file_path}")
                
        except Exception as e:
            print(f"❌ Error in file discovery: {e}")
    
    # Test with the full loader
    print("\n🚀 Testing Enhanced GitHub Loader...")
    
    try:
        loader = EnhancedGitHubLoader(
            repo_owner=repo_owner,
            repo_name=repo_name,
            branch="main",
            file_extensions=settings.github.file_extensions,
            github_token=settings.github.token,
        )
        
        documents = loader.load()
        print(f"Documents loaded: {len(documents)}")
        
        if documents:
            print("Sample document metadata:")
            doc = documents[0]
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
        else:
            print("❌ No documents loaded!")
            
    except Exception as e:
        print(f"❌ Error in loader: {e}")
        import traceback
        traceback.print_exc()

def test_file_extensions():
    """Test which files match our configured extensions."""
    print("\n🔍 Testing file extension matching...")
    
    repo_path = "temp_repo/anhhai680/car-notification-service"
    if not Path(repo_path).exists():
        print(f"❌ Repository path doesn't exist: {repo_path}")
        return
        
    # List all files in the repository
    all_files = []
    for root, dirs, files in os.walk(repo_path):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        for file in files:
            file_path = os.path.relpath(os.path.join(root, file), repo_path)
            all_files.append(file_path)
    
    print(f"Total files in repository: {len(all_files)}")
    
    # Show all files
    for file_path in all_files:
        extension = Path(file_path).suffix.lower()
        matches = extension in [ext.lower() for ext in settings.github.file_extensions]
        status = "✅" if matches else "❌"
        print(f"  {status} {file_path} (ext: {extension})")
    
    # Count matches
    matching_files = []
    for file_path in all_files:
        extension = Path(file_path).suffix.lower()
        if extension in [ext.lower() for ext in settings.github.file_extensions]:
            matching_files.append(file_path)
    
    print(f"\nMatching files: {len(matching_files)}")
    for file_path in matching_files:
        print(f"  ✅ {file_path}")

if __name__ == "__main__":
    print("🚀 Debug: Repository Indexing Issue")
    print("=" * 50)
    
    try:
        test_file_extensions()
        test_repository_loading()
    except Exception as e:
        print(f"❌ Error in debug script: {e}")
        import traceback
        traceback.print_exc()
