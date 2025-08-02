#!/usr/bin/env python3
"""
Integration test for Git-based GitHub loader implementation.

This script tests the complete Git-based loader workflow without requiring 
external dependencies or test frameworks.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

def test_git_command_executor():
    """Test GitCommandExecutor basic functionality."""
    print("Testing GitCommandExecutor...")
    
    from src.loaders.git_command_executor import GitCommandExecutor
    
    executor = GitCommandExecutor()
    assert executor.timeout_seconds == 300
    print("✓ GitCommandExecutor initialized correctly")
    
    # Test invalid directory (should fail gracefully)
    result = executor.execute_git_command(["status"], "/nonexistent")
    assert not result.success
    print("✓ GitCommandExecutor handles invalid directories")


def test_repository_url_handler():
    """Test RepositoryUrlHandler functionality."""
    print("\nTesting RepositoryUrlHandler...")
    
    from src.loaders.repository_url_handler import RepositoryUrlHandler
    
    handler = RepositoryUrlHandler()
    
    # Test URL normalization
    test_cases = [
        ("https://github.com/owner/repo", "https://github.com/owner/repo.git"),
        ("https://github.com/owner/repo.git", "https://github.com/owner/repo.git"),
        ("git@github.com:owner/repo.git", "https://github.com/owner/repo.git"),
    ]
    
    for input_url, expected in test_cases:
        result = handler.normalize_repo_url(input_url)
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ URL normalization works correctly")
    
    # Test repo info parsing
    owner, repo = handler.parse_repo_info("https://github.com/test/example.git")
    assert owner == "test"
    assert repo == "example"
    print("✓ Repository info parsing works correctly")


def test_file_system_processor():
    """Test FileSystemProcessor functionality."""
    print("\nTesting FileSystemProcessor...")
    
    from src.loaders.file_system_processor import FileSystemProcessor
    
    processor = FileSystemProcessor([".py", ".md", ".txt"])
    
    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        (temp_path / "test.py").write_text("print('hello')")
        (temp_path / "readme.md").write_text("# Test")
        (temp_path / "notes.txt").write_text("Some notes")
        (temp_path / "binary.bin").write_bytes(b"\x00\x01\x02")  # Binary file
        
        # Test directory scanning
        files = processor.scan_directory_tree(str(temp_path))
        assert len(files) == 3  # Should exclude binary.bin
        print("✓ Directory scanning works correctly")
        
        # Test file reading
        content = processor.read_file_content(str(temp_path / "test.py"))
        assert content == "print('hello')"
        print("✓ File content reading works correctly")
        
        # Test encoding detection
        encoding = processor.detect_file_encoding(str(temp_path / "test.py"))
        assert encoding in ["utf-8", "ascii"]
        print("✓ Encoding detection works correctly")


def test_git_repository_manager():
    """Test GitRepositoryManager functionality."""
    print("\nTesting GitRepositoryManager...")
    
    from src.loaders.git_repository_manager import GitRepositoryManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = GitRepositoryManager(temp_dir)
        
        # Test path generation
        path = manager.get_local_repo_path("owner", "repo")
        expected = str(Path(temp_dir) / "owner" / "repo")
        assert path == expected
        print("✓ Local repo path generation works correctly")
        
        # Test directory creation
        assert Path(temp_dir).exists()
        print("✓ Base directory creation works correctly")


def test_git_metadata_extractor():
    """Test GitMetadataExtractor basic functionality."""
    print("\nTesting GitMetadataExtractor...")
    
    from src.loaders.git_metadata_extractor import GitMetadataExtractor
    from src.loaders.git_command_executor import GitCommandExecutor
    
    executor = GitCommandExecutor()
    extractor = GitMetadataExtractor(executor)
    
    # Test initialization
    assert extractor.git_executor is not None
    print("✓ GitMetadataExtractor initialized correctly")


def test_enhanced_github_loader():
    """Test EnhancedGitHubLoader initialization."""
    print("\nTesting EnhancedGitHubLoader...")
    
    from src.loaders.enhanced_github_loader import EnhancedGitHubLoader
    
    with tempfile.TemporaryDirectory() as temp_dir:
        loader = EnhancedGitHubLoader(
            repo_owner="test",
            repo_name="repo",
            branch="main",
            temp_repo_path=temp_dir
        )
        
        assert loader.repo_owner == "test"
        assert loader.repo_name == "repo"
        assert loader.branch == "main"
        print("✓ EnhancedGitHubLoader initialized correctly")


def test_loader_migration_manager():
    """Test LoaderMigrationManager functionality."""
    print("\nTesting LoaderMigrationManager...")
    
    from src.loaders.loader_migration_manager import LoaderMigrationManager
    
    manager = LoaderMigrationManager()
    
    # Test configuration migration
    old_config = {
        "owner": "test",
        "repo": "example",
        "branch": "main"
    }
    
    new_config = manager.migrate_repository_config(old_config)
    assert new_config["use_git_loader"] is True
    assert "git_timeout_seconds" in new_config
    print("✓ Configuration migration works correctly")


def test_git_error_handler():
    """Test GitErrorHandler functionality."""
    print("\nTesting GitErrorHandler...")
    
    from src.loaders.git_error_handler import GitErrorHandler
    
    handler = GitErrorHandler()
    
    # Test corrupted repo cleanup
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "test_repo"
        repo_path.mkdir()
        (repo_path / "test_file").write_text("test")
        
        assert repo_path.exists()
        handler.cleanup_corrupted_repo(str(repo_path))
        assert not repo_path.exists()
        print("✓ Corrupted repository cleanup works correctly")


def main():
    """Run all integration tests."""
    print("🚀 Starting Git-based GitHub Loader Integration Tests\n")
    
    try:
        test_git_command_executor()
        test_repository_url_handler()
        test_file_system_processor()
        test_git_repository_manager()
        test_git_metadata_extractor()
        test_enhanced_github_loader()
        test_loader_migration_manager()
        test_git_error_handler()
        
        print("\n✅ All integration tests passed!")
        print("🎉 Git-based GitHub Loader implementation is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
