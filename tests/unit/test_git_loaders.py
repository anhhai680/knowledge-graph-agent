"""
Unit tests for Git-based GitHub loaders.

This module contains comprehensive tests for all Git-based loader components
including repository management, command execution, file processing, and
metadata extraction.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

import pytest
from langchain.schema import Document

from src.loaders.git_repository_manager import GitRepositoryManager
from src.loaders.git_command_executor import GitCommandExecutor, GitCommandResult
from src.loaders.file_system_processor import FileSystemProcessor
from src.loaders.git_metadata_extractor import GitMetadataExtractor
from src.loaders.repository_url_handler import RepositoryUrlHandler
from src.loaders.enhanced_github_loader import EnhancedGitHubLoader
from src.loaders.loader_migration_manager import LoaderMigrationManager
from src.loaders.git_error_handler import GitErrorHandler


class TestGitCommandExecutor:
    """Test suite for GitCommandExecutor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.executor = GitCommandExecutor()

    def test_init(self):
        """Test GitCommandExecutor initialization."""
        assert self.executor.timeout_seconds == 300

    @patch('subprocess.run')
    def test_execute_git_command_success(self, mock_run):
        """Test successful git command execution."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="test output",
            stderr="",
            args=["git", "status"]
        )
        
        result = self.executor.execute_git_command(["status"], "/tmp/test")
        
        assert result.success is True
        assert result.stdout == "test output"
        assert result.stderr == ""
        assert result.return_code == 0

    @patch('subprocess.run')
    def test_execute_git_command_failure(self, mock_run):
        """Test failed git command execution."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="fatal: not a git repository",
            args=["git", "status"]
        )
        
        result = self.executor.execute_git_command(["status"], "/tmp/test")
        
        assert result.success is False
        assert result.stderr == "fatal: not a git repository"
        assert result.return_code == 1

    @patch('subprocess.run')
    def test_clone_repository(self, mock_run):
        """Test repository cloning."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = self.executor.clone_repository(
            "https://github.com/test/repo.git",
            "/tmp/test",
            "main"
        )
        
        assert result is True
        mock_run.assert_called()

    @patch('subprocess.run')
    def test_pull_repository(self, mock_run):
        """Test repository pulling."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = self.executor.pull_repository("/tmp/test", "main")
        
        assert result is True
        mock_run.assert_called()


class TestGitRepositoryManager:
    """Test suite for GitRepositoryManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = GitRepositoryManager(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test GitRepositoryManager initialization."""
        assert self.manager.temp_repo_base_path == Path(self.temp_dir)
        assert isinstance(self.manager.git_executor, GitCommandExecutor)
        assert Path(self.temp_dir).exists()

    def test_get_local_repo_path(self):
        """Test local repository path generation."""
        path = self.manager.get_local_repo_path("owner", "repo")
        expected = str(Path(self.temp_dir) / "owner" / "repo")
        assert path == expected

    @patch.object(GitRepositoryManager, '_clone_repository')
    @patch.object(GitRepositoryManager, 'is_repository_valid')
    def test_clone_or_pull_new_repo(self, mock_is_valid, mock_clone):
        """Test cloning new repository."""
        mock_is_valid.return_value = False
        mock_clone.return_value = True
        
        result = self.manager.clone_or_pull_repository(
            "https://github.com/test/repo.git",
            "/tmp/test",
            "main"
        )
        
        assert result is True
        mock_clone.assert_called_once()

    @patch.object(GitRepositoryManager, '_pull_repository')
    @patch.object(GitRepositoryManager, 'is_repository_valid')
    def test_clone_or_pull_existing_repo(self, mock_is_valid, mock_pull):
        """Test pulling existing repository."""
        # Create test directory
        test_path = Path(self.temp_dir) / "test"
        test_path.mkdir()
        
        mock_is_valid.return_value = True
        mock_pull.return_value = True
        
        result = self.manager.clone_or_pull_repository(
            "https://github.com/test/repo.git",
            str(test_path),
            "main"
        )
        
        assert result is True
        mock_pull.assert_called_once()


class TestRepositoryUrlHandler:
    """Test suite for RepositoryUrlHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = RepositoryUrlHandler()

    def test_normalize_github_url(self):
        """Test GitHub URL normalization."""
        test_cases = [
            ("https://github.com/owner/repo", "https://github.com/owner/repo.git"),
            ("https://github.com/owner/repo.git", "https://github.com/owner/repo.git"),
            ("git@github.com:owner/repo.git", "https://github.com/owner/repo.git"),
        ]
        
        for input_url, expected in test_cases:
            result = self.handler.normalize_repo_url(input_url)
            assert result == expected

    def test_parse_repo_info(self):
        """Test repository information parsing."""
        url = "https://github.com/owner/repo.git"
        owner, repo = self.handler.parse_repo_info(url)
        
        assert owner == "owner"
        assert repo == "repo"

    def test_build_clone_url_with_token(self):
        """Test clone URL building with token."""
        url = self.handler.build_clone_url(
            "owner", "repo", "test_token"
        )
        
        assert "test_token" in url
        assert "owner/repo" in url


class TestFileSystemProcessor:
    """Test suite for FileSystemProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = FileSystemProcessor([".py", ".md", ".txt"])

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_scan_directory_tree(self):
        """Test directory tree scanning."""
        # Create test files
        test_file = Path(self.temp_dir) / "test.py"
        test_file.write_text("print('hello')")
        
        subdir = Path(self.temp_dir) / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("print('nested')")
        
        files = self.processor.scan_directory_tree(self.temp_dir)
        
        assert len(files) == 2
        assert any("test.py" in f for f in files)
        assert any("nested.py" in f for f in files)

    def test_read_file_content(self):
        """Test file content reading."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)
        
        content = self.processor.read_file_content(str(test_file))
        
        assert content == test_content

    def test_detect_file_encoding(self):
        """Test file encoding detection."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Hello", encoding="utf-8")
        
        encoding = self.processor.detect_file_encoding(str(test_file))
        
        assert encoding in ["utf-8", "ascii"]  # Both are valid for this content


class TestGitMetadataExtractor:
    """Test suite for GitMetadataExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_executor = Mock(spec=GitCommandExecutor)
        self.extractor = GitMetadataExtractor(self.mock_executor)

    def test_get_file_commit_info(self):
        """Test file commit information extraction."""
        self.mock_executor.get_file_commit_info.return_value = {
            "last_commit_sha": "abc123",
            "last_commit_author": "test_user",
            "last_commit_date": "2023-01-01",
            "last_commit_message": "Test commit"
        }
        
        info = self.extractor.get_file_commit_info("/repo/path", "test.py")
        
        assert info["last_commit_sha"] == "abc123"
        assert info["last_commit_author"] == "test_user"

    def test_get_repository_languages(self):
        """Test repository languages detection."""
        self.mock_executor.execute_git_command.return_value = GitCommandResult(
            success=True,
            stdout="100.0% Python\n",
            stderr="",
            return_code=0,
            command=["git", "ls-files"],
            execution_time=0.1
        )
        
        languages = self.extractor.get_repository_languages("/repo/path")
        
        # Should contain Python as detected language
        assert isinstance(languages, dict)


class TestEnhancedGitHubLoader:
    """Test suite for EnhancedGitHubLoader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('src.loaders.enhanced_github_loader.GitRepositoryManager')
    @patch('src.loaders.enhanced_github_loader.FileSystemProcessor')
    def test_init(self, mock_processor, mock_manager):
        """Test EnhancedGitHubLoader initialization."""
        loader = EnhancedGitHubLoader(
            repo_owner="test_owner",
            repo_name="test_repo",
            branch="test_branch"
        )
        
        assert loader.repo_owner == "test_owner"
        assert loader.repo_name == "test_repo"
        assert loader.branch == "test_branch"

    @patch('src.loaders.enhanced_github_loader.GitRepositoryManager')
    @patch('src.loaders.enhanced_github_loader.FileSystemProcessor')
    @patch('src.loaders.enhanced_github_loader.GitMetadataExtractor')
    def test_load_documents(self, mock_extractor, mock_processor, mock_manager):
        """Test document loading."""
        # Mock repository manager
        mock_manager_instance = mock_manager.return_value
        mock_manager_instance.clone_or_pull_repository.return_value = True
        mock_manager_instance.get_local_repo_path.return_value = "/tmp/test"
        
        # Mock file processor
        mock_processor_instance = mock_processor.return_value
        mock_processor_instance.scan_directory_tree.return_value = ["/tmp/test/file.py"]
        mock_processor_instance.read_file_content.return_value = "print('hello')"
        
        # Mock metadata extractor
        mock_extractor_instance = mock_extractor.return_value
        mock_extractor_instance.get_comprehensive_file_metadata.return_value = {
            "file_path": "file.py",
            "file_size": 100,
            "last_modified": "2023-01-01"
        }
        
        loader = EnhancedGitHubLoader(
            repo_owner="test_owner",
            repo_name="test_repo"
        )
        
        documents = loader.load()
        
        assert isinstance(documents, list)
        mock_manager_instance.clone_or_pull_repository.assert_called_once()


class TestLoaderMigrationManager:
    """Test suite for LoaderMigrationManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.migration_manager = LoaderMigrationManager()

    @patch('src.loaders.loader_migration_manager.EnhancedGitHubLoader')
    def test_create_git_loader(self, mock_git_loader):
        """Test Git loader creation."""
        loader = self.migration_manager.create_loader(
            "test_owner", "test_repo", use_git_loader=True
        )
        
        mock_git_loader.assert_called_once()

    @patch('src.loaders.loader_migration_manager.GitHubLoader')
    def test_create_api_loader(self, mock_api_loader):
        """Test API loader creation."""
        loader = self.migration_manager.create_loader(
            "test_owner", "test_repo", use_git_loader=False
        )
        
        mock_api_loader.assert_called_once()

    def test_migrate_repository_config(self):
        """Test repository configuration migration."""
        old_config = {
            "owner": "test_owner",
            "repo": "test_repo",
            "branch": "main"
        }
        
        new_config = self.migration_manager.migrate_repository_config(old_config)
        
        assert new_config["use_git_loader"] is True
        assert new_config["force_fresh_clone"] is False
        assert "git_timeout_seconds" in new_config

    @patch('src.loaders.loader_migration_manager.GitHubLoader')
    @patch('src.loaders.loader_migration_manager.EnhancedGitHubLoader')
    def test_benchmark_loaders(self, mock_git_loader, mock_api_loader):
        """Test loader benchmarking."""
        # Mock API loader
        mock_api_instance = Mock()
        mock_api_instance.load.return_value = [Mock(page_content="test")]
        mock_api_loader.return_value = mock_api_instance
        
        # Mock Git loader
        mock_git_instance = Mock()
        mock_git_instance.load.return_value = [Mock(page_content="test")]
        mock_git_loader.return_value = mock_git_instance
        
        repo_config = {
            "owner": "test_owner",
            "repo": "test_repo",
            "branch": "main"
        }
        
        results = self.migration_manager.benchmark_loaders(repo_config)
        
        assert "api_loader" in results
        assert "git_loader" in results
        assert "comparison" in results


class TestGitErrorHandler:
    """Test suite for GitErrorHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = GitErrorHandler()

    def test_handle_clone_error_authentication(self):
        """Test handling authentication errors during clone."""
        result = GitCommandResult(
            success=False,
            stdout="",
            stderr="fatal: Authentication failed",
            return_code=128,
            command=["git", "clone"],
            execution_time=1.0
        )
        
        recovery_result = self.error_handler.handle_clone_error(
            Exception("Authentication failed"), "https://github.com/test/repo.git"
        )
        
        assert recovery_result is None  # Returns None for failed recovery

    def test_cleanup_corrupted_repo(self):
        """Test corrupted repository cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a corrupted repo directory
            repo_path = Path(temp_dir) / "corrupted_repo"
            repo_path.mkdir()
            (repo_path / "invalid_file").write_text("invalid")
            
            self.error_handler.cleanup_corrupted_repo(str(repo_path))
            
            # Directory should be removed
            assert not repo_path.exists()


@pytest.fixture
def sample_repository_config():
    """Fixture providing sample repository configuration."""
    return {
        "owner": "test_owner",
        "repo": "test_repo", 
        "branch": "main"
    }


@pytest.fixture
def temp_git_repo():
    """Fixture providing a temporary Git repository."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "test_repo"
        repo_path.mkdir()
        
        # Initialize git repo
        os.system(f"cd {repo_path} && git init")
        os.system(f"cd {repo_path} && git config user.email 'test@example.com'")
        os.system(f"cd {repo_path} && git config user.name 'Test User'")
        
        # Create test file and commit
        test_file = repo_path / "test.py"
        test_file.write_text("print('hello world')")
        os.system(f"cd {repo_path} && git add . && git commit -m 'Initial commit'")
        
        yield str(repo_path)


class TestIntegration:
    """Integration tests for the complete Git loader system."""

    def test_end_to_end_loader_workflow(self, temp_git_repo, sample_repository_config):
        """Test complete workflow from repository to documents."""
        # This would test the full integration but requires actual Git operations
        # In a real test environment, you would set up a test repository
        pass

    def test_migration_manager_workflow(self, sample_repository_config):
        """Test complete migration manager workflow."""
        migration_manager = LoaderMigrationManager()
        
        # Test configuration migration
        migrated_config = migration_manager.migrate_repository_config(
            sample_repository_config
        )
        
        assert migrated_config["use_git_loader"] is True
        
        # Test loader creation with migrated config
        with patch('src.loaders.loader_migration_manager.EnhancedGitHubLoader'):
            loader = migration_manager.create_loader(
                migrated_config["owner"],
                migrated_config["repo"],
                use_git_loader=migrated_config["use_git_loader"]
            )
            
            assert loader is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
