"""
Unit tests for GitDiffService.

Tests the functionality of computing git diffs between commits
to support incremental re-indexing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.loaders.git_diff_service import GitDiffService, FileChangeType, FileChange, GitDiffResult
from src.loaders.git_command_executor import GitCommandResult


class TestGitDiffService:
    """Test cases for GitDiffService functionality."""

    @pytest.fixture
    def git_diff_service(self):
        """Create GitDiffService instance with mocked executor."""
        with patch('src.loaders.git_diff_service.GitCommandExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor
            service = GitDiffService()
            service.git_executor = mock_executor
            return service, mock_executor

    def test_get_current_commit_success(self, git_diff_service):
        """Test successful retrieval of current commit hash."""
        service, mock_executor = git_diff_service
        
        mock_executor.execute_git_command.return_value = GitCommandResult(
            success=True,
            stdout="abc123def456\n",
            stderr="",
            return_code=0,
            command=["git", "rev-parse", "origin/main"],
            execution_time=0.1
        )
        
        result = service.get_current_commit("/path/to/repo", "main")
        assert result == "abc123def456"
        
        mock_executor.execute_git_command.assert_called_once_with(
            ["git", "rev-parse", "origin/main"],
            cwd="/path/to/repo"
        )

    def test_get_current_commit_failure(self, git_diff_service):
        """Test handling of git command failure."""
        service, mock_executor = git_diff_service
        
        mock_executor.execute_git_command.return_value = GitCommandResult(
            success=False,
            stdout="",
            stderr="fatal: bad revision 'origin/main'",
            return_code=1,
            command=["git", "rev-parse", "origin/main"],
            execution_time=0.1
        )
        
        result = service.get_current_commit("/path/to/repo", "main")
        assert result is None

    def test_parse_diff_line_added(self, git_diff_service):
        """Test parsing of added file diff line."""
        service, _ = git_diff_service
        
        change = service._parse_diff_line("A\tsrc/new_file.py")
        assert change is not None
        assert change.change_type == FileChangeType.ADDED
        assert change.file_path == "src/new_file.py"
        assert change.old_file_path is None

    def test_parse_diff_line_modified(self, git_diff_service):
        """Test parsing of modified file diff line."""
        service, _ = git_diff_service
        
        change = service._parse_diff_line("M\tsrc/existing_file.py")
        assert change is not None
        assert change.change_type == FileChangeType.MODIFIED
        assert change.file_path == "src/existing_file.py"

    def test_parse_diff_line_deleted(self, git_diff_service):
        """Test parsing of deleted file diff line."""
        service, _ = git_diff_service
        
        change = service._parse_diff_line("D\tsrc/old_file.py")
        assert change is not None
        assert change.change_type == FileChangeType.DELETED
        assert change.file_path == "src/old_file.py"

    def test_parse_diff_line_renamed(self, git_diff_service):
        """Test parsing of renamed file diff line."""
        service, _ = git_diff_service
        
        change = service._parse_diff_line("R100\tsrc/old_name.py\tsrc/new_name.py")
        assert change is not None
        assert change.change_type == FileChangeType.RENAMED
        assert change.old_file_path == "src/old_name.py"
        assert change.file_path == "src/new_name.py"
        assert change.similarity_index == 100

    def test_parse_diff_line_copied(self, git_diff_service):
        """Test parsing of copied file diff line."""
        service, _ = git_diff_service
        
        change = service._parse_diff_line("C85\tsrc/template.py\tsrc/copy.py")
        assert change is not None
        assert change.change_type == FileChangeType.COPIED
        assert change.old_file_path == "src/template.py"
        assert change.file_path == "src/copy.py"
        assert change.similarity_index == 85

    def test_parse_diff_line_invalid(self, git_diff_service):
        """Test parsing of invalid diff line."""
        service, _ = git_diff_service
        
        # Test invalid format
        change = service._parse_diff_line("invalid line")
        assert change is None
        
        # Test incomplete rename
        change = service._parse_diff_line("R100\tonly_one_file.py")
        assert change is None

    def test_should_include_file_with_extensions(self, git_diff_service):
        """Test file inclusion filtering with extensions."""
        service, _ = git_diff_service
        
        extensions = [".py", ".js", ".md"]
        
        assert service._should_include_file("src/file.py", extensions) is True
        assert service._should_include_file("src/file.js", extensions) is True
        assert service._should_include_file("README.md", extensions) is True
        assert service._should_include_file("src/file.txt", extensions) is False
        assert service._should_include_file("src/file", extensions) is False

    def test_should_include_file_no_filter(self, git_diff_service):
        """Test file inclusion without extension filter."""
        service, _ = git_diff_service
        
        assert service._should_include_file("any_file.txt", None) is True
        assert service._should_include_file("no_extension", None) is True

    def test_get_changes_between_commits_success(self, git_diff_service):
        """Test successful diff computation between commits."""
        service, mock_executor = git_diff_service
        
        # Mock git diff output
        diff_output = """A\tsrc/new_file.py
M\tsrc/modified_file.py
D\tsrc/deleted_file.py
R100\tsrc/old_name.py\tsrc/new_name.py"""
        
        mock_executor.execute_git_command.return_value = GitCommandResult(
            success=True,
            stdout=diff_output,
            stderr="",
            return_code=0,
            command=["git", "diff", "--name-status", "-M", "abc123", "def456"],
            execution_time=0.2
        )
        
        result = service.get_changes_between_commits(
            "/path/to/repo",
            "abc123",
            "def456",
            file_extensions=[".py"]
        )
        
        assert result.from_commit == "abc123"
        assert result.to_commit == "def456"
        assert result.total_changes == 4
        assert len(result.added_files) == 1
        assert len(result.modified_files) == 1
        assert len(result.deleted_files) == 1
        assert len(result.renamed_files) == 1
        
        assert "src/new_file.py" in result.added_files
        assert "src/modified_file.py" in result.modified_files
        assert "src/deleted_file.py" in result.deleted_files
        assert ("src/old_name.py", "src/new_name.py") in result.renamed_files

    def test_get_changes_between_commits_failure(self, git_diff_service):
        """Test handling of git diff command failure."""
        service, mock_executor = git_diff_service
        
        mock_executor.execute_git_command.return_value = GitCommandResult(
            success=False,
            stdout="",
            stderr="fatal: bad revision 'abc123'",
            return_code=1,
            command=["git", "diff", "--name-status", "-M", "abc123", "def456"],
            execution_time=0.1
        )
        
        result = service.get_changes_between_commits("/path/to/repo", "abc123", "def456")
        
        assert result.total_changes == 0
        assert len(result.file_changes) == 0

    def test_get_files_to_process(self, git_diff_service):
        """Test getting files that need to be processed."""
        service, _ = git_diff_service
        
        # Create test diff result
        diff_result = GitDiffResult(
            from_commit="abc123",
            to_commit="def456",
            total_changes=4,
            changes_by_type={},
            file_changes=[],
            added_files=["new.py", "another.py"],
            modified_files=["changed.py"],
            deleted_files=["removed.py"],
            renamed_files=[("old.py", "renamed.py")]
        )
        
        files_to_process = service.get_files_to_process(diff_result)
        
        expected_files = {"new.py", "another.py", "changed.py", "renamed.py"}
        assert files_to_process == expected_files

    def test_get_files_to_remove(self, git_diff_service):
        """Test getting files that need to be removed from vector store."""
        service, _ = git_diff_service
        
        # Create test diff result
        diff_result = GitDiffResult(
            from_commit="abc123",
            to_commit="def456",
            total_changes=4,
            changes_by_type={},
            file_changes=[],
            added_files=["new.py"],
            modified_files=["changed.py"],
            deleted_files=["removed.py"],
            renamed_files=[("old.py", "renamed.py")]
        )
        
        files_to_remove = service.get_files_to_remove(diff_result)
        
        # Should include deleted files, old paths of renames, and modified files
        expected_files = {"removed.py", "old.py", "changed.py"}
        assert files_to_remove == expected_files

    def test_validate_commits_success(self, git_diff_service):
        """Test successful commit validation."""
        service, mock_executor = git_diff_service
        
        mock_executor.execute_git_command.return_value = GitCommandResult(
            success=True,
            stdout="",
            stderr="",
            return_code=0,
            command=["git", "cat-file", "-e", "abc123"],
            execution_time=0.1
        )
        
        result = service.validate_commits("/path/to/repo", "abc123", "def456")
        assert result is True
        
        # Should be called twice (once for each commit)
        assert mock_executor.execute_git_command.call_count == 2

    def test_validate_commits_failure(self, git_diff_service):
        """Test commit validation with invalid commit."""
        service, mock_executor = git_diff_service
        
        # First commit succeeds, second fails
        mock_executor.execute_git_command.side_effect = [
            GitCommandResult(success=True, stdout="", stderr="", return_code=0, command=[], execution_time=0.1),
            GitCommandResult(success=False, stdout="", stderr="fatal: Not a valid object name", return_code=1, command=[], execution_time=0.1)
        ]
        
        result = service.validate_commits("/path/to/repo", "abc123", "invalid")
        assert result is False

    def test_validate_commits_empty_hash(self, git_diff_service):
        """Test commit validation with empty commit hash."""
        service, mock_executor = git_diff_service
        
        mock_executor.execute_git_command.return_value = GitCommandResult(
            success=True,
            stdout="",
            stderr="",
            return_code=0,
            command=["git", "cat-file", "-e", "abc123"],
            execution_time=0.1
        )
        
        # Empty commit hash should be skipped
        result = service.validate_commits("/path/to/repo", "abc123", "")
        assert result is True
        
        # Should only be called once (for the non-empty commit)
        assert mock_executor.execute_git_command.call_count == 1

    def test_file_extension_filtering(self, git_diff_service):
        """Test that file extension filtering works correctly."""
        service, mock_executor = git_diff_service
        
        # Mock git diff output with mixed file types
        diff_output = """A\tsrc/file.py
A\tsrc/file.js
A\tsrc/file.txt
A\tREADME.md
A\tDockerfile"""
        
        mock_executor.execute_git_command.return_value = GitCommandResult(
            success=True,
            stdout=diff_output,
            stderr="",
            return_code=0,
            command=["git", "diff", "--name-status", "-M", "abc123", "def456"],
            execution_time=0.1
        )
        
        result = service.get_changes_between_commits(
            "/path/to/repo",
            "abc123",
            "def456",
            file_extensions=[".py", ".md"]
        )
        
        # Should only include .py and .md files
        assert result.total_changes == 2
        assert "src/file.py" in result.added_files
        assert "README.md" in result.added_files
        assert "src/file.js" not in result.added_files
        assert "src/file.txt" not in result.added_files
        assert "Dockerfile" not in result.added_files