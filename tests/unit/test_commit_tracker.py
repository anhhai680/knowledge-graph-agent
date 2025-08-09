"""
Unit tests for CommitTracker.

Tests the functionality of tracking last indexed commits for repositories
to support incremental re-indexing.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from src.utils.commit_tracker import CommitTracker


class TestCommitTracker:
    """Test cases for CommitTracker functionality."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create a temporary file for testing storage."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def commit_tracker(self, temp_storage_path):
        """Create CommitTracker instance with temporary storage."""
        return CommitTracker(storage_path=temp_storage_path)

    def test_initial_state(self, commit_tracker):
        """Test that CommitTracker starts with empty state."""
        assert commit_tracker.get_last_indexed_commit("test/repo") is None
        assert not commit_tracker.has_been_indexed("test/repo")

    def test_update_and_retrieve_commit(self, commit_tracker):
        """Test updating and retrieving commit hashes."""
        repository = "owner/repo"
        commit_hash = "abc123def456"
        branch = "main"

        # Update commit
        commit_tracker.update_last_indexed_commit(repository, commit_hash, branch)

        # Retrieve commit
        retrieved_commit = commit_tracker.get_last_indexed_commit(repository, branch)
        assert retrieved_commit == commit_hash

        # Check if indexed
        assert commit_tracker.has_been_indexed(repository, branch)

    def test_different_branches(self, commit_tracker):
        """Test tracking different branches of the same repository."""
        repository = "owner/repo"
        main_commit = "abc123"
        dev_commit = "def456"

        # Update different branches
        commit_tracker.update_last_indexed_commit(repository, main_commit, "main")
        commit_tracker.update_last_indexed_commit(repository, dev_commit, "develop")

        # Verify both are tracked separately
        assert commit_tracker.get_last_indexed_commit(repository, "main") == main_commit
        assert commit_tracker.get_last_indexed_commit(repository, "develop") == dev_commit

    def test_repository_info(self, commit_tracker):
        """Test retrieving complete repository information."""
        repository = "owner/repo"
        commit_hash = "abc123def456"
        metadata = {"files_processed": 100, "processing_mode": "incremental"}

        commit_tracker.update_last_indexed_commit(
            repository, commit_hash, "main", metadata
        )

        repo_info = commit_tracker.get_repository_info(repository, "main")
        assert repo_info is not None
        assert repo_info["repository"] == repository
        assert repo_info["branch"] == "main"
        assert repo_info["last_commit"] == commit_hash
        assert repo_info["metadata"] == metadata
        assert "last_indexed_at" in repo_info

    def test_list_tracked_repositories(self, commit_tracker):
        """Test listing all tracked repositories."""
        repos = [
            ("owner1/repo1", "commit1", "main"),
            ("owner2/repo2", "commit2", "develop"),
            ("owner1/repo3", "commit3", "main"),
        ]

        # Add repositories
        for repo, commit, branch in repos:
            commit_tracker.update_last_indexed_commit(repo, commit, branch)

        # List all repositories
        tracked_repos = commit_tracker.list_tracked_repositories()
        assert len(tracked_repos) == 3

        # Verify all repositories are tracked
        for repo, commit, branch in repos:
            repo_key = f"{repo}#{branch}"
            assert repo_key in tracked_repos
            assert tracked_repos[repo_key]["last_commit"] == commit

    def test_remove_repository(self, commit_tracker):
        """Test removing repository tracking."""
        repository = "owner/repo"
        commit_hash = "abc123"

        # Add repository
        commit_tracker.update_last_indexed_commit(repository, commit_hash)

        # Verify it exists
        assert commit_tracker.has_been_indexed(repository)

        # Remove it
        removed = commit_tracker.remove_repository(repository)
        assert removed is True

        # Verify it's gone
        assert not commit_tracker.has_been_indexed(repository)
        assert commit_tracker.get_last_indexed_commit(repository) is None

        # Try removing again
        removed_again = commit_tracker.remove_repository(repository)
        assert removed_again is False

    def test_persistence(self, temp_storage_path):
        """Test that data persists across instances."""
        repository = "owner/repo"
        commit_hash = "abc123def456"

        # Create first instance and add data
        tracker1 = CommitTracker(storage_path=temp_storage_path)
        tracker1.update_last_indexed_commit(repository, commit_hash)

        # Create second instance and verify data persists
        tracker2 = CommitTracker(storage_path=temp_storage_path)
        retrieved_commit = tracker2.get_last_indexed_commit(repository)
        assert retrieved_commit == commit_hash

    def test_statistics(self, commit_tracker):
        """Test indexing statistics functionality."""
        repos = [
            ("owner1/repo1", "main"),
            ("owner2/repo2", "develop"),
            ("owner1/repo3", "main"),
            ("owner3/repo4", "feature"),
        ]

        # Add repositories
        for i, (repo, branch) in enumerate(repos):
            commit_tracker.update_last_indexed_commit(repo, f"commit{i}", branch)

        # Get statistics
        stats = commit_tracker.get_indexing_statistics()
        assert stats["total_repositories"] == 4
        assert stats["repositories_by_branch"]["main"] == 2
        assert stats["repositories_by_branch"]["develop"] == 1
        assert stats["repositories_by_branch"]["feature"] == 1

    def test_invalid_json_recovery(self, temp_storage_path):
        """Test recovery from corrupted JSON file."""
        # Write invalid JSON to file
        with open(temp_storage_path, 'w') as f:
            f.write("invalid json content")

        # Create tracker - should recover gracefully
        tracker = CommitTracker(storage_path=temp_storage_path)
        assert tracker.get_last_indexed_commit("test/repo") is None

        # Should be able to add new data
        tracker.update_last_indexed_commit("test/repo", "abc123")
        assert tracker.get_last_indexed_commit("test/repo") == "abc123"

    def test_thread_safety(self, commit_tracker):
        """Test basic thread safety of commit tracker."""
        import threading
        import time

        repository = "owner/repo"
        results = []

        def update_commit(commit_hash):
            commit_tracker.update_last_indexed_commit(repository, commit_hash)
            retrieved = commit_tracker.get_last_indexed_commit(repository)
            results.append(retrieved)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_commit, args=[f"commit{i}"])
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have 5 results
        assert len(results) == 5

        # Final commit should be one of the updated values
        final_commit = commit_tracker.get_last_indexed_commit(repository)
        assert final_commit in [f"commit{i}" for i in range(5)]


def test_global_commit_tracker():
    """Test the global commit tracker instance."""
    from src.utils.commit_tracker import get_commit_tracker

    # Get global instance
    tracker1 = get_commit_tracker()
    tracker2 = get_commit_tracker()

    # Should be the same instance
    assert tracker1 is tracker2

    # Should work normally
    tracker1.update_last_indexed_commit("test/repo", "abc123")
    assert tracker2.get_last_indexed_commit("test/repo") == "abc123"