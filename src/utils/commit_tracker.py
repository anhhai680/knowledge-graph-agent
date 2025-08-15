"""
Commit Tracker for Git-based Incremental Re-indexing.

This module provides functionality to track the last indexed commit hash
for each repository/branch combination, enabling incremental re-indexing
based on git history.
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from loguru import logger


class CommitTracker:
    """
    Track last indexed commit hashes for repositories and branches.
    
    This class provides thread-safe persistence of commit tracking data
    to enable incremental re-indexing based on git history.
    """

    def __init__(self, storage_path: str = "indexed_commits.json"):
        """
        Initialize commit tracker.
        
        Args:
            storage_path: Path to store commit tracking data
        """
        self.storage_path = Path(storage_path)
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        
        # Load existing data
        self._load_data()
        
        logger.info(f"Initialized commit tracker with storage: {self.storage_path}")

    def _get_repo_key(self, repository: str, branch: str = "main") -> str:
        """
        Generate key for repository/branch combination.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            branch: Branch name
            
        Returns:
            Unique key for this repo/branch combination
        """
        return f"{repository}#{branch}"

    def _load_data(self) -> None:
        """Load commit tracking data from storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    self._data = json.load(f)
                logger.debug(f"Loaded commit tracking data for {len(self._data)} repositories")
            else:
                self._data = {}
                logger.debug("No existing commit tracking data found, starting fresh")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in commit tracking file: {e}")
            self._data = {}
        except Exception as e:
            logger.error(f"Failed to load commit tracking data: {e}")
            self._data = {}

    def _save_data(self) -> None:
        """Save commit tracking data to storage."""
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write data atomically
            temp_path = self.storage_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self._data, f, indent=2, default=str)
            
            # Atomic rename
            temp_path.replace(self.storage_path)
            
            logger.debug(f"Saved commit tracking data for {len(self._data)} repositories")
        except Exception as e:
            logger.error(f"Failed to save commit tracking data: {e}")
            raise

    def get_last_indexed_commit(self, repository: str, branch: str = "main") -> Optional[str]:
        """
        Get the last indexed commit hash for a repository/branch.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            branch: Branch name
            
        Returns:
            Last indexed commit hash, or None if never indexed
        """
        with self._lock:
            repo_key = self._get_repo_key(repository, branch)
            repo_data = self._data.get(repo_key, {})
            last_commit = repo_data.get("last_commit")
            
            if last_commit:
                logger.debug(f"Last indexed commit for {repository}#{branch}: {last_commit}")
            else:
                logger.debug(f"No previous index found for {repository}#{branch}")
                
            return last_commit

    def update_last_indexed_commit(
        self, 
        repository: str, 
        commit_hash: str, 
        branch: str = "main",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the last indexed commit hash for a repository/branch.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            commit_hash: Latest indexed commit hash
            branch: Branch name
            metadata: Additional metadata to store
        """
        with self._lock:
            repo_key = self._get_repo_key(repository, branch)
            
            # Create or update repository data
            if repo_key not in self._data:
                self._data[repo_key] = {}
            
            self._data[repo_key].update({
                "repository": repository,
                "branch": branch,
                "last_commit": commit_hash,
                "last_indexed_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            })
            
            # Save to storage
            self._save_data()
            
            logger.info(f"Updated last indexed commit for {repository}#{branch}: {commit_hash}")

    def get_repository_info(self, repository: str, branch: str = "main") -> Optional[Dict[str, Any]]:
        """
        Get complete tracking information for a repository/branch.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            branch: Branch name
            
        Returns:
            Complete tracking information or None if not found
        """
        with self._lock:
            repo_key = self._get_repo_key(repository, branch)
            return self._data.get(repo_key)

    def list_tracked_repositories(self) -> Dict[str, Dict[str, Any]]:
        """
        List all tracked repositories and their information.
        
        Returns:
            Dictionary of all tracked repositories
        """
        with self._lock:
            return dict(self._data)

    def remove_repository(self, repository: str, branch: str = "main") -> bool:
        """
        Remove tracking data for a repository/branch.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            branch: Branch name
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            repo_key = self._get_repo_key(repository, branch)
            
            if repo_key in self._data:
                del self._data[repo_key]
                self._save_data()
                logger.info(f"Removed tracking data for {repository}#{branch}")
                return True
            else:
                logger.warning(f"No tracking data found for {repository}#{branch}")
                return False

    def has_been_indexed(self, repository: str, branch: str = "main") -> bool:
        """
        Check if a repository/branch has been indexed before.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            branch: Branch name
            
        Returns:
            True if previously indexed, False otherwise
        """
        return self.get_last_indexed_commit(repository, branch) is not None

    def get_indexing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about indexed repositories.
        
        Returns:
            Statistics about tracked repositories
        """
        with self._lock:
            total_repos = len(self._data)
            repositories_by_branch = {}
            
            for repo_data in self._data.values():
                branch = repo_data.get("branch", "main")
                if branch not in repositories_by_branch:
                    repositories_by_branch[branch] = 0
                repositories_by_branch[branch] += 1
            
            return {
                "total_repositories": total_repos,
                "repositories_by_branch": repositories_by_branch,
                "last_updated": datetime.now().isoformat()
            }


# Global instance for application use
_commit_tracker: Optional[CommitTracker] = None


def get_commit_tracker() -> CommitTracker:
    """
    Get the global commit tracker instance.
    
    Returns:
        Global CommitTracker instance
    """
    global _commit_tracker
    if _commit_tracker is None:
        _commit_tracker = CommitTracker()
    return _commit_tracker