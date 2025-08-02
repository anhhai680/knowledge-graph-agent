"""
Git Repository Manager for the Knowledge Graph Agent.

This module provides functionality for managing local Git repositories,
including cloning, updating, and cleanup operations.
"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .git_command_executor import GitCommandExecutor, GitCommandResult


class GitRepositoryManager:
    """
    Manage local Git repository operations.
    
    This class handles cloning, updating, and managing local copies of
    GitHub repositories for the Knowledge Graph Agent.
    """

    def __init__(self, temp_repo_base_path: str = "temp_repo"):
        """
        Initialize Git repository manager.
        
        Args:
            temp_repo_base_path: Base directory for storing repositories
        """
        self.temp_repo_base_path = Path(temp_repo_base_path)
        self.git_executor = GitCommandExecutor()
        
        # Ensure base directory exists
        self.ensure_temp_repo_directory()
        
        logger.debug(f"Initialized Git repository manager with base path: {self.temp_repo_base_path}")

    def ensure_temp_repo_directory(self) -> None:
        """Create temp_repo directory if it doesn't exist."""
        try:
            self.temp_repo_base_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured temp repo directory exists: {self.temp_repo_base_path}")
        except Exception as e:
            logger.error(f"Failed to create temp repo directory: {e}")
            raise

    def get_local_repo_path(self, repo_owner: str, repo_name: str) -> str:
        """
        Generate local path for repository.
        
        Args:
            repo_owner: Repository owner/organization
            repo_name: Repository name
            
        Returns:
            Local path for the repository: temp_repo/{owner}/{repo}
        """
        repo_path = self.temp_repo_base_path / repo_owner / repo_name
        return str(repo_path)

    def clone_or_pull_repository(
        self, 
        repo_url: str, 
        local_path: str, 
        branch: str = "main"
    ) -> bool:
        """
        Clone repository if not exists, otherwise pull latest changes.
        
        Args:
            repo_url: Git URL for the repository
            local_path: Local path where repository should be stored
            branch: Branch to clone/pull (default: main)
            
        Returns:
            True if operation successful, False otherwise
        """
        local_path_obj = Path(local_path)
        
        try:
            if local_path_obj.exists() and self.is_repository_valid(local_path):
                # Repository exists and is valid, try to pull
                logger.info(f"Updating existing repository at {local_path}")
                pull_success = self._pull_repository(local_path, branch)
                if pull_success:
                    return True
                else:
                    # Pull failed, try fresh clone
                    logger.warning(f"Pull failed for {local_path}, attempting fresh clone")
                    return self._clone_repository(repo_url, local_path, branch)
            else:
                # Repository doesn't exist or is invalid, clone it
                logger.info(f"Cloning repository {repo_url} to {local_path}")
                return self._clone_repository(repo_url, local_path, branch)
                
        except Exception as e:
            logger.error(f"Failed to clone or pull repository {repo_url}: {e}")
            # Try cleanup and fresh clone as last resort
            try:
                if local_path_obj.exists():
                    logger.info(f"Attempting cleanup and fresh clone for {local_path}")
                    shutil.rmtree(local_path)
                return self._clone_repository(repo_url, local_path, branch)
            except Exception as cleanup_error:
                logger.error(f"Failed cleanup and fresh clone: {cleanup_error}")
                return False

    def _clone_repository(self, repo_url: str, local_path: str, branch: str) -> bool:
        """
        Clone a repository to local path.
        
        Args:
            repo_url: Git URL for the repository
            local_path: Local path where repository should be stored
            branch: Branch to clone
            
        Returns:
            True if clone successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Remove existing directory if it exists (always cleanup for fresh clone)
            if Path(local_path).exists():
                logger.info(f"Removing existing directory at {local_path} for fresh clone")
                try:
                    shutil.rmtree(local_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup existing directory: {cleanup_error}")
                    # Try to force cleanup with different permissions
                    import stat
                    for root, dirs, files in os.walk(local_path, topdown=False):
                        for name in files:
                            file_path = os.path.join(root, name)
                            os.chmod(file_path, stat.S_IWRITE)
                            os.remove(file_path)
                        for name in dirs:
                            dir_path = os.path.join(root, name)
                            os.chmod(dir_path, stat.S_IWRITE)
                            os.rmdir(dir_path)
                    os.rmdir(local_path)
            
            # Clone the repository
            result = self.git_executor.clone_repository(repo_url, local_path, branch)
            
            if result:
                logger.info(f"Successfully cloned repository to {local_path}")
                return True
            else:
                logger.error(f"Failed to clone repository to {local_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False

    def _pull_repository(self, local_path: str, branch: str) -> bool:
        """
        Pull latest changes for existing repository.
        
        Args:
            local_path: Local path of the repository
            branch: Branch to pull
            
        Returns:
            True if pull successful, False otherwise
        """
        try:
            result = self.git_executor.pull_repository(local_path, branch)
            
            if result:
                logger.info(f"Successfully updated repository at {local_path}")
                return True
            else:
                logger.warning(f"Failed to update repository at {local_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling repository: {e}")
            return False

    def is_repository_valid(self, local_path: str) -> bool:
        """
        Check if local repository is valid Git repo.
        
        Args:
            local_path: Path to check
            
        Returns:
            True if valid Git repository, False otherwise
        """
        try:
            local_path_obj = Path(local_path)
            
            # Check if directory exists
            if not local_path_obj.exists():
                return False
            
            # Check if .git directory exists
            git_dir = local_path_obj / ".git"
            if not git_dir.exists():
                return False
            
            # Try to run a simple git command
            result = self.git_executor.execute_git_command(
                ["rev-parse", "--git-dir"], 
                str(local_path_obj)
            )
            
            return result.success
            
        except Exception as e:
            logger.debug(f"Repository validation failed for {local_path}: {e}")
            return False

    def cleanup_repository(self, local_path: str) -> None:
        """
        Remove local repository directory.
        
        Args:
            local_path: Path to repository to remove
        """
        try:
            local_path_obj = Path(local_path)
            if local_path_obj.exists():
                shutil.rmtree(local_path_obj)
                logger.info(f"Cleaned up repository at {local_path}")
            else:
                logger.debug(f"Repository path does not exist: {local_path}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup repository at {local_path}: {e}")
            raise

    def get_repository_info(self, local_path: str) -> Dict[str, Any]:
        """
        Get basic repository information.
        
        Args:
            local_path: Path to local repository
            
        Returns:
            Dictionary containing repository information
        """
        try:
            if not self.is_repository_valid(local_path):
                return {}
            
            info = {}
            
            # Get current branch
            branch_result = self.git_executor.execute_git_command(
                ["branch", "--show-current"], 
                local_path
            )
            if branch_result.success:
                info["current_branch"] = branch_result.stdout.strip()
            
            # Get remote URL
            remote_result = self.git_executor.execute_git_command(
                ["remote", "get-url", "origin"], 
                local_path
            )
            if remote_result.success:
                info["remote_url"] = remote_result.stdout.strip()
            
            # Get last commit info
            commit_result = self.git_executor.execute_git_command(
                ["log", "-1", "--pretty=format:%H|%an|%ad|%s", "--date=iso"], 
                local_path
            )
            if commit_result.success:
                commit_parts = commit_result.stdout.strip().split("|", 3)
                if len(commit_parts) == 4:
                    info["last_commit"] = {
                        "sha": commit_parts[0],
                        "author": commit_parts[1],
                        "date": commit_parts[2],
                        "message": commit_parts[3]
                    }
            
            # Get repository stats
            stats = self.git_executor.get_repository_stats(local_path)
            info.update(stats)
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get repository info for {local_path}: {e}")
            return {}

    def cleanup_old_repositories(self, max_age_days: int = 7) -> int:
        """
        Clean up old cached repositories.
        
        Args:
            max_age_days: Maximum age in days for cached repositories
            
        Returns:
            Number of repositories cleaned up
        """
        try:
            if not self.temp_repo_base_path.exists():
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cleaned_count = 0
            
            # Walk through all repository directories
            for owner_dir in self.temp_repo_base_path.iterdir():
                if not owner_dir.is_dir():
                    continue
                    
                for repo_dir in owner_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue
                    
                    # Check if repository is older than cutoff
                    stat = repo_dir.stat()
                    repo_date = datetime.fromtimestamp(stat.st_mtime)
                    
                    if repo_date < cutoff_date:
                        try:
                            shutil.rmtree(repo_dir)
                            cleaned_count += 1
                            logger.info(f"Cleaned up old repository: {repo_dir}")
                        except Exception as e:
                            logger.error(f"Failed to cleanup {repo_dir}: {e}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old repositories: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache utilization statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        try:
            if not self.temp_repo_base_path.exists():
                return {
                    "total_repositories": 0,
                    "total_size_mb": 0,
                    "oldest_repository": None,
                    "newest_repository": None
                }
            
            total_repos = 0
            total_size = 0
            oldest_date = None
            newest_date = None
            
            # Walk through all repository directories
            for owner_dir in self.temp_repo_base_path.iterdir():
                if not owner_dir.is_dir():
                    continue
                    
                for repo_dir in owner_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue
                    
                    total_repos += 1
                    
                    # Calculate directory size
                    for file_path in repo_dir.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                    
                    # Track dates
                    stat = repo_dir.stat()
                    repo_date = datetime.fromtimestamp(stat.st_mtime)
                    
                    if oldest_date is None or repo_date < oldest_date:
                        oldest_date = repo_date
                    if newest_date is None or repo_date > newest_date:
                        newest_date = repo_date
            
            return {
                "total_repositories": total_repos,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "oldest_repository": oldest_date.isoformat() if oldest_date else None,
                "newest_repository": newest_date.isoformat() if newest_date else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                "total_repositories": 0,
                "total_size_mb": 0,
                "oldest_repository": None,
                "newest_repository": None,
                "error": str(e)
            }
