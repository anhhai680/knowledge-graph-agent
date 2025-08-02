"""
Git Command Executor for the Knowledge Graph Agent.

This module provides safe execution of Git commands with proper error handling,
timeout management, and result parsing.
"""

import subprocess
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from loguru import logger


@dataclass
class GitCommandResult:
    """Result of a Git command execution."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    command: List[str]
    execution_time: float


class GitCommandExecutor:
    """
    Execute Git commands safely with error handling.
    
    This class provides a safe interface for executing Git commands with
    timeout management, error handling, and structured result parsing.
    """

    def __init__(self, timeout_seconds: int = 300):
        """
        Initialize Git command executor.
        
        Args:
            timeout_seconds: Default timeout for Git commands
        """
        self.timeout_seconds = timeout_seconds
        logger.debug(f"Initialized Git command executor with timeout: {timeout_seconds}s")

    def execute_git_command(
        self, 
        command: List[str], 
        cwd: str,
        timeout: Optional[int] = None
    ) -> GitCommandResult:
        """
        Execute Git command and return structured result.
        
        Args:
            command: Git command arguments (without 'git' prefix)
            cwd: Working directory for the command
            timeout: Command timeout (uses default if None)
            
        Returns:
            GitCommandResult with execution details
        """
        import time
        
        # Prepare full command
        full_command = ["git"] + command
        timeout = timeout or self.timeout_seconds
        
        logger.debug(f"Executing Git command: {' '.join(full_command)} in {cwd}")
        
        start_time = time.time()
        
        try:
            # Execute command
            result = subprocess.run(
                full_command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False  # Don't raise exception on non-zero exit
            )
            
            execution_time = time.time() - start_time
            
            # Create result object
            git_result = GitCommandResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                command=full_command,
                execution_time=execution_time
            )
            
            if git_result.success:
                logger.debug(f"Git command completed successfully in {execution_time:.2f}s")
            else:
                logger.warning(
                    f"Git command failed with return code {result.returncode}: {result.stderr}"
                )
            
            return git_result
            
        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            logger.error(f"Git command timed out after {timeout}s")
            
            return GitCommandResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                return_code=-1,
                command=full_command,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing Git command: {e}")
            
            return GitCommandResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                command=full_command,
                execution_time=execution_time
            )

    def clone_repository(
        self, 
        repo_url: str, 
        local_path: str, 
        branch: str = "main"
    ) -> bool:
        """
        Execute git clone command.
        
        Args:
            repo_url: Repository URL to clone
            local_path: Local path where repository should be cloned
            branch: Branch to clone (default: main)
            
        Returns:
            True if clone successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Get just the repository name for the clone target
            repo_name = Path(local_path).name
            parent_dir = str(Path(local_path).parent)
            
            # Clone command with branch specification
            command = [
                "clone",
                "--branch", branch,
                "--single-branch",
                "--depth", "1",  # Shallow clone for performance
                repo_url,
                repo_name  # Use just the repo name, not the full path
            ]
            
            result = self.execute_git_command(
                command, 
                parent_dir,  # Execute from the parent directory
                timeout=600  # Longer timeout for clone operations
            )
            
            if result.success:
                logger.info(f"Successfully cloned {repo_url} to {local_path}")
                return True
            else:
                logger.error(f"Failed to clone {repo_url}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error cloning repository {repo_url}: {e}")
            return False

    def pull_repository(self, local_path: str, branch: str = "main") -> bool:
        """
        Execute git pull command.
        
        Args:
            local_path: Local path of the repository
            branch: Branch to pull (default: main)
            
        Returns:
            True if pull successful, False otherwise
        """
        try:
            # First, fetch the latest changes
            fetch_result = self.execute_git_command(
                ["fetch", "origin", branch],
                local_path
            )
            
            if not fetch_result.success:
                logger.warning(f"Fetch failed: {fetch_result.stderr}")
                # Continue with pull anyway
            
            # Pull the latest changes
            pull_result = self.execute_git_command(
                ["pull", "origin", branch],
                local_path
            )
            
            if pull_result.success:
                logger.info(f"Successfully pulled latest changes for {local_path}")
                return True
            else:
                logger.error(f"Failed to pull {local_path}: {pull_result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling repository {local_path}: {e}")
            return False

    def get_file_commit_info(self, file_path: str, repo_path: str) -> Dict[str, Any]:
        """
        Get latest commit info for specific file using git log.
        
        Args:
            file_path: Relative path to file within repository
            repo_path: Path to local repository
            
        Returns:
            Dictionary containing commit information
        """
        try:
            # Get latest commit for specific file
            command = [
                "log",
                "-1",
                "--pretty=format:%H|%an|%ae|%ad|%s",
                "--date=iso",
                "--",
                file_path
            ]
            
            result = self.execute_git_command(command, repo_path)
            
            if not result.success or not result.stdout.strip():
                return {}
            
            # Parse the output
            parts = result.stdout.strip().split("|", 4)
            if len(parts) < 5:
                return {}
            
            return {
                "commit_sha": parts[0],
                "commit_author": parts[1],
                "commit_author_email": parts[2],
                "commit_date": parts[3],
                "commit_message": parts[4]
            }
            
        except Exception as e:
            logger.error(f"Error getting commit info for {file_path}: {e}")
            return {}

    def get_repository_stats(self, repo_path: str) -> Dict[str, Any]:
        """
        Get repository statistics using git commands.
        
        Args:
            repo_path: Path to local repository
            
        Returns:
            Dictionary containing repository statistics
        """
        try:
            stats = {}
            
            # Get total number of commits
            commit_count_result = self.execute_git_command(
                ["rev-list", "--count", "HEAD"],
                repo_path
            )
            if commit_count_result.success:
                stats["total_commits"] = int(commit_count_result.stdout.strip())
            
            # Get number of contributors
            contributors_result = self.execute_git_command(
                ["shortlog", "-sn", "--all"],
                repo_path
            )
            if contributors_result.success:
                contributor_lines = contributors_result.stdout.strip().split("\n")
                stats["total_contributors"] = len([line for line in contributor_lines if line.strip()])
            
            # Get repository size (number of files)
            file_count_result = self.execute_git_command(
                ["ls-files"],
                repo_path
            )
            if file_count_result.success:
                file_lines = file_count_result.stdout.strip().split("\n")
                stats["total_files"] = len([line for line in file_lines if line.strip()])
            
            # Get first and last commit dates
            first_commit_result = self.execute_git_command(
                ["log", "--reverse", "--pretty=format:%ad", "--date=iso", "-1"],
                repo_path
            )
            if first_commit_result.success:
                stats["first_commit_date"] = first_commit_result.stdout.strip()
            
            last_commit_result = self.execute_git_command(
                ["log", "--pretty=format:%ad", "--date=iso", "-1"],
                repo_path
            )
            if last_commit_result.success:
                stats["last_commit_date"] = last_commit_result.stdout.strip()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting repository stats for {repo_path}: {e}")
            return {}

    def get_file_history(
        self, 
        file_path: str, 
        repo_path: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get commit history for a file.
        
        Args:
            file_path: Relative path to file within repository
            repo_path: Path to local repository
            limit: Maximum number of commits to return
            
        Returns:
            List of commit information dictionaries
        """
        try:
            command = [
                "log",
                f"-{limit}",
                "--pretty=format:%H|%an|%ae|%ad|%s",
                "--date=iso",
                "--",
                file_path
            ]
            
            result = self.execute_git_command(command, repo_path)
            
            if not result.success or not result.stdout.strip():
                return []
            
            commits = []
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|", 4)
                if len(parts) >= 5:
                    commits.append({
                        "commit_sha": parts[0],
                        "commit_author": parts[1],
                        "commit_author_email": parts[2],
                        "commit_date": parts[3],
                        "commit_message": parts[4]
                    })
            
            return commits
            
        except Exception as e:
            logger.error(f"Error getting file history for {file_path}: {e}")
            return []

    def get_branch_info(self, repo_path: str) -> Dict[str, str]:
        """
        Get current branch and remote information.
        
        Args:
            repo_path: Path to local repository
            
        Returns:
            Dictionary containing branch information
        """
        try:
            info = {}
            
            # Get current branch
            branch_result = self.execute_git_command(
                ["branch", "--show-current"],
                repo_path
            )
            if branch_result.success:
                info["current_branch"] = branch_result.stdout.strip()
            
            # Get remote URL
            remote_result = self.execute_git_command(
                ["remote", "get-url", "origin"],
                repo_path
            )
            if remote_result.success:
                info["remote_url"] = remote_result.stdout.strip()
            
            # Get remote branches
            remote_branches_result = self.execute_git_command(
                ["branch", "-r"],
                repo_path
            )
            if remote_branches_result.success:
                branches = []
                for line in remote_branches_result.stdout.strip().split("\n"):
                    branch = line.strip()
                    if branch and not branch.startswith("origin/HEAD"):
                        branches.append(branch)
                info["remote_branches"] = branches
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting branch info for {repo_path}: {e}")
            return {}

    def validate_git_installation(self) -> bool:
        """
        Validate that Git is properly installed and accessible.
        
        Returns:
            True if Git is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"Git is available: {version}")
                return True
            else:
                logger.error("Git is not available or not working properly")
                return False
                
        except Exception as e:
            logger.error(f"Failed to validate Git installation: {e}")
            return False
