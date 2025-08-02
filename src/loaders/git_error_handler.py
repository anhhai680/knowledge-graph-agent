"""
Git Error Handler for the Knowledge Graph Agent.

This module provides comprehensive error handling and recovery strategies
for Git operations, including timeouts, authentication failures, and corruption.
"""

import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger


class GitOperationError(Exception):
    """Custom exception for Git operation failures."""
    
    def __init__(self, message: str, operation: str, repo_path: str = "", original_error: Optional[Exception] = None):
        """
        Initialize Git operation error.
        
        Args:
            message: Error message
            operation: Git operation that failed
            repo_path: Repository path where error occurred
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.operation = operation
        self.repo_path = repo_path
        self.original_error = original_error


class GitErrorHandler:
    """
    Handle Git operation errors and recovery strategies.
    
    This class provides comprehensive error handling for various Git operation
    failures and implements recovery strategies to ensure robust operation.
    """

    def __init__(self):
        """Initialize Git error handler."""
        logger.debug("Initialized Git error handler")

    def handle_clone_error(self, error: Exception, repo_url: str) -> Optional[str]:
        """
        Handle repository clone failures with recovery strategies.
        
        Args:
            error: Exception that occurred during clone
            repo_url: Repository URL that failed to clone
            
        Returns:
            Recovery action taken, or None if no recovery possible
        """
        try:
            error_str = str(error).lower()
            
            # Authentication errors
            if any(keyword in error_str for keyword in [
                'authentication', 'permission denied', 'access denied',
                'unauthorized', 'forbidden', '403', '401'
            ]):
                logger.error(f"Authentication failed for {repo_url}")
                return self._suggest_auth_recovery(repo_url)
            
            # Network connectivity errors
            elif any(keyword in error_str for keyword in [
                'network', 'connection', 'timeout', 'unreachable',
                'dns', 'resolve', 'refused'
            ]):
                logger.error(f"Network connectivity issue for {repo_url}")
                return self._suggest_network_recovery(repo_url)
            
            # Repository not found
            elif any(keyword in error_str for keyword in [
                'not found', '404', 'does not exist', 'repository not found'
            ]):
                logger.error(f"Repository not found: {repo_url}")
                return "verify_repository_url"
            
            # Disk space issues
            elif any(keyword in error_str for keyword in [
                'no space', 'disk full', 'insufficient space'
            ]):
                logger.error(f"Insufficient disk space for cloning {repo_url}")
                return self._suggest_space_recovery()
            
            # Branch not found
            elif any(keyword in error_str for keyword in [
                'branch not found', 'reference not found', 'invalid branch'
            ]):
                logger.error(f"Branch not found for {repo_url}")
                return "try_default_branch"
            
            # Generic Git errors
            else:
                logger.error(f"Unhandled clone error for {repo_url}: {error}")
                return "retry_with_fallback"
                
        except Exception as e:
            logger.error(f"Error in clone error handler: {e}")
            return None

    def handle_pull_error(self, error: Exception, repo_path: str) -> bool:
        """
        Handle repository update failures.
        
        Args:
            error: Exception that occurred during pull
            repo_path: Local repository path
            
        Returns:
            True if recovery was attempted, False otherwise
        """
        try:
            error_str = str(error).lower()
            
            # Merge conflicts
            if any(keyword in error_str for keyword in [
                'merge conflict', 'conflict', 'diverged', 'cannot merge'
            ]):
                logger.warning(f"Merge conflict detected in {repo_path}, performing hard reset")
                return self._resolve_merge_conflict(repo_path)
            
            # Corrupted repository
            elif any(keyword in error_str for keyword in [
                'corrupt', 'damaged', 'invalid object', 'bad object'
            ]):
                logger.warning(f"Repository corruption detected in {repo_path}")
                return self._handle_corruption(repo_path)
            
            # Detached HEAD or branch issues
            elif any(keyword in error_str for keyword in [
                'detached head', 'not on any branch', 'invalid branch'
            ]):
                logger.warning(f"Branch issues detected in {repo_path}")
                return self._fix_branch_issues(repo_path)
            
            # Authentication expired
            elif any(keyword in error_str for keyword in [
                'authentication', 'credential', 'token'
            ]):
                logger.error(f"Authentication issue during pull for {repo_path}")
                return False  # Cannot recover from auth issues during pull
            
            else:
                logger.error(f"Unhandled pull error for {repo_path}: {error}")
                return False
                
        except Exception as e:
            logger.error(f"Error in pull error handler: {e}")
            return False

    def handle_command_timeout(self, command: str, repo_path: str) -> None:
        """
        Handle Git command timeouts.
        
        Args:
            command: Git command that timed out
            repo_path: Repository path where timeout occurred
        """
        try:
            logger.error(f"Git command timed out: {command} in {repo_path}")
            
            # For clone operations, suggest using shallow clone
            if "clone" in command.lower():
                logger.info("Consider using shallow clone for large repositories")
            
            # For pull operations, suggest fetching specific branch
            elif "pull" in command.lower():
                logger.info("Consider fetching specific branch instead of all branches")
            
            # For log operations, suggest limiting output
            elif "log" in command.lower():
                logger.info("Consider limiting log output with --max-count or date ranges")
            
        except Exception as e:
            logger.error(f"Error handling command timeout: {e}")

    def cleanup_corrupted_repo(self, repo_path: str) -> None:
        """
        Remove and prepare for re-clone of corrupted repository.
        
        Args:
            repo_path: Path to corrupted repository
        """
        try:
            repo_path_obj = Path(repo_path)
            
            if repo_path_obj.exists():
                logger.warning(f"Removing corrupted repository at {repo_path}")
                shutil.rmtree(repo_path_obj)
                logger.info(f"Corrupted repository removed: {repo_path}")
            else:
                logger.debug(f"Repository path does not exist: {repo_path}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup corrupted repository {repo_path}: {e}")
            raise GitOperationError(
                f"Cannot cleanup corrupted repository: {e}",
                "cleanup",
                repo_path,
                e
            )

    def suggest_recovery_action(self, error: GitOperationError) -> str:
        """
        Suggest recovery actions for different error types.
        
        Args:
            error: Git operation error
            
        Returns:
            Suggested recovery action
        """
        try:
            operation = error.operation.lower()
            
            if operation == "clone":
                if "authentication" in str(error).lower():
                    return "Check GitHub token and repository permissions"
                elif "not found" in str(error).lower():
                    return "Verify repository URL and existence"
                elif "network" in str(error).lower():
                    return "Check internet connection and GitHub status"
                else:
                    return "Try again or use API-based loader as fallback"
            
            elif operation == "pull":
                if "conflict" in str(error).lower():
                    return "Repository will be reset to remote state"
                elif "corrupt" in str(error).lower():
                    return "Repository will be re-cloned"
                else:
                    return "Try fresh clone of repository"
            
            elif operation == "timeout":
                return "Increase timeout or use shallow clone for large repositories"
            
            else:
                return "Contact system administrator or use API-based loader"
                
        except Exception as e:
            logger.error(f"Error suggesting recovery action: {e}")
            return "Unable to suggest recovery action"

    def _suggest_auth_recovery(self, repo_url: str) -> str:
        """Suggest authentication recovery strategies."""
        return "check_github_token"

    def _suggest_network_recovery(self, repo_url: str) -> str:
        """Suggest network recovery strategies."""
        return "check_connectivity"

    def _suggest_space_recovery(self) -> str:
        """Suggest disk space recovery strategies."""
        return "cleanup_old_repositories"

    def _resolve_merge_conflict(self, repo_path: str) -> bool:
        """
        Resolve merge conflicts by performing hard reset.
        
        Args:
            repo_path: Local repository path
            
        Returns:
            True if reset was successful
        """
        try:
            from .git_command_executor import GitCommandExecutor
            
            git_executor = GitCommandExecutor()
            
            # Reset to remote state
            reset_result = git_executor.execute_git_command(
                ["reset", "--hard", "HEAD"],
                repo_path
            )
            
            if reset_result.success:
                # Clean untracked files
                clean_result = git_executor.execute_git_command(
                    ["clean", "-fd"],
                    repo_path
                )
                
                logger.info(f"Successfully resolved merge conflicts in {repo_path}")
                return clean_result.success
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve merge conflict in {repo_path}: {e}")
            return False

    def _handle_corruption(self, repo_path: str) -> bool:
        """
        Handle repository corruption by removing and flagging for re-clone.
        
        Args:
            repo_path: Local repository path
            
        Returns:
            True if cleanup was successful
        """
        try:
            self.cleanup_corrupted_repo(repo_path)
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle corruption in {repo_path}: {e}")
            return False

    def _fix_branch_issues(self, repo_path: str) -> bool:
        """
        Fix branch-related issues in repository.
        
        Args:
            repo_path: Local repository path
            
        Returns:
            True if branch issues were fixed
        """
        try:
            from .git_command_executor import GitCommandExecutor
            
            git_executor = GitCommandExecutor()
            
            # Get the default branch
            branch_result = git_executor.execute_git_command(
                ["symbolic-ref", "refs/remotes/origin/HEAD"],
                repo_path
            )
            
            if branch_result.success:
                # Extract branch name
                default_branch = branch_result.stdout.strip().split("/")[-1]
                
                # Checkout the default branch
                checkout_result = git_executor.execute_git_command(
                    ["checkout", "-B", default_branch, f"origin/{default_branch}"],
                    repo_path
                )
                
                if checkout_result.success:
                    logger.info(f"Successfully fixed branch issues in {repo_path}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to fix branch issues in {repo_path}: {e}")
            return False

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.
        
        Returns:
            Dictionary containing error statistics
        """
        # This would be implemented with actual error tracking
        # For now, return empty statistics
        return {
            "total_errors": 0,
            "clone_errors": 0,
            "pull_errors": 0,
            "timeout_errors": 0,
            "auth_errors": 0,
            "corruption_errors": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0
        }

    def is_recoverable_error(self, error: Exception) -> bool:
        """
        Check if an error is recoverable.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is potentially recoverable
        """
        try:
            error_str = str(error).lower()
            
            # Non-recoverable errors
            non_recoverable = [
                'permission denied',
                'access denied',
                'forbidden',
                'unauthorized',
                'not found',
                'does not exist'
            ]
            
            # Recoverable errors
            recoverable = [
                'timeout',
                'connection',
                'network',
                'conflict',
                'corrupt',
                'detached head'
            ]
            
            if any(keyword in error_str for keyword in non_recoverable):
                return False
            
            if any(keyword in error_str for keyword in recoverable):
                return True
            
            # Default to potentially recoverable
            return True
            
        except Exception:
            return False
