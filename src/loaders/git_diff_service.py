"""
Git Diff Service for Incremental Re-indexing.

This module provides functionality to compute file changes between git commits,
enabling incremental re-indexing by identifying added, modified, deleted, and
renamed files.
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from loguru import logger

from src.loaders.git_command_executor import GitCommandExecutor, GitCommandResult


class FileChangeType(str, Enum):
    """Types of file changes detected by git diff."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"
    TYPE_CHANGED = "type_changed"


@dataclass
class FileChange:
    """Represents a single file change between commits."""
    change_type: FileChangeType
    file_path: str
    old_file_path: Optional[str] = None  # For renames/copies
    similarity_index: Optional[int] = None  # For renames/copies (0-100)


@dataclass
class GitDiffResult:
    """Result of git diff analysis between two commits."""
    from_commit: str
    to_commit: str
    total_changes: int
    changes_by_type: Dict[FileChangeType, int]
    file_changes: List[FileChange]
    added_files: List[str]
    modified_files: List[str]
    deleted_files: List[str]
    renamed_files: List[Tuple[str, str]]  # (old_path, new_path)


class GitDiffService:
    """
    Service for computing git diffs between commits.
    
    This service analyzes changes between git commits to support
    incremental re-indexing by identifying which files need to be
    processed, updated, or removed.
    """

    def __init__(self):
        """Initialize git diff service."""
        self.git_executor = GitCommandExecutor()
        logger.debug("Initialized Git diff service")

    def get_current_commit(self, repo_path: str, branch: str = "main") -> Optional[str]:
        """
        Get the current commit hash for a branch.
        
        Args:
            repo_path: Path to the git repository
            branch: Branch name to check
            
        Returns:
            Current commit hash or None if failed
        """
        result = self.git_executor.execute_git_command(
            ["git", "rev-parse", f"origin/{branch}"],
            cwd=repo_path
        )
        
        if result.success:
            commit_hash = result.stdout.strip()
            logger.debug(f"Current commit for {branch}: {commit_hash}")
            return commit_hash
        else:
            logger.error(f"Failed to get current commit for {branch}: {result.stderr}")
            return None

    def get_changes_between_commits(
        self, 
        repo_path: str, 
        from_commit: str, 
        to_commit: str = "HEAD",
        file_extensions: Optional[List[str]] = None
    ) -> GitDiffResult:
        """
        Get file changes between two commits.
        
        Args:
            repo_path: Path to the git repository
            from_commit: Starting commit hash
            to_commit: Ending commit hash (default: HEAD)
            file_extensions: Optional filter for specific file extensions
            
        Returns:
            GitDiffResult with analyzed changes
        """
        logger.info(f"Computing diff from {from_commit} to {to_commit}")
        
        # Get the raw diff with rename detection
        result = self.git_executor.execute_git_command(
            ["git", "diff", "--name-status", "-M", from_commit, to_commit],
            cwd=repo_path
        )
        
        if not result.success:
            logger.error(f"Git diff failed: {result.stderr}")
            return GitDiffResult(
                from_commit=from_commit,
                to_commit=to_commit,
                total_changes=0,
                changes_by_type={},
                file_changes=[],
                added_files=[],
                modified_files=[],
                deleted_files=[],
                renamed_files=[]
            )
        
        # Parse the diff output
        file_changes = self._parse_diff_output(result.stdout, file_extensions)
        
        # Categorize changes
        added_files = []
        modified_files = []
        deleted_files = []
        renamed_files = []
        changes_by_type = {}
        
        for change in file_changes:
            # Count changes by type
            if change.change_type not in changes_by_type:
                changes_by_type[change.change_type] = 0
            changes_by_type[change.change_type] += 1
            
            # Categorize for easy access
            if change.change_type == FileChangeType.ADDED:
                added_files.append(change.file_path)
            elif change.change_type == FileChangeType.MODIFIED:
                modified_files.append(change.file_path)
            elif change.change_type == FileChangeType.DELETED:
                deleted_files.append(change.file_path)
            elif change.change_type == FileChangeType.RENAMED:
                renamed_files.append((change.old_file_path or "", change.file_path))
        
        diff_result = GitDiffResult(
            from_commit=from_commit,
            to_commit=to_commit,
            total_changes=len(file_changes),
            changes_by_type=changes_by_type,
            file_changes=file_changes,
            added_files=added_files,
            modified_files=modified_files,
            deleted_files=deleted_files,
            renamed_files=renamed_files
        )
        
        logger.info(f"Found {len(file_changes)} changes: "
                   f"{len(added_files)} added, {len(modified_files)} modified, "
                   f"{len(deleted_files)} deleted, {len(renamed_files)} renamed")
        
        return diff_result

    def _parse_diff_output(
        self, 
        diff_output: str, 
        file_extensions: Optional[List[str]] = None
    ) -> List[FileChange]:
        """
        Parse git diff --name-status output.
        
        Args:
            diff_output: Raw output from git diff --name-status
            file_extensions: Optional filter for file extensions
            
        Returns:
            List of FileChange objects
        """
        changes = []
        
        for line in diff_output.strip().split('\n'):
            if not line.strip():
                continue
                
            change = self._parse_diff_line(line.strip())
            if change and self._should_include_file(change.file_path, file_extensions):
                changes.append(change)
        
        return changes

    def _parse_diff_line(self, line: str) -> Optional[FileChange]:
        """
        Parse a single line from git diff --name-status output.
        
        Git diff --name-status output format:
        - A<tab>file_path (added)
        - M<tab>file_path (modified)
        - D<tab>file_path (deleted)
        - R<similarity><tab>old_path<tab>new_path (renamed)
        - C<similarity><tab>old_path<tab>new_path (copied)
        - T<tab>file_path (type changed)
        
        Args:
            line: Single line from diff output
            
        Returns:
            FileChange object or None if parsing failed
        """
        parts = line.split('\t')
        if len(parts) < 2:
            return None
        
        status = parts[0]
        
        # Handle simple cases (A, M, D, T)
        if status in ['A', 'M', 'D', 'T']:
            change_type_map = {
                'A': FileChangeType.ADDED,
                'M': FileChangeType.MODIFIED,
                'D': FileChangeType.DELETED,
                'T': FileChangeType.TYPE_CHANGED
            }
            
            return FileChange(
                change_type=change_type_map[status],
                file_path=parts[1]
            )
        
        # Handle renames and copies (R<similarity>, C<similarity>)
        if status.startswith('R') or status.startswith('C'):
            if len(parts) < 3:
                return None
            
            change_type = FileChangeType.RENAMED if status.startswith('R') else FileChangeType.COPIED
            
            # Extract similarity index
            similarity_match = re.match(r'[RC](\d+)', status)
            similarity_index = int(similarity_match.group(1)) if similarity_match else None
            
            return FileChange(
                change_type=change_type,
                file_path=parts[2],  # new path
                old_file_path=parts[1],  # old path
                similarity_index=similarity_index
            )
        
        logger.warning(f"Unknown diff status: {status}")
        return None

    def _should_include_file(self, file_path: str, file_extensions: Optional[List[str]]) -> bool:
        """
        Check if a file should be included based on extension filter.
        
        Args:
            file_path: Path to the file
            file_extensions: Optional list of extensions to include
            
        Returns:
            True if file should be included
        """
        if not file_extensions:
            return True
        
        file_ext = Path(file_path).suffix.lower()
        return file_ext in file_extensions

    def get_files_to_process(self, diff_result: GitDiffResult) -> Set[str]:
        """
        Get the set of files that need to be processed for incremental indexing.
        
        This includes:
        - Added files
        - Modified files  
        - New paths of renamed files
        
        Args:
            diff_result: Result from git diff analysis
            
        Returns:
            Set of file paths to process
        """
        if not diff_result:
            logger.warning("diff_result is None, returning empty set")
            return set()
            
        files_to_process = set()
        
        # Add all added and modified files - with null checks
        if diff_result.added_files:
            files_to_process.update(diff_result.added_files)
        if diff_result.modified_files:
            files_to_process.update(diff_result.modified_files)
        
        # Add new paths from renamed files - with null check
        if diff_result.renamed_files:
            for old_path, new_path in diff_result.renamed_files:
                files_to_process.add(new_path)
        
        return files_to_process

    def get_files_to_remove(self, diff_result: GitDiffResult) -> Set[str]:
        """
        Get the set of files that need to be removed from vector store.
        
        This includes:
        - Deleted files
        - Old paths of renamed files
        - Modified files (will be re-added with new content)
        
        Args:
            diff_result: Result from git diff analysis
            
        Returns:
            Set of file paths to remove from vector store
        """
        if not diff_result:
            logger.warning("diff_result is None, returning empty set")
            return set()
            
        files_to_remove = set()
        
        # Add deleted files - with null check
        if diff_result.deleted_files:
            files_to_remove.update(diff_result.deleted_files)
        
        # Add old paths from renamed files - with null check
        if diff_result.renamed_files:
            for old_path, new_path in diff_result.renamed_files:
                files_to_remove.add(old_path)
        
        # Add modified files (they'll be re-added with new content) - with null check
        if diff_result.modified_files:
            files_to_remove.update(diff_result.modified_files)
        
        return files_to_remove

    def validate_commits(self, repo_path: str, *commit_hashes: str) -> bool:
        """
        Validate that commit hashes exist in the repository.
        
        Args:
            repo_path: Path to the git repository
            commit_hashes: Commit hashes to validate
            
        Returns:
            True if all commits are valid
        """
        for commit_hash in commit_hashes:
            if not commit_hash:
                continue
                
            result = self.git_executor.execute_git_command(
                ["git", "cat-file", "-e", commit_hash],
                cwd=repo_path
            )
            
            if not result.success:
                logger.error(f"Invalid commit hash: {commit_hash}")
                return False
        
        return True