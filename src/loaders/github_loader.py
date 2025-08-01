"""
GitHub integration module for the Knowledge Graph Agent.

This module provides functionality for loading content from GitHub repositories
using LangChain BaseLoader interface.
"""

import base64
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import github
from github import Github, GithubException, Repository
from github.ContentFile import ContentFile
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config.settings import settings


class GitHubLoader(BaseLoader):
    """
    Load documents from GitHub repositories.

    This loader fetches files from GitHub repositories and converts them into
    LangChain Document objects with appropriate metadata.
    """

    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        branch: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        github_token: Optional[str] = None,
    ):
        """
        Initialize the GitHub loader.

        Args:
            repo_owner: Owner of the GitHub repository
            repo_name: Name of the GitHub repository
            branch: Branch to load from (default: repository default branch)
            file_extensions: List of file extensions to load (default: settings.github.file_extensions)
            github_token: GitHub token for authentication (default: settings.github.token)
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.branch = branch
        self.file_extensions = file_extensions or settings.github.file_extensions
        self.github_token = github_token or settings.github.token

        # Initialize GitHub client
        self.client = Github(self.github_token)

        # Repository information cache
        self.repo: Optional[Repository.Repository] = None
        self.default_branch: Optional[str] = None

        logger.debug(f"Initialized GitHub loader for {repo_owner}/{repo_name}")

    def _check_rate_limit_safely(self, threshold: int = 10) -> bool:
        """
        Safely check GitHub API rate limit with backward compatibility.
        
        Args:
            threshold: Minimum remaining requests before rate limiting
            
        Returns:
            True if rate limit check passed or couldn't be determined, False if rate limited
        """
        try:
            rate_limit = self.client.get_rate_limit()
            core_rate = getattr(rate_limit, 'core', None)
            if core_rate and hasattr(core_rate, 'remaining'):
                remaining = core_rate.remaining
                logger.debug(f"GitHub API rate limit remaining: {remaining}")
                return remaining >= threshold
            else:
                # If we can't determine the structure, assume it's OK
                logger.debug("Unable to determine rate limit structure, proceeding")
                return True
        except Exception as e:
            logger.debug(f"Error checking rate limit: {e}")
            # If we can't check, assume it's OK to proceed
            return True

    def _handle_rate_limit_safely(self) -> None:
        """
        Safely handle GitHub API rate limiting with backward compatibility.
        """
        try:
            rate_limit = self.client.get_rate_limit()
            core_rate = getattr(rate_limit, 'core', None)
            if core_rate and hasattr(core_rate, 'reset'):
                reset_time = core_rate.reset
                # Handle both timezone-aware and naive datetime objects
                try:
                    if reset_time.tzinfo is not None:
                        # reset_time is timezone-aware, use timezone-aware utcnow
                        from datetime import timezone
                        current_time = datetime.now(timezone.utc)
                    else:
                        # reset_time is naive, use naive utcnow
                        current_time = datetime.utcnow()
                    
                    sleep_time = max(0, (reset_time - current_time).total_seconds() + 1)
                    logger.warning(
                        f"GitHub API rate limit reached. Sleeping for {sleep_time} seconds"
                    )
                    time.sleep(sleep_time)
                except Exception as dt_error:
                    logger.warning(f"Datetime calculation error: {dt_error}, waiting 60 seconds")
                    time.sleep(60)
            else:
                # Fallback: wait a fixed time if rate limit structure is unknown
                logger.warning("Rate limit reached but cannot determine reset time, waiting 60 seconds")
                time.sleep(60)
        except Exception as rate_limit_error:
            logger.warning(f"Rate limit handling error: {rate_limit_error}, waiting 60 seconds")
            time.sleep(60)

    def _get_repository(self) -> Repository.Repository:
        """
        Get the GitHub repository.

        Returns:
            GitHub repository object

        Raises:
            ValueError: If the repository cannot be accessed
        """
        if self.repo is not None:
            return self.repo

        try:
            self.repo = self.client.get_repo(f"{self.repo_owner}/{self.repo_name}")
            self.default_branch = self.repo.default_branch
            logger.debug(
                f"Successfully connected to repository {self.repo_owner}/{self.repo_name}"
            )
            return self.repo
        except GithubException as e:
            error_message = f"Error accessing repository {self.repo_owner}/{self.repo_name}: {str(e)}"
            logger.error(error_message)
            raise ValueError(error_message)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(GithubException),
    )
    def _get_content(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Get the content of a file from GitHub.

        Args:
            file_path: Path to the file in the repository

        Returns:
            Tuple of (file_content, metadata)

        Raises:
            GithubException: If the file cannot be accessed
        """
        repo = self._get_repository()
        branch = self.branch or self.default_branch

        try:
            # Get file content with proper branch handling
            ref_param = branch or self.default_branch or "main"
            file_content = repo.get_contents(file_path, ref=ref_param)

            # Handle case where file_content is a list (directory)
            if isinstance(file_content, list):
                logger.warning(f"Path {file_path} is a directory, not a file")
                return "", {}

            # Decode content
            if file_content.encoding == "base64":
                content = base64.b64decode(file_content.content).decode(
                    "utf-8", errors="replace"
                )
            else:
                content = file_content.content

            # Get file metadata
            metadata = self._get_file_metadata(file_content, file_path)

            logger.debug(f"Successfully loaded content from {file_path}")

            # Handle rate limiting with safe backward compatibility
            if not self._check_rate_limit_safely():
                self._handle_rate_limit_safely()

            return content, metadata

        except GithubException as e:
            if e.status == 403 and "rate limit" in str(e).lower():
                self._handle_rate_limit_safely()
                # The retry decorator will retry this call
                raise
            elif e.status == 404:
                logger.warning(f"File {file_path} not found in repository")
                return "", {}
            else:
                logger.error(f"Error getting content for {file_path}: {str(e)}")
                raise

    def _get_file_metadata(
        self, file_content: ContentFile, file_path: str
    ) -> Dict[str, Any]:
        """
        Get metadata for a file.

        Args:
            file_content: GitHub content file object
            file_path: Path to the file in the repository

        Returns:
            Dictionary with file metadata
        """
        repo = self._get_repository()
        branch = self.branch or self.default_branch

        # Extract language from file extension
        _, file_extension = os.path.splitext(file_path)
        language = self._detect_language(file_extension)

        try:
            # Get last commit information with PyGithub 2.x compatible approach
            # Ensure branch is not None before passing to API
            sha_param = branch or self.default_branch or "main"
            commits_paginated = repo.get_commits(path=file_path, sha=sha_param)
            # Get only the first commit to avoid loading all commits
            last_commit = None
            try:
                last_commit = next(iter(commits_paginated))
            except StopIteration:
                pass

            commit_info = {}
            if last_commit:
                commit_info = {
                    "commit_sha": last_commit.sha,
                    "commit_author": (
                        last_commit.commit.author.name
                        if last_commit.commit.author
                        else "Unknown"
                    ),
                    "commit_date": (
                        last_commit.commit.author.date.isoformat()
                        if last_commit.commit.author
                        else ""
                    ),
                    "commit_message": last_commit.commit.message,
                }
        except GithubException:
            commit_info = {}

        # Combine all metadata
        metadata = {
            "repository": f"{self.repo_owner}/{self.repo_name}",
            "file_path": file_path,
            "branch": branch,
            "language": language,
            "size_bytes": file_content.size,
            "sha": file_content.sha,
            "url": file_content.html_url,
            "source": "github",
            **commit_info,
        }

        return metadata

    def _detect_language(self, file_extension: str) -> str:
        """
        Detect programming language from file extension.

        Args:
            file_extension: File extension (e.g., ".py", ".js")

        Returns:
            Language name
        """
        # Simple mapping of file extensions to languages
        extension_to_language = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "jsx",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".html": "html",
            ".css": "css",
            ".cs": "csharp",
            ".csproj": "csharp-project",
            ".php": "php",
            ".md": "markdown",
            ".json": "json",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".csv": "csv",
            ".sh": "shell",
            ".bash": "shell",
            ".txt": "text",
            ".config": "config",
            "dockerfile": "dockerfile",
        }

        return extension_to_language.get(file_extension.lower(), "unknown")

    def _should_load_file(self, file_path: str) -> bool:
        """
        Check if a file should be loaded based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            True if the file should be loaded, False otherwise
        """
        # Check if the path is a directory
        if file_path.endswith("/"):
            return False

        # Get file extension
        _, file_extension = os.path.splitext(file_path)

        # Check if file_extension is in the allowed extensions
        for ext in self.file_extensions:
            if ext.lower() == file_extension.lower() or (
                # Special case for files without extension like "dockerfile"
                ext.lower()
                == file_path.lower().split("/")[-1]
            ):
                return True

        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(GithubException),
    )
    def _get_file_paths(self) -> List[str]:
        """
        Get all file paths in the repository.

        Returns:
            List of file paths

        Raises:
            GithubException: If the repository cannot be accessed
        """
        repo = self._get_repository()
        branch = self.branch or self.default_branch

        logger.info(
            f"Getting file paths from {self.repo_owner}/{self.repo_name} ({branch})"
        )

        def traverse_directory(path: str) -> List[str]:
            try:
                # Get directory contents with proper branch handling
                ref_param = branch or self.default_branch or "main"
                contents = repo.get_contents(path, ref=ref_param)
                file_paths = []
                
                # Handle both single file and directory contents
                if not isinstance(contents, list):
                    contents = [contents]

                for content in contents:
                    if content.type == "dir":
                        file_paths.extend(traverse_directory(content.path))
                    elif content.type == "file":
                        if self._should_load_file(content.path):
                            file_paths.append(content.path)

                # Handle rate limiting with safe backward compatibility
                if not self._check_rate_limit_safely():
                    self._handle_rate_limit_safely()

                return file_paths

            except GithubException as e:
                if e.status == 403 and "rate limit" in str(e).lower():
                    self._handle_rate_limit_safely()
                    # The retry decorator will retry this call
                    raise
                else:
                    logger.error(f"Error traversing directory {path}: {str(e)}")
                    return []

        return traverse_directory("")

    def load(self) -> List[Document]:
        """
        Load documents from the GitHub repository.

        Returns:
            List of LangChain Document objects
        """
        try:
            # Get all file paths
            file_paths = self._get_file_paths()
            logger.info(
                f"Found {len(file_paths)} files to load from {self.repo_owner}/{self.repo_name}"
            )

            # Load each file
            documents = []
            for file_path in file_paths:
                try:
                    content, metadata = self._get_content(file_path)
                    if content:
                        doc = Document(page_content=content, metadata=metadata)
                        documents.append(doc)
                        logger.debug(f"Loaded document from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")

            logger.info(
                f"Loaded {len(documents)} documents from {self.repo_owner}/{self.repo_name}"
            )
            return documents

        except Exception as e:
            logger.error(
                f"Error loading documents from {self.repo_owner}/{self.repo_name}: {str(e)}"
            )
            raise


class MultiRepositoryGitHubLoader(BaseLoader):
    """
    Load documents from multiple GitHub repositories.

    This loader coordinates loading documents from multiple GitHub repositories
    specified in the application settings.
    """

    def __init__(
        self,
        repositories: Optional[List[Dict[str, str]]] = None,
        github_token: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
    ):
        """
        Initialize the multi-repository GitHub loader.

        Args:
            repositories: List of repository configurations (default: settings.repositories)
            github_token: GitHub token for authentication (default: settings.github.token)
            file_extensions: List of file extensions to load (default: settings.github.file_extensions)
        """
        self.repositories = repositories or [
            {"owner": repo.owner, "repo": repo.repo, "branch": repo.branch}
            for repo in settings.repositories
        ]
        self.github_token = github_token or settings.github.token
        self.file_extensions = file_extensions or settings.github.file_extensions

        logger.debug(
            f"Initialized multi-repository GitHub loader with {len(self.repositories)} repositories"
        )

    def load(self) -> List[Document]:
        """
        Load documents from multiple GitHub repositories.

        Returns:
            List of LangChain Document objects
        """
        documents = []

        # Load documents from each repository
        for repo_config in self.repositories:
            owner = repo_config["owner"]
            repo = repo_config["repo"]
            branch = repo_config.get("branch")

            logger.info(f"Loading documents from {owner}/{repo}")

            try:
                # Create and use a single-repository loader
                loader = GitHubLoader(
                    repo_owner=owner,
                    repo_name=repo,
                    branch=branch,
                    github_token=self.github_token,
                    file_extensions=self.file_extensions,
                )

                repo_documents = loader.load()
                documents.extend(repo_documents)

                logger.info(
                    f"Loaded {len(repo_documents)} documents from {owner}/{repo}"
                )

            except Exception as e:
                logger.error(f"Error loading documents from {owner}/{repo}: {str(e)}")

        logger.info(
            f"Loaded {len(documents)} documents from {len(self.repositories)} repositories"
        )
        return documents
