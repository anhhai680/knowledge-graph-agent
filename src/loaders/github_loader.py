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
            # Get file content
            file_content = repo.get_contents(file_path, ref=branch)

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

            # Handle rate limiting
            if self.client.get_rate_limit().core.remaining < 10:
                reset_time = self.client.get_rate_limit().core.reset
                sleep_time = max(
                    0, (reset_time - datetime.utcnow()).total_seconds() + 1
                )
                logger.warning(
                    f"GitHub API rate limit almost reached. Sleeping for {sleep_time} seconds"
                )
                time.sleep(sleep_time)

            return content, metadata

        except GithubException as e:
            if e.status == 403 and "rate limit" in str(e).lower():
                reset_time = self.client.get_rate_limit().core.reset
                sleep_time = max(
                    0, (reset_time - datetime.utcnow()).total_seconds() + 1
                )
                logger.warning(
                    f"GitHub API rate limit reached. Sleeping for {sleep_time} seconds"
                )
                time.sleep(sleep_time)
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
            # Get last commit information
            commits = list(repo.get_commits(path=file_path, sha=branch, max_pages=1))
            last_commit = commits[0] if commits else None

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
                contents = repo.get_contents(path, ref=branch)
                file_paths = []

                for content in contents:
                    if content.type == "dir":
                        file_paths.extend(traverse_directory(content.path))
                    elif content.type == "file":
                        if self._should_load_file(content.path):
                            file_paths.append(content.path)

                # Handle rate limiting
                if self.client.get_rate_limit().core.remaining < 10:
                    reset_time = self.client.get_rate_limit().core.reset
                    sleep_time = max(
                        0, (reset_time - datetime.utcnow()).total_seconds() + 1
                    )
                    logger.warning(
                        f"GitHub API rate limit almost reached. Sleeping for {sleep_time} seconds"
                    )
                    time.sleep(sleep_time)

                return file_paths

            except GithubException as e:
                if e.status == 403 and "rate limit" in str(e).lower():
                    reset_time = self.client.get_rate_limit().core.reset
                    sleep_time = max(
                        0, (reset_time - datetime.utcnow()).total_seconds() + 1
                    )
                    logger.warning(
                        f"GitHub API rate limit reached. Sleeping for {sleep_time} seconds"
                    )
                    time.sleep(sleep_time)
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
