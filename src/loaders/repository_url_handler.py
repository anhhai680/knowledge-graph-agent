"""
Repository URL Handler for the Knowledge Graph Agent.

This module provides functionality for handling different repository URL formats,
authentication, and URL normalization for Git operations.
"""

import re
from typing import Tuple, Optional
from urllib.parse import urlparse, urlunparse
from loguru import logger


class RepositoryUrlHandler:
    """
    Handle different repository URL formats and authentication.
    
    This class provides functionality to normalize repository URLs,
    parse repository information, and handle authentication.
    """

    # URL patterns for different formats
    GITHUB_HTTPS_PATTERN = re.compile(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$")
    GITHUB_SSH_PATTERN = re.compile(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?/?$")
    GITHUB_GIT_PATTERN = re.compile(r"git://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$")

    def __init__(self):
        """Initialize repository URL handler."""
        logger.debug("Initialized repository URL handler")

    def normalize_repo_url(self, url: str, token: Optional[str] = None) -> str:
        """
        Convert various URL formats to authenticated clone URL.
        
        Args:
            url: Repository URL in any supported format
            token: GitHub token for authentication (optional)
            
        Returns:
            Normalized HTTPS URL with authentication if token provided
            
        Raises:
            ValueError: If URL format is not recognized
        """
        try:
            # Parse repository info first
            owner, repo = self.parse_repo_info(url)
            
            # Build normalized URL
            normalized_url = self.build_clone_url(owner, repo, token)
            
            logger.debug(f"Normalized URL: {url} -> {normalized_url}")
            return normalized_url
            
        except Exception as e:
            logger.error(f"Failed to normalize URL {url}: {e}")
            raise ValueError(f"Invalid repository URL format: {url}")

    def parse_repo_info(self, url: str) -> Tuple[str, str]:
        """
        Extract owner and repo name from URL.
        
        Args:
            url: Repository URL in any supported format
            
        Returns:
            Tuple of (owner, repo_name)
            
        Raises:
            ValueError: If URL format is not recognized
        """
        url = url.strip()
        
        # Try HTTPS format
        match = self.GITHUB_HTTPS_PATTERN.match(url)
        if match:
            return match.group(1), match.group(2)
        
        # Try SSH format
        match = self.GITHUB_SSH_PATTERN.match(url)
        if match:
            return match.group(1), match.group(2)
        
        # Try Git protocol format
        match = self.GITHUB_GIT_PATTERN.match(url)
        if match:
            return match.group(1), match.group(2)
        
        # Try to parse as owner/repo format
        if "/" in url and not url.startswith(("http", "git@", "git://")):
            parts = url.strip("/").split("/")
            if len(parts) >= 2:
                return parts[-2], parts[-1]
        
        raise ValueError(f"Unable to parse repository information from URL: {url}")

    def build_clone_url(
        self, 
        owner: str, 
        repo: str, 
        token: Optional[str] = None
    ) -> str:
        """
        Build proper HTTPS clone URL with token authentication.
        
        Args:
            owner: Repository owner/organization
            repo: Repository name
            token: GitHub token for authentication (optional)
            
        Returns:
            HTTPS clone URL with authentication if token provided
        """
        # Ensure repo name doesn't have .git suffix for URL building
        repo_clean = repo.rstrip(".git")
        
        if token:
            # Authenticated URL format
            return f"https://{token}@github.com/{owner}/{repo_clean}.git"
        else:
            # Public URL format
            return f"https://github.com/{owner}/{repo_clean}.git"

    def validate_repo_url(self, url: str) -> bool:
        """
        Validate repository URL format.
        
        Args:
            url: Repository URL to validate
            
        Returns:
            True if URL format is valid, False otherwise
        """
        try:
            # Try to parse the URL
            self.parse_repo_info(url)
            return True
        except ValueError:
            return False

    def is_github_url(self, url: str) -> bool:
        """
        Check if URL is a GitHub repository URL.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is a GitHub repository URL
        """
        url = url.lower().strip()
        
        return any([
            "github.com" in url,
            url.startswith("git@github.com:"),
            url.startswith("git://github.com/")
        ])

    def extract_repo_name_from_url(self, url: str) -> str:
        """
        Extract just the repository name from URL.
        
        Args:
            url: Repository URL
            
        Returns:
            Repository name without owner
            
        Raises:
            ValueError: If URL format is not recognized
        """
        try:
            _, repo = self.parse_repo_info(url)
            return repo.rstrip(".git")
        except Exception as e:
            raise ValueError(f"Unable to extract repository name from URL: {url}")

    def extract_owner_from_url(self, url: str) -> str:
        """
        Extract just the owner/organization from URL.
        
        Args:
            url: Repository URL
            
        Returns:
            Repository owner/organization
            
        Raises:
            ValueError: If URL format is not recognized
        """
        try:
            owner, _ = self.parse_repo_info(url)
            return owner
        except Exception as e:
            raise ValueError(f"Unable to extract owner from URL: {url}")

    def build_github_api_url(self, owner: str, repo: str) -> str:
        """
        Build GitHub API URL for repository.
        
        Args:
            owner: Repository owner/organization
            repo: Repository name
            
        Returns:
            GitHub API URL for the repository
        """
        repo_clean = repo.rstrip(".git")
        return f"https://api.github.com/repos/{owner}/{repo_clean}"

    def build_github_web_url(self, owner: str, repo: str) -> str:
        """
        Build GitHub web URL for repository.
        
        Args:
            owner: Repository owner/organization
            repo: Repository name
            
        Returns:
            GitHub web URL for the repository
        """
        repo_clean = repo.rstrip(".git")
        return f"https://github.com/{owner}/{repo_clean}"

    def convert_to_ssh_url(self, owner: str, repo: str) -> str:
        """
        Build SSH clone URL for repository.
        
        Args:
            owner: Repository owner/organization
            repo: Repository name
            
        Returns:
            SSH clone URL
        """
        repo_clean = repo.rstrip(".git")
        return f"git@github.com:{owner}/{repo_clean}.git"

    def mask_token_in_url(self, url: str) -> str:
        """
        Mask authentication token in URL for logging.
        
        Args:
            url: URL that may contain authentication token
            
        Returns:
            URL with token masked for safe logging
        """
        try:
            # Pattern to match GitHub URLs with tokens
            token_pattern = re.compile(r"(https://)([^@]+)(@github\.com/.+)")
            match = token_pattern.match(url)
            
            if match:
                # Replace token with masked version
                token = match.group(2)
                if len(token) > 8:
                    masked_token = token[:4] + "****" + token[-4:]
                else:
                    masked_token = "****"
                return f"{match.group(1)}{masked_token}{match.group(3)}"
            
            return url
            
        except Exception:
            # If anything goes wrong, return original URL
            return url

    def get_url_info(self, url: str) -> dict:
        """
        Get comprehensive information about a repository URL.
        
        Args:
            url: Repository URL
            
        Returns:
            Dictionary containing URL information
        """
        try:
            info = {
                "original_url": url,
                "is_valid": self.validate_repo_url(url),
                "is_github": self.is_github_url(url)
            }
            
            if info["is_valid"]:
                owner, repo = self.parse_repo_info(url)
                info.update({
                    "owner": owner,
                    "repo": repo,
                    "https_url": self.build_clone_url(owner, repo),
                    "ssh_url": self.convert_to_ssh_url(owner, repo),
                    "web_url": self.build_github_web_url(owner, repo),
                    "api_url": self.build_github_api_url(owner, repo)
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting URL info for {url}: {e}")
            return {
                "original_url": url,
                "is_valid": False,
                "is_github": False,
                "error": str(e)
            }
