"""
Enhanced GitHub Loader for the Knowledge Graph Agent.

This module provides functionality for loading content from GitHub repositories
using Git operations instead of API calls, eliminating rate limiting and
improving performance.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from loguru import logger

from src.config.settings import settings
from .git_repository_manager import GitRepositoryManager
from .git_command_executor import GitCommandExecutor
from .file_system_processor import FileSystemProcessor
from .git_metadata_extractor import GitMetadataExtractor
from .repository_url_handler import RepositoryUrlHandler


class EnhancedGitHubLoader(BaseLoader):
    """
    Load documents from GitHub repositories using Git operations instead of API.
    
    This loader eliminates rate limiting issues and improves performance by:
    - Cloning repositories locally instead of using API calls
    - Using file system operations for content access
    - Extracting metadata using Git commands
    - Providing configurable caching and cleanup
    """

    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        branch: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        github_token: Optional[str] = None,
        temp_repo_path: Optional[str] = None,
        force_fresh_clone: bool = False,
        cleanup_after_processing: bool = False
    ):
        """
        Initialize Git-based GitHub loader.
        
        Args:
            repo_owner: Owner of the GitHub repository
            repo_name: Name of the GitHub repository
            branch: Branch to load from (default: main)
            file_extensions: List of file extensions to load
            github_token: GitHub token for authentication
            temp_repo_path: Custom path for temporary repositories
            force_fresh_clone: Force a fresh clone even if repo exists
            cleanup_after_processing: Remove local repo after processing
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.branch = branch or "main"
        self.file_extensions = file_extensions or settings.github.file_extensions
        self.github_token = github_token or settings.github.token
        self.force_fresh_clone = force_fresh_clone
        self.cleanup_after_processing = cleanup_after_processing
        
        # Initialize components
        self.repo_manager = GitRepositoryManager(
            temp_repo_base_path=temp_repo_path or "temp_repo"
        )
        self.git_executor = GitCommandExecutor()
        self.file_processor = FileSystemProcessor(self.file_extensions)
        self.metadata_extractor = GitMetadataExtractor(self.git_executor)
        self.url_handler = RepositoryUrlHandler()
        
        # Repository information
        self.local_repo_path: Optional[str] = None
        self.repo_url: Optional[str] = None
        
        logger.info(f"Initialized Enhanced GitHub loader for {repo_owner}/{repo_name}")

    def load(self) -> List[Document]:
        """
        Load documents from GitHub repository using Git operations.
        
        Returns:
            List of LangChain Document objects
        """
        try:
            # Prepare local repository
            repo_path = self._prepare_local_repository()
            if not repo_path:
                logger.error("Failed to prepare local repository")
                return []
            
            # Discover files using file system operations
            file_paths = self._discover_files(repo_path)
            if not file_paths:
                logger.warning(f"No files found to process in {self.repo_owner}/{self.repo_name}")
                return []
            
            logger.info(f"Found {len(file_paths)} files to process")
            
            # Load documents from files
            documents = []
            for file_path in file_paths:
                try:
                    content, metadata = self._load_file_content(file_path, repo_path)
                    if content:
                        doc = Document(page_content=content, metadata=metadata)
                        documents.append(doc)
                        logger.debug(f"Loaded document from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
            
            logger.info(f"Successfully loaded {len(documents)} documents from {self.repo_owner}/{self.repo_name}")
            
            # Cleanup if requested
            self._cleanup_if_requested(repo_path)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from {self.repo_owner}/{self.repo_name}: {e}")
            raise

    def _prepare_local_repository(self) -> Optional[str]:
        """
        Clone or update local repository.
        
        Returns:
            Path to local repository, or None if failed
        """
        try:
            # Build repository URL
            self.repo_url = self.url_handler.build_clone_url(
                self.repo_owner, 
                self.repo_name, 
                self.github_token
            )
            
            # Get local path
            local_path = self.repo_manager.get_local_repo_path(
                self.repo_owner, 
                self.repo_name
            )
            
            # Force fresh clone if requested
            if self.force_fresh_clone and Path(local_path).exists():
                logger.info(f"Force fresh clone requested, removing existing repository at {local_path}")
                self.repo_manager.cleanup_repository(local_path)
            
            # Clone or pull repository
            success = self.repo_manager.clone_or_pull_repository(
                self.repo_url, 
                local_path, 
                self.branch
            )
            
            if success:
                self.local_repo_path = local_path
                logger.info(f"Repository prepared at {local_path}")
                return local_path
            else:
                logger.error(f"Failed to prepare repository at {local_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error preparing local repository: {e}")
            return None

    def _discover_files(self, repo_path: str) -> List[str]:
        """
        Discover files using file system operations.
        
        Args:
            repo_path: Path to local repository
            
        Returns:
            List of relative file paths to process
        """
        try:
            # Scan directory tree for matching files
            file_paths = self.file_processor.scan_directory_tree(repo_path)
            
            # Filter by size constraints
            filtered_paths = self.file_processor.filter_files_by_size(file_paths, repo_path)
            
            logger.info(f"Discovered {len(filtered_paths)} processable files out of {len(file_paths)} total files")
            return filtered_paths
            
        except Exception as e:
            logger.error(f"Error discovering files in {repo_path}: {e}")
            return []

    def _load_file_content(self, file_path: str, repo_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Load file content from local file system.
        
        Args:
            file_path: Relative path to file within repository
            repo_path: Path to local repository
            
        Returns:
            Tuple of (content, metadata)
        """
        try:
            # Full path to file
            full_file_path = Path(repo_path) / file_path
            
            # Read file content
            content = self.file_processor.read_file_content(str(full_file_path))
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, repo_path, file_path)
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Error loading file content for {file_path}: {e}")
            return "", {}

    def _extract_metadata(
        self, 
        file_path: str, 
        repo_path: str, 
        relative_path: str
    ) -> Dict[str, Any]:
        """
        Extract comprehensive metadata using Git and file system.
        
        Args:
            file_path: Relative path to file within repository
            repo_path: Path to local repository
            relative_path: Relative path for metadata
            
        Returns:
            Dictionary containing comprehensive metadata
        """
        try:
            metadata = {}
            
            # Basic repository information
            metadata.update({
                "repository": f"{self.repo_owner}/{self.repo_name}",
                "file_path": relative_path,
                "branch": self.branch,
                "source": "github_git",
                "loader_type": "enhanced_git"
            })
            
            # File system metadata
            full_file_path = Path(repo_path) / file_path
            file_stats = self.file_processor.get_file_stats(str(full_file_path))
            metadata.update(file_stats)
            
            # Language detection
            file_extension = Path(file_path).suffix
            metadata["language"] = self._detect_language(file_extension)
            
            # Git metadata
            git_metadata = self.metadata_extractor.get_comprehensive_file_metadata(
                file_path, repo_path
            )
            metadata.update(git_metadata)
            
            # Repository URLs
            metadata.update({
                "repository_url": self.url_handler.build_github_web_url(
                    self.repo_owner, self.repo_name
                ),
                "file_url": f"https://github.com/{self.repo_owner}/{self.repo_name}/blob/{self.branch}/{file_path}"
            })
            
            # Processing timestamp
            metadata["processed_at"] = datetime.utcnow().isoformat()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata for {file_path}: {e}")
            return {
                "repository": f"{self.repo_owner}/{self.repo_name}",
                "file_path": relative_path,
                "branch": self.branch,
                "source": "github_git",
                "error": str(e)
            }

    def _detect_language(self, file_extension: str) -> str:
        """
        Detect programming language from file extension.
        
        Args:
            file_extension: File extension including the dot
            
        Returns:
            Detected language name
        """
        extension_to_language = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.cs': 'csharp',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.sql': 'sql',
            '.sh': 'shell',
            '.bash': 'shell',
            '.ps1': 'powershell',
            '.r': 'r',
            '.m': 'objective-c',
            '.pl': 'perl',
            '.lua': 'lua',
            '.vim': 'vim',
            '.dockerfile': 'dockerfile',
            '.csproj': 'xml',
            '.config': 'xml',
            '.txt': 'text'
        }
        
        return extension_to_language.get(file_extension.lower(), 'unknown')

    def _cleanup_if_requested(self, repo_path: str) -> None:
        """
        Clean up local repository if configured.
        
        Args:
            repo_path: Path to local repository
        """
        try:
            if self.cleanup_after_processing:
                logger.info(f"Cleaning up repository at {repo_path}")
                self.repo_manager.cleanup_repository(repo_path)
            else:
                logger.debug(f"Repository retained at {repo_path} for future use")
                
        except Exception as e:
            logger.error(f"Error during cleanup of {repo_path}: {e}")

    def get_repository_info(self) -> Dict[str, Any]:
        """
        Get information about the repository.
        
        Returns:
            Dictionary containing repository information
        """
        try:
            if not self.local_repo_path:
                return {}
            
            # Get repository metadata
            repo_metadata = self.metadata_extractor.get_repository_metadata(self.local_repo_path)
            
            # Get directory statistics
            dir_stats = self.file_processor.get_directory_stats(self.local_repo_path)
            
            # Get repository languages
            languages = self.metadata_extractor.get_repository_languages(self.local_repo_path)
            
            return {
                "repository": f"{self.repo_owner}/{self.repo_name}",
                "branch": self.branch,
                "local_path": self.local_repo_path,
                "repository_metadata": repo_metadata,
                "directory_stats": dir_stats,
                "languages": languages,
                "loader_type": "enhanced_git"
            }
            
        except Exception as e:
            logger.error(f"Error getting repository info: {e}")
            return {"error": str(e)}

    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate that all required components are working.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "git_available": False,
            "temp_directory": False,
            "url_valid": False,
            "token_configured": False,
            "overall_status": "failed"
        }
        
        try:
            # Check Git installation
            validation_results["git_available"] = self.git_executor.validate_git_installation()
            
            # Check temp directory
            try:
                self.repo_manager.ensure_temp_repo_directory()
                validation_results["temp_directory"] = True
            except Exception:
                validation_results["temp_directory"] = False
            
            # Check URL format
            try:
                url_info = self.url_handler.get_url_info(
                    f"https://github.com/{self.repo_owner}/{self.repo_name}"
                )
                validation_results["url_valid"] = url_info["is_valid"]
            except Exception:
                validation_results["url_valid"] = False
            
            # Check token configuration
            validation_results["token_configured"] = bool(self.github_token and self.github_token != "your_github_token")
            
            # Overall status
            if all([
                validation_results["git_available"],
                validation_results["temp_directory"],
                validation_results["url_valid"]
            ]):
                validation_results["overall_status"] = "ready"
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            validation_results["error"] = str(e)
            return validation_results
