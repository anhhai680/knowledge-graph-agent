"""
Git Metadata Extractor for the Knowledge Graph Agent.

This module provides functionality for extracting rich metadata from Git repositories
using Git commands, including commit information, file history, and repository statistics.
"""

from typing import Dict, Any, List, Optional
from loguru import logger

from .git_command_executor import GitCommandExecutor


class GitMetadataExtractor:
    """
    Extract rich metadata using Git commands.
    
    This class provides functionality to extract comprehensive metadata
    from Git repositories including commit information, file history,
    and repository statistics.
    """

    def __init__(self, git_executor: GitCommandExecutor):
        """
        Initialize Git metadata extractor.
        
        Args:
            git_executor: Git command executor instance
        """
        self.git_executor = git_executor
        logger.debug("Initialized Git metadata extractor")

    def get_file_commit_info(self, file_path: str, repo_path: str) -> Dict[str, Any]:
        """
        Get latest commit information for file.
        
        Args:
            file_path: Relative path to file within repository
            repo_path: Path to local repository
            
        Returns:
            Dictionary containing commit information
        """
        try:
            return self.git_executor.get_file_commit_info(file_path, repo_path)
        except Exception as e:
            logger.error(f"Error getting file commit info for {file_path}: {e}")
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
            return self.git_executor.get_file_history(file_path, repo_path, limit)
        except Exception as e:
            logger.error(f"Error getting file history for {file_path}: {e}")
            return []

    def get_repository_metadata(self, repo_path: str) -> Dict[str, Any]:
        """
        Get overall repository metadata.
        
        Args:
            repo_path: Path to local repository
            
        Returns:
            Dictionary containing repository metadata
        """
        try:
            metadata = {}
            
            # Get basic repository stats
            stats = self.git_executor.get_repository_stats(repo_path)
            metadata.update(stats)
            
            # Get branch information
            branch_info = self.get_branch_info(repo_path)
            metadata.update(branch_info)
            
            # Get repository configuration
            config_info = self._get_repository_config(repo_path)
            metadata.update(config_info)
            
            # Get tag information
            tag_info = self._get_tag_info(repo_path)
            metadata.update(tag_info)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting repository metadata for {repo_path}: {e}")
            return {}

    def get_branch_info(self, repo_path: str) -> Dict[str, str]:
        """
        Get current branch and remote information.
        
        Args:
            repo_path: Path to local repository
            
        Returns:
            Dictionary containing branch information
        """
        try:
            return self.git_executor.get_branch_info(repo_path)
        except Exception as e:
            logger.error(f"Error getting branch info for {repo_path}: {e}")
            return {}

    def extract_author_info(self, commit_info: str) -> Dict[str, str]:
        """
        Parse author information from git log output.
        
        Args:
            commit_info: Raw git log output
            
        Returns:
            Dictionary containing parsed author information
        """
        try:
            # Parse format: "name <email>"
            author_info = {}
            
            if "<" in commit_info and ">" in commit_info:
                # Extract name and email
                parts = commit_info.split("<")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    email = parts[1].split(">")[0].strip()
                    
                    author_info["name"] = name
                    author_info["email"] = email
            else:
                # Just use the whole string as name
                author_info["name"] = commit_info.strip()
                author_info["email"] = ""
            
            return author_info
            
        except Exception as e:
            logger.debug(f"Error parsing author info '{commit_info}': {e}")
            return {"name": commit_info, "email": ""}

    def get_file_contributors(self, file_path: str, repo_path: str) -> List[Dict[str, Any]]:
        """
        Get list of contributors for a specific file.
        
        Args:
            file_path: Relative path to file within repository
            repo_path: Path to local repository
            
        Returns:
            List of contributor information
        """
        try:
            command = [
                "shortlog",
                "-sne",
                "--",
                file_path
            ]
            
            result = self.git_executor.execute_git_command(command, repo_path)
            
            if not result.success or not result.stdout.strip():
                return []
            
            contributors = []
            for line in result.stdout.strip().split("\n"):
                # Format: "     5  John Doe <john@example.com>"
                line = line.strip()
                if line:
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        commit_count = int(parts[0].strip())
                        author_info = self.extract_author_info(parts[1])
                        contributors.append({
                            "commit_count": commit_count,
                            "name": author_info["name"],
                            "email": author_info["email"]
                        })
            
            return contributors
            
        except Exception as e:
            logger.error(f"Error getting contributors for {file_path}: {e}")
            return []

    def get_file_blame_info(self, file_path: str, repo_path: str) -> Dict[str, Any]:
        """
        Get blame/annotation information for a file.
        
        Args:
            file_path: Relative path to file within repository
            repo_path: Path to local repository
            
        Returns:
            Dictionary containing blame information
        """
        try:
            command = [
                "blame",
                "--line-porcelain",
                file_path
            ]
            
            result = self.git_executor.execute_git_command(command, repo_path)
            
            if not result.success:
                return {}
            
            # Parse blame output to get statistics
            lines = result.stdout.split("\n")
            authors = {}
            total_lines = 0
            
            for line in lines:
                if line.startswith("author "):
                    author = line[7:]  # Remove "author " prefix
                    authors[author] = authors.get(author, 0) + 1
                    total_lines += 1
            
            return {
                "total_lines": total_lines,
                "authors": authors,
                "primary_author": max(authors.items(), key=lambda x: x[1])[0] if authors else None
            }
            
        except Exception as e:
            logger.debug(f"Error getting blame info for {file_path}: {e}")
            return {}

    def _get_repository_config(self, repo_path: str) -> Dict[str, Any]:
        """
        Get repository configuration information.
        
        Args:
            repo_path: Path to local repository
            
        Returns:
            Dictionary containing configuration information
        """
        try:
            config_info = {}
            
            # Get user name and email
            name_result = self.git_executor.execute_git_command(
                ["config", "user.name"], repo_path
            )
            if name_result.success:
                config_info["user_name"] = name_result.stdout.strip()
            
            email_result = self.git_executor.execute_git_command(
                ["config", "user.email"], repo_path
            )
            if email_result.success:
                config_info["user_email"] = email_result.stdout.strip()
            
            # Get core settings
            autocrlf_result = self.git_executor.execute_git_command(
                ["config", "core.autocrlf"], repo_path
            )
            if autocrlf_result.success:
                config_info["autocrlf"] = autocrlf_result.stdout.strip()
            
            return config_info
            
        except Exception as e:
            logger.debug(f"Error getting repository config for {repo_path}: {e}")
            return {}

    def _get_tag_info(self, repo_path: str) -> Dict[str, Any]:
        """
        Get tag information from repository.
        
        Args:
            repo_path: Path to local repository
            
        Returns:
            Dictionary containing tag information
        """
        try:
            # Get all tags
            tags_result = self.git_executor.execute_git_command(
                ["tag", "--sort=-version:refname"], repo_path
            )
            
            tag_info = {}
            
            if tags_result.success and tags_result.stdout.strip():
                tags = tags_result.stdout.strip().split("\n")
                tag_info["total_tags"] = len(tags)
                tag_info["latest_tag"] = tags[0] if tags else None
                tag_info["all_tags"] = tags[:10]  # Limit to first 10 tags
            else:
                tag_info["total_tags"] = 0
                tag_info["latest_tag"] = None
                tag_info["all_tags"] = []
            
            return tag_info
            
        except Exception as e:
            logger.debug(f"Error getting tag info for {repo_path}: {e}")
            return {"total_tags": 0, "latest_tag": None, "all_tags": []}

    def get_repository_languages(self, repo_path: str) -> Dict[str, int]:
        """
        Get programming languages used in repository by file count.
        
        Args:
            repo_path: Path to local repository
            
        Returns:
            Dictionary mapping language to file count
        """
        try:
            # Get all tracked files
            files_result = self.git_executor.execute_git_command(
                ["ls-files"], repo_path
            )
            
            if not files_result.success:
                return {}
            
            languages = {}
            extension_to_language = {
                '.py': 'Python',
                '.js': 'JavaScript',
                '.ts': 'TypeScript',
                '.jsx': 'JavaScript',
                '.tsx': 'TypeScript',
                '.cs': 'C#',
                '.java': 'Java',
                '.cpp': 'C++',
                '.c': 'C',
                '.h': 'C/C++',
                '.hpp': 'C++',
                '.php': 'PHP',
                '.rb': 'Ruby',
                '.go': 'Go',
                '.rs': 'Rust',
                '.swift': 'Swift',
                '.kt': 'Kotlin',
                '.scala': 'Scala',
                '.html': 'HTML',
                '.css': 'CSS',
                '.scss': 'SCSS',
                '.sass': 'Sass',
                '.less': 'Less',
                '.json': 'JSON',
                '.xml': 'XML',
                '.yaml': 'YAML',
                '.yml': 'YAML',
                '.md': 'Markdown',
                '.sql': 'SQL',
                '.sh': 'Shell',
                '.bash': 'Shell',
                '.ps1': 'PowerShell',
                '.r': 'R',
                '.m': 'Objective-C',
                '.mm': 'Objective-C++',
                '.pl': 'Perl',
                '.lua': 'Lua',
                '.vim': 'Vim script',
                '.dockerfile': 'Dockerfile'
            }
            
            for file_path in files_result.stdout.strip().split("\n"):
                if file_path.strip():
                    # Get file extension
                    if "." in file_path:
                        extension = "." + file_path.split(".")[-1].lower()
                    else:
                        # Handle files without extensions
                        filename = file_path.split("/")[-1].lower()
                        if filename in ['dockerfile', 'makefile']:
                            extension = f'.{filename}'
                        else:
                            continue
                    
                    language = extension_to_language.get(extension, 'Other')
                    languages[language] = languages.get(language, 0) + 1
            
            return languages
            
        except Exception as e:
            logger.error(f"Error getting repository languages for {repo_path}: {e}")
            return {}

    def get_comprehensive_file_metadata(
        self, 
        file_path: str, 
        repo_path: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive metadata for a single file.
        
        Args:
            file_path: Relative path to file within repository
            repo_path: Path to local repository
            
        Returns:
            Dictionary containing comprehensive file metadata
        """
        try:
            metadata = {}
            
            # Basic commit info
            commit_info = self.get_file_commit_info(file_path, repo_path)
            metadata.update(commit_info)
            
            # File contributors
            contributors = self.get_file_contributors(file_path, repo_path)
            if contributors:
                metadata["contributors"] = contributors
                metadata["total_contributors"] = len(contributors)
                metadata["primary_contributor"] = contributors[0]["name"] if contributors else None
            
            # Recent history
            history = self.get_file_history(file_path, repo_path, limit=3)
            if history:
                metadata["recent_history"] = history
                metadata["total_commits"] = len(history)
            
            # Blame information (for smaller files)
            try:
                blame_info = self.get_file_blame_info(file_path, repo_path)
                if blame_info:
                    metadata["blame_info"] = blame_info
            except Exception:
                # Skip blame info if it fails (file might be too large)
                pass
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting comprehensive metadata for {file_path}: {e}")
            return {}
