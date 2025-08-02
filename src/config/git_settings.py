"""
Git Settings Configuration for the Knowledge Graph Agent.

This module provides configuration settings for Git-based repository operations.
"""

from pydantic import BaseModel, Field
from typing import Optional


class GitSettings(BaseModel):
    """Configuration for Git-based repository operations."""
    
    # Repository Management
    temp_repo_base_path: str = Field(
        default="temp_repo", 
        description="Base directory for storing temporary repositories"
    )
    force_fresh_clone: bool = Field(
        default=False, 
        description="Always perform fresh clone instead of pull"
    )
    cleanup_after_processing: bool = Field(
        default=False, 
        description="Remove local repository after processing"
    )
    
    # Performance Settings
    git_timeout_seconds: int = Field(
        default=300, 
        description="Timeout for Git operations in seconds"
    )
    max_repo_size_mb: int = Field(
        default=500, 
        description="Maximum repository size in MB to process"
    )
    max_file_size_mb: int = Field(
        default=10, 
        description="Maximum individual file size in MB to process"
    )
    
    # Parallel Processing
    max_concurrent_repos: int = Field(
        default=2, 
        description="Maximum number of repositories to process concurrently"
    )
    max_concurrent_files: int = Field(
        default=10, 
        description="Maximum number of files to process concurrently per repository"
    )
    
    # Security Settings
    allow_private_repos: bool = Field(
        default=True, 
        description="Allow processing of private repositories"
    )
    verify_ssl: bool = Field(
        default=True, 
        description="Verify SSL certificates for Git operations"
    )
    shallow_clone: bool = Field(
        default=True, 
        description="Use shallow clone for better performance"
    )
    
    # Maintenance Settings
    cleanup_old_repos_days: int = Field(
        default=7, 
        description="Remove cached repositories older than this many days"
    )
    max_cached_repos: int = Field(
        default=50, 
        description="Maximum number of repositories to keep cached"
    )
    auto_cleanup_enabled: bool = Field(
        default=True, 
        description="Automatically cleanup old repositories"
    )
    
    # Git Configuration
    git_user_name: Optional[str] = Field(
        default=None, 
        description="Git user name for local operations"
    )
    git_user_email: Optional[str] = Field(
        default=None, 
        description="Git user email for local operations"
    )
    
    # Retry Settings
    max_retry_attempts: int = Field(
        default=3, 
        description="Maximum number of retry attempts for failed operations"
    )
    retry_delay_seconds: int = Field(
        default=1, 
        description="Base delay between retry attempts"
    )
    
    # Monitoring and Logging
    enable_performance_tracking: bool = Field(
        default=True, 
        description="Enable performance metrics tracking"
    )
    log_git_commands: bool = Field(
        default=False, 
        description="Log Git commands for debugging (security sensitive)"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "GIT_"
        case_sensitive = False
