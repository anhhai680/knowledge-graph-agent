"""
File System Processor for the Knowledge Graph Agent.

This module provides functionality for file system operations including
directory scanning, file content reading, encoding detection, and metadata extraction.
"""

import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
import chardet
from loguru import logger


class FileSystemProcessor:
    """
    Handle file system operations for repository processing.
    
    This class provides functionality for scanning directories, reading files,
    detecting encodings, and extracting file metadata.
    """

    def __init__(self, file_extensions: List[str]):
        """
        Initialize file system processor.
        
        Args:
            file_extensions: List of allowed file extensions to process
        """
        self.file_extensions = [ext.lower() for ext in file_extensions]
        # Ensure extensions start with dot
        self.file_extensions = [
            ext if ext.startswith('.') else f'.{ext}' 
            for ext in self.file_extensions
        ]
        
        # Maximum file size to process (10MB by default)
        self.max_file_size = 10 * 1024 * 1024
        
        # Binary file extensions to skip
        self.binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.bin', '.zip', '.tar', '.gz',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
        }
        
        logger.debug(f"Initialized file system processor with extensions: {self.file_extensions}")

    def scan_directory_tree(self, root_path: str) -> List[str]:
        """
        Recursively scan directory for matching files.
        
        Args:
            root_path: Root directory to scan
            
        Returns:
            List of relative file paths that match the criteria
        """
        try:
            root_path_obj = Path(root_path)
            if not root_path_obj.exists():
                logger.error(f"Root path does not exist: {root_path}")
                return []
            
            matching_files = []
            
            # Walk through directory tree
            for file_path in root_path_obj.rglob("*"):
                if file_path.is_file():
                    # Get relative path from root
                    relative_path = file_path.relative_to(root_path_obj)
                    
                    if self.should_process_file(str(relative_path)):
                        matching_files.append(str(relative_path))
            
            logger.info(f"Found {len(matching_files)} matching files in {root_path}")
            return matching_files
            
        except Exception as e:
            logger.error(f"Error scanning directory {root_path}: {e}")
            return []

    def should_process_file(self, file_path: str) -> bool:
        """
        Check if file should be processed based on extension and filters.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file should be processed, False otherwise
        """
        try:
            file_path_obj = Path(file_path)
            
            # Skip hidden files and directories
            if any(part.startswith('.') for part in file_path_obj.parts):
                # Allow specific common files
                allowed_hidden = {'.gitignore', '.env.example', '.env.template'}
                if file_path_obj.name not in allowed_hidden:
                    return False
            
            # Skip common non-source directories
            skip_dirs = {
                'node_modules', '__pycache__', '.git', '.vscode', '.idea',
                'bin', 'obj', 'build', 'dist', 'target', 'out',
                'vendor', 'packages', '.nuget'
            }
            
            if any(part in skip_dirs for part in file_path_obj.parts):
                return False
            
            # Check file extension
            file_extension = file_path_obj.suffix.lower()
            
            # Skip binary files
            if file_extension in self.binary_extensions:
                return False
            
            # Check if extension is in allowed list
            if file_extension in self.file_extensions:
                return True
            
            # Special case for files without extensions (like Dockerfile)
            if not file_extension:
                allowed_no_ext = {'dockerfile', 'makefile', 'readme', 'license'}
                if file_path_obj.name.lower() in allowed_no_ext:
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking if file should be processed {file_path}: {e}")
            return False

    def read_file_content(self, file_path: str) -> str:
        """
        Read file content with proper encoding detection.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is too large or can't be decoded
        """
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file size
            file_size = file_path_obj.stat().st_size
            if file_size > self.max_file_size:
                raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
            
            # Skip empty files
            if file_size == 0:
                return ""
            
            # Detect encoding
            encoding = self.detect_file_encoding(file_path)
            
            # Read file content
            with open(file_path_obj, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            logger.debug(f"Successfully read {file_size} bytes from {file_path}")
            return content
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    def detect_file_encoding(self, file_path: str) -> str:
        """
        Detect file encoding for proper content reading.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding name
        """
        try:
            file_path_obj = Path(file_path)
            
            # Read a sample of the file for encoding detection
            with open(file_path_obj, 'rb') as f:
                sample = f.read(8192)  # Read first 8KB
            
            # Use chardet to detect encoding
            result = chardet.detect(sample)
            
            if result and result['encoding']:
                encoding = result['encoding']
                confidence = result['confidence']
                
                # If confidence is low, fall back to common encodings
                if confidence < 0.7:
                    # Try common encodings in order of preference
                    common_encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
                    for test_encoding in common_encodings:
                        try:
                            with open(file_path_obj, 'r', encoding=test_encoding) as f:
                                f.read(1024)  # Try to read a bit
                            encoding = test_encoding
                            break
                        except UnicodeDecodeError:
                            continue
                
                logger.debug(f"Detected encoding {encoding} (confidence: {confidence:.2f}) for {file_path}")
                return encoding
            else:
                # Default to UTF-8 if detection fails
                logger.debug(f"Could not detect encoding for {file_path}, defaulting to utf-8")
                return 'utf-8'
                
        except Exception as e:
            logger.debug(f"Error detecting encoding for {file_path}: {e}")
            return 'utf-8'  # Default fallback

    def get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """
        Get file system metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata
        """
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                return {}
            
            stat = file_path_obj.stat()
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path_obj))
            
            metadata = {
                "file_name": file_path_obj.name,
                "file_extension": file_path_obj.suffix,
                "file_size_bytes": stat.st_size,
                "created_timestamp": stat.st_ctime,
                "modified_timestamp": stat.st_mtime,
                "accessed_timestamp": stat.st_atime,
                "mime_type": mime_type,
                "is_binary": self._is_binary_file(file_path),
                "encoding": self.detect_file_encoding(file_path)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting file stats for {file_path}: {e}")
            return {}

    def _is_binary_file(self, file_path: str) -> bool:
        """
        Check if file is binary by extension and content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is binary, False otherwise
        """
        try:
            file_path_obj = Path(file_path)
            
            # Check extension first
            if file_path_obj.suffix.lower() in self.binary_extensions:
                return True
            
            # Check content for binary indicators
            try:
                with open(file_path_obj, 'rb') as f:
                    chunk = f.read(1024)
                
                # Check for null bytes (common in binary files)
                if b'\x00' in chunk:
                    return True
                
                # Check for high ratio of non-printable characters
                if len(chunk) > 0:
                    printable_chars = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in [9, 10, 13])
                    ratio = printable_chars / len(chunk)
                    if ratio < 0.7:  # Less than 70% printable characters
                        return True
                
                return False
                
            except Exception:
                # If we can't read the file, assume it's binary
                return True
                
        except Exception as e:
            logger.debug(f"Error checking if file is binary {file_path}: {e}")
            return True  # Err on the side of caution

    def get_line_count(self, file_path: str) -> int:
        """
        Get number of lines in a text file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Number of lines in the file
        """
        try:
            if self._is_binary_file(file_path):
                return 0
            
            content = self.read_file_content(file_path)
            return len(content.splitlines())
            
        except Exception as e:
            logger.debug(f"Error counting lines in {file_path}: {e}")
            return 0

    def filter_files_by_size(self, file_paths: List[str], root_path: str) -> List[str]:
        """
        Filter files by size constraints.
        
        Args:
            file_paths: List of file paths to filter
            root_path: Root directory path
            
        Returns:
            Filtered list of file paths
        """
        filtered_files = []
        
        for file_path in file_paths:
            try:
                full_path = Path(root_path) / file_path
                if full_path.exists():
                    file_size = full_path.stat().st_size
                    if file_size <= self.max_file_size:
                        filtered_files.append(file_path)
                    else:
                        logger.debug(f"Skipping large file: {file_path} ({file_size} bytes)")
                        
            except Exception as e:
                logger.debug(f"Error checking file size for {file_path}: {e}")
        
        return filtered_files

    def get_directory_stats(self, root_path: str) -> Dict[str, Any]:
        """
        Get statistics about a directory.
        
        Args:
            root_path: Root directory to analyze
            
        Returns:
            Dictionary containing directory statistics
        """
        try:
            root_path_obj = Path(root_path)
            if not root_path_obj.exists():
                return {}
            
            total_files = 0
            total_size = 0
            file_types = {}
            
            for file_path in root_path_obj.rglob("*"):
                if file_path.is_file():
                    total_files += 1
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    # Count file types
                    extension = file_path.suffix.lower()
                    if not extension:
                        extension = "no_extension"
                    file_types[extension] = file_types.get(extension, 0) + 1
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types,
                "processable_files": len(self.scan_directory_tree(root_path))
            }
            
        except Exception as e:
            logger.error(f"Error getting directory stats for {root_path}: {e}")
            return {}
