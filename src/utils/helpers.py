"""
Helper utilities for the Knowledge Graph Agent.

This module provides utility functions used throughout the application.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import httpx
from langchain.schema import Document
from loguru import logger


def generate_unique_id() -> str:
    """
    Generate a unique identifier.
    
    Returns:
        Unique identifier string
    """
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """
    Get a formatted timestamp for the current time.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with the JSON contents
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file isn't valid JSON
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        raise


def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the JSON file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.debug(f"Saved JSON file: {file_path}")


def check_service_health(url: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Check if a service is healthy.
    
    Args:
        url: URL to check
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (is_healthy, message)
    """
    try:
        response = httpx.get(url, timeout=timeout)
        if response.status_code == 200:
            return True, "Service is healthy"
        else:
            return False, f"Service returned status code {response.status_code}"
    except httpx.TimeoutException:
        return False, f"Service timed out after {timeout} seconds"
    except httpx.RequestError as e:
        return False, f"Error connecting to service: {str(e)}"


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def format_document_for_prompt(doc: Document) -> str:
    """
    Format a document for inclusion in a prompt.
    
    Args:
        doc: LangChain Document
        
    Returns:
        Formatted document string
    """
    # Extract metadata fields
    repo = doc.metadata.get("repository", "Unknown Repository")
    file_path = doc.metadata.get("file_path", "Unknown File")
    language = doc.metadata.get("language", "")
    chunk_type = doc.metadata.get("chunk_type", "")
    
    # Format metadata header
    header = f"Source: {repo} - {file_path}"
    if language:
        header += f" (Language: {language})"
    if chunk_type:
        header += f" (Type: {chunk_type})"
    
    # Format content with header
    return f"{header}\n{'-' * 80}\n{doc.page_content}\n{'-' * 80}\n"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
