"""
Metadata extractor module for the Knowledge Graph Agent.

This module provides functionality for extracting metadata from document content.
"""

import os
import re
from typing import Any, Dict, Optional

from loguru import logger


def extract_metadata(
    content: str,
    language: str,
    file_path: str,
    chunk_index: int
) -> Dict[str, Any]:
    """
    Extract metadata from document content.
    
    Args:
        content: Document content
        language: Language of the document
        file_path: Path to the document
        chunk_index: Index of the chunk
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        "chunk_index": chunk_index,
    }
    
    # Extract line numbers
    line_count = content.count("\n") + 1
    metadata["line_count"] = line_count
    
    # Estimate token count (rough approximation: ~4 chars per token for English)
    token_count = len(content) // 4
    metadata["tokens"] = token_count
    
    # Language-specific metadata extraction
    if language == "csharp":
        metadata.update(extract_csharp_metadata(content))
    elif language in ["javascript", "typescript", "jsx", "tsx"]:
        metadata.update(extract_js_metadata(content))
    elif language == "python":
        metadata.update(extract_python_metadata(content))
    elif language in ["markdown", "md"]:
        metadata.update(extract_markdown_metadata(content))
    
    # Add chunk_type if not already present
    if "chunk_type" not in metadata:
        metadata["chunk_type"] = determine_chunk_type(content, language, file_path)
    
    return metadata


def determine_chunk_type(
    content: str,
    language: str,
    file_path: str
) -> str:
    """
    Determine the type of a chunk based on its content and language.
    
    Args:
        content: Chunk content
        language: Language of the chunk
        file_path: Path to the file
        
    Returns:
        Chunk type (e.g., "class", "method", "function", "component", "text")
    """
    # Check file type first
    if file_path.endswith(".md") or language in ["markdown", "md"]:
        return "text"
    
    # Check content patterns
    if language == "csharp":
        if "class " in content:
            return "class"
        elif "void " in content or "return " in content or "public " in content:
            return "method"
    elif language in ["javascript", "typescript", "jsx", "tsx"]:
        if "class " in content and " extends " in content:
            return "component"
        elif "function " in content or "=> {" in content or "return (" in content:
            return "function"
    elif language == "python":
        if "class " in content:
            return "class"
        elif "def " in content:
            return "function"
    
    # Default chunk type
    return "code"


def extract_csharp_metadata(content: str) -> Dict[str, Any]:
    """
    Extract metadata from C# content.
    
    Args:
        content: C# code content
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {}
    
    # Extract class name
    class_match = re.search(r"class\s+(\w+)", content)
    if class_match:
        metadata["class_name"] = class_match.group(1)
        metadata["chunk_type"] = "class"
    
    # Extract method name
    method_match = re.search(r"(?:public|private|protected|internal)?\s*(?:static|virtual|abstract|override)?\s*\w+\s+(\w+)\s*\(", content)
    if method_match:
        metadata["function_name"] = method_match.group(1)
        metadata["chunk_type"] = "method"
    
    # Extract namespace
    namespace_match = re.search(r"namespace\s+([\w\.]+)", content)
    if namespace_match:
        metadata["namespace"] = namespace_match.group(1)
    
    return metadata


def extract_js_metadata(content: str) -> Dict[str, Any]:
    """
    Extract metadata from JavaScript/TypeScript content.
    
    Args:
        content: JavaScript/TypeScript code content
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {}
    
    # Extract component name
    component_match = re.search(r"(?:function|const)\s+(\w+)\s*(?:=|\()", content)
    if component_match:
        component_name = component_match.group(1)
        if component_name and component_name[0].isupper():  # PascalCase naming convention for components
            metadata["component_name"] = component_name
            metadata["chunk_type"] = "component"
    
    # Extract class component
    class_component_match = re.search(r"class\s+(\w+)\s+extends\s+React", content)
    if class_component_match:
        metadata["component_name"] = class_component_match.group(1)
        metadata["chunk_type"] = "component"
    
    # Extract function name
    function_match = re.search(r"function\s+(\w+)\s*\(", content)
    if function_match:
        function_name = function_match.group(1)
        if not function_name[0].isupper():  # Skip component functions
            metadata["function_name"] = function_name
            metadata["chunk_type"] = "function"
    
    # Extract arrow function
    arrow_function_match = re.search(r"const\s+(\w+)\s*=\s*(?:\(.*?\))?\s*=>", content)
    if arrow_function_match:
        function_name = arrow_function_match.group(1)
        if not function_name[0].isupper():  # Skip component functions
            metadata["function_name"] = function_name
            metadata["chunk_type"] = "function"
    
    # Extract imports/exports
    if re.search(r"(import|export)\s+", content):
        metadata["has_imports"] = True
    
    return metadata


def extract_python_metadata(content: str) -> Dict[str, Any]:
    """
    Extract metadata from Python content.
    
    Args:
        content: Python code content
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {}
    
    # Extract class name
    class_match = re.search(r"class\s+(\w+)(?:\(.*?\))?:", content)
    if class_match:
        metadata["class_name"] = class_match.group(1)
        metadata["chunk_type"] = "class"
    
    # Extract function name
    function_match = re.search(r"def\s+(\w+)\s*\(", content)
    if function_match:
        metadata["function_name"] = function_match.group(1)
        metadata["chunk_type"] = "function"
    
    # Extract imports
    if re.search(r"(import|from)\s+", content):
        metadata["has_imports"] = True
    
    return metadata


def extract_markdown_metadata(content: str) -> Dict[str, Any]:
    """
    Extract metadata from Markdown content.
    
    Args:
        content: Markdown content
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        "chunk_type": "text"
    }
    
    # Extract heading
    heading_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if heading_match:
        metadata["heading"] = heading_match.group(1).strip()
    
    # Extract subheading
    subheading_match = re.search(r"^##\s+(.+)$", content, re.MULTILINE)
    if subheading_match:
        metadata["subheading"] = subheading_match.group(1).strip()
    
    # Check for code blocks
    if re.search(r"```\w*\n", content):
        metadata["has_code_blocks"] = True
    
    return metadata
