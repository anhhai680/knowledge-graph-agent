"""
Chunking strategy module for the Knowledge Graph Agent.

This module provides language-aware chunking strategies for documents.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
    TextSplitter
)
from loguru import logger


class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.
    
    This class defines the interface for all chunking strategies.
    """
    
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
    ):
        """
        Initialize the chunking strategy.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def split_document(self, document: Document) -> List[Document]:
        """
        Split a document into chunks.
        
        Args:
            document: LangChain Document to split
            
        Returns:
            List of Document chunks
        """
        pass
    
    def _preserve_metadata(self, original_document: Document, chunks: List[str]) -> List[Document]:
        """
        Preserve metadata when creating chunks.
        
        Args:
            original_document: Original document
            chunks: List of text chunks
            
        Returns:
            List of Document objects with preserved metadata
        """
        chunk_documents = []
        
        for i, chunk in enumerate(chunks):
            # Clone the original metadata
            metadata = dict(original_document.metadata)
            
            # Add chunk-specific metadata
            metadata["chunk_index"] = i
            metadata["chunk_count"] = len(chunks)
            
            # Create a new Document with the chunk and metadata
            chunk_document = Document(
                page_content=chunk,
                metadata=metadata
            )
            
            chunk_documents.append(chunk_document)
        
        return chunk_documents


class GenericChunkingStrategy(ChunkingStrategy):
    """
    Generic chunking strategy using RecursiveCharacterTextSplitter.
    
    This strategy is used for languages without a specific strategy.
    """
    
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        language: Optional[str] = None,
    ):
        """
        Initialize the generic chunking strategy.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            language: Language of the document (used to select appropriate separators)
        """
        super().__init__(chunk_size, chunk_overlap)
        self.language = language
    
    def split_document(self, document: Document) -> List[Document]:
        """
        Split a document into chunks using RecursiveCharacterTextSplitter.
        
        Args:
            document: LangChain Document to split
            
        Returns:
            List of Document chunks
        """
        # Map language to LangChain Language enum if possible
        lang_mapping = {
            "python": Language.PYTHON,
            "javascript": Language.JS,
            "typescript": Language.TS,
            "jsx": Language.JS,
            "tsx": Language.TS,
            "html": Language.HTML,
            "css": Language.CSS,
            "csharp": Language.CSHARP,
            "php": Language.PHP,
            "markdown": Language.MARKDOWN,
            "json": Language.JSON,
            "yaml": Language.YAML,
            "shell": Language.BASH,
        }
        
        language = self.language or document.metadata.get("language", "")
        lang_enum = lang_mapping.get(language)
        
        # Create the appropriate text splitter
        if lang_enum:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang_enum,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            # Default separators for unknown languages
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        
        # Split the document
        chunks = splitter.split_text(document.page_content)
        
        # Create Document objects for each chunk with preserved metadata
        return self._preserve_metadata(document, chunks)


class CSharpChunkingStrategy(ChunkingStrategy):
    """
    C# specific chunking strategy.
    
    This strategy chunks C# code based on class and method boundaries.
    """
    
    def split_document(self, document: Document) -> List[Document]:
        """
        Split a C# document into chunks based on class and method boundaries.
        
        Args:
            document: LangChain Document to split
            
        Returns:
            List of Document chunks
        """
        content = document.page_content
        
        # Extract classes and methods
        class_chunks = self._extract_classes(content)
        
        if not class_chunks:
            # If no classes found, use the generic strategy
            generic_strategy = GenericChunkingStrategy(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                language="csharp"
            )
            return generic_strategy.split_document(document)
        
        # Create Document objects for each chunk with preserved metadata
        chunk_documents = []
        
        for class_name, class_content in class_chunks.items():
            # Extract methods from the class
            method_chunks = self._extract_methods(class_content)
            
            if not method_chunks:
                # If no methods found, use the class as a chunk
                metadata = dict(document.metadata)
                metadata["chunk_type"] = "class"
                metadata["class_name"] = class_name
                
                chunk_document = Document(
                    page_content=class_content,
                    metadata=metadata
                )
                
                chunk_documents.append(chunk_document)
            else:
                # Create a chunk for each method
                for method_name, method_content in method_chunks.items():
                    metadata = dict(document.metadata)
                    metadata["chunk_type"] = "method"
                    metadata["class_name"] = class_name
                    metadata["function_name"] = method_name
                    
                    chunk_document = Document(
                        page_content=method_content,
                        metadata=metadata
                    )
                    
                    chunk_documents.append(chunk_document)
        
        # If no chunks were created, use the generic strategy
        if not chunk_documents:
            generic_strategy = GenericChunkingStrategy(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                language="csharp"
            )
            return generic_strategy.split_document(document)
        
        return chunk_documents
    
    def _extract_classes(self, content: str) -> Dict[str, str]:
        """
        Extract classes from C# content.
        
        Args:
            content: C# code content
            
        Returns:
            Dictionary mapping class names to class content
        """
        # Regex pattern for C# classes
        class_pattern = r"(public|private|protected|internal)?\s*(static|abstract|sealed)?\s*class\s+(\w+)(?:<.+?>)?\s*(?::\s*\w+(?:\s*,\s*\w+)*)?\s*\{([\s\S]*?)(?=\}\s*(?:public|private|protected|internal)?\s*(?:static|abstract|sealed)?\s*class|\}\s*$|\z)"
        
        classes = {}
        for match in re.finditer(class_pattern, content):
            class_name = match.group(3)
            class_content = match.group(0)
            classes[class_name] = class_content
        
        return classes
    
    def _extract_methods(self, class_content: str) -> Dict[str, str]:
        """
        Extract methods from C# class content.
        
        Args:
            class_content: C# class content
            
        Returns:
            Dictionary mapping method names to method content
        """
        # Regex pattern for C# methods
        method_pattern = r"(public|private|protected|internal)?\s*(static|virtual|abstract|override|async)?\s*\w+(?:<.+?>)?\s+(\w+)\s*\([\s\S]*?\)\s*(?:where\s+.+?(?=\{))?\s*\{([\s\S]*?)(?=\}\s*(?:public|private|protected|internal)?\s*|\}\s*$|\z)"
        
        methods = {}
        for match in re.finditer(method_pattern, class_content):
            method_name = match.group(3)
            method_content = match.group(0)
            methods[method_name] = method_content
        
        return methods


class ReactChunkingStrategy(ChunkingStrategy):
    """
    React specific chunking strategy.
    
    This strategy chunks React code based on component and function boundaries.
    """
    
    def split_document(self, document: Document) -> List[Document]:
        """
        Split a React document into chunks based on component and function boundaries.
        
        Args:
            document: LangChain Document to split
            
        Returns:
            List of Document chunks
        """
        content = document.page_content
        
        # Extract components and functions
        component_chunks = self._extract_components(content)
        function_chunks = self._extract_functions(content)
        
        if not component_chunks and not function_chunks:
            # If no components or functions found, use the generic strategy
            language = "javascript"
            if document.metadata.get("language") in ["typescript", "tsx"]:
                language = "typescript"
            
            generic_strategy = GenericChunkingStrategy(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                language=language
            )
            return generic_strategy.split_document(document)
        
        # Create Document objects for each chunk with preserved metadata
        chunk_documents = []
        
        # Process component chunks
        for component_name, component_content in component_chunks.items():
            metadata = dict(document.metadata)
            metadata["chunk_type"] = "component"
            metadata["component_name"] = component_name
            
            chunk_document = Document(
                page_content=component_content,
                metadata=metadata
            )
            
            chunk_documents.append(chunk_document)
        
        # Process function chunks
        for function_name, function_content in function_chunks.items():
            # Skip functions that are already part of a component
            if any(function_content in component for component in component_chunks.values()):
                continue
            
            metadata = dict(document.metadata)
            metadata["chunk_type"] = "function"
            metadata["function_name"] = function_name
            
            chunk_document = Document(
                page_content=function_content,
                metadata=metadata
            )
            
            chunk_documents.append(chunk_document)
        
        # If no chunks were created, use the generic strategy
        if not chunk_documents:
            language = "javascript"
            if document.metadata.get("language") in ["typescript", "tsx"]:
                language = "typescript"
            
            generic_strategy = GenericChunkingStrategy(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                language=language
            )
            return generic_strategy.split_document(document)
        
        return chunk_documents
    
    def _extract_components(self, content: str) -> Dict[str, str]:
        """
        Extract React components from content.
        
        Args:
            content: React code content
            
        Returns:
            Dictionary mapping component names to component content
        """
        # Regex pattern for React function components
        func_component_pattern = r"(?:export\s+)?(?:const|function)\s+(\w+)\s*(?:=\s*(?:\(\s*\{[^}]*\}\s*\)|\([^)]*\))\s*=>|=>\s*\{|\([^)]*\)\s*\{|\{)"
        
        # Regex pattern for React class components
        class_component_pattern = r"(?:export\s+)?class\s+(\w+)\s+extends\s+React\.Component"
        
        components = {}
        
        # Extract function components
        for match in re.finditer(func_component_pattern, content):
            component_name = match.group(1)
            
            # Check if it's likely a component (PascalCase naming convention)
            if component_name and component_name[0].isupper():
                # Find the component body
                start_pos = match.start()
                brace_count = 0
                in_component = False
                end_pos = start_pos
                
                for i in range(start_pos, len(content)):
                    if content[i] == "{":
                        brace_count += 1
                        in_component = True
                    elif content[i] == "}":
                        brace_count -= 1
                        if in_component and brace_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > start_pos:
                    component_content = content[start_pos:end_pos]
                    components[component_name] = component_content
        
        # Extract class components
        for match in re.finditer(class_component_pattern, content):
            component_name = match.group(1)
            
            # Find the component body
            start_pos = match.start()
            brace_count = 0
            in_component = False
            end_pos = start_pos
            
            for i in range(start_pos, len(content)):
                if content[i] == "{":
                    brace_count += 1
                    in_component = True
                elif content[i] == "}":
                    brace_count -= 1
                    if in_component and brace_count == 0:
                        end_pos = i + 1
                        break
            
            if end_pos > start_pos:
                component_content = content[start_pos:end_pos]
                components[component_name] = component_content
        
        return components
    
    def _extract_functions(self, content: str) -> Dict[str, str]:
        """
        Extract JavaScript/TypeScript functions from content.
        
        Args:
            content: JavaScript/TypeScript code content
            
        Returns:
            Dictionary mapping function names to function content
        """
        # Regex pattern for functions
        function_pattern = r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{|(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>"
        
        functions = {}
        
        for match in re.finditer(function_pattern, content):
            function_name = match.group(1) or match.group(2)
            
            # Skip component functions (they're already extracted)
            if function_name and not function_name[0].isupper():
                # Find the function body
                start_pos = match.start()
                brace_count = 0
                in_function = False
                end_pos = start_pos
                
                for i in range(start_pos, len(content)):
                    if content[i] == "{":
                        brace_count += 1
                        in_function = True
                    elif content[i] == "}":
                        brace_count -= 1
                        if in_function and brace_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > start_pos:
                    function_content = content[start_pos:end_pos]
                    functions[function_name] = function_content
        
        return functions


def get_chunking_strategy(
    language: str,
    chunk_size: int,
    chunk_overlap: int
) -> ChunkingStrategy:
    """
    Get the appropriate chunking strategy for a language.
    
    Args:
        language: Language of the document
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        ChunkingStrategy instance
    """
    # Select strategy based on language
    if language in ["csharp", "csharp-project"]:
        return CSharpChunkingStrategy(chunk_size, chunk_overlap)
    elif language in ["javascript", "typescript", "jsx", "tsx"]:
        return ReactChunkingStrategy(chunk_size, chunk_overlap)
    else:
        return GenericChunkingStrategy(chunk_size, chunk_overlap, language)
