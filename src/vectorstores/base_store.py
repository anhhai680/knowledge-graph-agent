"""
Base vector store module for the Knowledge Graph Agent.

This module provides the base class for vector store implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain.embeddings.base import Embeddings
from langchain.schema import Document



class BaseStore(ABC):
    """
    Abstract base class for vector store implementations.

    This class defines the interface for all vector store implementations.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[Embeddings] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Documents to add
            embeddings: Embeddings instance for generating embeddings
            batch_size: Batch size for adding documents
        """
        pass

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[Document]:
        """
        Search for documents similar to a query.

        Args:
            query: Query string
            k: Number of results to return
            filter: Filter to apply to search results
            **kwargs: Additional arguments for the search

        Returns:
            List of similar documents
        """
        pass

    @abstractmethod
    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to a query with similarity scores.

        Args:
            query: Query string
            k: Number of results to return
            filter: Filter to apply to search results
            **kwargs: Additional arguments for the search

        Returns:
            List of tuples with similar documents and their scores
        """
        pass

    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete
            filter: Filter to apply to documents to delete
        """
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.

        Returns:
            Dictionary with collection statistics
        """
        pass

    @abstractmethod
    def get_repository_metadata(self) -> List[Dict[str, Any]]:
        """
        Get repository metadata from all indexed documents.

        Returns:
            List of dictionaries containing repository metadata
        """
        pass

    @abstractmethod
    def health_check(self) -> Tuple[bool, str]:
        """
        Check if the vector store is healthy.

        Returns:
            Tuple of (is_healthy, message)
        """
        pass

    @staticmethod
    @abstractmethod
    def from_documents(
        documents: List[Document], embeddings: Embeddings, **kwargs
    ) -> "BaseStore":
        """
        Create a vector store from documents.

        Args:
            documents: List of LangChain Documents
            embeddings: Embeddings instance for generating embeddings
            **kwargs: Additional arguments for the vector store

        Returns:
            BaseStore instance
        """
        pass

    @staticmethod
    @abstractmethod
    def from_existing(**kwargs) -> "BaseStore":
        """
        Create a vector store from an existing collection.

        Args:
            **kwargs: Arguments for connecting to the existing vector store

        Returns:
            BaseStore instance
        """
        pass

    @staticmethod
    def process_metadata_for_storage(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata to ensure it's compatible with the vector store.

        Args:
            metadata: Document metadata

        Returns:
            Processed metadata
        """
        processed = {}

        # Process each metadata field to ensure compatibility
        for key, value in metadata.items():
            # Convert lists and dicts to strings
            if isinstance(value, (list, dict)):
                processed[key] = str(value)
            # Convert None to empty string
            elif value is None:
                processed[key] = ""
            # Pass through other values
            else:
                processed[key] = value

        return processed

    @staticmethod
    def format_filter(filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a filter for the vector store.

        Args:
            filter: Filter to format

        Returns:
            Formatted filter
        """
        # Default implementation just returns the filter
        return filter
