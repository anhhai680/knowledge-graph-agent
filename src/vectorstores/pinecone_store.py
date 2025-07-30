"""
Pinecone vector store module for the Knowledge Graph Agent.

This module provides a Pinecone implementation of the vector store.
"""

from typing import Any, Dict, List, Optional, Tuple

import pinecone
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import Pinecone as LangchainPinecone
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config.settings import settings
from src.llm.embedding_factory import EmbeddingFactory
from src.llm.llm_constants import LLM_CONSTANTS
from src.vectorstores.base_store import BaseStore


class PineconeStore(BaseStore):
    """
    Pinecone implementation of the vector store.

    This class provides methods for interacting with a Pinecone vector store.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        """
        Initialize the Pinecone vector store.

        Args:
            api_key: Pinecone API key
            collection_name: Name of the collection
            embeddings: Embeddings instance for generating embeddings
        """
        # Get settings from config if not provided
        self.api_key = api_key or settings.pinecone.api_key
        self.collection_name = collection_name or settings.pinecone.collection_name

        # Create embeddings if not provided
        self.embeddings = embeddings or EmbeddingFactory.create()

        # Initialize Pinecone
        pinecone.init(api_key=self.api_key)

        # Get or create index
        try:
            # Check if index exists
            indexes = pinecone.list_indexes()

            if self.collection_name not in indexes:
                # Create index if it doesn't exist
                logger.info(f"Creating Pinecone index: {self.collection_name}")

                # Create index with appropriate settings
                # Dimension based on the embedding model (OpenAI ada = 1536)
                pinecone.create_index(
                    name=self.collection_name,
                    dimension=LLM_CONSTANTS.EMBEDDING_DIMENSION.value,  # Dimension of OpenAI embeddings
                    metric="cosine",
                )

            # Connect to the index
            self.index = pinecone.Index(self.collection_name)
            logger.debug(f"Connected to Pinecone index: {self.collection_name}")

            # Create LangChain Pinecone instance
            self.langchain_store = LangchainPinecone(
                index=self.index, embedding_function=self.embeddings, text_key="text"
            )

        except Exception as e:
            error_message = f"Error connecting to Pinecone: {str(e)}"
            logger.error(error_message)
            raise ValueError(error_message)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
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
        if not documents:
            logger.warning("No documents to add to Pinecone")
            return

        # Use provided embeddings or default
        embedding_func = embeddings or self.embeddings

        # Use provided batch size or default
        batch_size = batch_size or settings.embedding.batch_size

        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            try:
                # Process metadata for each document
                for doc in batch:
                    doc.metadata = self.process_metadata_for_storage(doc.metadata)

                # Add documents to the store
                self.langchain_store.add_documents(batch)
                logger.debug(f"Added batch of {len(batch)} documents to Pinecone")

            except Exception as e:
                logger.error(f"Error adding documents to Pinecone: {str(e)}")
                # The retry decorator will retry this batch
                raise

        logger.info(
            f"Added {len(documents)} documents to Pinecone index {self.collection_name}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
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
        try:
            # Format filter if provided
            formatted_filter = self.format_filter(filter) if filter else None

            # Search for similar documents
            results = self.langchain_store.similarity_search(
                query=query, k=k, filter=formatted_filter, **kwargs
            )

            logger.debug(
                f"Found {len(results)} documents similar to query: {query[:50]}..."
            )
            return results

        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            # The retry decorator will retry this search
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
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
        try:
            # Format filter if provided
            formatted_filter = self.format_filter(filter) if filter else None

            # Search for similar documents with scores
            results = self.langchain_store.similarity_search_with_score(
                query=query, k=k, filter=formatted_filter, **kwargs
            )

            logger.debug(
                f"Found {len(results)} documents with scores similar to query: {query[:50]}..."
            )
            return results

        except Exception as e:
            logger.error(f"Error searching Pinecone with scores: {str(e)}")
            # The retry decorator will retry this search
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
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
        try:
            if ids:
                # Delete documents by ID
                self.index.delete(ids=ids)
                logger.debug(f"Deleted {len(ids)} documents from Pinecone")
            elif filter:
                # Format filter
                formatted_filter = self.format_filter(filter)

                # Fetch IDs matching the filter
                query_results = self.index.query(
                    vector=[0]
                    * LLM_CONSTANTS.EMBEDDING_DIMENSION.value,  # Dummy vector for metadata-only query
                    filter=formatted_filter,
                    top_k=10000,
                    include_metadata=False,
                )

                if query_results and "matches" in query_results:
                    ids_to_delete = [match["id"] for match in query_results["matches"]]
                    if ids_to_delete:
                        self.index.delete(ids=ids_to_delete)
                        logger.debug(
                            f"Deleted {len(ids_to_delete)} documents from Pinecone based on filter"
                        )
            else:
                logger.warning("No IDs or filter provided for deletion")

        except Exception as e:
            logger.error(f"Error deleting documents from Pinecone: {str(e)}")
            # The retry decorator will retry this deletion
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get index stats
            stats = self.index.describe_index_stats()

            # Extract relevant information
            namespaces = stats.get("namespaces", {})
            total_count = sum(ns.get("vector_count", 0) for ns in namespaces.values())

            result = {
                "name": self.collection_name,
                "count": total_count,
                "dimension": stats.get("dimension"),
                "namespaces": len(namespaces),
            }

            return result

        except Exception as e:
            logger.error(f"Error getting Pinecone index stats: {str(e)}")
            return {"name": self.collection_name, "error": str(e)}

    def health_check(self) -> Tuple[bool, str]:
        """
        Check if the vector store is healthy.

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            # Attempt to get index stats
            _ = self.index.describe_index_stats()
            return True, "Pinecone vector store is healthy"

        except Exception as e:
            error_message = f"Pinecone vector store health check failed: {str(e)}"
            logger.error(error_message)
            return False, error_message

    @staticmethod
    def format_filter(filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a filter for Pinecone.

        Args:
            filter: Filter to format

        Returns:
            Formatted filter
        """
        # Pinecone uses a dictionary of metadata filters
        # Most filters can be passed through as-is
        return filter

    @staticmethod
    def from_documents(
        documents: List[Document],
        embeddings: Embeddings,
        collection_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "PineconeStore":
        """
        Create a Pinecone vector store from documents.

        Args:
            documents: List of LangChain Documents
            embeddings: Embeddings instance for generating embeddings
            collection_name: Name of the collection
            api_key: Pinecone API key
            **kwargs: Additional arguments for the vector store

        Returns:
            PineconeStore instance
        """
        # Create a new PineconeStore instance
        store = PineconeStore(
            api_key=api_key, collection_name=collection_name, embeddings=embeddings
        )

        # Add documents to the store
        store.add_documents(documents, embeddings=embeddings)

        return store

    @staticmethod
    def from_existing(
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "PineconeStore":
        """
        Create a PineconeStore from an existing collection.

        Args:
            collection_name: Name of the collection
            embeddings: Embeddings instance for generating embeddings
            api_key: Pinecone API key
            **kwargs: Additional arguments for the vector store

        Returns:
            PineconeStore instance
        """
        # Create a new PineconeStore instance connected to the existing index
        store = PineconeStore(
            api_key=api_key, collection_name=collection_name, embeddings=embeddings
        )

        return store
