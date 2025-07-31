"""
Chroma vector store module for the Knowledge Graph Agent.

This module provides a Chroma implementation of the vector store.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config.settings import settings
from src.llm.embedding_factory import EmbeddingFactory
from src.vectorstores.base_store import BaseStore


class ChromaStore(BaseStore):
    """
    Chroma implementation of the vector store.

    This class provides methods for interacting with a Chroma vector store.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the Chroma vector store.

        Args:
            client: Chroma client
            collection_name: Name of the collection
            embeddings: Embeddings instance for generating embeddings
            persist_directory: Directory for persisting the vector store
        """
        # Get settings from config if not provided
        self.collection_name = collection_name or settings.chroma.collection_name

        # Create embeddings if not provided
        self.embeddings = embeddings or EmbeddingFactory.create()

        # Create client if not provided
        if client is None:
            if persist_directory:
                # Use local persistence
                self.client = chromadb.PersistentClient(path=persist_directory)
            else:
                # Use Chroma server
                self.client = chromadb.HttpClient(
                    host=settings.chroma.host, port=settings.chroma.port
                )
        else:
            self.client = client

        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(self.collection_name)
            logger.debug(f"Connected to Chroma collection: {self.collection_name}")
        except Exception as e:
            error_message = f"Error connecting to Chroma collection: {str(e)}"
            logger.error(error_message)
            raise ValueError(error_message)

        # Create LangChain Chroma instance
        self.langchain_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

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
            logger.warning("No documents to add to Chroma")
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
                logger.debug(f"Added batch of {len(batch)} documents to Chroma")

            except Exception as e:
                logger.error(f"Error adding documents to Chroma: {str(e)}")
                # The retry decorator will retry this batch
                raise

        logger.info(
            f"Added {len(documents)} documents to Chroma collection {self.collection_name}"
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
            logger.error(f"Error searching Chroma: {str(e)}")
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
            logger.error(f"Error searching Chroma with scores: {str(e)}")
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
                self.collection.delete(ids=ids)
                logger.debug(f"Deleted {len(ids)} documents from Chroma")
            elif filter:
                # Format filter
                formatted_filter = self.format_filter(filter)

                # Get documents matching the filter
                query_results = self.collection.query(
                    query_texts=[""], where=formatted_filter, n_results=10000
                )

                if query_results and "ids" in query_results:
                    ids_to_delete = query_results["ids"][0]
                    if ids_to_delete:
                        self.collection.delete(ids=ids_to_delete)
                        logger.debug(
                            f"Deleted {len(ids_to_delete)} documents from Chroma based on filter"
                        )
            else:
                logger.warning("No IDs or filter provided for deletion")

        except Exception as e:
            logger.error(f"Error deleting documents from Chroma: {str(e)}")
            # The retry decorator will retry this deletion
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get collection count
            count = self.collection.count()

            # Get collection metadata (if any)
            collection_info = self.client.get_collection(self.collection_name)

            stats = {
                "name": self.collection_name,
                "count": count,
                "dimension": (
                    collection_info.metadata.get("dimension")
                    if hasattr(collection_info, "metadata")
                    else None
                ),
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting Chroma collection stats: {str(e)}")
            return {"name": self.collection_name, "error": str(e)}

    def get_repository_metadata(self) -> List[Dict[str, Any]]:
        """
        Get repository metadata from all indexed documents.

        Returns:
            List of dictionaries containing repository metadata
        """
        try:
            # Query all documents to analyze repository metadata
            # We'll get a sample of documents and aggregate repository information
            query_results = self.collection.query(
                query_texts=[""],  # Empty query to get all documents
                n_results=10000,  # Large number to get all documents
                include=["metadatas", "documents"]
            )

            if not query_results or "metadatas" not in query_results:
                logger.warning("No documents found in Chroma collection")
                return []

            # Aggregate repository information from document metadata
            repo_stats = {}
            metadatas_list = query_results.get("metadatas")
            documents_list = query_results.get("documents")
            
            metadatas = metadatas_list[0] if metadatas_list and len(metadatas_list) > 0 else []
            documents = documents_list[0] if documents_list and len(documents_list) > 0 else []

            for i, metadata in enumerate(metadatas):
                if not metadata:
                    continue

                repo_url = metadata.get("repository_url", "")
                repo_name = metadata.get("repository", "")
                
                # Ensure we have string values
                if not isinstance(repo_url, str):
                    repo_url = str(repo_url) if repo_url else ""
                if not isinstance(repo_name, str):
                    repo_name = str(repo_name) if repo_name else ""
                
                # Skip if no repository information
                if not repo_url and not repo_name:
                    continue

                # Use repository URL as key, fallback to name
                repo_key = repo_url or repo_name
                
                if repo_key not in repo_stats:
                    # Extract repository display name
                    display_name = repo_name
                    if not display_name and repo_url and isinstance(repo_url, str) and "/" in repo_url:
                        url_parts = repo_url.rstrip("/").split("/")
                        if len(url_parts) >= 2:
                            display_name = f"{url_parts[-2]}/{url_parts[-1]}"
                    
                    repo_stats[repo_key] = {
                        "name": display_name or repo_key,
                        "url": repo_url,
                        "branch": metadata.get("branch", "main"),
                        "file_count": 0,
                        "document_count": 0,
                        "languages": set(),
                        "total_size": 0,
                        "last_indexed": None,
                        "files": set()
                    }

                # Update statistics
                repo_data = repo_stats[repo_key]
                repo_data["document_count"] += 1
                
                # Track unique files
                source_file = metadata.get("source", "")
                if source_file and isinstance(source_file, str):
                    repo_data["files"].add(source_file)

                # Track languages
                language = metadata.get("language", "")
                if language and isinstance(language, str):
                    repo_data["languages"].add(language)

                # Track file size
                file_size = metadata.get("size", 0)
                if isinstance(file_size, (int, float)):
                    repo_data["total_size"] += file_size

                # Track last modification (use as proxy for indexing time)
                last_modified = metadata.get("last_modified")
                if last_modified:
                    if not repo_data["last_indexed"] or last_modified > repo_data["last_indexed"]:
                        repo_data["last_indexed"] = last_modified

            # Convert to final format
            repositories = []
            for repo_key, repo_data in repo_stats.items():
                # Calculate file count from unique files
                repo_data["file_count"] = len(repo_data["files"])
                
                # Convert languages set to list
                repo_data["languages"] = list(repo_data["languages"])
                
                # Convert size to MB
                size_mb = repo_data["total_size"] / (1024 * 1024) if repo_data["total_size"] > 0 else 0.0
                
                repositories.append({
                    "name": repo_data["name"],
                    "url": repo_data["url"],
                    "branch": repo_data["branch"],
                    "last_indexed": repo_data["last_indexed"],
                    "file_count": repo_data["file_count"],
                    "document_count": repo_data["document_count"],
                    "languages": repo_data["languages"],
                    "size_mb": round(size_mb, 2)
                })

            logger.info(f"Retrieved metadata for {len(repositories)} repositories from Chroma")
            return repositories

        except Exception as e:
            logger.error(f"Error getting repository metadata from Chroma: {str(e)}")
            return []

    def health_check(self) -> Tuple[bool, str]:
        """
        Check if the vector store is healthy.

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            # Attempt to get collection stats
            _ = self.collection.count()
            return True, "Chroma vector store is healthy"

        except Exception as e:
            error_message = f"Chroma vector store health check failed: {str(e)}"
            logger.error(error_message)
            return False, error_message

    @staticmethod
    def format_filter(filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a filter for Chroma.

        Args:
            filter: Filter to format

        Returns:
            Formatted filter
        """
        # Chroma uses a 'where' filter with a specific format
        # Most filters can be passed through as-is
        return filter

    @staticmethod
    def from_documents(
        documents: List[Document],
        embeddings: Embeddings,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        **kwargs,
    ) -> "ChromaStore":
        """
        Create a Chroma vector store from documents.

        Args:
            documents: List of LangChain Documents
            embeddings: Embeddings instance for generating embeddings
            collection_name: Name of the collection
            persist_directory: Directory for persisting the vector store
            **kwargs: Additional arguments for the vector store

        Returns:
            ChromaStore instance
        """
        # Create a new ChromaStore instance
        store = ChromaStore(
            collection_name=collection_name,
            embeddings=embeddings,
            persist_directory=persist_directory,
        )

        # Add documents to the store
        store.add_documents(documents, embeddings=embeddings)

        return store

    @staticmethod
    def from_existing(
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        **kwargs,
    ) -> "ChromaStore":
        """
        Create a ChromaStore from an existing collection.

        Args:
            collection_name: Name of the collection
            embeddings: Embeddings instance for generating embeddings
            persist_directory: Directory for persisting the vector store
            **kwargs: Additional arguments for the vector store

        Returns:
            ChromaStore instance
        """
        # Create a new ChromaStore instance connected to the existing collection
        store = ChromaStore(
            collection_name=collection_name,
            embeddings=embeddings,
            persist_directory=persist_directory,
        )

        return store
