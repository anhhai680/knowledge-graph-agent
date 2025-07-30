"""
Vector store factory module for the Knowledge Graph Agent.

This module provides a factory pattern for creating vector store instances.
"""

from typing import Dict, Any, List, Optional, Tuple, Union

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from loguru import logger

from src.config.settings import DatabaseType, settings
from src.llm.embedding_factory import EmbeddingFactory
from src.vectorstores.base_store import BaseStore
from src.vectorstores.chroma_store import ChromaStore
from src.vectorstores.pinecone_store import PineconeStore


class VectorStoreFactory:
    """
    Factory class for creating vector store instances.

    This class provides methods for creating different types of vector store instances
    based on the application configuration.
    """

    @staticmethod
    def create(
        database_type: Optional[DatabaseType] = None,
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        **kwargs,
    ) -> BaseStore:
        """
        Create a vector store instance based on the database type.

        Args:
            database_type: Type of vector database to use (default: from settings)
            collection_name: Name of the collection
            embeddings: Embeddings instance for generating embeddings
            **kwargs: Additional arguments for the vector store

        Returns:
            BaseStore instance

        Raises:
            ValueError: If the database type is not supported
        """
        # Get database type from settings if not provided
        db_type = database_type or settings.database_type

        # Create embeddings if not provided
        embed_func = embeddings or EmbeddingFactory.create()

        # Create vector store based on database type
        if db_type == DatabaseType.CHROMA:
            return VectorStoreFactory._create_chroma_store(
                collection_name=collection_name, embeddings=embed_func, **kwargs
            )
        elif db_type == DatabaseType.PINECONE:
            return VectorStoreFactory._create_pinecone_store(
                collection_name=collection_name, embeddings=embed_func, **kwargs
            )
        else:
            error_message = f"Unsupported database type: {db_type}"
            logger.error(error_message)
            raise ValueError(error_message)

    @staticmethod
    def _create_chroma_store(
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        **kwargs,
    ) -> ChromaStore:
        """
        Create a Chroma vector store.

        Args:
            collection_name: Name of the collection
            embeddings: Embeddings instance for generating embeddings
            persist_directory: Directory for persisting the vector store
            **kwargs: Additional arguments for the vector store

        Returns:
            ChromaStore instance
        """
        # Use collection name from settings if not provided
        coll_name = collection_name or settings.chroma.collection_name

        logger.debug(f"Creating Chroma vector store with collection {coll_name}")

        return ChromaStore.from_existing(
            collection_name=coll_name,
            embeddings=embeddings,
            persist_directory=persist_directory,
            **kwargs,
        )

    @staticmethod
    def _create_pinecone_store(
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> PineconeStore:
        """
        Create a Pinecone vector store.

        Args:
            collection_name: Name of the collection
            embeddings: Embeddings instance for generating embeddings
            api_key: Pinecone API key
            **kwargs: Additional arguments for the vector store

        Returns:
            PineconeStore instance
        """
        # Use collection name and API key from settings if not provided
        coll_name = (
            collection_name or settings.pinecone.collection_name
            if settings.pinecone
            else None
        )
        api_key = api_key or settings.pinecone.api_key if settings.pinecone else None

        if not api_key:
            error_message = "Pinecone API key is required"
            logger.error(error_message)
            raise ValueError(error_message)

        if not coll_name:
            error_message = "Pinecone collection name is required"
            logger.error(error_message)
            raise ValueError(error_message)

        logger.debug(f"Creating Pinecone vector store with collection {coll_name}")

        return PineconeStore.from_existing(
            collection_name=coll_name, embeddings=embeddings, api_key=api_key, **kwargs
        )

    @staticmethod
    def from_documents(
        documents: List[Document],
        database_type: Optional[DatabaseType] = None,
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        **kwargs,
    ) -> BaseStore:
        """
        Create a vector store from documents.

        Args:
            documents: List of LangChain Documents
            database_type: Type of vector database to use (default: from settings)
            collection_name: Name of the collection
            embeddings: Embeddings instance for generating embeddings
            **kwargs: Additional arguments for the vector store

        Returns:
            BaseStore instance
        """
        # Get database type from settings if not provided
        db_type = database_type or settings.database_type

        # Create embeddings if not provided
        embed_func = embeddings or EmbeddingFactory.create()

        # Create vector store based on database type
        if db_type == DatabaseType.CHROMA:
            coll_name = collection_name or settings.chroma.collection_name
            return ChromaStore.from_documents(
                documents=documents,
                embeddings=embed_func,
                collection_name=coll_name,
                **kwargs,
            )
        elif db_type == DatabaseType.PINECONE:
            coll_name = (
                collection_name or settings.pinecone.collection_name
                if settings.pinecone
                else None
            )
            api_key = (
                kwargs.get("api_key") or settings.pinecone.api_key
                if settings.pinecone
                else None
            )

            if not api_key:
                error_message = "Pinecone API key is required"
                logger.error(error_message)
                raise ValueError(error_message)

            if not coll_name:
                error_message = "Pinecone collection name is required"
                logger.error(error_message)
                raise ValueError(error_message)

            return PineconeStore.from_documents(
                documents=documents,
                embeddings=embed_func,
                collection_name=coll_name,
                api_key=api_key,
                **kwargs,
            )
        else:
            error_message = f"Unsupported database type: {db_type}"
            logger.error(error_message)
            raise ValueError(error_message)
