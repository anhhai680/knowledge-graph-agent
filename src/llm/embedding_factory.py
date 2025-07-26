"""
Embedding factory module for the Knowledge Graph Agent.

This module provides a factory pattern for creating embedding instances.
"""

from typing import Dict, Any, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings.base import Embeddings
from loguru import logger

from src.config.settings import EmbeddingProvider, settings


class EmbeddingFactory:
    """
    Factory class for creating embedding instances.
    
    This class provides methods for creating different types of embedding instances
    based on the application configuration.
    """
    
    @staticmethod
    def create(
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        callbacks: Optional[list[BaseCallbackHandler]] = None,
        **kwargs
    ) -> Embeddings:
        """
        Create an embedding instance based on the provider specified in settings.
        
        Args:
            model_name: Name of the model to use (default: from settings)
            batch_size: Batch size for embedding generation (default: from settings)
            callbacks: List of callback handlers (default: None)
            **kwargs: Additional arguments to pass to the embedding model
            
        Returns:
            LangChain Embeddings instance
            
        Raises:
            ValueError: If the embedding provider is not supported
        """
        # Get parameters from settings if not provided
        model_name = model_name or settings.embedding.model
        batch_size = batch_size or settings.embedding.batch_size
        
        # Create embedding based on provider
        if settings.embedding.provider == EmbeddingProvider.OPENAI:
            return EmbeddingFactory._create_openai_embedding(
                model_name=model_name,
                batch_size=batch_size,
                callbacks=callbacks,
                **kwargs
            )
        else:
            error_message = f"Unsupported embedding provider: {settings.embedding.provider}"
            logger.error(error_message)
            raise ValueError(error_message)
    
    @staticmethod
    def _create_openai_embedding(
        model_name: str,
        batch_size: int,
        callbacks: Optional[list[BaseCallbackHandler]] = None,
        **kwargs
    ) -> Embeddings:
        """
        Create an OpenAI embedding instance.
        
        Args:
            model_name: Name of the model to use
            batch_size: Batch size for embedding generation
            callbacks: List of callback handlers
            **kwargs: Additional arguments to pass to the embedding model
            
        Returns:
            LangChain OpenAI embeddings instance
        """
        from langchain_openai import OpenAIEmbeddings
        
        logger.debug(f"Creating OpenAI embeddings with model {model_name}")
        
        return OpenAIEmbeddings(
            model=model_name,
            chunk_size=batch_size,
            openai_api_key=settings.openai.api_key,
            base_url=settings.llm_api_base_url,
            default_headers={"User-Agent": "Knowledge-Graph-Agent"},
            **kwargs
        )
    
    @staticmethod
    def batch_embed_documents(
        documents: List[str],
        embeddings: Optional[Embeddings] = None,
        batch_size: Optional[int] = None,
        max_tokens_per_batch: Optional[int] = None,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Embed a list of documents in batches to avoid rate limits.
        
        Args:
            documents: List of documents to embed
            embeddings: Embeddings instance (default: create a new one)
            batch_size: Batch size for embedding generation (default: from settings)
            max_tokens_per_batch: Maximum tokens per batch (default: from settings)
            show_progress: Whether to show progress (default: True)
            
        Returns:
            List of embeddings
        """
        import math
        from tqdm import tqdm
        
        # Get parameters from settings if not provided
        batch_size = batch_size or settings.embedding.batch_size
        max_tokens_per_batch = max_tokens_per_batch or settings.embedding.max_tokens_per_batch
        
        # Create embeddings if not provided
        if embeddings is None:
            embeddings = EmbeddingFactory.create()
        
        # Calculate approximate token counts for each document
        # This is a rough estimate: ~4 chars per token for English text
        doc_token_counts = [math.ceil(len(doc) / 4) for doc in documents]
        
        # Group documents into batches based on token counts and batch size
        batches = []
        current_batch = []
        current_batch_tokens = 0
        
        for doc, token_count in zip(documents, doc_token_counts):
            # If adding this document would exceed max_tokens_per_batch, finalize the current batch
            if current_batch and current_batch_tokens + token_count > max_tokens_per_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_tokens = 0
            
            # Add document to the current batch
            current_batch.append(doc)
            current_batch_tokens += token_count
            
            # If current batch reaches batch_size, finalize it
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
                current_batch_tokens = 0
        
        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)
        
        # Process batches
        all_embeddings = []
        total_batches = len(batches)
        
        logger.info(f"Embedding {len(documents)} documents in {total_batches} batches")
        
        # Show progress if requested
        batch_iterator = tqdm(batches, desc="Generating embeddings") if show_progress else batches
        
        for i, batch in enumerate(batch_iterator):
            try:
                batch_embeddings = embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"Embedded batch {i+1}/{total_batches} ({len(batch)} documents)")
            except Exception as e:
                logger.error(f"Error embedding batch {i+1}/{total_batches}: {str(e)}")
                # Retry with smaller batches
                half_size = len(batch) // 2
                if half_size > 0:
                    logger.info(f"Retrying with smaller batches of size {half_size}")
                    first_half = batch[:half_size]
                    second_half = batch[half_size:]
                    
                    try:
                        first_embeddings = embeddings.embed_documents(first_half)
                        all_embeddings.extend(first_embeddings)
                        logger.debug(f"Embedded first half of batch {i+1} ({len(first_half)} documents)")
                    except Exception as e2:
                        logger.error(f"Error embedding first half of batch {i+1}: {str(e2)}")
                        # If still failing, embed one by one
                        for doc in first_half:
                            try:
                                single_embedding = embeddings.embed_documents([doc])
                                all_embeddings.extend(single_embedding)
                            except Exception as e3:
                                logger.error(f"Error embedding individual document: {str(e3)}")
                                # Add a zero vector as placeholder
                                all_embeddings.append([0.0] * 1536)  # Assuming OpenAI's embedding size
                    
                    try:
                        second_embeddings = embeddings.embed_documents(second_half)
                        all_embeddings.extend(second_embeddings)
                        logger.debug(f"Embedded second half of batch {i+1} ({len(second_half)} documents)")
                    except Exception as e2:
                        logger.error(f"Error embedding second half of batch {i+1}: {str(e2)}")
                        # If still failing, embed one by one
                        for doc in second_half:
                            try:
                                single_embedding = embeddings.embed_documents([doc])
                                all_embeddings.extend(single_embedding)
                            except Exception as e3:
                                logger.error(f"Error embedding individual document: {str(e3)}")
                                # Add a zero vector as placeholder
                                all_embeddings.append([0.0] * 1536)  # Assuming OpenAI's embedding size
                else:
                    # If batch size is 1, we can't split further, so add a zero vector
                    logger.error(f"Cannot split batch further, adding zero vectors for {len(batch)} documents")
                    all_embeddings.extend([[0.0] * 1536] * len(batch))  # Assuming OpenAI's embedding size
        
        logger.info(f"Embedded {len(all_embeddings)}/{len(documents)} documents")
        
        return all_embeddings
