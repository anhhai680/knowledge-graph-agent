"""
Document processor module for the Knowledge Graph Agent.

This module provides functionality for processing documents with language-aware chunking.
"""

from typing import List, Optional

from langchain.schema import Document
from loguru import logger

from src.config.settings import settings
from src.processors.chunking_strategy import (
    get_chunking_strategy,
)
from src.processors.metadata_extractor import extract_metadata


class DocumentProcessor:
    """
    Process documents with language-aware chunking.

    This class provides methods for chunking documents based on their language
    and extracting metadata.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of each chunk (default: from settings)
            chunk_overlap: Overlap between chunks (default: from settings)
        """
        self.chunk_size = chunk_size or settings.document_processing.chunk_size
        self.chunk_overlap = chunk_overlap or settings.document_processing.chunk_overlap

        logger.debug(
            f"Initialized document processor with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        )

    def process_document(self, document: Document) -> List[Document]:
        """
        Process a document with language-aware chunking.

        Args:
            document: LangChain Document to process

        Returns:
            List of processed Document objects
        """
        # Extract language from metadata
        language = document.metadata.get("language", "unknown")
        file_path = document.metadata.get("file_path", "")

        logger.debug(
            f"Processing document with language={language}, file_path={file_path}"
        )

        # Get appropriate chunking strategy
        chunking_strategy = get_chunking_strategy(
            language, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        # Split document into chunks
        chunks = chunking_strategy.split_document(document)

        # Extract metadata for each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Extract additional metadata from the chunk content
            additional_metadata = extract_metadata(
                chunk.page_content, language, file_path, chunk_index=i
            )

            # Update chunk metadata
            updated_metadata = {**chunk.metadata, **additional_metadata}
            processed_chunk = Document(
                page_content=chunk.page_content, metadata=updated_metadata
            )

            processed_chunks.append(processed_chunk)

        logger.debug(f"Processed document into {len(processed_chunks)} chunks")

        return processed_chunks

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process a list of documents with language-aware chunking.

        Args:
            documents: List of LangChain Documents to process

        Returns:
            List of processed Document objects
        """
        all_chunks = []

        for document in documents:
            try:
                chunks = self.process_document(document)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                file_path = document.metadata.get("file_path", "unknown")
                logger.error(f"Failed to process document: {file_path}")

        logger.info(
            f"Processed {len(documents)} documents into {len(all_chunks)} chunks"
        )

        return all_chunks
