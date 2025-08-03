"""
Test Chroma dimension mismatch fixes.

This module tests the fixes for Chroma vector store dimension mismatch issues.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.vectorstores.chroma_store import ChromaStore
from src.llm.embedding_factory import EmbeddingFactory


class TestChromaDimensionFixes:
    """Test Chroma dimension mismatch fixes."""

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings with 1536 dimensions."""
        mock_emb = Mock()
        mock_emb.embed_query.return_value = [0.1] * 1536  # 1536 dimensions
        mock_emb.embed_documents.return_value = [[0.1] * 1536]  # 1536 dimensions
        return mock_emb

    @pytest.fixture
    def mock_chroma_client(self):
        """Create mock Chroma client."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.get_collection.return_value = Mock(metadata={"dimension": 1536})
        return mock_client, mock_collection

    def test_get_collection_stats_with_none_metadata(self, mock_embeddings, mock_chroma_client):
        """Test get_collection_stats handles None metadata gracefully."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock collection with None metadata
        mock_collection_info = Mock()
        mock_collection_info.metadata = None
        mock_client.get_collection.return_value = mock_collection_info
        
        store = ChromaStore(client=mock_client, embeddings=mock_embeddings)
        
        stats = store.get_collection_stats()
        
        assert stats["name"] == "knowledge-base-graph"
        assert stats["count"] == 100
        assert stats["dimension"] is None
        assert "error" not in stats

    def test_get_collection_stats_with_valid_metadata(self, mock_embeddings, mock_chroma_client):
        """Test get_collection_stats with valid metadata."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock collection with valid metadata
        mock_collection_info = Mock()
        mock_collection_info.metadata = {"dimension": 1536}
        mock_client.get_collection.return_value = mock_collection_info
        
        store = ChromaStore(client=mock_client, embeddings=mock_embeddings)
        
        stats = store.get_collection_stats()
        
        assert stats["name"] == "knowledge-base-graph"
        assert stats["count"] == 100
        assert stats["dimension"] == 1536

    def test_check_embedding_dimension_compatibility_match(self, mock_embeddings, mock_chroma_client):
        """Test embedding dimension compatibility when dimensions match."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock collection with matching dimension
        mock_collection_info = Mock()
        mock_collection_info.metadata = {"dimension": 1536}
        mock_client.get_collection.return_value = mock_collection_info
        
        store = ChromaStore(client=mock_client, embeddings=mock_embeddings)
        
        is_compatible, msg = store.check_embedding_dimension_compatibility()
        
        assert is_compatible is True
        assert "Embedding dimensions match: 1536" in msg

    def test_check_embedding_dimension_compatibility_mismatch(self, mock_embeddings, mock_chroma_client):
        """Test embedding dimension compatibility when dimensions don't match."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock collection with different dimension
        mock_collection_info = Mock()
        mock_collection_info.metadata = {"dimension": 384}
        mock_client.get_collection.return_value = mock_collection_info
        
        store = ChromaStore(client=mock_client, embeddings=mock_embeddings)
        
        is_compatible, msg = store.check_embedding_dimension_compatibility()
        
        assert is_compatible is False
        assert "Dimension mismatch: expected 384, got 1536" in msg

    def test_get_dimension_mismatch_guidance(self, mock_embeddings, mock_chroma_client):
        """Test dimension mismatch guidance."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock collection with different dimension
        mock_collection_info = Mock()
        mock_collection_info.metadata = {"dimension": 384}
        mock_client.get_collection.return_value = mock_collection_info
        
        store = ChromaStore(client=mock_client, embeddings=mock_embeddings)
        
        guidance = store.get_dimension_mismatch_guidance()
        
        assert guidance["current_dimension"] == 1536
        assert guidance["expected_dimension"] == 384
        assert guidance["has_mismatch"] is True
        assert len(guidance["solutions"]) > 0

    def test_recreate_collection_with_correct_dimension(self, mock_embeddings, mock_chroma_client):
        """Test recreating collection with correct dimension."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock successful recreation
        mock_client.delete_collection.return_value = None
        mock_client.create_collection.return_value = mock_collection
        
        store = ChromaStore(client=mock_client, embeddings=mock_embeddings)
        
        success = store.recreate_collection_with_correct_dimension()
        
        assert success is True
        mock_client.delete_collection.assert_called_once_with("knowledge-base-graph")
        mock_client.create_collection.assert_called_once_with(
            name="knowledge-base-graph",
            metadata={"dimension": 1536}
        )

    def test_health_check_with_dimension_mismatch(self, mock_embeddings, mock_chroma_client):
        """Test health check detects dimension mismatch."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock collection with different dimension
        mock_collection_info = Mock()
        mock_collection_info.metadata = {"dimension": 384}
        mock_client.get_collection.return_value = mock_collection_info
        
        store = ChromaStore(client=mock_client, embeddings=mock_embeddings)
        
        is_healthy, msg = store.health_check()
        
        assert is_healthy is False
        assert "dimension mismatch" in msg.lower()

    def test_health_check_without_dimension_info(self, mock_embeddings, mock_chroma_client):
        """Test health check when no dimension information is available."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock collection with no dimension info
        mock_collection_info = Mock()
        mock_collection_info.metadata = None
        mock_client.get_collection.return_value = mock_collection_info
        
        store = ChromaStore(client=mock_client, embeddings=mock_embeddings)
        
        is_healthy, msg = store.health_check()
        
        assert is_healthy is True
        assert "healthy" in msg.lower()

    def test_get_repository_metadata_with_dimension_mismatch(self, mock_embeddings, mock_chroma_client):
        """Test get_repository_metadata handles dimension mismatch gracefully."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock collection with different dimension
        mock_collection_info = Mock()
        mock_collection_info.metadata = {"dimension": 384}
        mock_client.get_collection.return_value = mock_collection_info
        
        store = ChromaStore(client=mock_client, embeddings=mock_embeddings)
        
        repositories = store.get_repository_metadata()
        
        # Should return empty list when dimension mismatch is detected
        assert repositories == []


if __name__ == "__main__":
    pytest.main([__file__]) 