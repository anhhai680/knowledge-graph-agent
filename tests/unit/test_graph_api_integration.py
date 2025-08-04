"""
Unit tests for graph API integration.

This module tests the integration between the graph store and API endpoints.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.utils.feature_flags import FeatureFlags


class TestGraphAPIIntegration:
    """Test cases for graph API integration."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @patch('src.utils.feature_flags.FeatureFlags.is_enabled')
    def test_graph_query_endpoint_disabled(self, mock_is_enabled, client):
        """Test graph query endpoint when features are disabled."""
        mock_is_enabled.return_value = False
        
        response = client.post(
            "/api/v1/graph/query",
            json={"query": "MATCH (n) RETURN n LIMIT 5"}
        )
        
        assert response.status_code == 400
        assert "Graph features are not enabled" in response.json()["detail"]
    
    @patch('src.utils.feature_flags.FeatureFlags.is_enabled')
    @patch('src.api.main.get_graph_store')
    def test_graph_query_endpoint_enabled(self, mock_get_graph_store, mock_is_enabled, client):
        """Test graph query endpoint when features are enabled."""
        mock_is_enabled.return_value = True
        
        # Mock graph store
        from src.api.models import GraphQueryResult
        
        mock_graph_store = Mock()
        mock_graph_store.execute_query.return_value = GraphQueryResult(
            data=[{"node": {"id": 1, "name": "test"}}],
            metadata={"node_count": 1, "relationship_count": 0},
            execution_time_ms=10.5,
            query="MATCH (n) RETURN n LIMIT 5"
        )
        mock_get_graph_store.return_value = mock_graph_store
        
        response = client.post(
            "/api/v1/graph/query",
            json={"query": "MATCH (n) RETURN n LIMIT 5"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["query"] == "MATCH (n) RETURN n LIMIT 5"
        assert data["result"]["data"] == [{"node": {"id": 1, "name": "test"}}]
    
    @patch('src.utils.feature_flags.FeatureFlags.is_enabled')
    def test_graph_info_endpoint_disabled(self, mock_is_enabled, client):
        """Test graph info endpoint when features are disabled."""
        mock_is_enabled.return_value = False
        
        response = client.get("/api/v1/graph/info")
        
        assert response.status_code == 400
        assert "Graph features are not enabled" in response.json()["detail"]
    
    @patch('src.utils.feature_flags.FeatureFlags.is_enabled')
    @patch('src.api.main.get_graph_store')
    def test_graph_info_endpoint_enabled(self, mock_get_graph_store, mock_is_enabled, client):
        """Test graph info endpoint when features are enabled."""
        mock_is_enabled.return_value = True
        
        # Mock graph store
        mock_graph_store = Mock()
        mock_graph_store.get_graph_info.return_value = {
            "connected": True,
            "node_count": 100,
            "relationship_count": 250,
            "database_type": "MemGraph"
        }
        mock_get_graph_store.return_value = mock_graph_store
        
        response = client.get("/api/v1/graph/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["node_count"] == 100
        assert data["relationship_count"] == 250
        assert data["database_type"] == "MemGraph"
    
    def test_graph_query_validation(self, client):
        """Test graph query validation."""
        # Test invalid query
        response = client.post(
            "/api/v1/graph/query",
            json={"query": "INVALID QUERY"}
        )
        
        assert response.status_code == 400  # Bad request due to validation error
    
    def test_graph_query_request_model(self):
        """Test graph query request model validation."""
        from src.api.models import GraphQueryRequest
        
        # Valid request
        valid_request = GraphQueryRequest(
            query="MATCH (n) RETURN n LIMIT 5",
            parameters={"limit": 5},
            timeout_seconds=30
        )
        assert valid_request.query == "MATCH (n) RETURN n LIMIT 5"
        assert valid_request.parameters == {"limit": 5}
        assert valid_request.timeout_seconds == 30
        
        # Test validation error for invalid query
        with pytest.raises(ValueError, match="Query must be a valid Cypher query"):
            GraphQueryRequest(query="INVALID QUERY")
    
    def test_graph_query_response_model(self):
        """Test graph query response model."""
        from src.api.models import GraphQueryResponse, GraphQueryResult
        
        result = GraphQueryResult(
            data=[{"node": {"id": 1}}],
            metadata={"node_count": 1},
            execution_time_ms=10.5,
            query="MATCH (n) RETURN n"
        )
        
        response = GraphQueryResponse(
            success=True,
            result=result,
            error=None,
            processing_time=0.1,
            query="MATCH (n) RETURN n"
        )
        
        assert response.success is True
        assert response.result.data == [{"node": {"id": 1}}]
        assert response.error is None
        assert response.processing_time == 0.1 