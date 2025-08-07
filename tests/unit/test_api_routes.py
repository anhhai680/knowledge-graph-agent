"""
Unit tests for API routes with comprehensive endpoint testing.

This module provides comprehensive unit tests for all API endpoints,
including request validation, response formatting, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.models import (
    IndexRepositoryRequest,
    BatchIndexRequest,
    QueryRequest,
    QueryIntent,
    SearchStrategy,
    WorkflowStatus
)


@pytest.fixture
def client():
    """Create test client for FastAPI application."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_indexing_workflow():
    """Mock indexing workflow for testing."""
    workflow = AsyncMock()
    workflow.ainvoke.return_value = {
        "processed_files": 150,
        "embeddings_generated": 500,
        "errors": []
    }
    return workflow


@pytest.fixture
def mock_query_workflow():
    """Mock query workflow for testing."""
    workflow = AsyncMock()
    # Mock the run method that's actually called in the routes
    workflow.run.return_value = {
        "query_intent": QueryIntent.CODE_SEARCH,
        "search_strategy": "hybrid", 
        "context_documents": [
            {
                "page_content": "def test_function(): pass",
                "metadata": {
                    "source": "test.py",
                    "repository": "test/repo",
                    "language": "python",
                    "score": 0.95
                }
            }
        ],
        "confidence_score": 0.85,
        "suggestions": ["Try more specific terms"],
        "generated_response": "Here's how to implement authentication...",
        "results": [
            {
                "content": "def test_function(): pass",
                "metadata": {
                    "source": "test.py",
                    "repository": "test/repo", 
                    "language": "python",
                    "score": 0.95
                }
            }
        ]
    }
    return workflow


class TestAPIRoutes:
    """Test suite for API routes."""

    def test_root_endpoint(self, client):
        """Test the root welcome endpoint."""
        response = client.get("/api/v1/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["version"] == "1.0.0"

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        with patch("src.api.main.get_indexing_workflow") as mock_indexing, \
             patch("src.api.main.get_query_workflow") as mock_query, \
             patch("src.api.main.get_vector_store") as mock_vector_store:
            
            # Mock the workflows
            mock_indexing.return_value = MagicMock()
            mock_query.return_value = MagicMock()
            
            # Mock the vector store to have a health_check method
            mock_store = MagicMock()
            mock_store.health_check.return_value = (True, "healthy")
            mock_vector_store.return_value = mock_store
            
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert "version" in data
            assert "components" in data
            assert data["version"] == "1.0.0"

    def test_stats_endpoint(self, client):
        """Test the statistics endpoint."""
        with patch("src.api.main.get_indexing_workflow"):
            response = client.get("/api/v1/stats")
            assert response.status_code == 200
            
            data = response.json()
            assert "total_repositories" in data
            assert "total_documents" in data
            assert "total_files" in data
            assert "system_health" in data
            assert isinstance(data["total_repositories"], int)

    def test_repositories_endpoint(self, client):
        """Test the repositories listing endpoint."""
        with patch("src.api.main.get_indexing_workflow"):
            response = client.get("/api/v1/repositories")
            assert response.status_code == 200
            
            data = response.json()
            assert "repositories" in data
            assert "total_count" in data
            assert "last_updated" in data
            assert isinstance(data["repositories"], list)

    @patch("src.api.main.get_indexing_workflow")
    def test_index_repository_endpoint(self, mock_workflow_dep, client, mock_indexing_workflow):
        """Test single repository indexing endpoint."""
        mock_workflow_dep.return_value = mock_indexing_workflow
        
        request_data = {
            "repository_url": "https://github.com/test/repo",
            "branch": "main",
            "force_reindex": False
        }
        
        response = client.post("/api/v1/index/repository", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "workflow_id" in data
        assert "repository" in data
        assert "status" in data
        assert data["repository"] == request_data["repository_url"]
        assert data["status"] == "pending"

    @patch("src.api.main.get_indexing_workflow")
    def test_index_repository_invalid_url(self, mock_indexing, client):
        """Test repository indexing with invalid URL."""
        mock_indexing.return_value = MagicMock()
        
        request_data = {
            "repository_url": "invalid-url",
            "branch": "main"
        }
        
        response = client.post("/api/v1/index/repository", json=request_data)
        assert response.status_code == 422  # Validation error

    @patch("src.api.main.get_query_workflow")
    def test_query_endpoint(self, mock_workflow_dep, client, mock_query_workflow):
        """Test query processing endpoint."""
        mock_workflow_dep.return_value = mock_query_workflow
        
        request_data = {
            "query": "How to implement authentication?",
            "intent": "code_search",
            "top_k": 5,
            "search_strategy": "hybrid",
            "include_metadata": True
        }
        
        response = client.post("/api/v1/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "query" in data
        assert "intent" in data
        assert "strategy" in data
        assert "results" in data
        assert "total_results" in data
        assert "processing_time" in data
        assert "confidence_score" in data
        
        assert data["query"] == request_data["query"]
        assert data["intent"] == "code_search"
        assert isinstance(data["results"], list)

    @patch("src.api.main.get_query_workflow")
    def test_query_endpoint_validation(self, mock_query, client):
        """Test query endpoint with validation errors."""
        mock_query.return_value = MagicMock()
        
        # Empty query
        response = client.post("/api/v1/query", json={"query": ""})
        assert response.status_code == 422
        
        # Query too long
        long_query = "x" * 3000
        response = client.post("/api/v1/query", json={"query": long_query})
        assert response.status_code == 422

    def test_workflow_status_endpoint(self, client):
        """Test workflow status endpoint."""
        # First create a workflow
        with patch("src.api.main.get_indexing_workflow"):
            request_data = {
                "repository_url": "https://github.com/test/repo"
            }
            response = client.post("/api/v1/index/repository", json=request_data)
            workflow_id = response.json()["workflow_id"]
        
        # Then check its status
        response = client.get(f"/api/v1/workflows/{workflow_id}/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "workflow_id" in data
        assert "workflow_type" in data
        assert "status" in data
        assert "started_at" in data
        assert data["workflow_id"] == workflow_id

    def test_workflow_status_not_found(self, client):
        """Test workflow status endpoint with non-existent workflow."""
        response = client.get("/api/v1/workflows/non-existent-id/status")
        assert response.status_code == 404

    def test_list_workflows_endpoint(self, client):
        """Test workflow listing endpoint."""
        # Create some workflows first
        with patch("src.api.main.get_indexing_workflow"):
            for i in range(3):
                request_data = {
                    "repository_url": f"https://github.com/test/repo{i}"
                }
                client.post("/api/v1/index/repository", json=request_data)
        
        # List all workflows
        response = client.get("/api/v1/workflows")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 3
        
        # Test with status filter
        response = client.get("/api/v1/workflows?status=pending")
        assert response.status_code == 200

    @patch("builtins.open")
    @patch("json.load")
    @patch("src.api.main.get_indexing_workflow")
    def test_batch_index_endpoint(self, mock_workflow_dep, mock_json_load, mock_open, client, mock_indexing_workflow):
        """Test batch indexing endpoint."""
        mock_workflow_dep.return_value = mock_indexing_workflow
        
        # Mock appSettings.json content
        mock_json_load.return_value = {
            "repositories": [
                {"url": "https://github.com/test/repo1", "branch": "main"},
                {"url": "https://github.com/test/repo2", "branch": "develop"}
            ]
        }
        
        response = client.post("/api/v1/index")
        assert response.status_code == 200
        
        data = response.json()
        assert "workflows" in data
        assert "batch_id" in data
        assert "total_repositories" in data
        assert len(data["workflows"]) == 2
        assert data["total_repositories"] == 2

    @patch("src.api.main.get_indexing_workflow")
    @patch("builtins.open")
    def test_batch_index_no_settings_file(self, mock_open, mock_indexing, client):
        """Test batch indexing when appSettings.json is missing."""
        mock_indexing.return_value = MagicMock()
        mock_open.side_effect = FileNotFoundError()
        
        response = client.post("/api/v1/index")
        assert response.status_code == 404

    @patch("src.api.main.get_indexing_workflow")
    @patch("builtins.open")
    @patch("json.load")
    def test_batch_index_empty_repositories(self, mock_json_load, mock_open, mock_indexing, client):
        """Test batch indexing with empty repositories list."""
        mock_indexing.return_value = MagicMock()
        mock_json_load.return_value = {"repositories": []}
        
        response = client.post("/api/v1/index")
        assert response.status_code == 400


class TestAPIModels:
    """Test suite for API models validation."""

    def test_index_repository_request_validation(self):
        """Test IndexRepositoryRequest validation."""
        # Valid request
        request = IndexRepositoryRequest(
            repository_url="https://github.com/test/repo",
            branch="main"
        )
        assert request.repository_url == "https://github.com/test/repo"
        assert request.branch == "main"
        assert request.force_reindex is False

    def test_index_repository_request_invalid_url(self):
        """Test IndexRepositoryRequest with invalid URL."""
        with pytest.raises(ValueError):
            IndexRepositoryRequest(
                repository_url="not-a-github-url"
            )

    def test_query_request_validation(self):
        """Test QueryRequest validation."""
        # Valid request
        request = QueryRequest(
            query="Test query",
            intent=QueryIntent.CODE_SEARCH,
            top_k=10,
            search_strategy=SearchStrategy.HYBRID
        )
        assert request.query == "Test query"
        assert request.intent == QueryIntent.CODE_SEARCH
        assert request.top_k == 10
        assert request.search_strategy == SearchStrategy.HYBRID

    def test_query_request_length_validation(self):
        """Test QueryRequest query length validation."""
        # Too short query (empty)
        with pytest.raises(ValueError):
            QueryRequest(query="")
        
        # Too long query
        with pytest.raises(ValueError):
            QueryRequest(query="x" * 3000)

    def test_batch_index_request_validation(self):
        """Test BatchIndexRequest validation."""
        repositories = [
            IndexRepositoryRequest(repository_url="https://github.com/test/repo1"),
            IndexRepositoryRequest(repository_url="https://github.com/test/repo2")
        ]
        
        request = BatchIndexRequest(
            repositories=repositories,
            parallel_jobs=2
        )
        assert len(request.repositories) == 2
        assert request.parallel_jobs == 2

    def test_batch_index_request_empty_repositories(self):
        """Test BatchIndexRequest with empty repositories."""
        with pytest.raises(ValueError):
            BatchIndexRequest(repositories=[])


class TestAPIIntegration:
    """Integration tests for API functionality."""

    @patch("src.api.main.get_indexing_workflow")
    @patch("src.api.main.get_query_workflow")
    def test_full_workflow_integration(self, mock_query_dep, mock_index_dep, client, mock_indexing_workflow, mock_query_workflow):
        """Test full workflow from indexing to querying."""
        mock_index_dep.return_value = mock_indexing_workflow
        mock_query_dep.return_value = mock_query_workflow
        
        # 1. Index a repository
        index_request = {
            "repository_url": "https://github.com/test/repo",
            "branch": "main"
        }
        index_response = client.post("/api/v1/index/repository", json=index_request)
        assert index_response.status_code == 200
        
        workflow_id = index_response.json()["workflow_id"]
        
        # 2. Check workflow status
        status_response = client.get(f"/api/v1/workflows/{workflow_id}/status")
        assert status_response.status_code == 200
        
        # 3. Query the indexed content
        query_request = {
            "query": "How to implement authentication?",
            "repositories": ["test/repo"],
            "top_k": 5
        }
        query_response = client.post("/api/v1/query", json=query_request)
        assert query_response.status_code == 200
        
        query_data = query_response.json()
        assert len(query_data["results"]) > 0
        assert query_data["confidence_score"] > 0

    @patch("src.api.main.get_indexing_workflow")
    def test_error_handling(self, mock_indexing, client):
        """Test API error handling."""
        mock_indexing.return_value = MagicMock()
        
        # Test 404 for non-existent workflow
        response = client.get("/api/v1/workflows/non-existent/status")
        assert response.status_code == 404
        
        # Test validation error for invalid request
        response = client.post("/api/v1/index/repository", json={})
        assert response.status_code == 422
        
        # Test invalid query parameters
        response = client.get("/api/v1/workflows?status=invalid_status")
        # Should still return 200 but with empty results
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
