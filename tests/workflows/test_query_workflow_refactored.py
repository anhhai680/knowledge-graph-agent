"""
Unit tests for refactored QueryWorkflow.

Tests the main QueryWorkflow class to ensure backward compatibility
and proper integration with the modular orchestrator.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.workflows.query_workflow import QueryWorkflow, execute_query
from src.workflows.workflow_states import (
    QueryState, 
    ProcessingStatus,
    create_query_state
)


class TestQueryWorkflowRefactored:
    """Test suite for refactored QueryWorkflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.workflow = QueryWorkflow(
            collection_name="test-collection",
            default_k=4
        )

    def test_initialization(self):
        """Test workflow initialization."""
        assert self.workflow.collection_name == "test-collection"
        assert self.workflow.default_k == 4
        assert self.workflow.orchestrator is not None
        
        # Verify backward compatibility components
        assert self.workflow.vector_store_factory is not None
        assert self.workflow.embedding_factory is not None
        assert self.workflow.llm_factory is not None

    def test_define_steps(self):
        """Test that workflow defines steps via orchestrator."""
        steps = self.workflow.define_steps()
        expected_steps = [
            "parse_and_analyze",
            "search_documents", 
            "process_context",
            "generate_response",
            "finalize_response"
        ]
        assert steps == expected_steps

    def test_validate_state(self):
        """Test state validation via orchestrator."""
        state = create_query_state(
            workflow_id="test-123",
            original_query="test query"
        )
        assert self.workflow.validate_state(dict(state)) is True

    def test_execute_step(self):
        """Test step execution via orchestrator."""
        with patch.object(self.workflow.orchestrator, 'execute_step') as mock_execute:
            mock_execute.return_value = {"test": "result"}
            
            state = {"original_query": "test"}
            result = self.workflow.execute_step("parse_and_analyze", state)
            
            assert result == {"test": "result"}
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_method(self):
        """Test main run method delegates to orchestrator."""
        with patch.object(self.workflow.orchestrator, 'execute_workflow') as mock_execute:
            mock_state = create_query_state(
                workflow_id="test-123",
                original_query="test query"
            )
            mock_state["status"] = ProcessingStatus.COMPLETED
            mock_state["total_query_time"] = 1.5
            mock_state["llm_generation"] = {"generated_response": "test response"}
            mock_execute.return_value = mock_state
            
            result = await self.workflow.run(
                query="test query",
                repositories=["repo1"],
                languages=["python"],
                k=5
            )
            
            assert result["original_query"] == "test query"
            assert result["status"] == ProcessingStatus.COMPLETED
            mock_execute.assert_called_once_with(
                query="test query",
                repositories=["repo1"],
                languages=["python"],
                file_types=None,
                k=5
            )

    @pytest.mark.asyncio
    async def test_run_method_error_handling(self):
        """Test error handling in run method."""
        with patch.object(self.workflow.orchestrator, 'execute_workflow') as mock_execute:
            mock_execute.side_effect = Exception("Test error")
            
            with pytest.raises(Exception, match="Test error"):
                await self.workflow.run(query="test query")

    def test_backward_compatibility_get_vector_store(self):
        """Test backward compatibility for _get_vector_store method."""
        with patch.object(self.workflow.orchestrator.search_handler, '_get_vector_store') as mock_get:
            mock_vector_store = Mock()
            mock_get.return_value = mock_vector_store
            
            result = self.workflow._get_vector_store()
            
            assert result == mock_vector_store
            mock_get.assert_called_once()

    def test_backward_compatibility_get_llm(self):
        """Test backward compatibility for _get_llm method."""
        with patch.object(self.workflow.orchestrator.llm_handler, '_get_llm') as mock_get:
            mock_llm = Mock()
            mock_get.return_value = mock_llm
            
            result = self.workflow._get_llm()
            
            assert result == mock_llm
            mock_get.assert_called_once()

    def test_backward_compatibility_get_embeddings(self):
        """Test backward compatibility for _get_embeddings method."""
        with patch.object(self.workflow.embedding_factory, 'create') as mock_create:
            mock_embeddings = Mock()
            mock_create.return_value = mock_embeddings
            
            result = self.workflow._get_embeddings()
            
            assert result == mock_embeddings
            mock_create.assert_called_once()

    def test_backward_compatibility_get_retriever(self):
        """Test backward compatibility for _get_retriever method."""
        with patch.object(self.workflow, '_get_vector_store') as mock_get_vs:
            mock_vector_store = Mock()
            mock_vector_store.similarity_search.return_value = ["doc1", "doc2"]
            mock_get_vs.return_value = mock_vector_store
            
            retriever = self.workflow._get_retriever(k=5, filter_dict={"test": "filter"})
            documents = retriever("test query")
            
            assert documents == ["doc1", "doc2"]
            mock_vector_store.similarity_search.assert_called_once_with(
                "test query", k=5, filter={"test": "filter"}
            )

    def test_collection_name_determination(self):
        """Test collection name determination logic."""
        # Test with explicit collection name
        workflow1 = QueryWorkflow(collection_name="explicit-collection")
        assert workflow1.collection_name == "explicit-collection"
        
        # Test with default collection name (would use settings)
        workflow2 = QueryWorkflow()
        assert workflow2.collection_name is not None

    def test_configuration_parameters(self):
        """Test configuration parameter handling."""
        workflow = QueryWorkflow(
            default_k=8,
            max_k=50,
            min_context_length=200,
            max_context_length=10000,
            response_quality_threshold=0.8
        )
        
        assert workflow.default_k == 8
        assert workflow.max_k == 50
        assert workflow.min_context_length == 200
        assert workflow.max_context_length == 10000
        assert workflow.response_quality_threshold == 0.8
        
        # Verify orchestrator gets same configuration
        assert workflow.orchestrator.default_k == 8
        assert workflow.orchestrator.max_k == 50


class TestExecuteQueryHelper:
    """Test suite for execute_query helper function."""

    @pytest.mark.asyncio
    async def test_execute_query_success(self):
        """Test successful query execution."""
        with patch('src.workflows.query_workflow.QueryWorkflow') as mock_workflow_class:
            # Mock workflow instance
            mock_workflow = Mock()
            mock_workflow_class.return_value = mock_workflow
            
            # Mock run method
            mock_state = create_query_state(
                workflow_id="test-123",
                original_query="test query"
            )
            mock_state["llm_generation"] = {"generated_response": "test response"}
            mock_state["response_sources"] = [{"file_path": "test.py"}]
            mock_state["response_quality_score"] = 0.9
            mock_state["total_query_time"] = 2.5
            mock_state["document_retrieval"] = {"retrieved_documents": ["doc1", "doc2"]}
            
            mock_workflow.run = AsyncMock(return_value=mock_state)
            
            result = await execute_query(
                query="test query",
                repositories=["repo1"],
                languages=["python"]
            )
            
            assert result["query"] == "test query"
            assert result["response"] == "test response"
            assert result["sources"] == [{"file_path": "test.py"}]
            assert result["quality_score"] == 0.9
            assert result["processing_time"] == 2.5
            assert result["documents_retrieved"] == 2
            assert result["workflow_id"] == "test-123"

    @pytest.mark.asyncio
    async def test_execute_query_with_kwargs(self):
        """Test query execution with additional kwargs."""
        with patch('src.workflows.query_workflow.QueryWorkflow') as mock_workflow_class:
            mock_workflow = Mock()
            mock_workflow_class.return_value = mock_workflow
            
            mock_state = create_query_state(
                workflow_id="test-123",
                original_query="test query"
            )
            mock_state["llm_generation"] = {"generated_response": "response"}
            mock_state["response_sources"] = []
            mock_state["response_quality_score"] = 0.8
            mock_state["total_query_time"] = 1.0
            mock_state["document_retrieval"] = {"retrieved_documents": []}
            
            mock_workflow.run = AsyncMock(return_value=mock_state)
            
            result = await execute_query(
                query="test query",
                default_k=10,
                max_k=30
            )
            
            # Verify workflow was created with kwargs
            mock_workflow_class.assert_called_once_with(default_k=10, max_k=30)
            
            # Verify run was called with correct parameters
            mock_workflow.run.assert_called_once_with(
                query="test query",
                repositories=None,
                languages=None,
                default_k=10,
                max_k=30
            )
