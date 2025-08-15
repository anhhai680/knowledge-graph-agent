"""
Integration tests for RAG Agent with workflow and prompt manager integration.

This module provides integration tests for the RAG Agent,
testing its interaction with workflows and prompt managers.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agents.rag_agent import RAGAgent
from src.workflows.query_workflow import QueryWorkflow
from src.utils.prompt_manager import PromptManager
from src.workflows.workflow_states import QueryIntent


@pytest.fixture
def mock_workflow():
    """Create a mock workflow for testing."""
    workflow = AsyncMock(spec=QueryWorkflow)
    workflow.run.return_value = {
        "document_retrieval": {
            "retrieved_documents": []
        },
        "llm_generation": {
            "generated_response": "Mock response"
        },
        "query_intent": QueryIntent.CODE_SEARCH,
        "processing_time": 0.1
    }
    return workflow


@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager for testing."""
    prompt_manager = Mock(spec=PromptManager)
    prompt_manager.create_query_prompt.return_value = {
        "confidence_score": 0.8,
        "template_type": "code_search",
        "system_prompt_type": "technical"
    }
    prompt_manager.get_supported_intents.return_value = [
        QueryIntent.CODE_SEARCH,
        QueryIntent.DOCUMENTATION,
        QueryIntent.EXPLANATION,
        QueryIntent.DEBUGGING,
        QueryIntent.ARCHITECTURE
    ]
    prompt_manager.get_template_statistics.return_value = {
        "total_templates": 5,
        "template_types": ["code_search", "documentation"]
    }
    return prompt_manager


@pytest.fixture
def rag_agent(mock_workflow, mock_prompt_manager):
    """Create a RAG agent instance for testing."""
    return RAGAgent(
        workflow=mock_workflow,
        prompt_manager=mock_prompt_manager,
        default_top_k=5,
        confidence_threshold=0.3
    )


class TestRAGAgent:
    """Integration test suite for RAG Agent functionality."""

    def test_rag_agent_initialization(self, mock_workflow, mock_prompt_manager):
        """Test RAG agent initialization with workflow and prompt manager."""
        agent = RAGAgent(
            workflow=mock_workflow,
            prompt_manager=mock_prompt_manager
        )
        
        assert agent.workflow == mock_workflow
        assert agent.prompt_manager == mock_prompt_manager
        assert agent.default_top_k == 5
        assert agent.confidence_threshold == 0.3

    def test_rag_agent_initialization_with_filters(self, mock_workflow, mock_prompt_manager):
        """Test RAG agent initialization with repository and language filters."""
        agent = RAGAgent(
            workflow=mock_workflow,
            prompt_manager=mock_prompt_manager,
            repository_filter=["repo1", "repo2"],
            language_filter=["python", "javascript"]
        )
        
        assert agent.repository_filter == ["repo1", "repo2"]
        assert agent.language_filter == ["python", "javascript"]

    def test_validate_input_string(self, rag_agent):
        """Test input validation with string input."""
        assert rag_agent._validate_input("test query") is True
        assert rag_agent._validate_input("") is False
        assert rag_agent._validate_input("   ") is False

    def test_validate_input_dict(self, rag_agent):
        """Test input validation with dictionary input."""
        valid_input = {"query": "test query"}
        assert rag_agent._validate_input(valid_input) is True
        
        empty_query = {"query": ""}
        assert rag_agent._validate_input(empty_query) is True
        
        invalid_input = {"not_query": "test"}
        assert rag_agent._validate_input(invalid_input) is False

    def test_validate_input_invalid_types(self, rag_agent):
        """Test input validation with invalid types."""
        assert rag_agent._validate_input(None) is False
        assert rag_agent._validate_input(123) is False
        assert rag_agent._validate_input([]) is False

    @pytest.mark.asyncio
    async def test_process_input_string(self, rag_agent):
        """Test processing string input with workflow integration."""
        result = await rag_agent._process_input("test query")
        
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert "query_intent" in result
        assert "context_summary" in result
        assert "prompt_metadata" in result
        assert "processing_time" in result
        
        # The confidence should be from the successful workflow processing
        assert result["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_process_input_dict(self, rag_agent):
        """Test processing dictionary input with workflow integration."""
        input_data = {
            "query": "test query",
            "top_k": 10,
            "repository_filter": ["repo1"],
            "language_filter": ["python"]
        }
        
        result = await rag_agent._process_input(input_data)
        
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert "query_intent" in result
        assert "context_summary" in result
        assert "prompt_metadata" in result
        assert "processing_time" in result
        
        # The query_intent should be 'code_search' from fallback processing
        assert result["query_intent"] == QueryIntent.CODE_SEARCH

    @pytest.mark.asyncio
    async def test_fallback_processing(self, rag_agent):
        """Test fallback processing when workflow fails."""
        rag_agent.workflow.run.side_effect = Exception("Workflow error")
        
        result = await rag_agent._fallback_processing("test query")
        
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert result.get("error") is True
        assert result["confidence"] == 0.1

    def test_format_sources_empty(self, rag_agent):
        """Test formatting empty sources list."""
        sources = rag_agent._format_sources([])
        assert sources == []

    def test_format_sources_single(self, rag_agent):
        """Test formatting single source."""
        from langchain.schema import Document
        
        documents = [
            Document(
                page_content="test content",
                metadata={
                    "file_path": "test.py",
                    "repository": "test/repo",
                    "language": "python",
                    "line_start": 1,
                    "line_end": 10
                }
            )
        ]
        
        sources = rag_agent._format_sources(documents)
        
        assert len(sources) == 1
        assert sources[0]["id"] == 1
        assert sources[0]["content"] == "test content"
        assert sources[0]["metadata"]["file_path"] == "test.py"

    def test_format_sources_multiple(self, rag_agent):
        """Test formatting multiple sources."""
        from langchain.schema import Document
        
        documents = [
            Document(
                page_content="content 1",
                metadata={"file_path": "test1.py", "repository": "repo1"}
            ),
            Document(
                page_content="content 2",
                metadata={"file_path": "test2.py", "repository": "repo2"}
            )
        ]
        
        sources = rag_agent._format_sources(documents)
        
        assert len(sources) == 2
        assert sources[0]["id"] == 1
        assert sources[1]["id"] == 2

    def test_update_filters(self, rag_agent):
        """Test updating repository and language filters."""
        rag_agent.update_filters(
            repository_filter=["repo1", "repo2"],
            language_filter=["python", "javascript"]
        )
        
        assert rag_agent.repository_filter == ["repo1", "repo2"]
        assert rag_agent.language_filter == ["python", "javascript"]

    def test_update_filters_partial(self, rag_agent):
        """Test updating only one filter type."""
        rag_agent.update_filters(repository_filter=["repo1"])
        
        assert rag_agent.repository_filter == ["repo1"]
        assert rag_agent.language_filter == []  # Should remain unchanged

    @pytest.mark.asyncio
    async def test_query_with_context(self, rag_agent):
        """Test querying with provided context documents."""
        from langchain.schema import Document
        
        context_docs = [
            Document(
                page_content="context content",
                metadata={"repository": "test/repo", "language": "python"}
            )
        ]
        
        result = await rag_agent.query_with_context(
            query="test query",
            context_documents=context_docs,
            query_intent=QueryIntent.CODE_SEARCH
        )
        
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert "query_intent" in result
        assert result["query_intent"] == QueryIntent.CODE_SEARCH

    @pytest.mark.asyncio
    async def test_batch_query(self, rag_agent):
        """Test batch query processing."""
        queries = ["query 1", "query 2"]
        
        results = await rag_agent.batch_query(queries, max_concurrent=2)
        
        assert len(results) == 2
        for result in results:
            assert "answer" in result
            assert "sources" in result
            assert "confidence" in result

    @pytest.mark.asyncio
    async def test_batch_query_disabled(self):
        """Test batch query when disabled."""
        agent = RAGAgent(enable_batch_processing=False)
        
        with pytest.raises(ValueError, match="Batch processing is disabled"):
            await agent.batch_query(["query 1"], max_concurrent=1)

    @pytest.mark.asyncio
    async def test_batch_query_with_exceptions(self, rag_agent):
        """Test batch query with some queries failing."""
        rag_agent.workflow.run.side_effect = [
            Exception("Error 1"),
            {
                "document_retrieval": {"retrieved_documents": []},
                "llm_generation": {"generated_response": "Success"},
                "query_intent": QueryIntent.CODE_SEARCH,
                "processing_time": 0.1
            }
        ]
        
        queries = ["query 1", "query 2"]
        results = await rag_agent.batch_query(queries, max_concurrent=2)
        
        assert len(results) == 2
        assert results[0].get("error") is True
        assert "answer" in results[1]

    def test_get_supported_query_types(self, rag_agent):
        """Test getting supported query types."""
        types = rag_agent.get_supported_query_types()
        
        assert isinstance(types, list)
        assert "code_search" in types
        assert "documentation" in types
        assert "explanation" in types
        assert "debugging" in types
        assert "architecture" in types

    def test_get_agent_statistics(self, rag_agent):
        """Test getting agent statistics."""
        stats = rag_agent.get_agent_statistics()
        
        assert "agent_type" in stats
        assert "default_top_k" in stats
        assert "confidence_threshold" in stats
        assert "repository_filter" in stats
        assert "language_filter" in stats
        assert "batch_processing_enabled" in stats
        assert "supported_query_types" in stats
        assert stats["agent_type"] == "RAGAgent"

    def test_invoke_method(self, rag_agent):
        """Test the invoke method inherited from BaseAgent."""
        result = rag_agent.invoke("test query")
        
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result

    def test_prompt_manager_integration(self, rag_agent):
        """Test integration with prompt manager."""
        # Test that prompt manager methods are called correctly
        types = rag_agent.get_supported_query_types()
        stats = rag_agent.get_agent_statistics()
        
        assert isinstance(types, list)
        assert "prompt_manager_stats" in stats
