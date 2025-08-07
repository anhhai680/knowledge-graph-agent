"""
Unit tests for RAG Agent with comprehensive functionality testing.

This module provides comprehensive unit tests for the RAG Agent,
including input validation, query processing, and response formatting.
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
    """Test suite for RAG Agent functionality."""

    def test_agent_initialization(self, mock_workflow, mock_prompt_manager):
        """Test RAG agent initialization."""
        agent = RAGAgent(
            workflow=mock_workflow,
            prompt_manager=mock_prompt_manager
        )
        
        assert agent.workflow == mock_workflow
        assert agent.prompt_manager == mock_prompt_manager
        assert agent.default_top_k == 5
        assert agent.confidence_threshold == 0.3
        assert agent.name == "RAGAgent"

    def test_validate_input_string(self, rag_agent):
        """Test input validation with valid string."""
        assert rag_agent._validate_input("test query") is True
        assert rag_agent._validate_input("") is False
        assert rag_agent._validate_input("   ") is False

    def test_validate_input_dict(self, rag_agent):
        """Test input validation with valid dictionary."""
        valid_input = {"query": "test query"}
        assert rag_agent._validate_input(valid_input) is True
        
        # Empty query should still be valid (validation happens in processing)
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
    async def test_process_input_string_success(self, rag_agent):
        """Test processing string input successfully."""
        result = await rag_agent._process_input("test query")
        
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert "query_intent" in result
        assert "context_summary" in result
        assert "prompt_metadata" in result
        assert "processing_time" in result

    @pytest.mark.asyncio
    async def test_process_input_dict_success(self, rag_agent):
        """Test processing dictionary input successfully."""
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

    @pytest.mark.asyncio
    async def test_process_input_workflow_failure(self, rag_agent):
        """Test processing input when workflow fails."""
        rag_agent.workflow.run.side_effect = Exception("Workflow error")
        
        result = await rag_agent._process_input("test query")
        
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert result.get("error") is True

    @pytest.mark.asyncio
    async def test_process_input_no_workflow(self):
        """Test processing input without workflow."""
        agent = RAGAgent(workflow=None)
        
        result = await agent._process_input("test query")
        
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        # When no workflow is provided, a default workflow is created
        assert result.get("error") is None
        assert result.get("query_intent") is not None

    @pytest.mark.asyncio
    async def test_process_input_exception(self, rag_agent):
        """Test processing input with exception."""
        rag_agent.workflow.run.side_effect = Exception("Test exception")
        
        result = await rag_agent._process_input("test query")
        
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert result.get("error") is True

    def test_format_sources_empty(self, rag_agent):
        """Test formatting empty sources list."""
        sources = rag_agent._format_sources([])
        assert sources == []

    def test_format_sources(self, rag_agent):
        """Test formatting sources with documents."""
        from langchain.schema import Document
        
        documents = [
            Document(
                page_content="test content 1",
                metadata={
                    "file_path": "test1.py",
                    "repository": "test/repo1",
                    "language": "python",
                    "line_start": 1,
                    "line_end": 10
                }
            ),
            Document(
                page_content="test content 2",
                metadata={
                    "file_path": "test2.py",
                    "repository": "test/repo2",
                    "language": "javascript",
                    "line_start": 5,
                    "line_end": 15
                }
            )
        ]
        
        sources = rag_agent._format_sources(documents)
        
        assert len(sources) == 2
        assert sources[0]["id"] == 1
        assert sources[0]["content"] == "test content 1"
        assert sources[0]["metadata"]["file_path"] == "test1.py"
        assert sources[1]["id"] == 2
        assert sources[1]["metadata"]["language"] == "javascript"

    def test_update_filters(self, rag_agent):
        """Test updating repository and language filters."""
        rag_agent.update_filters(
            repository_filter=["repo1", "repo2"],
            language_filter=["python", "javascript"]
        )
        
        assert rag_agent.repository_filter == ["repo1", "repo2"]
        assert rag_agent.language_filter == ["python", "javascript"]

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
        queries = ["query 1", "query 2", "query 3"]
        
        results = await rag_agent.batch_query(queries, max_concurrent=2)
        
        assert len(results) == 3
        for result in results:
            assert "answer" in result
            assert "sources" in result
            assert "confidence" in result

    @pytest.mark.asyncio
    async def test_batch_workflow_query(self, rag_agent):
        """Test batch query with workflow integration."""
        queries = ["query 1", "query 2"]
        
        results = await rag_agent.batch_query(queries, max_concurrent=3)
        
        assert len(results) == 2
        assert rag_agent.workflow.run.call_count == 2

    def test_get_supported_query_types(self, rag_agent):
        """Test getting supported query types."""
        types = rag_agent.get_supported_query_types()
        
        assert isinstance(types, list)
        assert "code_search" in types
        assert "documentation" in types
        assert "explanation" in types
        assert "debugging" in types
        assert "architecture" in types

    def get_agent_statistics(self, rag_agent):
        """Test getting agent statistics."""
        stats = rag_agent.get_agent_statistics()
        
        assert "agent_type" in stats
        assert "default_top_k" in stats
        assert "confidence_threshold" in stats
        assert "repository_filter" in stats
        assert "language_filter" in stats
        assert "batch_processing_enabled" in stats
        assert "supported_query_types" in stats
        assert "prompt_manager_stats" in stats
        assert stats["agent_type"] == "RAGAgent"
