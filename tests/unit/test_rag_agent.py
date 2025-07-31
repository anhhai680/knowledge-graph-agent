"""
Unit tests for RAG agent implementation.

This module contains tests for the RAGAgent class and its integration
with query workflows and prompt manager.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain.schema import Document

from src.agents.rag_agent import RAGAgent
from src.utils.prompt_manager import PromptManager
from src.workflows.query_workflow import QueryWorkflow
from src.workflows.workflow_states import QueryIntent, QueryState


@pytest.fixture
def mock_prompt_manager():
    """Create a mock PromptManager."""
    manager = MagicMock(spec=PromptManager)
    manager.create_query_prompt.return_value = {
        "template_type": "ChatPromptTemplate",
        "confidence_score": 0.8,
        "system_prompt_type": QueryIntent.CODE_SEARCH,
        "metadata": {"test": "value"},
    }
    manager.get_supported_intents.return_value = [
        QueryIntent.CODE_SEARCH,
        QueryIntent.DOCUMENTATION,
        QueryIntent.EXPLANATION,
    ]
    manager.get_template_statistics.return_value = {
        "system_prompts": 6,
        "query_templates": 3,
        "supported_intents": 5,
    }
    manager._create_error_recovery_prompt.return_value = {
        "template_type": "error_recovery",
        "metadata": {"recovery_mode": True},
    }
    return manager


class TestRAGAgent:
    """Test cases for RAGAgent functionality."""

    def test_rag_agent_initialization(self, mock_prompt_manager):
        """Test RAG agent initialization."""
        agent = RAGAgent(
            prompt_manager=mock_prompt_manager,
            default_top_k=5,
            confidence_threshold=0.3,
        )
        
        assert agent.name == "RAGAgent"
        assert agent.default_top_k == 5
        assert agent.confidence_threshold == 0.3
        assert agent.repository_filter == []
        assert agent.language_filter == []
        assert agent.prompt_manager == mock_prompt_manager

    def test_rag_agent_initialization_with_filters(self, mock_prompt_manager):
        """Test RAG agent initialization with filters."""
        agent = RAGAgent(
            prompt_manager=mock_prompt_manager,
            repository_filter=["repo1", "repo2"],
            language_filter=["python", "javascript"],
        )
        
        assert agent.repository_filter == ["repo1", "repo2"]
        assert agent.language_filter == ["python", "javascript"]

    def test_validate_input_string(self, mock_prompt_manager):
        """Test input validation for string queries."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        assert agent._validate_input("test query") is True
        assert agent._validate_input("") is False
        assert agent._validate_input("   ") is False

    def test_validate_input_dict(self):
        """Test input validation for structured queries."""
        agent = RAGAgent()
        
        # Valid structured query
        valid_query = {
            "query": "test query",
            "k": 5,
            "repository_filter": ["repo1"],
            "language_filter": ["python"],
        }
        assert agent._validate_input(valid_query) is True
        
        # Missing query field
        invalid_query = {"k": 5}
        assert agent._validate_input(invalid_query) is False
        
        # Empty query
        empty_query = {"query": ""}
        assert agent._validate_input(empty_query) is False
        
        # Invalid k value
        invalid_k = {"query": "test", "k": -1}
        assert agent._validate_input(invalid_k) is False

    def test_validate_input_invalid_types(self):
        """Test input validation for invalid types."""
        agent = RAGAgent()
        
        assert agent._validate_input(123) is False
        assert agent._validate_input(None) is False
        assert agent._validate_input([]) is False

    @pytest.mark.asyncio
    async def test_process_input_string_success(self):
        """Test processing string input successfully."""
        mock_workflow = MagicMock(spec=QueryWorkflow)
        mock_workflow.invoke.return_value = {
            "status": "completed",
            "response": "Test response",
            "retrieved_documents": [
                Document(page_content="test content", metadata={"file_path": "test.py"})
            ],
            "query_intent": "code_search",
            "search_strategy": "semantic",
            "response_quality_score": 0.85,
            "processing_time": 1.5,
            "completed_steps": ["parse_query", "vector_search"],
            "errors": [],
        }
        
        agent = RAGAgent(workflow=mock_workflow)
        result = await agent._process_input("test query")
        
        assert result["success"] is True
        assert result["data"]["answer"] == "Test response"
        assert len(result["data"]["sources"]) == 1
        assert result["data"]["metadata"]["query_intent"] == "code_search"
        assert result["data"]["metadata"]["documents_retrieved"] == 1

    @pytest.mark.asyncio
    async def test_process_input_dict_success(self):
        """Test processing structured input successfully."""
        mock_workflow = MagicMock(spec=QueryWorkflow)
        mock_workflow.invoke.return_value = {
            "status": "completed",
            "response": "Test response",
            "retrieved_documents": [],
            "query_intent": "documentation",
            "search_strategy": "hybrid",
            "response_quality_score": 0.75,
            "processing_time": 2.0,
            "completed_steps": ["parse_query"],
            "errors": [],
        }
        
        agent = RAGAgent(workflow=mock_workflow)
        input_data = {
            "query": "test query",
            "k": 10,
            "repository_filter": ["repo1"],
            "include_metadata": False,
        }
        
        result = await agent._process_input(input_data)
        
        assert result["success"] is True
        assert result["data"]["answer"] == "Test response"
        assert result["data"]["sources"] == []

    @pytest.mark.asyncio
    async def test_process_input_workflow_failure(self):
        """Test processing when workflow fails."""
        mock_workflow = MagicMock(spec=QueryWorkflow)
        mock_workflow.invoke.return_value = {
            "status": "failed",
            "error": "Vector store connection failed",
            "errors": ["Connection timeout"],
        }
        
        agent = RAGAgent(workflow=mock_workflow)
        result = await agent._process_input("test query")
        
        assert result["success"] is False
        assert "Vector store connection failed" in result["error"]
        assert result["metadata"]["workflow_status"] == "failed"

    @pytest.mark.asyncio
    async def test_process_input_no_workflow(self):
        """Test processing without workflow (fallback mode)."""
        agent = RAGAgent(workflow=None)
        result = await agent._process_input("test query")
        
        assert result["success"] is False
        assert "RAG workflow not available" in result["error"]
        assert result["metadata"]["fallback_mode"] is True

    @pytest.mark.asyncio
    async def test_process_input_exception(self):
        """Test processing with exception."""
        mock_workflow = MagicMock(spec=QueryWorkflow)
        mock_workflow.invoke.side_effect = Exception("Test exception")
        
        agent = RAGAgent(workflow=mock_workflow)
        result = await agent._process_input("test query")
        
        assert result["success"] is False
        assert "Internal error during query processing" in result["error"]

    def test_format_sources(self):
        """Test source formatting."""
        agent = RAGAgent()
        documents = [
            Document(
                page_content="Short content",
                metadata={
                    "file_path": "src/test.py",
                    "repository": "test/repo",
                    "language": "python",
                    "chunk_type": "function",
                },
            ),
            Document(
                page_content="Long content " * 100,  # Exceeds 500 chars
                metadata={"file_path": "src/long.py"},
            ),
        ]
        
        sources = agent._format_sources(documents)
        
        assert len(sources) == 2
        assert sources[0]["source_id"] == 1
        assert sources[0]["content"] == "Short content"
        assert sources[0]["file_path"] == "src/test.py"
        assert sources[0]["repository"] == "test/repo"
        assert sources[0]["language"] == "python"
        assert sources[0]["chunk_type"] == "function"
        
        assert sources[1]["source_id"] == 2
        assert sources[1]["content"].endswith("...")  # Truncated
        assert sources[1]["file_path"] == "src/long.py"

    def test_format_sources_empty(self):
        """Test source formatting with empty list."""
        agent = RAGAgent()
        sources = agent._format_sources([])
        
        assert sources == []

    def test_update_filters(self):
        """Test filter updates."""
        agent = RAGAgent()
        
        # Update repository filter
        agent.update_filters(repository_filter=["repo1", "repo2"])
        assert agent.repository_filter == ["repo1", "repo2"]
        assert agent.language_filter == []  # Unchanged
        
        # Update language filter
        agent.update_filters(language_filter=["python", "javascript"])
        assert agent.repository_filter == ["repo1", "repo2"]  # Unchanged
        assert agent.language_filter == ["python", "javascript"]
        
        # Update both
        agent.update_filters(
            repository_filter=["repo3"],
            language_filter=["typescript"],
        )
        assert agent.repository_filter == ["repo3"]
        assert agent.language_filter == ["typescript"]

    
@pytest.mark.asyncio
    async def test_query_with_context(self, mock_prompt_manager):
        """Test context-specific querying."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        context_docs = [
            Document(
                page_content="def test_function():
    return True",
                metadata={
                    "file_path": "test.py",
                    "repository": "specific_repo",
                    "language": "python",
                },
            ),
        ]
        
        result = await agent.query_with_context(
            query="test query",
            context_documents=context_docs,
            query_intent=QueryIntent.CODE_SEARCH,
        )
        
        assert "answer" in result
        assert len(result["sources"]) == 1
        assert result["query_intent"] == QueryIntent.CODE_SEARCH

    @pytest.mark.asyncio
    async def test_batch_query(self, mock_prompt_manager):
        """Test batch query processing."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        queries = ["query 1", "query 2"]
        
        results = await agent.batch_query(queries, max_concurrent=2)
        
        assert len(results) == 2
        assert all("answer" in result for result in results)

    @pytest.mark.asyncio
    async def test_batch_query(self):
        """Test batch query processing."""
        mock_workflow = MagicMock(spec=QueryWorkflow)
        mock_workflow.invoke.return_value = {
            "status": "completed",
            "response": "Batch response",
            "retrieved_documents": [],
            "query_intent": "code_search",
            "search_strategy": "semantic",
            "response_quality_score": 0.8,
            "processing_time": 1.0,
            "completed_steps": [],
            "errors": [],
        }
        
        agent = RAGAgent(workflow=mock_workflow)
        queries = ["query 1", "query 2"]
        shared_context = {"repository_filter": ["repo1"], "k": 6}
        
        results = await agent.batch_query(queries, shared_context)
        
        assert len(results) == 2
        assert all(result["success"] for result in results)

    def test_get_supported_query_types(self):
        """Test supported query types."""
        agent = RAGAgent()
        types = agent.get_supported_query_types()
        
        assert "code_search" in types
        assert "documentation_search" in types
        assert "concept_explanation" in types
        assert "implementation_help" in types
        assert "troubleshooting" in types

    def test_get_agent_statistics(self):
        """Test agent statistics."""
        agent = RAGAgent(
            default_k=8,
            max_k=25,
            repository_filter=["repo1"],
            language_filter=["python"],
        )
        
        stats = agent.get_agent_statistics()
        
        assert stats["agent_name"] == "RAGAgent"
        assert stats["configuration"]["default_k"] == 8
        assert stats["configuration"]["max_k"] == 25
        assert stats["configuration"]["repository_filter"] == ["repo1"]
        assert stats["configuration"]["language_filter"] == ["python"]
        assert "supported_query_types" in stats
