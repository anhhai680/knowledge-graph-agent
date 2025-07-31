"""
Unit tests for RAG agent implementation.

This module contains tests for the RAGAgent class and its integration
with query workflows and prompt manager.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

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


@pytest.fixture  
def mock_workflow():
    """Create a mock QueryWorkflow."""
    workflow = MagicMock(spec=QueryWorkflow)
    workflow.ainvoke = AsyncMock(return_value={
        "retrieved_documents": [
            Document(
                page_content="def example_function():\n    return 'Hello, World!'",
                metadata={
                    "file_path": "example.py",
                    "repository": "test/repo",
                    "language": "python",
                    "chunk_type": "function",
                },
            ),
        ],
        "answer": "This is a test function that returns a greeting.",
        "processing_time": 0.1,
    })
    return workflow


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

    def test_validate_input_dict(self, mock_prompt_manager):
        """Test input validation for dictionary queries."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        valid_dict = {"query": "test query", "top_k": 5}
        invalid_dict = {"top_k": 5}  # Missing required 'query' field
        
        assert agent._validate_input(valid_dict) is True
        assert agent._validate_input(invalid_dict) is False

    def test_validate_input_invalid_types(self, mock_prompt_manager):
        """Test input validation for invalid types."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        assert agent._validate_input(None) is False
        assert agent._validate_input(123) is False
        assert agent._validate_input([]) is False

    @pytest.mark.asyncio
    async def test_process_input_string(self, mock_prompt_manager):
        """Test processing string input."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        result = await agent._process_input("test query")
        
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert result["confidence"] == 0.8  # From mock prompt manager
        mock_prompt_manager.create_query_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_input_dict(self, mock_prompt_manager):
        """Test processing dictionary input."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        input_dict = {
            "query": "test query",
            "top_k": 8,
            "query_intent": QueryIntent.DOCUMENTATION,
        }
        
        result = await agent._process_input(input_dict)
        
        assert "answer" in result
        assert result["query_intent"] == QueryIntent.DOCUMENTATION
        mock_prompt_manager.create_query_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_processing(self, mock_prompt_manager):
        """Test fallback processing when main pipeline fails."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        result = await agent._fallback_processing("test query")
        
        assert result["error"] is True
        assert result["confidence"] == 0.1
        assert "apologize" in result["answer"].lower()
        mock_prompt_manager._create_error_recovery_prompt.assert_called_once()

    def test_format_sources_empty(self, mock_prompt_manager):
        """Test formatting empty source list."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        sources = agent._format_sources([])
        
        assert sources == []

    def test_format_sources_single(self, mock_prompt_manager):
        """Test formatting single source document."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        doc = Document(
            page_content="def test():\n    return True",
            metadata={
                "file_path": "test.py",
                "repository": "test/repo",
                "language": "python",
                "chunk_type": "function",
            },
        )
        
        sources = agent._format_sources([doc])
        
        assert len(sources) == 1
        assert sources[0]["id"] == 1
        assert "def test()" in sources[0]["content"]
        assert sources[0]["metadata"]["file_path"] == "test.py"

    def test_format_sources_multiple(self, mock_prompt_manager):
        """Test formatting multiple source documents."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        docs = [
            Document(
                page_content="def test1():\n    return True",
                metadata={"file_path": "test1.py", "repository": "repo1"},
            ),
            Document(
                page_content="def test2():\n    return False",
                metadata={"file_path": "test2.py", "repository": "repo2"},
            ),
        ]
        
        sources = agent._format_sources(docs)
        
        assert len(sources) == 2
        assert sources[0]["id"] == 1
        assert sources[1]["id"] == 2
        assert sources[0]["metadata"]["file_path"] == "test1.py"
        assert sources[1]["metadata"]["file_path"] == "test2.py"

    def test_update_filters(self, mock_prompt_manager):
        """Test updating filter criteria."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        agent.update_filters(
            repository_filter=["new_repo"],
            language_filter=["typescript"],
        )
        
        assert agent.repository_filter == ["new_repo"]
        assert agent.language_filter == ["typescript"]

    def test_update_filters_partial(self, mock_prompt_manager):
        """Test updating only some filters."""
        agent = RAGAgent(
            prompt_manager=mock_prompt_manager,
            repository_filter=["old_repo"],
            language_filter=["python"],
        )
        
        agent.update_filters(repository_filter=["new_repo"])
        
        assert agent.repository_filter == ["new_repo"]
        assert agent.language_filter == ["python"]  # Should remain unchanged

    @pytest.mark.asyncio
    async def test_query_with_context(self, mock_prompt_manager):
        """Test query processing with provided context."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        context_docs = [
            Document(
                page_content="def test_function():\n    return True",
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
        assert result["context_summary"]["documents_found"] == 1

    @pytest.mark.asyncio
    async def test_batch_query(self, mock_prompt_manager):
        """Test batch query processing."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        queries = ["query 1", "query 2"]  # List of strings which is valid Union[str, Dict[str, Any]]
        
        results = await agent.batch_query(queries, max_concurrent=2)
        
        assert len(results) == 2
        assert all("answer" in result for result in results)
        assert mock_prompt_manager.create_query_prompt.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_query_disabled(self, mock_prompt_manager):
        """Test batch query when disabled."""
        agent = RAGAgent(
            prompt_manager=mock_prompt_manager,
            enable_batch_processing=False,
        )
        queries = ["query 1", "query 2"]
        
        with pytest.raises(ValueError, match="Batch processing is disabled"):
            await agent.batch_query(queries)

    @pytest.mark.asyncio
    async def test_batch_query_with_exceptions(self, mock_prompt_manager):
        """Test batch query handling exceptions."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        # Mock one query to raise an exception
        with patch.object(agent, '_process_input') as mock_process:
            mock_process.side_effect = [
                {"answer": "success", "sources": []},  # First query succeeds
                Exception("Test error"),  # Second query fails
            ]
            
            queries = ["query 1", "query 2"]
            results = await agent.batch_query(queries)
            
            assert len(results) == 2
            assert results[0]["answer"] == "success"
            assert results[1]["error"] is True

    def test_get_supported_query_types(self, mock_prompt_manager):
        """Test getting supported query types."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        query_types = agent.get_supported_query_types()
        
        expected_types = [intent.value for intent in mock_prompt_manager.get_supported_intents.return_value]
        assert query_types == expected_types

    def test_get_agent_statistics(self, mock_prompt_manager):
        """Test getting agent statistics."""
        agent = RAGAgent(
            prompt_manager=mock_prompt_manager,
            default_top_k=10,
            confidence_threshold=0.5,
        )
        
        stats = agent.get_agent_statistics()
        
        assert stats["agent_type"] == "RAGAgent"
        assert stats["default_top_k"] == 10
        assert stats["confidence_threshold"] == 0.5
        assert "prompt_manager_stats" in stats

    @pytest.mark.asyncio
    async def test_invoke_method(self, mock_prompt_manager):
        """Test the invoke method from BaseAgent."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        result = await agent.ainvoke("test query")
        
        # BaseAgent.ainvoke returns an AgentResponse, which contains success and data
        assert "answer" in result
        mock_prompt_manager.create_query_prompt.assert_called_once()

    def test_prompt_manager_integration(self, mock_prompt_manager):
        """Test integration with prompt manager."""
        agent = RAGAgent(prompt_manager=mock_prompt_manager)
        
        # Verify prompt manager is properly integrated
        assert agent.prompt_manager == mock_prompt_manager
        
        # Test that prompt manager methods are accessible
        intents = agent.get_supported_query_types()
        assert len(intents) > 0
        
        stats = agent.get_agent_statistics()
        assert "prompt_manager_stats" in stats
