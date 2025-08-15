"""
Unit tests for LangChain Prompt Manager Integration.

This module contains tests for the PromptManager class and its integration
with LangChain PromptTemplate components.
"""

import pytest
from unittest.mock import MagicMock, patch

from langchain.schema import Document

from src.utils.prompt_manager import PromptManager, CodeQueryResponse
from src.workflows.workflow_states import QueryIntent


class TestPromptManager:
    """Test cases for PromptManager functionality."""

    def test_prompt_manager_initialization(self):
        """Test prompt manager initialization."""
        manager = PromptManager()
        
        assert manager.base_system_prompt is not None
        assert len(manager.intent_system_prompts) == 5
        assert manager.main_query_template is not None
        assert manager.output_parser is not None

    def test_supported_intents(self):
        """Test supported query intents."""
        manager = PromptManager()
        intents = manager.get_supported_intents()
        
        expected_intents = [
            QueryIntent.CODE_SEARCH,
            QueryIntent.DOCUMENTATION,
            QueryIntent.EXPLANATION,
            QueryIntent.DEBUGGING,
            QueryIntent.ARCHITECTURE,
        ]
        
        assert len(intents) == len(expected_intents)
        assert all(intent in intents for intent in expected_intents)

    def test_format_context_documents_empty(self):
        """Test formatting empty document list."""
        manager = PromptManager()
        result = manager._format_context_documents([])
        
        assert result == "No relevant context documents found."

    def test_format_context_documents_single(self):
        """Test formatting single document."""
        manager = PromptManager()
        doc = Document(
            page_content="def test_function():\n    return True",
            metadata={
                "file_path": "test.py",
                "repository": "test/repo",
                "language": "python",
                "chunk_type": "function",
            },
        )
        
        result = manager._format_context_documents([doc])
        
        assert "**Source 1: test.py**" in result
        assert "Repository: test/repo" in result
        assert "Language: python" in result
        assert "Type: function" in result
        assert "def test_function()" in result

    def test_format_context_documents_multiple(self):
        """Test formatting multiple documents."""
        manager = PromptManager()
        docs = [
            Document(
                page_content="class TestClass:\n    pass",
                metadata={
                    "file_path": "test1.py",
                    "repository": "test/repo",
                    "language": "python",
                    "chunk_type": "class",
                },
            ),
            Document(
                page_content="function testFunction() { return true; }",
                metadata={
                    "file_path": "test2.js",
                    "repository": "test/repo",
                    "language": "javascript",
                    "chunk_type": "function",
                },
            ),
        ]
        
        result = manager._format_context_documents(docs)
        
        assert "**Source 1: test1.py**" in result
        assert "**Source 2: test2.js**" in result
        assert "class TestClass" in result
        assert "function testFunction" in result

    def test_format_context_documents_truncation(self):
        """Test document content truncation."""
        manager = PromptManager()
        # Create content that exceeds the 2000 character limit
        long_content = "def long_function():\n    " + "# very long comment line that takes up significant space\n    " * 50 + "return True"
        doc = Document(
            page_content=long_content,
            metadata={
                "file_path": "long.py",
                "repository": "test/repo",
                "language": "python",
                "chunk_type": "function",
            },
        )
        
        result = manager._format_context_documents([doc])
        
        # If content is very long, it should be truncated
        if len(long_content) > 2000:
            assert "[truncated]" in result
        assert len(result) > 0  # Should have some content

    def test_assess_context_confidence_empty(self):
        """Test confidence assessment with empty documents."""
        manager = PromptManager()
        confidence = manager._assess_context_confidence([], "test query")
        
        assert confidence == 0.0

    def test_assess_context_confidence_single_doc(self):
        """Test confidence assessment with single document."""
        manager = PromptManager()
        doc = Document(
            page_content="def test():\n    return True",
            metadata={
                "file_path": "test.py",
                "repository": "test/repo",
                "language": "python",
                "chunk_type": "function",
            },
        )
        
        confidence = manager._assess_context_confidence([doc], "test query")
        
        assert 0.0 < confidence < 1.0

    def test_assess_context_confidence_multiple_docs(self):
        """Test confidence assessment with multiple documents."""
        manager = PromptManager()
        docs = [
            Document(
                page_content="def test1():\n    return True",
                metadata={
                    "file_path": "test1.py",
                    "repository": "test/repo",
                    "language": "python",
                    "chunk_type": "function",
                },
            ),
            Document(
                page_content="def test2():\n    return False",
                metadata={
                    "file_path": "test2.py",
                    "repository": "test/repo",
                    "language": "python",
                    "chunk_type": "function",
                },
            ),
        ]
        
        confidence = manager._assess_context_confidence(docs, "test query")
        
        # Multiple docs should have higher confidence than single doc
        single_confidence = manager._assess_context_confidence([docs[0]], "test query")
        assert confidence > single_confidence

    def test_create_query_prompt_no_context(self):
        """Test query prompt creation with no context."""
        manager = PromptManager()
        result = manager.create_query_prompt(
            query="What is this function?",
            context_documents=[],
        )
        
        assert result["template_type"] == "ChatPromptTemplate"
        assert result["confidence_score"] == 0.0
        assert result["context_documents_count"] == 0
        assert "Question" in str(result["prompt"])  # Should contain the formatted question

    def test_create_query_prompt_with_context(self):
        """Test query prompt creation with context."""
        manager = PromptManager()
        docs = [
            Document(
                page_content="def hello_world():\n    return 'Hello, World!'",
                metadata={
                    "file_path": "hello.py",
                    "repository": "test/repo",
                    "language": "python",
                    "chunk_type": "function",
                },
            ),
        ]
        
        result = manager.create_query_prompt(
            query="What does this function do?",
            context_documents=docs,
            query_intent=QueryIntent.CODE_SEARCH,
        )
        
        assert result["template_type"] == "ChatPromptTemplate"
        assert result["confidence_score"] > 0.0
        assert result["context_documents_count"] == 1
        assert result["system_prompt_type"] == QueryIntent.CODE_SEARCH

    def test_create_query_prompt_with_filters(self):
        """Test query prompt creation with filters."""
        manager = PromptManager()
        docs = [
            Document(
                page_content="function test() { return true; }",
                metadata={
                    "file_path": "test.js",
                    "repository": "frontend/repo",
                    "language": "javascript",
                    "chunk_type": "function",
                },
            ),
        ]
        
        result = manager.create_query_prompt(
            query="How does this work?",
            context_documents=docs,
            query_intent=QueryIntent.EXPLANATION,
            repository_filter=["frontend/repo"],
            language_filter=["javascript"],
            top_k=8,
        )
        
        assert result["metadata"]["repository_filter"] == ["frontend/repo"]
        assert result["metadata"]["language_filter"] == ["javascript"]
        assert result["metadata"]["top_k"] == 8
        assert result["metadata"]["query_intent"] == QueryIntent.EXPLANATION

    def test_create_query_prompt_high_confidence(self):
        """Test query prompt creation with high confidence context."""
        manager = PromptManager()
        
        # Create multiple documents with rich content for high confidence
        docs = []
        for i in range(5):
            doc_content = f"def function_{i}():\n    " + "# detailed implementation\n    " * 20 + f"return {i}"
            docs.append(
                Document(
                    page_content=doc_content,
                    metadata={
                        "file_path": f"module_{i}.py",
                        "repository": "test/repo",
                        "language": "python",
                        "chunk_type": "function",
                    },
                )
            )
        
        result = manager.create_query_prompt(
            query="Explain these functions",
            context_documents=docs,
            confidence_threshold=0.5,
        )
        
        # Should have high confidence due to multiple docs with rich metadata
        assert result["confidence_score"] >= 0.5
        assert result["context_documents_count"] == 5

    def test_create_response_formatting_prompt(self):
        """Test response formatting prompt creation."""
        manager = PromptManager()
        source_docs = [
            Document(
                page_content="def example():\n    pass",
                metadata={
                    "file_path": "example.py",
                    "repository": "test/repo",
                    "language": "python",
                },
            ),
        ]
        
        result = manager.create_response_formatting_prompt(
            raw_response="This function does nothing but demonstrates syntax.",
            source_documents=source_docs,
            query_intent=QueryIntent.EXPLANATION,
        )
        
        assert result["template_type"] == "response_formatting"
        assert result["source_count"] == 1
        assert result["metadata"]["query_intent"] == QueryIntent.EXPLANATION
        assert result["metadata"]["formatting_mode"] is True

    def test_create_error_recovery_prompt(self):
        """Test error recovery prompt creation."""
        manager = PromptManager()
        result = manager._create_error_recovery_prompt(
            query="Test query",
            error_info="Test error occurred",
        )
        
        assert result["template_type"] == "error_recovery"
        assert result["confidence_score"] == 0.1
        assert result["metadata"]["recovery_mode"] is True
        assert "Test error occurred" in result["metadata"]["error"]

    def test_get_template_statistics(self):
        """Test template statistics retrieval."""
        manager = PromptManager()
        stats = manager.get_template_statistics()
        
        assert stats["system_prompts"] == 6  # 5 intent + 1 base
        assert stats["query_templates"] == 3
        assert stats["fallback_templates"] == 2
        assert len(stats["supported_intents"]) == 5
        assert stats["output_parser"] == "PydanticOutputParser"

    @patch('src.utils.prompt_manager.logger')
    def test_create_query_prompt_exception_handling(self, mock_logger):
        """Test exception handling in query prompt creation."""
        manager = PromptManager()
        
        # Create a scenario that would trigger exception handling
        # by using extremely long query that might cause issues
        extremely_long_query = "What is this function? " * 10000  # Very long query
        
        result = manager.create_query_prompt(
            query=extremely_long_query,
            context_documents=[],
        )
        
        # Should still return a valid result even with long query
        assert result["template_type"] in ["ChatPromptTemplate", "error_recovery"]
        assert "confidence_score" in result

    def test_code_query_response_model(self):
        """Test CodeQueryResponse model validation."""
        response = CodeQueryResponse(
            answer="Test answer",
            confidence=0.85,
            sources_used=["file1.py", "file2.js"],
            recommendations=["Use better variable names"],
        )
        
        assert response.answer == "Test answer"
        assert response.confidence == 0.85
        assert len(response.sources_used) == 2
        assert response.recommendations is not None
        assert len(response.recommendations) == 1

    def test_code_query_response_model_optional_fields(self):
        """Test CodeQueryResponse model with optional fields."""
        response = CodeQueryResponse(
            answer="Test answer",
            confidence=0.75,
            sources_used=["file1.py"],
        )
        
        assert response.recommendations is None

    def test_query_intent_coverage(self):
        """Test that all query intents have corresponding system prompts."""
        manager = PromptManager()
        
        # Check that supported QueryIntent values have corresponding system prompts
        # Note: EVENT_FLOW intent is not currently supported in the PromptManager
        supported_intents = [
            QueryIntent.CODE_SEARCH,
            QueryIntent.DOCUMENTATION,
            QueryIntent.EXPLANATION,
            QueryIntent.DEBUGGING,
            QueryIntent.ARCHITECTURE
        ]
        
        for intent in supported_intents:
            assert intent in manager.intent_system_prompts, f"Missing system prompt for {intent}"
        
        # Verify that EVENT_FLOW is not supported (as per current implementation)
        assert QueryIntent.EVENT_FLOW not in manager.intent_system_prompts, "EVENT_FLOW should not be supported in current implementation"

    def test_prompt_template_types(self):
        """Test that prompt templates are of correct LangChain types."""
        from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
        
        manager = PromptManager()
        
        # Check system prompt types
        assert isinstance(manager.base_system_prompt, SystemMessagePromptTemplate)
        for intent_prompt in manager.intent_system_prompts.values():
            assert isinstance(intent_prompt, SystemMessagePromptTemplate)
        
        # Check query template types
        assert isinstance(manager.main_query_template, ChatPromptTemplate)
        assert isinstance(manager.high_confidence_template, ChatPromptTemplate)
        assert isinstance(manager.low_confidence_template, ChatPromptTemplate)
        assert isinstance(manager.no_context_template, ChatPromptTemplate)
        assert isinstance(manager.error_recovery_template, ChatPromptTemplate)
