"""
Unit tests for GenericQAAgent.

Tests the Generic Q&A Agent functionality including question classification,
response generation, and template management.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from src.agents.generic_qa_agent import (
    GenericQAAgent, 
    QuestionCategory, 
    GenericQAResponse
)
from src.workflows.generic_qa_workflow import GenericQAWorkflow


class TestGenericQAAgent:
    """Test cases for GenericQAAgent."""
    
    @pytest.fixture
    def mock_workflow(self):
        """Create mock workflow for testing."""
        workflow = Mock(spec=GenericQAWorkflow)
        workflow.invoke = Mock(return_value={
            "answer": "Test answer",
            "analysis_components": ["test_component"],
            "confidence_score": 0.8,
            "template_used": "test_template",
            "metadata": {"test": "data"}
        })
        return workflow
    
    @pytest.fixture
    def qa_agent(self, mock_workflow):
        """Create GenericQAAgent instance for testing."""
        return GenericQAAgent(workflow=mock_workflow)
    
    def test_question_classification_business_capability(self, qa_agent):
        """Test classification of business capability questions."""
        question = "What business capability does this service own?"
        category = qa_agent.classify_question(question)
        assert category == QuestionCategory.BUSINESS_CAPABILITY
    
    def test_question_classification_api_endpoints(self, qa_agent):
        """Test classification of API endpoint questions."""
        question = "What are the main API endpoints and their status codes?"
        category = qa_agent.classify_question(question)
        assert category == QuestionCategory.API_ENDPOINTS
    
    def test_question_classification_data_modeling(self, qa_agent):
        """Test classification of data modeling questions."""
        question = "How is data modeling and persistence implemented?"
        category = qa_agent.classify_question(question)
        assert category == QuestionCategory.DATA_MODELING
    
    def test_question_classification_workflows(self, qa_agent):
        """Test classification of workflow questions."""
        question = "What's the end-to-end workflow for creating entities?"
        category = qa_agent.classify_question(question)
        assert category == QuestionCategory.WORKFLOWS
    
    def test_question_classification_architecture(self, qa_agent):
        """Test classification of architecture questions."""
        question = "What's the overall architecture and layer structure?"
        category = qa_agent.classify_question(question)
        assert category == QuestionCategory.ARCHITECTURE
    
    def test_question_classification_default(self, qa_agent):
        """Test default classification for unclear questions."""
        question = "Random unclear question without keywords"
        category = qa_agent.classify_question(question)
        assert category == QuestionCategory.ARCHITECTURE  # Default category
    
    def test_get_supported_categories(self, qa_agent):
        """Test getting supported categories."""
        categories = qa_agent.get_supported_categories()
        
        assert len(categories) == 5
        assert all("id" in cat for cat in categories)
        assert all("name" in cat for cat in categories)
        assert all("description" in cat for cat in categories)
        assert all("examples" in cat for cat in categories)
        
        # Check specific categories exist
        category_ids = [cat["id"] for cat in categories]
        expected_ids = [
            QuestionCategory.BUSINESS_CAPABILITY,
            QuestionCategory.API_ENDPOINTS,
            QuestionCategory.DATA_MODELING,
            QuestionCategory.WORKFLOWS,
            QuestionCategory.ARCHITECTURE
        ]
        
        for expected_id in expected_ids:
            assert expected_id in category_ids
    
    def test_get_available_templates(self, qa_agent):
        """Test getting available templates."""
        templates = qa_agent.get_available_templates()
        
        assert len(templates) >= 3  # At least the basic templates
        assert all("id" in template for template in templates)
        assert all("name" in template for template in templates)
        assert all("description" in template for template in templates)
        assert all("categories" in template for template in templates)
        assert all("file_patterns" in template for template in templates)
        
        # Check for expected templates
        template_ids = [template["id"] for template in templates]
        expected_templates = [
            "dotnet_clean_architecture",
            "react_spa", 
            "python_fastapi"
        ]
        
        for expected_template in expected_templates:
            assert expected_template in template_ids
    
    @pytest.mark.asyncio
    async def test_process_input_string(self, qa_agent, mock_workflow):
        """Test processing string input."""
        question = "What business capability does this service own?"
        
        result = await qa_agent._process_input(question)
        
        assert result["success"] is True
        assert result["question"] == question
        assert result["category"] == QuestionCategory.BUSINESS_CAPABILITY.value
        assert result["answer"] == "Test answer"
        assert result["agent"] == "GenericQAAgent"
        
        # Verify workflow was called
        mock_workflow.invoke.assert_called_once()
        call_args = mock_workflow.invoke.call_args[0][0]
        assert call_args["question"] == question
        assert call_args["category"] == QuestionCategory.BUSINESS_CAPABILITY.value
    
    @pytest.mark.asyncio
    async def test_process_input_dict(self, qa_agent, mock_workflow):
        """Test processing dictionary input."""
        input_data = {
            "question": "What are the API endpoints?",
            "category": "api_endpoints",
            "repository_context": {"test": "context"}
        }
        
        result = await qa_agent._process_input(input_data)
        
        assert result["success"] is True
        assert result["question"] == input_data["question"]
        assert result["category"] == "api_endpoints"
        
        # Verify workflow was called with correct parameters
        mock_workflow.invoke.assert_called_once()
        call_args = mock_workflow.invoke.call_args[0][0]
        assert call_args["question"] == input_data["question"]
        assert call_args["category"] == "api_endpoints"
        assert call_args["repository_context"] == input_data["repository_context"]
    
    @pytest.mark.asyncio
    async def test_process_input_without_workflow(self):
        """Test processing without workflow (fallback mode)."""
        qa_agent = GenericQAAgent()  # No workflow
        question = "What business capability does this service own?"
        
        result = await qa_agent._process_input(question)
        
        assert result["success"] is True
        assert result["question"] == question
        assert result["category"] == QuestionCategory.BUSINESS_CAPABILITY.value
        assert "fallback" in result["answer"] or "workflow" in result["answer"]
        assert result["confidence_score"] == 0.5  # Fallback confidence
    
    @pytest.mark.asyncio
    async def test_process_input_empty_question(self, qa_agent):
        """Test processing empty question."""
        result = await qa_agent._process_input("")
        
        assert result["success"] is False
        assert "Question is required" in result["error"]
    
    @pytest.mark.asyncio
    async def test_process_input_workflow_error(self, qa_agent):
        """Test handling workflow errors."""
        qa_agent.workflow.invoke.side_effect = Exception("Workflow error")
        
        question = "Test question"
        result = await qa_agent._process_input(question)
        
        assert result["success"] is False
        assert "Workflow error" in result["error"]
    
    def test_validate_input_string(self, qa_agent):
        """Test input validation for strings."""
        assert qa_agent._validate_input("Valid question") is True
        assert qa_agent._validate_input("") is False
        assert qa_agent._validate_input("   ") is False
    
    def test_validate_input_dict(self, qa_agent):
        """Test input validation for dictionaries."""
        valid_dict = {"question": "Valid question"}
        invalid_dict_empty = {"question": ""}
        invalid_dict_missing = {"other": "data"}
        
        assert qa_agent._validate_input(valid_dict) is True
        assert qa_agent._validate_input(invalid_dict_empty) is False
        assert qa_agent._validate_input(invalid_dict_missing) is False
    
    def test_validate_input_invalid_types(self, qa_agent):
        """Test input validation for invalid types."""
        assert qa_agent._validate_input(123) is False
        assert qa_agent._validate_input(None) is False
        assert qa_agent._validate_input([]) is False
    
    def test_generate_fallback_answer(self, qa_agent):
        """Test fallback answer generation."""
        question = "Test question"
        category = QuestionCategory.BUSINESS_CAPABILITY
        
        answer = qa_agent._generate_fallback_answer(question, category)
        
        assert isinstance(answer, str)
        assert len(answer) > 50  # Should be substantial
        assert "business capabilities" in answer.lower()
        assert category.value in answer
    
    @pytest.mark.asyncio
    async def test_analyze_project_structure(self, qa_agent):
        """Test project structure analysis."""
        repository_path = "/test/path"
        
        result = await qa_agent.analyze_project_structure(repository_path)
        
        assert result["success"] is True
        assert result["analysis"]["repository_path"] == repository_path
        assert result["analysis"]["analysis_depth"] == "standard"
        assert "supported_categories" in result["analysis"]
    
    @pytest.mark.asyncio
    async def test_analyze_project_structure_with_depth(self, qa_agent):
        """Test project structure analysis with specific depth."""
        repository_path = "/test/path"
        analysis_depth = "comprehensive"
        
        result = await qa_agent.analyze_project_structure(
            repository_path, 
            analysis_depth
        )
        
        assert result["success"] is True
        assert result["analysis"]["analysis_depth"] == analysis_depth
    
    def test_generic_qa_response_creation(self):
        """Test GenericQAResponse dataclass creation."""
        response = GenericQAResponse(
            category=QuestionCategory.ARCHITECTURE,
            question="Test question",
            answer="Test answer",
            analysis_components=["component1", "component2"],
            confidence_score=0.85,
            template_used="test_template",
            metadata={"key": "value"}
        )
        
        assert response.category == QuestionCategory.ARCHITECTURE
        assert response.question == "Test question"
        assert response.answer == "Test answer"
        assert len(response.analysis_components) == 2
        assert response.confidence_score == 0.85
        assert response.template_used == "test_template"
        assert response.metadata["key"] == "value"


class TestQuestionCategory:
    """Test cases for QuestionCategory enum."""
    
    def test_question_category_values(self):
        """Test question category enum values."""
        assert QuestionCategory.BUSINESS_CAPABILITY == "business_capability"
        assert QuestionCategory.API_ENDPOINTS == "api_endpoints"
        assert QuestionCategory.DATA_MODELING == "data_modeling"
        assert QuestionCategory.WORKFLOWS == "workflows"
        assert QuestionCategory.ARCHITECTURE == "architecture"
    
    def test_question_category_count(self):
        """Test that we have the expected number of categories."""
        categories = list(QuestionCategory)
        assert len(categories) == 5


if __name__ == "__main__":
    pytest.main([__file__])