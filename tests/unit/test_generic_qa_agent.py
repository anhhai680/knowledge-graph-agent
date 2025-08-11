"""
Unit tests for Generic Q&A Agent.

This module tests the GenericQAAgent functionality including question processing,
template-based responses, and project analysis capabilities.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.generic_qa_agent import GenericQAAgent, QuestionCategory
from src.workflows.generic_qa_workflow import GenericQAWorkflow


class TestGenericQAAgent:
    """Test cases for GenericQAAgent."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow for testing."""
        workflow = MagicMock(spec=GenericQAWorkflow)
        workflow.invoke.return_value = {
            "answer": "Test answer",
            "detected_category": "architecture",
            "project_analysis": {"patterns": ["layered_architecture"]},
            "confidence_score": 0.9,
            "template_used": "python_fastapi",
            "processing_time": 1.5,
            "executed_steps": ["classify_question", "generate_response"]
        }
        return workflow

    @pytest.fixture
    def agent(self, mock_workflow):
        """Create a GenericQAAgent instance for testing."""
        return GenericQAAgent(
            workflow=mock_workflow,
            default_template="python_fastapi"
        )

    def test_initialization(self):
        """Test agent initialization."""
        agent = GenericQAAgent()
        
        assert agent.agent_name == "GenericQAAgent"
        assert agent.default_template == "python_fastapi"
        assert len(agent.supported_categories) == 5
        assert QuestionCategory.BUSINESS_CAPABILITY in agent.supported_categories

    def test_validate_input_string(self, agent):
        """Test input validation with string input."""
        # Valid string input
        assert agent._validate_input("What is the architecture?")
        
        # Empty string
        assert not agent._validate_input("")
        assert not agent._validate_input("   ")

    def test_validate_input_dict(self, agent):
        """Test input validation with dictionary input."""
        # Valid dict input
        valid_input = {
            "question": "What is the architecture?",
            "template": "python_fastapi",
            "category": "architecture"
        }
        assert agent._validate_input(valid_input)
        
        # Invalid dict - no question
        invalid_input = {"template": "python_fastapi"}
        assert not agent._validate_input(invalid_input)
        
        # Invalid dict - empty question
        invalid_input = {"question": ""}
        assert not agent._validate_input(invalid_input)
        
        # Invalid dict - invalid category
        invalid_input = {
            "question": "Test question",
            "category": "invalid_category"
        }
        assert not agent._validate_input(invalid_input)

    @pytest.mark.asyncio
    async def test_process_input_string(self, agent, mock_workflow):
        """Test processing string input."""
        question = "What is the architecture?"
        
        result = await agent._process_input(question)
        
        assert result["success"] is True
        assert result["question"] == question
        assert result["answer"] == "Test answer"
        assert result["category"] == "architecture"
        assert result["template"] == "python_fastapi"
        assert result["confidence_score"] == 0.9
        
        # Verify workflow was called
        mock_workflow.invoke.assert_called_once()
        call_args = mock_workflow.invoke.call_args[0][0]
        assert call_args["question"] == question
        assert call_args["project_template"] == "python_fastapi"

    @pytest.mark.asyncio
    async def test_process_input_dict(self, agent, mock_workflow):
        """Test processing dictionary input."""
        input_data = {
            "question": "What are the API endpoints?",
            "template": "dotnet_clean_architecture",
            "category": "api_endpoints"
        }
        
        result = await agent._process_input(input_data)
        
        assert result["success"] is True
        assert result["question"] == input_data["question"]
        
        # Verify workflow was called with correct parameters
        call_args = mock_workflow.invoke.call_args[0][0]
        assert call_args["question"] == input_data["question"]
        assert call_args["project_template"] == "dotnet_clean_architecture"
        assert call_args["category"] == "api_endpoints"

    @pytest.mark.asyncio
    async def test_process_input_error_handling(self, agent, mock_workflow):
        """Test error handling during input processing."""
        mock_workflow.invoke.side_effect = Exception("Workflow error")
        
        result = await agent._process_input("Test question")
        
        assert result["success"] is False
        assert "Failed to process question" in result["error"]
        assert result["agent"] == "GenericQAAgent"

    @pytest.mark.asyncio
    async def test_ainvoke_success(self, agent, mock_workflow):
        """Test successful ainvoke call."""
        question = "What is the data modeling approach?"
        
        result = await agent.ainvoke(question)
        
        assert result["success"] is True
        assert "answer" in result
        assert "category" in result

    @pytest.mark.asyncio
    async def test_ainvoke_validation_failure(self, agent):
        """Test ainvoke with invalid input."""
        result = await agent.ainvoke("")
        
        assert result["success"] is False
        assert "Invalid input format" in result["error"]

    def test_get_supported_categories(self, agent):
        """Test getting supported categories."""
        categories = agent.get_supported_categories()
        
        assert len(categories) == 5
        assert "business_capability" in categories
        assert "api_endpoints" in categories
        assert "data_modeling" in categories
        assert "workflows" in categories
        assert "architecture" in categories

    def test_get_available_templates(self, agent, mock_workflow):
        """Test getting available templates."""
        mock_workflow.get_available_templates.return_value = [
            "python_fastapi", "dotnet_clean_architecture", "react_spa"
        ]
        
        templates = agent.get_available_templates()
        
        assert len(templates) == 3
        assert "python_fastapi" in templates
        assert "dotnet_clean_architecture" in templates
        assert "react_spa" in templates

    @pytest.mark.asyncio
    async def test_analyze_project_structure(self, agent, mock_workflow):
        """Test project structure analysis."""
        mock_workflow.analyze_project = AsyncMock(return_value={
            "success": True,
            "detected_template": "python_fastapi",
            "architecture_patterns": ["layered_architecture"],
            "confidence": 0.8
        })
        
        result = await agent.analyze_project_structure(
            project_path="/path/to/project",
            template_hint="python_fastapi"
        )
        
        assert result["success"] is True
        assert result["detected_template"] == "python_fastapi"
        assert "layered_architecture" in result["architecture_patterns"]
        assert result["confidence"] == 0.8
        
        mock_workflow.analyze_project.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_project_structure_error(self, agent, mock_workflow):
        """Test project structure analysis error handling."""
        mock_workflow.analyze_project = AsyncMock(side_effect=Exception("Analysis error"))
        
        result = await agent.analyze_project_structure(project_path="/path/to/project")
        
        assert result["success"] is False
        assert "Failed to analyze project" in result["error"]

    def test_get_question_examples(self, agent):
        """Test getting question examples."""
        examples = agent.get_question_examples()
        
        assert isinstance(examples, dict)
        assert "business_capability" in examples
        assert "api_endpoints" in examples
        assert len(examples["business_capability"]) > 0
        assert len(examples["api_endpoints"]) > 0

    def test_get_question_examples_specific_category(self, agent):
        """Test getting question examples for specific category."""
        examples = agent.get_question_examples(category="architecture")
        
        assert len(examples) == 1
        assert "architecture" in examples
        assert len(examples["architecture"]) > 0

    def test_invoke_fallback_for_sync_context(self, agent):
        """Test invoke method fallback behavior."""
        # Mock the running event loop scenario
        with patch('asyncio.get_running_loop', side_effect=RuntimeError("No running loop")):
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = {"success": True, "answer": "Test"}
                
                result = agent.invoke("Test question")
                
                mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_state_creation(self, agent, mock_workflow):
        """Test that workflow state is created correctly."""
        input_data = {
            "question": "Test question",
            "template": "react_spa",
            "category": "workflows"
        }
        
        await agent._process_input(input_data)
        
        call_args = mock_workflow.invoke.call_args[0][0]
        assert call_args["question"] == "Test question"
        assert call_args["project_template"] == "react_spa"
        assert call_args["category"] == "workflows"
        assert call_args["supported_categories"] == agent.get_supported_categories()
        assert "agent_config" in call_args
        assert call_args["agent_config"]["default_template"] == "python_fastapi"
        assert call_args["agent_config"]["agent_name"] == "GenericQAAgent"


class TestQuestionCategory:
    """Test cases for QuestionCategory enum."""

    def test_question_category_values(self):
        """Test question category enum values."""
        assert QuestionCategory.BUSINESS_CAPABILITY.value == "business_capability"
        assert QuestionCategory.API_ENDPOINTS.value == "api_endpoints"
        assert QuestionCategory.DATA_MODELING.value == "data_modeling"
        assert QuestionCategory.WORKFLOWS.value == "workflows"
        assert QuestionCategory.ARCHITECTURE.value == "architecture"

    def test_question_category_creation(self):
        """Test creating QuestionCategory from string."""
        category = QuestionCategory("architecture")
        assert category == QuestionCategory.ARCHITECTURE
        
        with pytest.raises(ValueError):
            QuestionCategory("invalid_category")