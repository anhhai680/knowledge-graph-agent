"""
Unit tests for Generic Q&A Workflow.

This module tests the GenericQAWorkflow functionality including step execution,
question classification, project analysis, and response generation.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.workflows.generic_qa_workflow import GenericQAWorkflow, GenericQAStep


class TestGenericQAWorkflow:
    """Test cases for GenericQAWorkflow."""

    @pytest.fixture
    def workflow(self):
        """Create a GenericQAWorkflow instance for testing."""
        return GenericQAWorkflow(
            enable_project_analysis=True,
            max_response_length=1000
        )

    def test_initialization(self):
        """Test workflow initialization."""
        workflow = GenericQAWorkflow()
        
        assert workflow.enable_project_analysis is True
        assert workflow.max_response_length == 5000
        assert workflow.architecture_detector is not None
        assert workflow.business_analyzer is not None
        assert workflow.api_analyzer is not None
        assert workflow.data_analyzer is not None
        assert workflow.operational_analyzer is not None

    def test_define_steps(self, workflow):
        """Test workflow step definitions."""
        steps = workflow.define_steps()
        
        expected_steps = [
            GenericQAStep.CLASSIFY_QUESTION.value,
            GenericQAStep.ANALYZE_PROJECT.value,
            GenericQAStep.GENERATE_RESPONSE.value,
            GenericQAStep.VALIDATE_RESPONSE.value,
        ]
        
        assert steps == expected_steps

    def test_validate_state_valid(self, workflow):
        """Test state validation with valid state."""
        valid_state = {
            "question": "What is the architecture?",
            "project_template": "python_fastapi"
        }
        
        assert workflow.validate_state(valid_state) is True

    def test_validate_state_missing_question(self, workflow):
        """Test state validation with missing question."""
        invalid_state = {
            "project_template": "python_fastapi"
        }
        
        assert workflow.validate_state(invalid_state) is False

    def test_validate_state_empty_question(self, workflow):
        """Test state validation with empty question."""
        invalid_state = {
            "question": "",
            "project_template": "python_fastapi"
        }
        
        assert workflow.validate_state(invalid_state) is False

    def test_classify_question_with_category(self, workflow):
        """Test question classification when category is provided."""
        state = {
            "question": "What are the API endpoints?",
            "category": "api_endpoints",
            "supported_categories": ["api_endpoints", "architecture"]
        }
        
        result_state = workflow._classify_question(state)
        
        assert result_state["detected_category"] == "api_endpoints"
        assert result_state["classification_confidence"] == 1.0

    def test_classify_question_auto_detect(self, workflow):
        """Test automatic question classification."""
        state = {
            "question": "What are the main API endpoints and how is pagination implemented?",
            "supported_categories": ["api_endpoints", "architecture"]
        }
        
        result_state = workflow._classify_question(state)
        
        assert result_state["detected_category"] == "api_endpoints"
        assert result_state["classification_confidence"] == 0.9

    def test_classify_question_invalid_category(self, workflow):
        """Test question classification with invalid category."""
        state = {
            "question": "What is the architecture?",
            "category": "invalid_category",
            "supported_categories": ["architecture", "api_endpoints"]
        }
        
        result_state = workflow._classify_question(state)
        
        # Should fall back to auto-detection
        assert result_state["detected_category"] == "architecture"
        assert result_state["classification_confidence"] == 0.8

    def test_analyze_project_enabled(self, workflow):
        """Test project analysis when enabled."""
        state = {
            "project_template": "python_fastapi",
            "detected_category": "architecture"
        }
        
        with patch.object(workflow.architecture_detector, 'analyze') as mock_analyze:
            mock_analyze.return_value = {"patterns": ["layered_architecture"]}
            
            result_state = workflow._analyze_project(state)
            
            assert "project_analysis" in result_state
            assert "patterns" in result_state["project_analysis"]
            mock_analyze.assert_called_once_with(template="python_fastapi")

    def test_analyze_project_disabled(self, workflow):
        """Test project analysis when disabled."""
        workflow.enable_project_analysis = False
        state = {"project_template": "python_fastapi"}
        
        result_state = workflow._analyze_project(state)
        
        assert result_state["project_analysis"] == {}

    def test_analyze_project_business_capability(self, workflow):
        """Test project analysis for business capability questions."""
        state = {
            "project_template": "python_fastapi",
            "detected_category": "business_capability"
        }
        
        with patch.object(workflow.business_analyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = {"scope": "user_management"}
            
            result_state = workflow._analyze_project(state)
            
            assert "scope" in result_state["project_analysis"]
            mock_analyze.assert_called_once_with(template="python_fastapi")

    def test_generate_response_with_template(self, workflow):
        """Test response generation with template."""
        state = {
            "question": "What is the architecture?",
            "detected_category": "architecture",
            "project_template": "python_fastapi",
            "project_analysis": {"patterns": ["layered_architecture"]}
        }
        
        result_state = workflow._generate_response(state)
        
        assert "answer" in result_state
        assert "template_used" in result_state
        assert "response_length" in result_state
        assert result_state["template_used"] == "python_fastapi"
        assert len(result_state["answer"]) > 0

    def test_generate_response_fallback(self, workflow):
        """Test response generation with fallback when no template."""
        # Clear templates to trigger fallback
        workflow.templates = {}
        
        state = {
            "question": "What is the architecture?",
            "detected_category": "architecture",
            "project_template": "unknown_template",
            "project_analysis": {}
        }
        
        result_state = workflow._generate_response(state)
        
        assert "answer" in result_state
        assert "Based on your question about architecture" in result_state["answer"]

    def test_generate_response_length_limit(self, workflow):
        """Test response generation with length limit."""
        workflow.max_response_length = 50
        
        state = {
            "question": "What is the architecture?",
            "detected_category": "architecture",
            "project_template": "python_fastapi",
            "project_analysis": {}
        }
        
        result_state = workflow._generate_response(state)
        
        assert len(result_state["answer"]) <= 53  # 50 + "..."

    def test_validate_response_success(self, workflow):
        """Test response validation with valid response."""
        state = {
            "answer": "This is a comprehensive answer about the architecture patterns and layers used in the system.",
            "detected_category": "architecture"
        }
        
        result_state = workflow._validate_response(state)
        
        assert result_state["validation_results"]["is_valid"] is True
        assert result_state["confidence_score"] > 0.5
        assert len(result_state["validation_results"]["issues"]) == 0

    def test_validate_response_too_short(self, workflow):
        """Test response validation with too short response."""
        state = {
            "answer": "Short answer",
            "detected_category": "architecture"
        }
        
        result_state = workflow._validate_response(state)
        
        assert "Response too short" in result_state["validation_results"]["issues"]
        assert result_state["confidence_score"] < 1.0

    def test_validate_response_placeholder_text(self, workflow):
        """Test response validation with placeholder text."""
        state = {
            "answer": "This is a response with [TODO] placeholder that needs to be completed.",
            "detected_category": "architecture"
        }
        
        result_state = workflow._validate_response(state)
        
        assert "Contains placeholder text" in result_state["validation_results"]["issues"]
        assert result_state["confidence_score"] < 1.0

    def test_validate_response_category_specific(self, workflow):
        """Test response validation with category-specific checks."""
        # Test API endpoints category
        state = {
            "answer": "This response talks about the system but doesn't mention any endpoints.",
            "detected_category": "api_endpoints"
        }
        
        result_state = workflow._validate_response(state)
        
        assert any("endpoint" in issue for issue in result_state["validation_results"]["issues"])
        assert result_state["confidence_score"] < 1.0

    def test_detect_category_business(self, workflow):
        """Test category detection for business questions."""
        question = "What business capability does the service own and what are the core entities?"
        
        category = workflow._detect_category(question)
        
        assert category == "business_capability"

    def test_detect_category_api(self, workflow):
        """Test category detection for API questions."""
        question = "What are the API endpoints and expected behaviors with status codes?"
        
        category = workflow._detect_category(question)
        
        assert category == "api_endpoints"

    def test_detect_category_data(self, workflow):
        """Test category detection for data modeling questions."""
        question = "How is data modeled and persisted with repository patterns?"
        
        category = workflow._detect_category(question)
        
        assert category == "data_modeling"

    def test_detect_category_workflow(self, workflow):
        """Test category detection for workflow questions."""
        question = "What's the end-to-end workflow for create and update operations?"
        
        category = workflow._detect_category(question)
        
        assert category == "workflows"

    def test_detect_category_architecture(self, workflow):
        """Test category detection for architecture questions."""
        question = "What's the architecture and what operational concerns apply?"
        
        category = workflow._detect_category(question)
        
        assert category == "architecture"

    def test_detect_category_fallback(self, workflow):
        """Test category detection fallback."""
        question = "This is a generic question without specific keywords."
        
        category = workflow._detect_category(question)
        
        assert category == "architecture"  # Default category

    def test_load_templates(self, workflow):
        """Test template loading."""
        templates = workflow._load_templates()
        
        assert isinstance(templates, dict)
        assert "python_fastapi" in templates
        assert "dotnet_clean_architecture" in templates
        assert "react_spa" in templates
        
        # Check template structure
        python_template = templates["python_fastapi"]
        assert "business_capability" in python_template
        assert "api_endpoints" in python_template
        assert "architecture" in python_template

    def test_apply_template(self, workflow):
        """Test template application."""
        question = "What is the architecture?"
        template = {
            "structure": "Layered architecture with clean separation",
            "patterns": ["API layer", "Business layer", "Data layer"],
            "examples": ["FastAPI", "SQLAlchemy", "Pydantic"]
        }
        analysis = {"detected_patterns": ["layered"]}
        
        response = workflow._apply_template(question, template, analysis)
        
        assert "Structure:" in response
        assert "Common Patterns:" in response
        assert "Examples:" in response
        assert "Analysis:" in response
        assert "API layer" in response

    def test_generate_fallback_response(self, workflow):
        """Test fallback response generation."""
        question = "What is the architecture?"
        category = "architecture"
        analysis = {"patterns": ["layered"]}
        
        response = workflow._generate_fallback_response(question, category, analysis)
        
        assert "Based on your question about architecture" in response
        assert "Project Analysis:" in response

    def test_get_available_templates(self, workflow):
        """Test getting available templates."""
        templates = workflow.get_available_templates()
        
        assert isinstance(templates, list)
        assert "python_fastapi" in templates
        assert "dotnet_clean_architecture" in templates
        assert "react_spa" in templates

    @pytest.mark.asyncio
    async def test_analyze_project_method(self, workflow):
        """Test the analyze_project method."""
        analysis_state = {
            "project_path": "/path/to/project",
            "repository_url": "https://github.com/user/repo",
            "template_hint": "python_fastapi"
        }
        
        with patch.object(workflow.architecture_detector, 'detect_patterns') as mock_detect:
            with patch.object(workflow.business_analyzer, 'extract_capabilities') as mock_extract:
                mock_detect.return_value = ["layered_architecture"]
                mock_extract.return_value = ["user_management"]
                
                result = await workflow.analyze_project(analysis_state)
                
                assert result["success"] is True
                assert result["detected_template"] == "python_fastapi"
                assert "layered_architecture" in result["architecture_patterns"]
                assert "user_management" in result["business_capabilities"]

    @pytest.mark.asyncio
    async def test_analyze_project_error(self, workflow):
        """Test analyze_project error handling."""
        analysis_state = {"project_path": "/invalid/path"}
        
        with patch.object(workflow.architecture_detector, 'detect_patterns', side_effect=Exception("Analysis error")):
            result = await workflow.analyze_project(analysis_state)
            
            assert result["success"] is False
            assert "Analysis error" in result["error"]

    def test_execute_step_classify_question(self, workflow):
        """Test executing classify question step."""
        state = {"question": "What is the architecture?"}
        
        result = workflow.execute_step(GenericQAStep.CLASSIFY_QUESTION.value, state)
        
        assert "detected_category" in result
        assert "classification_confidence" in result

    def test_execute_step_unknown(self, workflow):
        """Test executing unknown step."""
        state = {"question": "What is the architecture?"}
        
        with pytest.raises(ValueError, match="Unknown step"):
            workflow.execute_step("unknown_step", state)

    def test_full_workflow_execution(self, workflow):
        """Test complete workflow execution."""
        initial_state = {
            "question": "What is the architecture and what operational concerns apply?",
            "project_template": "python_fastapi"
        }
        
        result_state = workflow.invoke(initial_state)
        
        # Check that all steps were executed
        assert "detected_category" in result_state
        assert "project_analysis" in result_state
        assert "answer" in result_state
        assert "validation_results" in result_state
        assert "confidence_score" in result_state
        
        # Check workflow metadata
        assert workflow.status.value in ["completed", "failed"]
        assert len(workflow.metadata.executed_steps) > 0