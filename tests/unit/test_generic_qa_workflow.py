"""
Unit tests for GenericQAWorkflow.

Tests the Generic Q&A Workflow functionality including workflow execution,
step processing, and template management.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.workflows.generic_qa_workflow import (
    GenericQAWorkflow,
    GenericQAStep
)
from src.agents.generic_qa_agent import QuestionCategory


class TestGenericQAWorkflow:
    """Test cases for GenericQAWorkflow."""
    
    @pytest.fixture
    def workflow(self):
        """Create GenericQAWorkflow instance for testing."""
        return GenericQAWorkflow(
            workflow_id="test_workflow",
            enable_persistence=False  # Disable for testing
        )
    
    @pytest.fixture
    def basic_state(self):
        """Create basic workflow state for testing."""
        return {
            "question": "What business capability does this service own?",
            "category": QuestionCategory.BUSINESS_CAPABILITY.value,
            "workflow_type": "generic_qa"
        }
    
    def test_define_steps(self, workflow):
        """Test workflow step definition."""
        steps = workflow.define_steps()
        
        expected_steps = [
            GenericQAStep.INITIALIZE,
            GenericQAStep.CLASSIFY_QUESTION,
            GenericQAStep.LOAD_TEMPLATE,
            GenericQAStep.ANALYZE_CONTEXT,
            GenericQAStep.GENERATE_RESPONSE,
            GenericQAStep.VALIDATE_RESPONSE,
            GenericQAStep.FINALIZE,
        ]
        
        assert steps == expected_steps
        assert len(steps) == 7
    
    def test_validate_state_valid(self, workflow):
        """Test state validation with valid state."""
        valid_state = {
            "question": "Valid question",
            "category": QuestionCategory.ARCHITECTURE.value
        }
        
        assert workflow.validate_state(valid_state) is True
    
    def test_validate_state_missing_question(self, workflow):
        """Test state validation with missing question."""
        invalid_state = {
            "category": QuestionCategory.ARCHITECTURE.value
        }
        
        assert workflow.validate_state(invalid_state) is False
    
    def test_validate_state_empty_question(self, workflow):
        """Test state validation with empty question."""
        invalid_state = {
            "question": "",
            "category": QuestionCategory.ARCHITECTURE.value
        }
        
        assert workflow.validate_state(invalid_state) is False
    
    def test_validate_state_invalid_category(self, workflow):
        """Test state validation with invalid category."""
        invalid_state = {
            "question": "Valid question",
            "category": "invalid_category"
        }
        
        assert workflow.validate_state(invalid_state) is False
    
    def test_execute_step_initialize(self, workflow, basic_state):
        """Test initialize step execution."""
        result_state = workflow.execute_step(GenericQAStep.INITIALIZE, basic_state)
        
        assert result_state["workflow_type"] == "generic_qa"
        assert "start_time" in result_state
        assert "analysis_components" in result_state
        assert "completed_steps" in result_state
        assert GenericQAStep.INITIALIZE in result_state["completed_steps"]
    
    def test_execute_step_classify_question(self, workflow, basic_state):
        """Test question classification step."""
        # Initialize first
        state = workflow.execute_step(GenericQAStep.INITIALIZE, basic_state)
        
        # Then classify
        result_state = workflow.execute_step(GenericQAStep.CLASSIFY_QUESTION, state)
        
        assert result_state["category"] == QuestionCategory.BUSINESS_CAPABILITY.value
        assert "question_classification" in result_state["analysis_components"]
        assert GenericQAStep.CLASSIFY_QUESTION in result_state["completed_steps"]
    
    def test_execute_step_classify_question_provided_category(self, workflow):
        """Test question classification with pre-provided valid category."""
        state = {
            "question": "Test question",
            "category": QuestionCategory.API_ENDPOINTS.value,
            "analysis_components": [],
            "completed_steps": []
        }
        
        result_state = workflow.execute_step(GenericQAStep.CLASSIFY_QUESTION, state)
        
        assert result_state["category"] == QuestionCategory.API_ENDPOINTS.value
    
    def test_execute_step_load_template(self, workflow, basic_state):
        """Test template loading step."""
        # Initialize and classify first
        state = workflow.execute_step(GenericQAStep.INITIALIZE, basic_state)
        state = workflow.execute_step(GenericQAStep.CLASSIFY_QUESTION, state)
        
        # Load template
        result_state = workflow.execute_step(GenericQAStep.LOAD_TEMPLATE, state)
        
        assert "template" in result_state
        assert "template_type" in result_state
        assert "template_used" in result_state
        assert "template_loading" in result_state["analysis_components"]
        assert GenericQAStep.LOAD_TEMPLATE in result_state["completed_steps"]
    
    def test_execute_step_analyze_context(self, workflow, basic_state):
        """Test context analysis step."""
        # Set up state
        state = workflow.execute_step(GenericQAStep.INITIALIZE, basic_state)
        state = workflow.execute_step(GenericQAStep.CLASSIFY_QUESTION, state)
        
        # Analyze context
        result_state = workflow.execute_step(GenericQAStep.ANALYZE_CONTEXT, state)
        
        assert "context_analysis" in result_state
        assert "confidence_score" in result_state
        assert GenericQAStep.ANALYZE_CONTEXT in result_state["completed_steps"]
    
    def test_execute_step_generate_response(self, workflow, basic_state):
        """Test response generation step."""
        # Set up state through previous steps
        state = workflow.execute_step(GenericQAStep.INITIALIZE, basic_state)
        state = workflow.execute_step(GenericQAStep.CLASSIFY_QUESTION, state)
        state = workflow.execute_step(GenericQAStep.LOAD_TEMPLATE, state)
        state = workflow.execute_step(GenericQAStep.ANALYZE_CONTEXT, state)
        
        # Generate response
        result_state = workflow.execute_step(GenericQAStep.GENERATE_RESPONSE, state)
        
        assert "answer" in result_state
        assert len(result_state["answer"]) > 0
        assert "response_generation" in result_state["analysis_components"]
        assert GenericQAStep.GENERATE_RESPONSE in result_state["completed_steps"]
    
    def test_execute_step_validate_response(self, workflow, basic_state):
        """Test response validation step."""
        # Set up state through all previous steps
        state = workflow.execute_step(GenericQAStep.INITIALIZE, basic_state)
        state = workflow.execute_step(GenericQAStep.CLASSIFY_QUESTION, state)
        state = workflow.execute_step(GenericQAStep.LOAD_TEMPLATE, state)
        state = workflow.execute_step(GenericQAStep.ANALYZE_CONTEXT, state)
        state = workflow.execute_step(GenericQAStep.GENERATE_RESPONSE, state)
        
        # Validate response
        result_state = workflow.execute_step(GenericQAStep.VALIDATE_RESPONSE, state)
        
        assert "validation_result" in result_state
        assert "response_validation" in result_state["analysis_components"]
        assert GenericQAStep.VALIDATE_RESPONSE in result_state["completed_steps"]
    
    def test_execute_step_finalize(self, workflow, basic_state):
        """Test finalize step execution."""
        # Set up state through all previous steps
        state = workflow.execute_step(GenericQAStep.INITIALIZE, basic_state)
        state = workflow.execute_step(GenericQAStep.CLASSIFY_QUESTION, state)
        state = workflow.execute_step(GenericQAStep.LOAD_TEMPLATE, state)
        state = workflow.execute_step(GenericQAStep.ANALYZE_CONTEXT, state)
        state = workflow.execute_step(GenericQAStep.GENERATE_RESPONSE, state)
        state = workflow.execute_step(GenericQAStep.VALIDATE_RESPONSE, state)
        
        # Finalize
        result_state = workflow.execute_step(GenericQAStep.FINALIZE, state)
        
        assert result_state["status"] == "completed"
        assert "metadata" in result_state
        assert "processing_time" in result_state["metadata"]
        assert GenericQAStep.FINALIZE in result_state["completed_steps"]
    
    def test_execute_step_unknown_step(self, workflow, basic_state):
        """Test execution of unknown step."""
        with pytest.raises(ValueError, match="Unknown workflow step"):
            workflow.execute_step("unknown_step", basic_state)
    
    def test_full_workflow_execution(self, workflow, basic_state):
        """Test full workflow execution."""
        result_state = workflow.invoke(basic_state)
        
        # Verify final state
        assert result_state["status"] == "completed"
        assert "answer" in result_state
        assert len(result_state["answer"]) > 0
        assert "metadata" in result_state
        assert result_state["category"] == QuestionCategory.BUSINESS_CAPABILITY.value
        
        # Verify all steps were completed
        expected_steps = workflow.define_steps()
        for step in expected_steps:
            assert step in result_state["completed_steps"]
    
    def test_workflow_execution_missing_question(self, workflow):
        """Test workflow execution with missing question."""
        invalid_state = {"category": QuestionCategory.ARCHITECTURE.value}
        
        with pytest.raises(ValueError, match="Question is required"):
            workflow.invoke(invalid_state)
    
    def test_classification_patterns_initialization(self, workflow):
        """Test that classification patterns are properly initialized."""
        # Check that template cache is populated
        assert len(workflow.template_cache) > 0
        assert "dotnet_clean_architecture" in workflow.template_cache
        assert "python_fastapi" in workflow.template_cache
        assert "generic" in workflow.template_cache
        
        # Check that each template type has all categories
        for template_type, templates in workflow.template_cache.items():
            for category in QuestionCategory:
                assert category in templates
    
    def test_determine_template_type_dotnet(self, workflow):
        """Test template type determination for .NET projects."""
        repository_context = {
            "file_patterns": ["*.cs", "*.csproj", "*.sln"]
        }
        
        template_type = workflow._determine_template_type(repository_context)
        assert template_type == "dotnet_clean_architecture"
    
    def test_determine_template_type_python(self, workflow):
        """Test template type determination for Python projects."""
        repository_context = {
            "file_patterns": ["*.py", "requirements.txt", "pyproject.toml"]
        }
        
        template_type = workflow._determine_template_type(repository_context)
        assert template_type == "python_fastapi"
    
    def test_determine_template_type_react(self, workflow):
        """Test template type determination for React projects."""
        repository_context = {
            "file_patterns": ["*.js", "*.jsx", "*.ts", "*.tsx", "package.json"]
        }
        
        template_type = workflow._determine_template_type(repository_context)
        assert template_type == "react_spa"
    
    def test_determine_template_type_generic(self, workflow):
        """Test template type determination for unknown projects."""
        repository_context = {
            "file_patterns": ["*.unknown", "some_file.txt"]
        }
        
        template_type = workflow._determine_template_type(repository_context)
        assert template_type == "generic"
    
    def test_determine_template_type_no_context(self, workflow):
        """Test template type determination with no context."""
        template_type = workflow._determine_template_type(None)
        assert template_type == "generic"
    
    def test_get_template(self, workflow):
        """Test template retrieval."""
        template = workflow._get_template(
            QuestionCategory.BUSINESS_CAPABILITY,
            "dotnet_clean_architecture"
        )
        
        assert "structure" in template
        assert "components" in template
        assert isinstance(template["structure"], str)
        assert isinstance(template["components"], list)
    
    def test_get_template_fallback(self, workflow):
        """Test template retrieval with fallback."""
        # Request non-existent template type
        template = workflow._get_template(
            QuestionCategory.BUSINESS_CAPABILITY,
            "non_existent_type"
        )
        
        # Should fall back to generic template
        assert "structure" in template
        assert "components" in template
    
    def test_perform_context_analysis_with_context(self, workflow):
        """Test context analysis with repository context."""
        repository_context = {"file_patterns": ["*.py"]}
        
        analysis = workflow._perform_context_analysis(
            QuestionCategory.ARCHITECTURE,
            repository_context
        )
        
        assert analysis["context_available"] is True
        assert analysis["quality_score"] > 0.5  # Should be higher with context
        assert len(analysis["components_analyzed"]) > 0
    
    def test_perform_context_analysis_without_context(self, workflow):
        """Test context analysis without repository context."""
        analysis = workflow._perform_context_analysis(
            QuestionCategory.ARCHITECTURE,
            None
        )
        
        assert analysis["context_available"] is False
        assert analysis["quality_score"] <= 0.5  # Lower without context
    
    def test_generate_response_from_template(self, workflow):
        """Test response generation from template."""
        template = {
            "structure": "## Analysis\n\n{analysis}\n\n## Components\n\n{components}",
            "components": ["test_component"]
        }
        context_analysis = {
            "findings": ["Test finding"]
        }
        
        response = workflow._generate_response_from_template(
            template,
            context_analysis,
            "Test question",
            QuestionCategory.ARCHITECTURE
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "##" in response  # Should have structured headers
        assert "architecture" in response.lower()
    
    def test_assess_response_quality(self, workflow):
        """Test response quality assessment."""
        good_response = "## Architecture Analysis\n\nThis is a comprehensive analysis of the system architecture with detailed explanations."
        poor_response = "Short answer"
        
        good_quality = workflow._assess_response_quality(good_response, "architecture question")
        poor_quality = workflow._assess_response_quality(poor_response, "architecture question")
        
        assert good_quality > poor_quality
        assert good_quality <= 0.8  # Should be capped
        assert poor_quality >= 0.0
    
    def test_validate_response_quality_good_response(self, workflow):
        """Test response validation for good response."""
        good_response = "## Business Capability Analysis\n\nThis service owns the user management business capability, handling user registration, authentication, and profile management."
        
        validation = workflow._validate_response_quality(
            good_response,
            "What business capability does this service own?",
            QuestionCategory.BUSINESS_CAPABILITY
        )
        
        assert validation["needs_enhancement"] is False
        assert validation["confidence_adjustment"] >= 0.0
    
    def test_validate_response_quality_poor_response(self, workflow):
        """Test response validation for poor response."""
        poor_response = "Short"
        
        validation = workflow._validate_response_quality(
            poor_response,
            "What business capability does this service own?",
            QuestionCategory.BUSINESS_CAPABILITY
        )
        
        assert validation["needs_enhancement"] is True
        assert "Response is too short" in validation["suggestions"]
        assert validation["confidence_adjustment"] < 0.0
    
    def test_enhance_response(self, workflow):
        """Test response enhancement."""
        short_response = "Basic answer"
        suggestions = ["Response is too short", "Add structured headers"]
        
        enhanced = workflow._enhance_response(short_response, suggestions)
        
        assert len(enhanced) > len(short_response)
        assert "##" in enhanced  # Should add headers
        assert "Additional Notes" in enhanced


class TestGenericQAStep:
    """Test cases for GenericQAStep enum."""
    
    def test_generic_qa_step_values(self):
        """Test Generic Q&A step enum values."""
        assert GenericQAStep.INITIALIZE == "initialize"
        assert GenericQAStep.CLASSIFY_QUESTION == "classify_question"
        assert GenericQAStep.LOAD_TEMPLATE == "load_template"
        assert GenericQAStep.ANALYZE_CONTEXT == "analyze_context"
        assert GenericQAStep.GENERATE_RESPONSE == "generate_response"
        assert GenericQAStep.VALIDATE_RESPONSE == "validate_response"
        assert GenericQAStep.FINALIZE == "finalize"
    
    def test_generic_qa_step_count(self):
        """Test that we have the expected number of steps."""
        steps = list(GenericQAStep)
        assert len(steps) == 7


if __name__ == "__main__":
    pytest.main([__file__])