"""
Unit tests for base agent architecture.

This module contains tests for the BaseAgent class and AgentResponse functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.base_agent import BaseAgent, AgentResponse
from src.workflows.base_workflow import BaseWorkflow


class MockAgent(BaseAgent):
    """Mock implementation of BaseAgent for testing."""

    def _validate_input(self, input_data):
        """Test validation - accepts strings only."""
        return isinstance(input_data, str) and len(input_data.strip()) > 0

    async def _process_input(self, input_data, config=None):
        """Test processing - returns uppercase."""
        if input_data == "error":
            raise ValueError("Test error")
        
        return {
            "success": True,
            "result": str(input_data).upper(),
            "agent": self.agent_name,
        }


class TestBaseAgent:
    """Test cases for BaseAgent functionality."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = MockAgent(agent_name="TestAgent")
        
        assert agent.agent_name == "TestAgent"
        assert agent.workflow is None
        assert isinstance(agent.logger, type(agent.logger))

    def test_agent_initialization_with_workflow(self):
        """Test agent initialization with workflow."""
        mock_workflow = MagicMock(spec=BaseWorkflow)
        agent = MockAgent(workflow=mock_workflow, agent_name="TestAgent")
        
        assert agent.workflow == mock_workflow
        assert agent.agent_name == "TestAgent"

    def test_invoke_success(self):
        """Test successful synchronous invocation."""
        agent = MockAgent(agent_name="TestAgent")
        result = agent.invoke("test input")
        
        assert result["success"] is True
        assert result["result"] == "TEST INPUT"
        assert result["agent"] == "TestAgent"

    def test_invoke_invalid_input(self):
        """Test invocation with invalid input."""
        agent = MockAgent(agent_name="TestAgent")
        result = agent.invoke("")  # Empty string should be invalid
        
        assert result["success"] is False
        assert result["error"] == "Invalid input format"
        assert result["agent"] == "TestAgent"

    def test_invoke_processing_error(self):
        """Test invocation with processing error."""
        agent = MockAgent(agent_name="TestAgent")
        result = agent.invoke("error")  # Triggers ValueError
        
        assert result["success"] is False
        assert "Test error" in result["error"]
        assert result["agent"] == "TestAgent"

    @pytest.mark.asyncio
    async def test_ainvoke_success(self):
        """Test successful asynchronous invocation."""
        agent = MockAgent(agent_name="TestAgent")
        result = await agent.ainvoke("test input")
        
        assert result["success"] is True
        assert result["result"] == "TEST INPUT"
        assert result["agent"] == "TestAgent"

    @pytest.mark.asyncio
    async def test_ainvoke_invalid_input(self):
        """Test asynchronous invocation with invalid input."""
        agent = MockAgent(agent_name="TestAgent")
        result = await agent.ainvoke("")
        
        assert result["success"] is False
        assert result["error"] == "Invalid input format"

    @pytest.mark.asyncio
    async def test_ainvoke_processing_error(self):
        """Test asynchronous invocation with processing error."""
        agent = MockAgent(agent_name="TestAgent")
        result = await agent.ainvoke("error")
        
        assert result["success"] is False
        assert "Test error" in result["error"]

    def test_batch_processing(self):
        """Test batch processing."""
        agent = MockAgent(agent_name="TestAgent")
        inputs = ["input1", "input2", ""]  # Last one is invalid
        
        results = agent.batch(inputs)
        
        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[0]["result"] == "INPUT1"
        assert results[1]["success"] is True
        assert results[1]["result"] == "INPUT2"
        assert results[2]["success"] is False  # Invalid input

    @pytest.mark.asyncio
    async def test_abatch_processing(self):
        """Test asynchronous batch processing."""
        agent = MockAgent(agent_name="TestAgent")
        inputs = ["input1", "input2", "error"]  # Last one triggers error
        
        results = await agent.abatch(inputs)
        
        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is True
        assert results[2]["success"] is False

    def test_get_agent_info(self):
        """Test agent information retrieval."""
        mock_workflow = MagicMock(spec=BaseWorkflow)
        mock_workflow.__class__.__name__ = "TestWorkflow"
        
        agent = MockAgent(workflow=mock_workflow, agent_name="TestAgent")
        info = agent.get_agent_info()
        
        assert info["agent_name"] == "TestAgent"
        assert info["agent_type"] == "MockAgent"
        assert info["has_workflow"] is True
        assert info["workflow_type"] == "TestWorkflow"

    def test_get_agent_info_no_workflow(self):
        """Test agent information without workflow."""
        agent = MockAgent(agent_name="TestAgent")
        info = agent.get_agent_info()
        
        assert info["has_workflow"] is False
        assert info["workflow_type"] is None

    def test_set_workflow(self):
        """Test workflow setting."""
        agent = MockAgent(agent_name="TestAgent")
        mock_workflow = MagicMock(spec=BaseWorkflow)
        
        agent.set_workflow(mock_workflow)
        
        assert agent.workflow == mock_workflow

    @pytest.mark.asyncio
    async def test_get_workflow_status(self):
        """Test workflow status retrieval."""
        mock_workflow = MagicMock(spec=BaseWorkflow)
        mock_workflow.__class__.__name__ = "TestWorkflow"
        mock_workflow.workflow_id = "test_id"
        
        agent = MockAgent(workflow=mock_workflow, agent_name="TestAgent")
        status = await agent.get_workflow_status()
        
        assert status is not None
        assert status["workflow_name"] == "TestWorkflow"
        assert status["workflow_id"] == "test_id"

    @pytest.mark.asyncio
    async def test_get_workflow_status_no_workflow(self):
        """Test workflow status without workflow."""
        agent = MockAgent(agent_name="TestAgent")
        status = await agent.get_workflow_status()
        
        assert status is None


class TestAgentResponse:
    """Test cases for AgentResponse functionality."""

    def test_success_response_creation(self):
        """Test successful response creation."""
        response = AgentResponse.success_response(
            data={"result": "test"},
            metadata={"key": "value"},
            agent_name="TestAgent",
        )
        
        assert response.success is True
        assert response.data == {"result": "test"}
        assert response.metadata == {"key": "value"}
        assert response.agent_name == "TestAgent"
        assert response.error is None

    def test_error_response_creation(self):
        """Test error response creation."""
        response = AgentResponse.error_response(
            error="Test error",
            metadata={"key": "value"},
            agent_name="TestAgent",
        )
        
        assert response.success is False
        assert response.error == "Test error"
        assert response.metadata == {"key": "value"}
        assert response.agent_name == "TestAgent"
        assert response.data is None

    def test_response_to_dict(self):
        """Test response dictionary conversion."""
        response = AgentResponse(
            success=True,
            data={"result": "test"},
            metadata={"key": "value"},
            agent_name="TestAgent",
        )
        
        result_dict = response.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["data"] == {"result": "test"}
        assert result_dict["metadata"] == {"key": "value"}
        assert result_dict["agent"] == "TestAgent"
        assert "error" not in result_dict

    def test_error_response_to_dict(self):
        """Test error response dictionary conversion."""
        response = AgentResponse(
            success=False,
            error="Test error",
            metadata={"key": "value"},
            agent_name="TestAgent",
        )
        
        result_dict = response.to_dict()
        
        assert result_dict["success"] is False
        assert result_dict["error"] == "Test error"
        assert result_dict["metadata"] == {"key": "value"}
        assert result_dict["agent"] == "TestAgent"
