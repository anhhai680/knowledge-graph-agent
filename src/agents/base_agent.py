"""
Base Agent Architecture with LangChain Runnable Integration.

This module provides the foundational agent architecture that integrates
with LangGraph workflows and LangChain components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain.schema.runnable import Runnable, RunnableConfig
from loguru import logger

from src.workflows.base_workflow import BaseWorkflow


class BaseAgent(Runnable, ABC):
    """
    Base agent class implementing LangChain Runnable interface.
    
    This class provides the foundation for all agents in the system,
    enabling integration with LangGraph workflows and LangChain components.
    """

    def __init__(
        self,
        workflow: Optional[BaseWorkflow] = None,
        agent_name: str = "BaseAgent",
        **kwargs,
    ):
        """
        Initialize base agent.

        Args:
            workflow: LangGraph workflow instance for stateful processing
            agent_name: Name identifier for the agent
            **kwargs: Additional initialization parameters
        """
        super().__init__(**kwargs)
        self.workflow = workflow
        self.agent_name = agent_name
        self.logger = logger.bind(agent=agent_name)

    @abstractmethod
    async def _process_input(
        self, 
        input_data: Union[str, Dict[str, Any]], 
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Process input data and return structured response.

        Args:
            input_data: Input data to process
            config: Optional configuration for processing

        Returns:
            Structured response dictionary
        """
        pass

    @abstractmethod
    def _validate_input(self, input_data: Any) -> bool:
        """
        Validate input data format and content.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise
        """
        pass

    async def ainvoke(
        self, 
        input: Union[str, Dict[str, Any]], 
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Invoke the agent with input data (asynchronous).

        Args:
            input: Input data to process
            config: Optional configuration

        Returns:
            Agent response
        """
        try:
            # Validate input
            if not self._validate_input(input):
                return {
                    "success": False,
                    "error": "Invalid input format",
                    "agent": self.agent_name,
                }

            # Process asynchronously
            result = await self._process_input(input, config)
            return result

        except Exception as e:
            self.logger.error(f"Error in agent ainvoke: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "agent": self.agent_name,
            }

    def batch(
        self, 
        inputs: List[Union[str, Dict[str, Any]]], 
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple inputs in batch (synchronous).

        Args:
            inputs: List of input data to process
            config: Optional configuration(s)
            **kwargs: Additional arguments

        Returns:
            List of agent responses
        """
        results = []
        configs = config if isinstance(config, list) else [config] * len(inputs)
        
        for i, input_data in enumerate(inputs):
            try:
                result = self.invoke(input_data, configs[i] if configs else None)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing batch item {i}: {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "agent": self.agent_name,
                    "batch_index": i,
                })
        
        return results

    async def abatch(
        self, 
        inputs: List[Union[str, Dict[str, Any]]], 
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple inputs in batch (asynchronous).

        Args:
            inputs: List of input data to process
            config: Optional configuration(s)
            **kwargs: Additional arguments

        Returns:
            List of agent responses
        """
        import asyncio
        
        configs = config if isinstance(config, list) else [config] * len(inputs)
        
        tasks = []
        for i, input_data in enumerate(inputs):
            task = self.ainvoke(input_data, configs[i] if configs else None)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing batch item {i}: {str(result)}")
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "agent": self.agent_name,
                    "batch_index": i,
                })
            else:
                processed_results.append(result)
        
        return processed_results

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information and status.

        Returns:
            Agent information dictionary
        """
        return {
            "agent_name": self.agent_name,
            "agent_type": self.__class__.__name__,
            "has_workflow": self.workflow is not None,
            "workflow_type": self.workflow.__class__.__name__ if self.workflow else None,
        }

    def set_workflow(self, workflow: BaseWorkflow) -> None:
        """
        Set or update the workflow for this agent.

        Args:
            workflow: LangGraph workflow instance
        """
        self.workflow = workflow
        self.logger.info(f"Workflow updated for agent {self.agent_name}")

    async def get_workflow_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current workflow status if workflow is available.

        Returns:
            Workflow status information or None
        """
        if not self.workflow:
            return None

        try:
            # Get workflow state information
            return {
                "workflow_name": self.workflow.__class__.__name__,
                "workflow_id": getattr(self.workflow, 'workflow_id', 'unknown'),
                "has_state_manager": hasattr(self.workflow, 'state_manager'),
            }
        except Exception as e:
            self.logger.error(f"Error getting workflow status: {str(e)}")
            return None


class AgentResponse:
    """
    Structured response object for agent outputs.
    """

    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
    ):
        """
        Initialize agent response.

        Args:
            success: Whether the operation was successful
            data: Response data
            error: Error message if operation failed
            metadata: Additional metadata
            agent_name: Name of the agent that generated this response
        """
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.agent_name = agent_name

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to dictionary format.

        Returns:
            Response as dictionary
        """
        result = {
            "success": self.success,
            "data": self.data,
            "metadata": self.metadata,
        }

        if self.error:
            result["error"] = self.error

        if self.agent_name:
            result["agent"] = self.agent_name

        return result

    @classmethod
    def success_response(
        cls,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
    ) -> "AgentResponse":
        """
        Create a successful response.

        Args:
            data: Response data
            metadata: Additional metadata
            agent_name: Name of the agent

        Returns:
            Success response instance
        """
        return cls(
            success=True,
            data=data,
            metadata=metadata,
            agent_name=agent_name,
        )

    @classmethod
    def error_response(
        cls,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
    ) -> "AgentResponse":
        """
        Create an error response.

        Args:
            error: Error message
            metadata: Additional metadata
            agent_name: Name of the agent

        Returns:
            Error response instance
        """
        return cls(
            success=False,
            error=error,
            metadata=metadata,
            agent_name=agent_name,
        )
