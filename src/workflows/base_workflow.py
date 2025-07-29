"""
Base workflow infrastructure for LangGraph integration.

This module provides the foundation for stateful workflows using LangGraph with
LangChain Runnable interface integration.
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar
from enum import Enum

from langchain.schema.runnable import Runnable
from loguru import logger
from tenacity import (
    Retrying,
    RetryCallState,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.utils.logging import get_logger
from src.vectorstores.store_factory import VectorStoreFactory

# Type variable for workflow state
StateType = TypeVar('StateType', bound=Dict[str, Any])


class WorkflowStatus(str, Enum):
    """Workflow execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowStep(str, Enum):
    """Common workflow step enumeration."""
    INITIALIZE = "initialize"
    VALIDATE = "validate"
    PROCESS = "process"
    FINALIZE = "finalize"
    ERROR_HANDLING = "error_handling"
    RETRY = "retry"
    CLEANUP = "cleanup"


class WorkflowMetadata:
    """
    Workflow metadata tracking class.
    
    Tracks workflow execution metadata including ID, status, executed steps,
    duration, and error information.
    """
    
    def __init__(self, workflow_id: Optional[str] = None):
        """
        Initialize workflow metadata.
        
        Args:
            workflow_id: Optional workflow ID, generates UUID if not provided
        """
        self.id = workflow_id or str(uuid.uuid4())
        self.status = WorkflowStatus.PENDING
        self.executed_steps: List[str] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.errors: List[Dict[str, Any]] = []
        self.retry_count = 0
        self.current_step: Optional[str] = None
        self.progress_percentage = 0.0
        
    def start(self) -> None:
        """Mark workflow as started."""
        self.start_time = time.time()
        self.status = WorkflowStatus.RUNNING
        logger.info(f"Workflow {self.id} started")
        
    def complete(self) -> None:
        """Mark workflow as completed."""
        self.end_time = time.time()
        self.status = WorkflowStatus.COMPLETED
        self.progress_percentage = 100.0
        if self.start_time:  
            self.duration = self.end_time - self.start_time  
            logger.info(f"Workflow {self.id} completed in {self.duration:.2f}s")  
        else:  
            logger.info(f"Workflow {self.id} completed")
        
    def fail(self, error: Exception, step: Optional[str] = None) -> None:
        """
        Mark workflow as failed.
        
        Args:
            error: Exception that caused the failure
            step: Optional step where failure occurred
        """
        self.end_time = time.time()
        self.status = WorkflowStatus.FAILED
        if self.start_time:
            self.duration = self.end_time - self.start_time
            
        error_info = {
            "timestamp": self.end_time,
            "step": step or self.current_step,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "retry_count": self.retry_count
        }
        step_info = f" at step {step}" if step else ""  
        self.errors.append(error_info)  
        logger.error(f"Workflow {self.id} failed{step_info}: {error}")
        
    def add_step(self, step: str) -> None:
        """
        Add executed step to workflow.
        
        Args:
            step: Step name to add
        """
        self.current_step = step
        if step not in self.executed_steps:
            self.executed_steps.append(step)
        logger.debug(f"Workflow {self.id} executing step: {step}")
        
    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1
        logger.warning(f"Workflow {self.id} retry attempt {self.retry_count}")
        
    def update_progress(self, percentage: float) -> None:
        """
        Update workflow progress percentage.
        
        Args:
            percentage: Progress percentage (0-100)
        """
        self.progress_percentage = max(0.0, min(100.0, percentage))
        logger.debug(f"Workflow {self.id} progress: {self.progress_percentage:.1f}%")
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary.
        
        Returns:
            Dictionary representation of metadata
        """
        return {
            "id": self.id,
            "status": self.status.value,
            "executed_steps": self.executed_steps,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "errors": self.errors,
            "retry_count": self.retry_count,
            "current_step": self.current_step,
            "progress_percentage": self.progress_percentage
        }


class BaseWorkflow(Runnable[StateType, StateType], ABC):
    """
    Abstract base class for LangGraph workflows implementing LangChain Runnable interface.
    
    This class provides the foundation for stateful workflows with error handling,
    retry logic, progress tracking, and integration with vector store factory.
    """
    
    def __init__(
        self,
        workflow_id: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        enable_persistence: bool = True
    ):
        """
        Initialize base workflow.
        
        Args:
            workflow_id: Optional workflow ID
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            enable_persistence: Enable workflow state persistence
        """
        self.metadata = WorkflowMetadata(workflow_id)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_persistence = enable_persistence
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize vector store factory for switching collections within the same factory at runtime
        self.vector_store_factory = VectorStoreFactory()
        
        # Initialize retry configuration using instance attributes
        self.retrier = Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(min=self.retry_delay, max=60),
            retry=retry_if_exception_type((ConnectionError, TimeoutError)),
            before_sleep=self._log_before_retry,
            reraise=True
        )
        
    @property
    def workflow_id(self) -> str:
        """Get workflow ID."""
        return self.metadata.id
        
    @property
    def status(self) -> WorkflowStatus:
        """Get current workflow status."""
        return self.metadata.status
        
    @property
    def progress(self) -> float:
        """Get current progress percentage."""
        return self.metadata.progress_percentage
        
    def _log_before_retry(self, retry_state: RetryCallState) -> None:
        """
        Log before retry callback for tenacity.
        
        Args:
            retry_state: The retry state information
        """
        self.metadata.increment_retry()
        step = retry_state.args[0] if retry_state.args else "unknown"
        exception = retry_state.outcome.exception() if retry_state.outcome else "unknown error"
        self.logger.warning(f"Retrying step {step} due to: {exception}")
        
    @abstractmethod
    def define_steps(self) -> List[str]:
        """
        Define the workflow steps.
        
        Returns:
            List of step names in execution order
        """
        pass
        
    @abstractmethod
    def execute_step(self, step: str, state: StateType) -> StateType:
        """
        Execute a single workflow step.
        
        Args:
            step: Step name to execute
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        pass
        
    @abstractmethod
    def validate_state(self, state: StateType) -> bool:
        """
        Validate workflow state.
        
        Args:
            state: Workflow state to validate
            
        Returns:
            True if state is valid, False otherwise
        """
        pass
        
    def invoke(
        self,
        input_state: StateType,
        config: Optional[Dict[str, Any]] = None
    ) -> StateType:
        """
        Execute the workflow (LangChain Runnable interface).
        
        Args:
            input_state: Initial workflow state
            config: Optional configuration
            
        Returns:
            Final workflow state
        """
        try:
            self.metadata.start()
            return self._execute_workflow(input_state)
        except Exception as e:
            self.metadata.fail(e)
            raise
            
    def _execute_workflow(self, state: StateType) -> StateType:
        """
        Internal workflow execution logic.
        
        Args:
            state: Initial workflow state
            
        Returns:
            Final workflow state
        """
        current_state = state.copy()
        steps = self.define_steps()
        total_steps = len(steps)
        
        for i, step in enumerate(steps):
            try:
                self.metadata.add_step(step)
                current_state = self.retrier(self.execute_step, step, current_state)
                
                # Update progress
                progress = ((i + 1) / total_steps) * 100
                self.metadata.update_progress(progress)
                
                # Persist state if enabled
                if self.enable_persistence:
                    self._persist_state(current_state)
                    
            except Exception as e:
                self.logger.error(f"Step {step} failed: {e}")
                current_state = self._handle_step_error(step, current_state, e)
                
        self.metadata.complete()
        return current_state
        
    def _handle_step_error(
        self,
        step: str,
        state: StateType,
        error: Exception
    ) -> StateType:
        """
        Handle step execution error.
        
        Args:
            step: Step name that failed
            state: Current workflow state
            error: Exception that occurred
            
        Returns:
            Updated workflow state with error handling
        """
        error_state = state.copy()
        error_state.setdefault("errors", []).append({
            "step": step,
            "error": str(error),
            "timestamp": time.time()
        })
        
        # Allow subclasses to implement custom error handling
        try:
            return self.handle_error(step, error_state, error)
        except Exception as handle_error:
            self.logger.error(f"Error handler failed: {handle_error}")
            return error_state
            
    def handle_error(
        self,
        step: str,
        state: StateType,
        error: Exception
    ) -> StateType:
        """
        Handle workflow errors (override in subclasses).
        
        Args:
            step: Step name that failed
            state: Current workflow state
            error: Exception that occurred
            
        Returns:
            Updated workflow state
        """
        # Default implementation - mark error and continue
        self.logger.warning(f"Using default error handling for step {step}")
        return state
        
    def _persist_state(self, state: StateType) -> None:
        """
        Persist workflow state (placeholder for future implementation).
        
        Args:
            state: Current workflow state
        """
        if not self.enable_persistence:
            return
            
        # TODO: Implement state persistence to database/file system
        self.logger.debug(f"Persisting state for workflow {self.workflow_id}")
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get workflow metadata.
        
        Returns:
            Workflow metadata dictionary
        """
        return self.metadata.to_dict()
        
    def pause(self) -> None:
        """Pause workflow execution."""
        self.metadata.status = WorkflowStatus.PAUSED
        self.logger.info(f"Workflow {self.workflow_id} paused")
        
    def resume(self) -> None:
        """Resume workflow execution."""
        if self.metadata.status == WorkflowStatus.PAUSED:
            self.metadata.status = WorkflowStatus.RUNNING
            self.logger.info(f"Workflow {self.workflow_id} resumed")
            
    def cancel(self) -> None:
        """Cancel workflow execution."""
        self.metadata.status = WorkflowStatus.CANCELLED
        self.metadata.end_time = time.time()
        if self.metadata.start_time:
            self.metadata.duration = self.metadata.end_time - self.metadata.start_time
        self.logger.info(f"Workflow {self.workflow_id} cancelled")
        
    def get_vector_store(self, collection_name: Optional[str] = None):
        """
        Get vector store instance using factory.
        
        Args:
            collection_name: Optional collection name
            
        Returns:
            Vector store instance
        """
        try:
            return self.vector_store_factory.create(collection_name=collection_name)
        except Exception as e:
            self.logger.error(f"Failed to create vector store: {e}")
            raise
