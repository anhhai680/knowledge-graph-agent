"""
Workflow state management module for LangGraph workflows.

This module provides state persistence and management capabilities for workflows,
including state validation, serialization, and recovery mechanisms.
"""

import json
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar
from enum import Enum
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from src.config.settings import settings
from src.utils.logging import get_logger

# Type variable for state types
StateType = TypeVar("StateType", bound=Dict[str, Any])


class StateBackend(str, Enum):
    """State persistence backend enumeration."""

    MEMORY = "memory"
    FILE = "file"
    DATABASE = "database"


class StateSerializationFormat(str, Enum):
    """State serialization format enumeration."""

    JSON = "json"
    PICKLE = "pickle"


class WorkflowStateMetadata(BaseModel):
    """
    Workflow state metadata model.

    Contains metadata about the workflow state including timestamps,
    version information, and validation checksums.
    """

    workflow_id: str = Field(..., description="Unique workflow identifier")
    state_version: int = Field(default=1, description="State version number")
    created_at: float = Field(
        default_factory=time.time, description="Creation timestamp"
    )
    updated_at: float = Field(
        default_factory=time.time, description="Last update timestamp"
    )
    state_checksum: Optional[str] = Field(None, description="State validation checksum")
    serialization_format: StateSerializationFormat = Field(
        default=StateSerializationFormat.JSON, description="Serialization format used"
    )

    model_config = ConfigDict(use_enum_values=True)


class StateManager(ABC):
    """
    Abstract base class for workflow state management.

    Provides interface for state persistence, retrieval, and validation
    across different backend implementations.
    """

    def __init__(
        self,
        backend: StateBackend = StateBackend.MEMORY,
        serialization_format: StateSerializationFormat = StateSerializationFormat.JSON,
    ):
        """
        Initialize state manager.

        Args:
            backend: State persistence backend to use
            serialization_format: Serialization format for state data
        """
        self.backend = backend
        self.serialization_format = serialization_format
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def save_state(
        self,
        workflow_id: str,
        state: StateType,
        metadata: Optional[WorkflowStateMetadata] = None,
    ) -> bool:
        """
        Save workflow state.

        Args:
            workflow_id: Unique workflow identifier
            state: Workflow state to save
            metadata: Optional state metadata

        Returns:
            True if save successful, False otherwise
        """
        pass

    @abstractmethod
    def load_state(self, workflow_id: str) -> Optional[StateType]:
        """
        Load workflow state.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            Workflow state if found, None otherwise
        """
        pass

    @abstractmethod
    def delete_state(self, workflow_id: str) -> bool:
        """
        Delete workflow state.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            True if delete successful, False otherwise
        """
        pass

    @abstractmethod
    def list_states(self) -> List[str]:
        """
        List all stored workflow IDs.

        Returns:
            List of workflow identifiers
        """
        pass

    def get_state_metadata(self, workflow_id: str) -> Optional[WorkflowStateMetadata]:
        """
        Get state metadata.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            State metadata if found, None otherwise
        """
        # Default implementation - override in subclasses
        return None

    def validate_state(self, state: StateType) -> bool:
        """
        Validate state structure and content.

        Args:
            state: Workflow state to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(state, dict):
            return False

        # Basic validation - subclasses can implement more specific validation
        required_fields = ["status", "workflow_id"]
        return all(field in state for field in required_fields)

    def _serialize_state(self, state: StateType) -> bytes:
        """
        Serialize state data.

        Args:
            state: State to serialize

        Returns:
            Serialized state data
        """
        if self.serialization_format == StateSerializationFormat.JSON:
            return json.dumps(state, default=str).encode("utf-8")
        elif self.serialization_format == StateSerializationFormat.PICKLE:
            return pickle.dumps(state)
        else:
            raise ValueError(
                f"Unsupported serialization format: {self.serialization_format}"
            )

    def _deserialize_state(self, data: bytes) -> StateType:
        """
        Deserialize state data.

        Args:
            data: Serialized state data

        Returns:
            Deserialized state
        """
        if self.serialization_format == StateSerializationFormat.JSON:
            return json.loads(data.decode("utf-8"))
        elif self.serialization_format == StateSerializationFormat.PICKLE:
            return pickle.loads(data)
        else:
            raise ValueError(
                f"Unsupported serialization format: {self.serialization_format}"
            )


class MemoryStateManager(StateManager):
    """
    In-memory state manager implementation.

    Stores workflow states in memory. States are lost when the application restarts.
    Suitable for development and testing environments.
    """

    def __init__(self, **kwargs):
        """Initialize memory state manager."""
        super().__init__(backend=StateBackend.MEMORY, **kwargs)
        self._states: Dict[str, StateType] = {}
        self._metadata: Dict[str, WorkflowStateMetadata] = {}

    def save_state(
        self,
        workflow_id: str,
        state: StateType,
        metadata: Optional[WorkflowStateMetadata] = None,
    ) -> bool:
        """
        Save workflow state to memory.

        Args:
            workflow_id: Unique workflow identifier
            state: Workflow state to save
            metadata: Optional state metadata

        Returns:
            True if save successful, False otherwise
        """
        try:
            if not self.validate_state(state):
                self.logger.error(f"Invalid state for workflow {workflow_id}")
                return False

            self._states[workflow_id] = state.copy()

            # Create or update metadata
            if metadata is None:
                if workflow_id in self._metadata:
                    metadata = self._metadata[workflow_id]
                    metadata.updated_at = time.time()
                    metadata.state_version += 1
                else:
                    metadata = WorkflowStateMetadata(workflow_id=workflow_id)

            self._metadata[workflow_id] = metadata

            self.logger.debug(f"Saved state for workflow {workflow_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save state for workflow {workflow_id}: {e}")
            return False

    def load_state(self, workflow_id: str) -> Optional[StateType]:
        """
        Load workflow state from memory.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            Workflow state if found, None otherwise
        """
        try:
            state = self._states.get(workflow_id)
            if state is not None:
                self.logger.debug(f"Loaded state for workflow {workflow_id}")
                return state.copy()
            return None

        except Exception as e:
            self.logger.error(f"Failed to load state for workflow {workflow_id}: {e}")
            return None

    def delete_state(self, workflow_id: str) -> bool:
        """
        Delete workflow state from memory.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            True if delete successful, False otherwise
        """
        try:
            if workflow_id in self._states:
                del self._states[workflow_id]

            if workflow_id in self._metadata:
                del self._metadata[workflow_id]

            self.logger.debug(f"Deleted state for workflow {workflow_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete state for workflow {workflow_id}: {e}")
            return False

    def list_states(self) -> List[str]:
        """
        List all stored workflow IDs.

        Returns:
            List of workflow identifiers
        """
        return list(self._states.keys())

    def get_state_metadata(self, workflow_id: str) -> Optional[WorkflowStateMetadata]:
        """
        Get state metadata from memory.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            State metadata if found, None otherwise
        """
        return self._metadata.get(workflow_id)


class FileStateManager(StateManager):
    """
    File-based state manager implementation.

    Stores workflow states as files on the filesystem. States persist across
    application restarts. Suitable for single-instance deployments.
    """

    def __init__(self, state_dir: Optional[str] = None, **kwargs):
        """
        Initialize file state manager.

        Args:
            state_dir: Directory to store state files
        """
        super().__init__(backend=StateBackend.FILE, **kwargs)
        self.state_dir = Path(state_dir or "workflow_states")
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _get_state_file_path(self, workflow_id: str) -> Path:
        """
        Get file path for workflow state.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            Path to state file
        """
        extension = (
            "json"
            if self.serialization_format == StateSerializationFormat.JSON
            else "pkl"
        )
        return self.state_dir / f"{workflow_id}.{extension}"

    def _get_metadata_file_path(self, workflow_id: str) -> Path:
        """
        Get file path for workflow metadata.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            Path to metadata file
        """
        return self.state_dir / f"{workflow_id}_metadata.json"

    def save_state(
        self,
        workflow_id: str,
        state: StateType,
        metadata: Optional[WorkflowStateMetadata] = None,
    ) -> bool:
        """
        Save workflow state to file.

        Args:
            workflow_id: Unique workflow identifier
            state: Workflow state to save
            metadata: Optional state metadata

        Returns:
            True if save successful, False otherwise
        """
        try:
            if not self.validate_state(state):
                self.logger.error(f"Invalid state for workflow {workflow_id}")
                return False

            # Save state data
            state_file = self._get_state_file_path(workflow_id)
            serialized_data = self._serialize_state(state)

            with open(state_file, "wb") as f:
                f.write(serialized_data)

            # Save metadata
            if metadata is None:
                metadata = self.get_state_metadata(workflow_id)
                if metadata:
                    # Existing metadata found, update it
                    metadata.updated_at = time.time()
                    metadata.state_version += 1
                else:
                    # No existing metadata, create new
                    metadata = WorkflowStateMetadata(
                        workflow_id=workflow_id, state_checksum=None
                    )
            else:
                # Metadata was passed in, update it
                metadata.updated_at = time.time()
                metadata.state_version += 1

            metadata_file = self._get_metadata_file_path(workflow_id)
            with open(metadata_file, "w") as f:
                json.dump(metadata.model_dump(), f, indent=2)

            self.logger.debug(f"Saved state for workflow {workflow_id} to {state_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save state for workflow {workflow_id}: {e}")
            return False

    def load_state(self, workflow_id: str) -> Optional[StateType]:
        """
        Load workflow state from file.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            Workflow state if found, None otherwise
        """
        try:
            state_file = self._get_state_file_path(workflow_id)

            if not state_file.exists():
                return None

            with open(state_file, "rb") as f:
                data = f.read()

            state = self._deserialize_state(data)
            self.logger.debug(
                f"Loaded state for workflow {workflow_id} from {state_file}"
            )
            return state

        except Exception as e:
            self.logger.error(f"Failed to load state for workflow {workflow_id}: {e}")
            return None

    def delete_state(self, workflow_id: str) -> bool:
        """
        Delete workflow state files.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            True if delete successful, False otherwise
        """
        try:
            state_file = self._get_state_file_path(workflow_id)
            metadata_file = self._get_metadata_file_path(workflow_id)

            if state_file.exists():
                state_file.unlink()

            if metadata_file.exists():
                metadata_file.unlink()

            self.logger.debug(f"Deleted state files for workflow {workflow_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete state for workflow {workflow_id}: {e}")
            return False

    def list_states(self) -> List[str]:
        """
        List all stored workflow IDs from files.

        Returns:
            List of workflow identifiers
        """
        try:
            workflow_ids = []
            extension = (
                "json"
                if self.serialization_format == StateSerializationFormat.JSON
                else "pkl"
            )

            for file_path in self.state_dir.glob(f"*.{extension}"):
                if not file_path.name.endswith("_metadata.json"):
                    workflow_id = file_path.stem
                    workflow_ids.append(workflow_id)

            return workflow_ids

        except Exception as e:
            self.logger.error(f"Failed to list states: {e}")
            return []

    def get_state_metadata(self, workflow_id: str) -> Optional[WorkflowStateMetadata]:
        """
        Get state metadata from file.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            State metadata if found, None otherwise
        """
        try:
            metadata_file = self._get_metadata_file_path(workflow_id)

            if not metadata_file.exists():
                return None

            with open(metadata_file, "r") as f:
                metadata_data = json.load(f)

            return WorkflowStateMetadata(**metadata_data)

        except Exception as e:
            self.logger.error(
                f"Failed to load metadata for workflow {workflow_id}: {e}"
            )
            return None


class StateManagerFactory:
    """
    Factory class for creating state manager instances.

    Creates appropriate state manager based on configuration settings.
    """

    @staticmethod
    def create(backend: Optional[StateBackend] = None, **kwargs) -> StateManager:
        """
        Create state manager instance.

        Args:
            backend: State persistence backend (defaults to settings)
            **kwargs: Additional arguments for state manager

        Returns:
            StateManager instance

        Raises:
            ValueError: If backend is not supported
        """
        if backend is None:
            # Get from settings or default to memory
            backend_str = getattr(settings, "WORKFLOW_STATE_BACKEND", "memory")
            if hasattr(settings, "WORKFLOW_STATE_BACKEND"):
                backend_str = settings.workflow.state_backend
            else:
                backend_str = "memory"
            backend = StateBackend(backend_str)

        if backend == StateBackend.MEMORY:
            return MemoryStateManager(**kwargs)
        elif backend == StateBackend.FILE:
            return FileStateManager(**kwargs)
        elif backend == StateBackend.DATABASE:
            # TODO: Implement database state manager
            logger.warning(
                "Database state manager not implemented, falling back to file"
            )
            return FileStateManager(**kwargs)
        else:
            raise ValueError(f"Unsupported state backend: {backend}")
