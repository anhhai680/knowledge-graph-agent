"""
Unit tests for workflow infrastructure.

Tests for base workflow, state manager, and workflow state schemas.
"""

import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

import pytest

from src.workflows.base_workflow import (
    BaseWorkflow,
    WorkflowStatus,
    WorkflowStep,
    WorkflowMetadata,
)
from src.workflows.state_manager import (
    StateManager,
    MemoryStateManager,
    FileStateManager,
    StateBackend,
    StateSerializationFormat,
    WorkflowStateMetadata,
    StateManagerFactory,
)
from src.workflows.workflow_states import (
    WorkflowType,
    ProcessingStatus,
    IndexingState,
    QueryState,
    create_indexing_state,
    create_query_state,
    update_workflow_progress,
    add_workflow_error,
    is_workflow_complete,
    get_workflow_duration,
)


class TestWorkflowMetadata:
    """Test cases for WorkflowMetadata class."""

    def test_metadata_initialization(self):
        """Test metadata initialization."""
        metadata = WorkflowMetadata()

        assert isinstance(metadata.id, str)
        assert metadata.status == WorkflowStatus.PENDING
        assert metadata.executed_steps == []
        assert metadata.start_time is None
        assert metadata.end_time is None
        assert metadata.duration is None
        assert metadata.errors == []
        assert metadata.retry_count == 0
        assert metadata.current_step is None
        assert metadata.progress_percentage == 0.0

    def test_metadata_with_custom_id(self):
        """Test metadata initialization with custom ID."""
        custom_id = "test-workflow-123"
        metadata = WorkflowMetadata(custom_id)

        assert metadata.id == custom_id

    def test_start_workflow(self):
        """Test workflow start."""
        metadata = WorkflowMetadata()
        start_time = time.time()

        metadata.start()

        assert metadata.status == WorkflowStatus.RUNNING
        assert metadata.start_time >= start_time

    def test_complete_workflow(self):
        """Test workflow completion."""
        metadata = WorkflowMetadata()
        metadata.start()
        time.sleep(0.01)  # Small delay for duration calculation

        metadata.complete()

        assert metadata.status == WorkflowStatus.COMPLETED
        assert metadata.progress_percentage == 100.0
        assert metadata.end_time is not None
        assert metadata.duration is not None
        assert metadata.duration > 0

    def test_fail_workflow(self):
        """Test workflow failure."""
        metadata = WorkflowMetadata()
        metadata.start()
        error = ValueError("Test error")
        step = "test_step"

        metadata.fail(error, step)

        assert metadata.status == WorkflowStatus.FAILED
        assert len(metadata.errors) == 1
        assert metadata.errors[0]["error_type"] == "ValueError"
        assert metadata.errors[0]["error_message"] == "Test error"
        assert metadata.errors[0]["step"] == step

    def test_add_step(self):
        """Test adding executed steps."""
        metadata = WorkflowMetadata()

        metadata.add_step("step1")
        metadata.add_step("step2")
        metadata.add_step("step1")  # Duplicate should not be added again

        assert metadata.executed_steps == ["step1", "step2"]
        assert metadata.current_step == "step1"

    def test_increment_retry(self):
        """Test retry increment."""
        metadata = WorkflowMetadata()

        metadata.increment_retry()
        metadata.increment_retry()

        assert metadata.retry_count == 2

    def test_update_progress(self):
        """Test progress update."""
        metadata = WorkflowMetadata()

        metadata.update_progress(50.5)
        assert metadata.progress_percentage == 50.5

        # Test bounds
        metadata.update_progress(-10)
        assert metadata.progress_percentage == 0.0

        metadata.update_progress(150)
        assert metadata.progress_percentage == 100.0

    def test_to_dict(self):
        """Test metadata serialization."""
        metadata = WorkflowMetadata("test-id")
        metadata.start()
        metadata.add_step("step1")

        result = metadata.to_dict()

        assert result["id"] == "test-id"
        assert result["status"] == WorkflowStatus.RUNNING.value
        assert result["executed_steps"] == ["step1"]
        assert result["current_step"] == "step1"


class TestWorkflow(BaseWorkflow):
    """Test implementation of BaseWorkflow for testing."""

    def __init__(self, fail_step: str = None, **kwargs):
        super().__init__(**kwargs)
        self.fail_step = fail_step
        self.steps_executed = []

    def define_steps(self) -> List[str]:
        return ["initialize", "process", "finalize"]

    def execute_step(self, step: str, state: Dict[str, Any]) -> Dict[str, Any]:
        self.steps_executed.append(step)

        if self.fail_step == step:
            raise ValueError(f"Intentional failure at {step}")

        new_state = state.copy()
        new_state[f"{step}_completed"] = True
        return new_state

    def validate_state(self, state: Dict[str, Any]) -> bool:
        return isinstance(state, dict)


class TestBaseWorkflow:
    """Test cases for BaseWorkflow class."""

    def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = TestWorkflow()

        assert isinstance(workflow.workflow_id, str)
        assert workflow.status == WorkflowStatus.PENDING
        assert workflow.progress == 0.0
        assert workflow.max_retries == 3
        assert workflow.retry_delay == 5.0
        assert workflow.enable_persistence is True

    def test_workflow_execution_success(self):
        """Test successful workflow execution."""
        workflow = TestWorkflow()
        initial_state = {"test": True}

        result = workflow.invoke(initial_state)

        assert workflow.status == WorkflowStatus.COMPLETED
        assert workflow.progress == 100.0
        assert result["initialize_completed"] is True
        assert result["process_completed"] is True
        assert result["finalize_completed"] is True
        assert len(workflow.steps_executed) == 3

    def test_workflow_execution_failure(self):
        """Test workflow execution with failure."""
        workflow = TestWorkflow(fail_step="process")
        initial_state = {"test": True}

        with pytest.raises(ValueError):
            workflow.invoke(initial_state)

        assert workflow.status == WorkflowStatus.FAILED
        assert len(workflow.metadata.errors) > 0

    def test_workflow_pause_resume_cancel(self):
        """Test workflow pause, resume, and cancel operations."""
        workflow = TestWorkflow()

        # Test pause
        workflow.pause()
        assert workflow.status == WorkflowStatus.PAUSED

        # Test resume
        workflow.resume()
        assert workflow.status == WorkflowStatus.RUNNING

        # Test cancel
        workflow.cancel()
        assert workflow.status == WorkflowStatus.CANCELLED
        assert workflow.metadata.end_time is not None

    @patch("src.workflows.base_workflow.VectorStoreFactory")
    def test_get_vector_store(self, mock_factory):
        """Test vector store integration."""
        mock_store = Mock()
        mock_factory.return_value.create.return_value = mock_store

        workflow = TestWorkflow()
        result = workflow.get_vector_store("test_collection")

        mock_factory.return_value.create.assert_called_with(
            collection_name="test_collection"
        )
        assert result == mock_store


class TestMemoryStateManager:
    """Test cases for MemoryStateManager class."""

    def test_save_and_load_state(self):
        """Test saving and loading state."""
        manager = MemoryStateManager()
        workflow_id = "test-workflow"
        state = {"status": "running", "workflow_id": workflow_id, "progress": 50}

        # Save state
        success = manager.save_state(workflow_id, state)
        assert success is True

        # Load state
        loaded_state = manager.load_state(workflow_id)
        assert loaded_state == state

    def test_delete_state(self):
        """Test deleting state."""
        manager = MemoryStateManager()
        workflow_id = "test-workflow"
        state = {"status": "running", "workflow_id": workflow_id}

        manager.save_state(workflow_id, state)
        success = manager.delete_state(workflow_id)
        assert success is True

        loaded_state = manager.load_state(workflow_id)
        assert loaded_state is None

    def test_list_states(self):
        """Test listing states."""
        manager = MemoryStateManager()
        workflow_ids = ["workflow1", "workflow2", "workflow3"]

        for wid in workflow_ids:
            state = {"status": "running", "workflow_id": wid}
            manager.save_state(wid, state)

        listed_ids = manager.list_states()
        assert set(listed_ids) == set(workflow_ids)

    def test_get_state_metadata(self):
        """Test getting state metadata."""
        manager = MemoryStateManager()
        workflow_id = "test-workflow"
        state = {"status": "running", "workflow_id": workflow_id}

        manager.save_state(workflow_id, state)
        metadata = manager.get_state_metadata(workflow_id)

        assert metadata is not None
        assert metadata.workflow_id == workflow_id
        assert metadata.state_version == 1

    def test_invalid_state(self):
        """Test handling invalid state."""
        manager = MemoryStateManager()
        workflow_id = "test-workflow"
        invalid_state = {"invalid": True}  # Missing required fields

        success = manager.save_state(workflow_id, invalid_state)
        assert success is False


class TestFileStateManager:
    """Test cases for FileStateManager class."""

    def test_save_and_load_state(self):
        """Test saving and loading state to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = FileStateManager(state_dir=temp_dir)
            workflow_id = "test-workflow"
            state = {"status": "running", "workflow_id": workflow_id, "data": [1, 2, 3]}

            # Save state
            success = manager.save_state(workflow_id, state)
            assert success is True

            # Verify files exist
            state_file = Path(temp_dir) / f"{workflow_id}.json"
            metadata_file = Path(temp_dir) / f"{workflow_id}_metadata.json"
            assert state_file.exists()
            assert metadata_file.exists()

            # Load state
            loaded_state = manager.load_state(workflow_id)
            assert loaded_state == state

    def test_delete_state(self):
        """Test deleting state files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = FileStateManager(state_dir=temp_dir)
            workflow_id = "test-workflow"
            state = {"status": "running", "workflow_id": workflow_id}

            manager.save_state(workflow_id, state)
            success = manager.delete_state(workflow_id)
            assert success is True

            # Verify files are deleted
            state_file = Path(temp_dir) / f"{workflow_id}.json"
            metadata_file = Path(temp_dir) / f"{workflow_id}_metadata.json"
            assert not state_file.exists()
            assert not metadata_file.exists()

    def test_list_states(self):
        """Test listing states from files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = FileStateManager(state_dir=temp_dir)
            workflow_ids = ["workflow1", "workflow2", "workflow3"]

            for wid in workflow_ids:
                state = {"status": "running", "workflow_id": wid}
                manager.save_state(wid, state)

            listed_ids = manager.list_states()
            assert set(listed_ids) == set(workflow_ids)


class TestStateManagerFactory:
    """Test cases for StateManagerFactory class."""

    def test_create_memory_state_manager(self):
        """Test creating memory state manager."""
        manager = StateManagerFactory.create(StateBackend.MEMORY)
        assert isinstance(manager, MemoryStateManager)

    def test_create_file_state_manager(self):
        """Test creating file state manager."""
        manager = StateManagerFactory.create(StateBackend.FILE)
        assert isinstance(manager, FileStateManager)

    def test_create_default_state_manager(self):
        """Test creating default state manager."""
        manager = StateManagerFactory.create()
        assert isinstance(manager, MemoryStateManager)


class TestWorkflowStates:
    """Test cases for workflow state schemas."""

    def test_create_indexing_state(self):
        """Test creating indexing state."""
        workflow_id = str(uuid.uuid4())
        repositories = ["repo1", "repo2"]

        state = create_indexing_state(workflow_id, repositories)

        assert state["workflow_id"] == workflow_id
        assert state["workflow_type"] == WorkflowType.INDEXING
        assert state["repositories"] == repositories
        assert state["status"] == ProcessingStatus.NOT_STARTED
        assert state["processed_files"] == 0
        assert state["total_files"] == 0
        assert state["embeddings_generated"] == 0

    def test_create_query_state(self):
        """Test creating query state."""
        workflow_id = str(uuid.uuid4())
        query = "test query"

        state = create_query_state(workflow_id, query)

        assert state["workflow_id"] == workflow_id
        assert state["workflow_type"] == WorkflowType.QUERY
        assert state["original_query"] == query
        assert state["status"] == ProcessingStatus.NOT_STARTED

    def test_update_workflow_progress(self):
        """Test updating workflow progress."""
        state = create_indexing_state("test", ["repo1"])
        original_time = state["updated_at"]

        time.sleep(0.01)  # Small delay
        updated_state = update_workflow_progress(state, 75.5, "test_step")

        assert updated_state["progress_percentage"] == 75.5
        assert updated_state["current_step"] == "test_step"
        assert updated_state["updated_at"] > original_time

    def test_add_workflow_error(self):
        """Test adding workflow error."""
        state = create_indexing_state("test", ["repo1"])
        error_message = "Test error"
        step = "test_step"

        updated_state = add_workflow_error(state, error_message, step)

        assert len(updated_state["errors"]) == 1
        assert updated_state["errors"][0]["message"] == error_message
        assert updated_state["errors"][0]["step"] == step

    def test_is_workflow_complete(self):
        """Test workflow completion check."""
        state = create_indexing_state("test", ["repo1"])

        assert is_workflow_complete(state) is False

        state["status"] = ProcessingStatus.COMPLETED
        assert is_workflow_complete(state) is True

    def test_get_workflow_duration(self):
        """Test getting workflow duration."""
        state = create_indexing_state("test", ["repo1"])

        # Simulate some processing time
        time.sleep(0.01)
        state["updated_at"] = time.time()

        duration = get_workflow_duration(state)
        assert duration is not None
        assert duration > 0


if __name__ == "__main__":
    pytest.main([__file__])
