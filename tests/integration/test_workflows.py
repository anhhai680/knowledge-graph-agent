"""
Integration tests for workflow infrastructure.

Tests for end-to-end workflow execution with real components.
"""

import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

import pytest

from src.workflows.base_workflow import BaseWorkflow, WorkflowStatus
from src.workflows.state_manager import (
    StateManagerFactory,
    StateBackend,
    MemoryStateManager,
    FileStateManager,
)
from src.workflows.workflow_states import (
    WorkflowType,
    ProcessingStatus,
    SearchStrategy,
    create_indexing_state,
    create_query_state,
    update_workflow_progress,
)


class IntegrationTestWorkflow(BaseWorkflow):
    """Test workflow implementation for integration testing."""

    def __init__(self, simulate_processing_time: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.simulate_processing_time = simulate_processing_time
        self.processing_results = []

    def define_steps(self) -> List[str]:
        return [
            "initialize",
            "load_data",
            "process_data",
            "generate_embeddings",
            "store_results",
            "finalize",
        ]

    def execute_step(self, step: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow step with realistic processing simulation."""
        new_state = state.copy()

        if self.simulate_processing_time:
            time.sleep(0.01)  # Simulate processing time

        if step == "initialize":
            new_state.update(
                {
                    "initialized": True,
                    "data_sources": ["source1", "source2", "source3"],
                    "total_items": 100,
                }
            )

        elif step == "load_data":
            new_state.update(
                {"loaded_items": new_state.get("total_items", 0), "data_loaded": True}
            )

        elif step == "process_data":
            total_items = new_state.get("loaded_items", 0)
            processed_items = int(total_items * 0.8)  # Simulate 80% success rate
            new_state.update(
                {
                    "processed_items": processed_items,
                    "failed_items": total_items - processed_items,
                    "data_processed": True,
                }
            )

        elif step == "generate_embeddings":
            processed_items = new_state.get("processed_items", 0)
            embeddings_generated = int(
                processed_items * 0.95
            )  # Simulate 95% success rate
            new_state.update(
                {
                    "embeddings_generated": embeddings_generated,
                    "embedding_failures": processed_items - embeddings_generated,
                    "embeddings_complete": True,
                }
            )

        elif step == "store_results":
            embeddings = new_state.get("embeddings_generated", 0)
            new_state.update(
                {"stored_embeddings": embeddings, "storage_complete": True}
            )

        elif step == "finalize":
            new_state.update({"workflow_complete": True, "final_status": "success"})

        # Record processing result
        self.processing_results.append(
            {
                "step": step,
                "timestamp": time.time(),
                "state_keys": list(new_state.keys()),
            }
        )

        return new_state

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate workflow state structure."""
        required_fields = ["workflow_id"]
        return all(field in state for field in required_fields)

    def handle_error(
        self, step: str, state: Dict[str, Any], error: Exception
    ) -> Dict[str, Any]:
        """Custom error handling for integration tests."""
        error_state = state.copy()
        error_state.update(
            {
                "error_handled": True,
                "error_step": step,
                "error_message": str(error),
                "recovery_attempted": True,
            }
        )
        return error_state


class TestWorkflowIntegration:
    """Integration tests for workflow system."""

    def test_complete_workflow_execution(self):
        """Test complete workflow execution with all components."""
        workflow = IntegrationTestWorkflow()
        initial_state = {
            "workflow_id": str(uuid.uuid4()),
            "workflow_type": "integration_test",
        }

        # Execute workflow
        result_state = workflow.invoke(initial_state)

        # Verify workflow completion
        assert workflow.status == WorkflowStatus.COMPLETED
        assert workflow.progress == 100.0
        assert len(workflow.processing_results) == 6  # All steps executed

        # Verify final state
        assert result_state["workflow_complete"] is True
        assert result_state["final_status"] == "success"
        assert result_state["embeddings_generated"] > 0
        assert result_state["stored_embeddings"] > 0

        # Verify metadata
        metadata = workflow.get_metadata()
        assert metadata["status"] == WorkflowStatus.COMPLETED.value
        assert metadata["progress_percentage"] == 100.0
        assert metadata["duration"] > 0
        assert len(metadata["executed_steps"]) == 6

    @pytest.mark.xfail(
        reason="Persistence is not yet implemented in BaseWorkflow._persist_state"
    )
    def test_workflow_with_state_persistence(self):
        """Test workflow execution with state persistence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create workflow with file-based state persistence
            workflow = IntegrationTestWorkflow(
                enable_persistence=True, simulate_processing_time=False
            )

            initial_state = {
                "workflow_id": workflow.workflow_id,
                "workflow_type": "persistence_test",
            }

            # Execute workflow
            result_state = workflow.invoke(initial_state)

            # Verify execution
            assert workflow.status == WorkflowStatus.COMPLETED
            assert result_state["workflow_complete"] is True

    def test_workflow_error_handling_and_recovery(self):
        """Test workflow error handling and recovery mechanisms."""

        class FailingWorkflow(IntegrationTestWorkflow):
            def execute_step(self, step: str, state: Dict[str, Any]) -> Dict[str, Any]:
                if step == "process_data":
                    raise ValueError("Simulated processing error")
                return super().execute_step(step, state)

        workflow = FailingWorkflow(simulate_processing_time=False)
        initial_state = {
            "workflow_id": str(uuid.uuid4()),
            "workflow_type": "error_test",
        }

        # Execute workflow (should handle error)
        result_state = workflow.invoke(initial_state)

        # Verify error was handled
        assert "error_handled" in result_state
        assert result_state["error_step"] == "process_data"
        assert result_state["recovery_attempted"] is True

        # Verify workflow failed after error handling (actual behavior)
        assert workflow.status == WorkflowStatus.FAILED

    @patch("src.workflows.base_workflow.VectorStoreFactory")
    def test_workflow_vector_store_integration(self, mock_factory):
        """Test workflow integration with vector store factory."""
        mock_store = Mock()
        mock_factory.return_value.create.return_value = mock_store

        workflow = IntegrationTestWorkflow()

        # Test vector store access
        vector_store = workflow.get_vector_store("test_collection")

        # Verify factory was called correctly
        mock_factory.return_value.create.assert_called_with(
            collection_name="test_collection"
        )
        assert vector_store == mock_store

    def test_concurrent_workflow_execution(self):
        """Test concurrent execution of multiple workflows."""
        import threading
        import queue

        results_queue = queue.Queue()

        def execute_workflow(workflow_id: str):
            """Execute workflow in thread."""
            try:
                workflow = IntegrationTestWorkflow(simulate_processing_time=False)
                initial_state = {
                    "workflow_id": workflow_id,
                    "workflow_type": "concurrent_test",
                }

                result_state = workflow.invoke(initial_state)
                results_queue.put(
                    {
                        "workflow_id": workflow_id,
                        "status": workflow.status,
                        "success": result_state.get("workflow_complete", False),
                    }
                )

            except Exception as e:
                results_queue.put(
                    {"workflow_id": workflow_id, "status": "error", "error": str(e)}
                )

        # Start multiple workflows concurrently
        threads = []
        workflow_ids = [f"workflow-{i}" for i in range(5)]

        for wid in workflow_ids:
            thread = threading.Thread(target=execute_workflow, args=(wid,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Verify all workflows completed successfully
        assert len(results) == 5
        for result in results:
            # Normalize status to enum for consistent comparison
            normalized_status = (
                WorkflowStatus(result["status"])
                if isinstance(result["status"], str)
                else result["status"]
            )
            assert normalized_status == WorkflowStatus.COMPLETED
            if "success" in result:
                assert result["success"] is True


class TestStateManagerIntegration:
    """Integration tests for state manager components."""

    def test_state_manager_with_workflows(self):
        """Test state manager integration with workflows."""
        state_manager = StateManagerFactory.create(StateBackend.MEMORY)

        # Create and save workflow states
        indexing_state = create_indexing_state(
            "indexing-workflow", ["repo1", "repo2"], "chroma", "test-collection"
        )

        query_state = create_query_state(
            "query-workflow", "test query", SearchStrategy.SEMANTIC
        )

        # Save states
        assert state_manager.save_state("indexing-workflow", indexing_state) is True
        assert state_manager.save_state("query-workflow", query_state) is True

        # Load and verify states
        loaded_indexing = state_manager.load_state("indexing-workflow")
        loaded_query = state_manager.load_state("query-workflow")

        assert loaded_indexing["workflow_type"] == WorkflowType.INDEXING
        assert loaded_indexing["repositories"] == ["repo1", "repo2"]

        assert loaded_query["workflow_type"] == WorkflowType.QUERY
        assert loaded_query["original_query"] == "test query"

        # Test state updates
        update_workflow_progress(loaded_indexing, 50.0, "processing")
        assert state_manager.save_state("indexing-workflow", loaded_indexing) is True

        updated_state = state_manager.load_state("indexing-workflow")
        assert updated_state["progress_percentage"] == 50.0
        assert updated_state["current_step"] == "processing"

    def test_file_state_manager_persistence(self):
        """Test file-based state manager persistence across instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first manager instance and save state
            manager1 = FileStateManager(state_dir=temp_dir)
            workflow_id = "persistent-workflow"
            state = create_indexing_state(workflow_id, ["repo1"])

            assert manager1.save_state(workflow_id, state) is True

            # Create second manager instance and load state
            manager2 = FileStateManager(state_dir=temp_dir)
            loaded_state = manager2.load_state(workflow_id)

            assert loaded_state is not None
            assert loaded_state["workflow_id"] == workflow_id
            assert loaded_state["repositories"] == ["repo1"]

            # Verify metadata persistence
            metadata1 = manager1.get_state_metadata(workflow_id)
            metadata2 = manager2.get_state_metadata(workflow_id)

            assert metadata1 is not None
            assert metadata2 is not None
            assert metadata1.workflow_id == metadata2.workflow_id
            assert metadata1.state_version == metadata2.state_version

    def test_state_manager_error_scenarios(self):
        """Test state manager error handling scenarios."""
        manager = MemoryStateManager()

        # Test loading non-existent state
        result = manager.load_state("non-existent")
        assert result is None

        # Test saving invalid state
        invalid_state = {"invalid": True}  # Missing required fields
        success = manager.save_state("invalid", invalid_state)
        assert success is False

        # Test deleting non-existent state (should succeed gracefully)
        success = manager.delete_state("non-existent")
        assert success is True


class TestWorkflowStateIntegration:
    """Integration tests for workflow state schemas."""

    def test_indexing_state_lifecycle(self):
        """Test complete indexing state lifecycle."""
        workflow_id = str(uuid.uuid4())
        repositories = ["repo1", "repo2", "repo3"]

        # Create initial state
        state = create_indexing_state(workflow_id, repositories)
        assert state["status"] == ProcessingStatus.NOT_STARTED

        # Start processing
        state["status"] = ProcessingStatus.IN_PROGRESS
        update_workflow_progress(state, 10.0, "initialize")

        # Process repositories
        for i, repo in enumerate(repositories):
            state["current_repo"] = repo
            progress = ((i + 1) / len(repositories)) * 80  # 80% for processing
            update_workflow_progress(state, progress, f"processing_{repo}")

            # Simulate file processing
            state["processed_files"] += 50
            state["total_files"] += 60
            state["embeddings_generated"] += 45

        # Finalize
        state["status"] = ProcessingStatus.COMPLETED
        update_workflow_progress(state, 100.0, "finalize")

        # Verify final state
        assert state["status"] == ProcessingStatus.COMPLETED
        assert state["progress_percentage"] == 100.0
        assert state["processed_files"] == 150
        assert state["embeddings_generated"] == 135

    def test_query_state_lifecycle(self):
        """Test complete query state lifecycle."""
        workflow_id = str(uuid.uuid4())
        query = "How to implement authentication?"

        # Create initial state
        state = create_query_state(workflow_id, query)
        assert state["original_query"] == query

        # Process query
        state["status"] = ProcessingStatus.IN_PROGRESS
        state["processed_query"] = query.lower()
        update_workflow_progress(state, 20.0, "parse_query")

        # Retrieve documents
        state["document_retrieval"]["retrieved_documents"] = [
            {"content": "doc1", "score": 0.9},
            {"content": "doc2", "score": 0.8},
        ]
        state["document_retrieval"]["relevance_scores"] = [0.9, 0.8]
        update_workflow_progress(state, 60.0, "retrieve_documents")

        # Generate response
        state["llm_generation"][
            "generated_response"
        ] = "Authentication can be implemented using..."
        state["llm_generation"]["token_usage"] = {
            "prompt_tokens": 150,
            "completion_tokens": 200,
            "total_tokens": 350,
        }
        update_workflow_progress(state, 90.0, "generate_response")

        # Finalize
        state["status"] = ProcessingStatus.COMPLETED
        state["response_quality_score"] = 0.85
        update_workflow_progress(state, 100.0, "finalize")

        # Verify final state
        assert state["status"] == ProcessingStatus.COMPLETED
        assert state["llm_generation"]["generated_response"] is not None
        assert len(state["document_retrieval"]["retrieved_documents"]) == 2
        assert state["response_quality_score"] == 0.85


if __name__ == "__main__":
    pytest.main([__file__])
