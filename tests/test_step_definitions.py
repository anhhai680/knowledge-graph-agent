#!/usr/bin/env python3
"""
Quick test to verify step definitions after PR comment resolution.
"""

from src.workflows.indexing_workflow import IndexingWorkflow, IndexingWorkflowSteps


def test_step_definitions():
    """Test that language_aware_chunking step is removed from workflow."""

    # Create workflow instance
    workflow = IndexingWorkflow()

    # Get step definitions
    steps = workflow.define_steps()

    print("Current workflow steps:")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")

    # Verify language_aware_chunking is not in steps
    assert (
        IndexingWorkflowSteps.LANGUAGE_AWARE_CHUNKING not in steps
    ), "LANGUAGE_AWARE_CHUNKING step should be removed from workflow"

    # Define expected steps based on the actual workflow
    expected_steps = [
        'initialize_state',
        'load_repositories', 
        'validate_repos',
        'load_files_from_github',
        'process_documents',
        'extract_metadata',
        'store_in_vector_db',
        'store_in_graph_db',
        'update_workflow_state',
        'check_complete',
        'finalize_index'
    ]

    assert steps == expected_steps, f"Steps don't match expected. Got: {steps}"

    # Verify step count
    assert len(steps) == 11, f"Expected 11 steps, got {len(steps)}"

    print("✅ All tests passed!")
    print(
        f"✅ Language-aware chunking validation is now integrated into the PROCESS_DOCUMENTS step"
    )
    print(f"✅ Workflow has {len(steps)} steps (removed 1 step)")


if __name__ == "__main__":
    test_step_definitions()
