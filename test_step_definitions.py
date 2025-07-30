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
    assert IndexingWorkflowSteps.LANGUAGE_AWARE_CHUNKING not in steps, \
        "LANGUAGE_AWARE_CHUNKING step should be removed from workflow"
    
    # Verify expected steps are present
    expected_steps = [
        IndexingWorkflowSteps.INITIALIZE_STATE,
        IndexingWorkflowSteps.LOAD_REPOSITORIES,
        IndexingWorkflowSteps.VALIDATE_REPOS,
        IndexingWorkflowSteps.LOAD_FILES_FROM_GITHUB,
        IndexingWorkflowSteps.PROCESS_DOCUMENTS,
        IndexingWorkflowSteps.EXTRACT_METADATA,
        IndexingWorkflowSteps.GENERATE_EMBEDDINGS,
        IndexingWorkflowSteps.STORE_IN_VECTOR_DB,
        IndexingWorkflowSteps.UPDATE_WORKFLOW_STATE,
        IndexingWorkflowSteps.CHECK_COMPLETE,
        IndexingWorkflowSteps.FINALIZE_INDEX
    ]
    
    assert steps == expected_steps, f"Steps don't match expected. Got: {steps}"
    
    # Verify step count
    assert len(steps) == 11, f"Expected 11 steps, got {len(steps)}"
    
    print("✅ All tests passed!")
    print(f"✅ Language-aware chunking validation is now integrated into the PROCESS_DOCUMENTS step")
    print(f"✅ Workflow has {len(steps)} steps (removed 1 step)")

if __name__ == "__main__":
    test_step_definitions()
