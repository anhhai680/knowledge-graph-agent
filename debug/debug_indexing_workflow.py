#!/usr/bin/env python3
"""
Debug script to test the indexing workflow and identify where it fails.

This script will run the indexing workflow step by step to identify where
the documents are getting lost.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.workflows.indexing_workflow import IndexingWorkflow
from src.workflows.workflow_states import create_indexing_state

def test_indexing_workflow():
    """Test the indexing workflow to identify where it fails."""
    print("🚀 Testing indexing workflow...")
    
    try:
        # Create indexing workflow (test with just one repository)
        workflow = IndexingWorkflow(
            repositories=["car-notification-service"],  # Test with one repo
            vector_store_type="chroma"
        )
        
        print(f"Created workflow for: {workflow.target_repositories}")
        
        # Create initial state
        initial_state = create_indexing_state(
            workflow_id="debug-test",
            repositories=["car-notification-service"]
        )
        
        print(f"Created initial state")
        
        # Run workflow steps one by one to see where it fails
        steps = workflow.define_steps()
        print(f"Workflow steps: {steps}")
        
        current_state = initial_state
        
        for i, step in enumerate(steps):
            print(f"\n{'='*50}")
            print(f"Step {i+1}/{len(steps)}: {step}")
            print(f"{'='*50}")
            
            try:
                current_state = workflow.execute_step(step, current_state)
                
                # Check state after each step
                print(f"✅ Step {step} completed successfully")
                print(f"   Status: {current_state.get('status', 'unknown')}")
                print(f"   Progress: {current_state.get('progress_percentage', 0):.1f}%")
                print(f"   Errors: {len(current_state.get('errors', []))}")
                
                # Show specific information for key steps
                if step == "load_files_from_github":
                    loaded_docs = current_state.get("metadata", {}).get("loaded_documents", [])
                    print(f"   Loaded documents: {len(loaded_docs)}")
                    print(f"   Total files: {current_state.get('total_files', 0)}")
                    
                elif step == "process_documents":
                    processed_docs = current_state.get("metadata", {}).get("processed_documents", [])
                    print(f"   Processed documents: {len(processed_docs)}")
                    print(f"   Total chunks: {current_state.get('total_chunks', 0)}")
                    
                elif step == "store_in_vector_db":
                    storage_stats = current_state.get("metadata", {}).get("storage_stats", {})
                    print(f"   Storage stats: {storage_stats}")
                    print(f"   Successful embeddings: {current_state.get('successful_embeddings', 0)}")
                    
                # Check repository states
                repo_states = current_state.get("repository_states", {})
                for repo_name, repo_state in repo_states.items():
                    print(f"   Repository {repo_name}: {repo_state.get('status', 'unknown')}")
                    if repo_state.get("errors"):
                        print(f"     Errors: {repo_state['errors']}")
                
            except Exception as e:
                print(f"❌ Step {step} failed: {e}")
                
                # Show detailed error information
                import traceback
                traceback.print_exc()
                
                # Show state errors
                errors = current_state.get("errors", [])
                if errors:
                    print(f"\nState errors ({len(errors)}):")
                    for error in errors[-3:]:  # Show last 3 errors
                        print(f"  - {error.get('message', 'Unknown error')}")
                
                break
                
        print(f"\n{'='*50}")
        print("FINAL STATE SUMMARY")
        print(f"{'='*50}")
        print(f"Final status: {current_state.get('status', 'unknown')}")
        print(f"Total errors: {len(current_state.get('errors', []))}")
        print(f"Total files: {current_state.get('total_files', 0)}")
        print(f"Total chunks: {current_state.get('total_chunks', 0)}")
        print(f"Successful embeddings: {current_state.get('successful_embeddings', 0)}")
        
        # Show repository final states
        repo_states = current_state.get("repository_states", {})
        for repo_name, repo_state in repo_states.items():
            print(f"\nRepository {repo_name}:")
            print(f"  Status: {repo_state.get('status', 'unknown')}")
            print(f"  Total files: {repo_state.get('total_files', 0)}")
            print(f"  Processed files: {repo_state.get('processed_files', 0)}")
            if repo_state.get("errors"):
                print(f"  Errors: {repo_state['errors']}")
                
    except Exception as e:
        print(f"❌ Error in indexing workflow test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Debug: Indexing Workflow Test")
    print("=" * 60)
    
    try:
        test_indexing_workflow()
    except Exception as e:
        print(f"❌ Error in debug script: {e}")
        import traceback
        traceback.print_exc()
