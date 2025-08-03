"""
Performance comparison tests for refactored QueryWorkflow.

Tests to ensure the refactored system maintains or improves performance
compared to the original monolithic implementation.
"""

import time
import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.workflows.query_workflow import QueryWorkflow


class TestPerformanceComparison:
    """Test suite for performance comparison."""

    def setup_method(self):
        """Set up test fixtures."""
        self.workflow = QueryWorkflow(collection_name="test-collection")

    def test_initialization_performance(self):
        """Test that workflow initialization is fast."""
        start_time = time.time()
        
        # Create multiple workflow instances
        workflows = []
        for i in range(10):
            workflow = QueryWorkflow(collection_name=f"test-{i}")
            workflows.append(workflow)
        
        end_time = time.time()
        initialization_time = end_time - start_time
        
        # Should initialize 10 workflows in less than 1 second
        assert initialization_time < 1.0, f"Initialization took {initialization_time:.3f}s"
        print(f"✓ Initialized 10 workflows in {initialization_time:.3f}s")

    def test_step_definition_performance(self):
        """Test that step definition is fast."""
        start_time = time.time()
        
        # Call define_steps multiple times
        for _ in range(100):
            steps = self.workflow.define_steps()
        
        end_time = time.time()
        definition_time = end_time - start_time
        
        # Should define steps 100 times in less than 0.1 seconds
        assert definition_time < 0.1, f"Step definition took {definition_time:.3f}s"
        print(f"✓ Defined steps 100 times in {definition_time:.3f}s")

    def test_state_validation_performance(self):
        """Test that state validation is fast."""
        test_state = {
            "workflow_id": "test-123",
            "original_query": "test query",
            "workflow_type": "query",
            "status": "not_started"
        }
        
        start_time = time.time()
        
        # Validate state multiple times
        for _ in range(1000):
            is_valid = self.workflow.validate_state(test_state)
        
        end_time = time.time()
        validation_time = end_time - start_time
        
        # Should validate state 1000 times in less than 0.1 seconds
        assert validation_time < 0.1, f"State validation took {validation_time:.3f}s"
        print(f"✓ Validated state 1000 times in {validation_time:.3f}s")

    def test_orchestrator_delegation_overhead(self):
        """Test that orchestrator delegation doesn't add significant overhead."""
        test_state = {
            "workflow_id": "test-123",
            "original_query": "test query",
            "processed_query": "test query"
        }
        
        with patch.object(self.workflow.orchestrator, 'execute_step') as mock_execute:
            mock_execute.return_value = test_state
            
            start_time = time.time()
            
            # Execute steps multiple times
            for _ in range(100):
                result = self.workflow.execute_step("parse_and_analyze", test_state)
            
            end_time = time.time()
            delegation_time = end_time - start_time
            
            # Should execute 100 delegated steps in less than 0.1 seconds
            assert delegation_time < 0.1, f"Delegation took {delegation_time:.3f}s"
            print(f"✓ Executed 100 delegated steps in {delegation_time:.3f}s")

    def test_memory_usage_efficiency(self):
        """Test that the refactored system doesn't use excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple workflow instances
        workflows = []
        for i in range(50):
            workflow = QueryWorkflow(collection_name=f"test-{i}")
            workflows.append(workflow)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not increase memory by more than 100MB for 50 workflows
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
        print(f"✓ Memory increase for 50 workflows: {memory_increase:.1f}MB")

    @pytest.mark.asyncio
    async def test_workflow_execution_performance(self):
        """Test that workflow execution performance is maintained."""
        with patch.object(self.workflow.orchestrator, 'execute_workflow') as mock_execute:
            # Mock a fast response
            mock_state = {
                "workflow_id": "test-123",
                "original_query": "test query",
                "status": "completed",
                "total_query_time": 0.1,
                "llm_generation": {"generated_response": "test response"}
            }
            mock_execute.return_value = mock_state
            
            start_time = time.time()
            
            # Execute workflow
            result = await self.workflow.run(query="test query")
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should execute in less than 0.01 seconds (excluding actual LLM/vector calls)
            assert execution_time < 0.01, f"Workflow execution took {execution_time:.3f}s"
            print(f"✓ Workflow execution completed in {execution_time:.3f}s")

    def test_component_isolation_performance(self):
        """Test that component isolation doesn't impact performance."""
        # Test that each handler can be used independently without overhead
        from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler
        from src.workflows.query.handlers.vector_search_handler import VectorSearchHandler
        from src.workflows.query.handlers.llm_generation_handler import LLMGenerationHandler
        from src.workflows.query.handlers.context_processing_handler import ContextProcessingHandler
        
        start_time = time.time()
        
        # Create multiple handler instances
        for _ in range(20):
            parsing_handler = QueryParsingHandler()
            search_handler = VectorSearchHandler()
            llm_handler = LLMGenerationHandler()
            context_handler = ContextProcessingHandler()
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Should create 80 handlers (20 * 4) in less than 0.5 seconds
        assert creation_time < 0.5, f"Handler creation took {creation_time:.3f}s"
        print(f"✓ Created 80 handlers in {creation_time:.3f}s")

    def test_backward_compatibility_performance(self):
        """Test that backward compatibility methods don't add overhead."""
        start_time = time.time()
        
        # Test backward compatibility methods multiple times
        for _ in range(100):
            # These should delegate efficiently
            self.workflow.validate_state({"original_query": "test"})
            steps = self.workflow.define_steps()
        
        end_time = time.time()
        compatibility_time = end_time - start_time
        
        # Should execute 200 compatibility calls in less than 0.1 seconds
        assert compatibility_time < 0.1, f"Compatibility calls took {compatibility_time:.3f}s"
        print(f"✓ Executed 200 compatibility calls in {compatibility_time:.3f}s")

    def test_configuration_overhead(self):
        """Test that configuration doesn't add significant overhead."""
        start_time = time.time()
        
        # Create workflows with different configurations
        configs = [
            {"default_k": 4, "max_k": 20},
            {"default_k": 8, "max_k": 40},
            {"min_context_length": 100, "max_context_length": 8000},
            {"response_quality_threshold": 0.7},
            {"collection_name": "test-collection"}
        ]
        
        workflows = []
        for config in configs * 10:  # 50 total workflows
            workflow = QueryWorkflow(**config)
            workflows.append(workflow)
        
        end_time = time.time()
        config_time = end_time - start_time
        
        # Should create 50 configured workflows in less than 2 seconds
        assert config_time < 2.0, f"Configuration took {config_time:.3f}s"
        print(f"✓ Created 50 configured workflows in {config_time:.3f}s")


def run_performance_benchmark():
    """Run a comprehensive performance benchmark."""
    print("🚀 Running Performance Benchmark for Refactored QueryWorkflow")
    print("=" * 60)
    
    test_instance = TestPerformanceComparison()
    test_instance.setup_method()
    
    # Run all performance tests
    tests = [
        test_instance.test_initialization_performance,
        test_instance.test_step_definition_performance,
        test_instance.test_state_validation_performance,
        test_instance.test_orchestrator_delegation_overhead,
        test_instance.test_memory_usage_efficiency,
        test_instance.test_component_isolation_performance,
        test_instance.test_backward_compatibility_performance,
        test_instance.test_configuration_overhead,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
    
    print("=" * 60)
    print("✅ Performance benchmark completed!")


if __name__ == "__main__":
    run_performance_benchmark()
