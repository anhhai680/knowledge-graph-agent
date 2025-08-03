"""
Query workflow orchestrator extending BaseWorkflow.

This module implements the main orchestrator that composes step handlers
while maintaining LangGraph integration and existing state management.
"""

import time
from typing import List, Dict, Any, Optional
from src.workflows.base_workflow import BaseWorkflow
from src.workflows.workflow_states import (
    QueryState, 
    ProcessingStatus,
    SearchStrategy,
    create_query_state, 
    update_workflow_progress
)
from ..handlers.query_parsing_handler import QueryParsingHandler
from ..handlers.vector_search_handler import VectorSearchHandler
from ..handlers.llm_generation_handler import LLMGenerationHandler
from ..handlers.context_processing_handler import ContextProcessingHandler


class QueryWorkflowOrchestrator(BaseWorkflow[QueryState]):
    """
    Main orchestrator for query workflow execution.
    
    Extends BaseWorkflow to leverage existing error handling, retry logic,
    and progress tracking while composing step handlers for modular execution.
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        default_k: int = 4,
        max_k: int = 20,
        min_context_length: int = 100,
        max_context_length: int = 8000,
        response_quality_threshold: float = 0.7,
        **kwargs
    ):
        """Initialize query workflow orchestrator."""
        super().__init__(workflow_id="query-workflow-orchestrator", **kwargs)
        
        # Store configuration
        self.collection_name = collection_name
        self.default_k = default_k
        self.max_k = max_k
        self.min_context_length = min_context_length
        self.max_context_length = max_context_length
        self.response_quality_threshold = response_quality_threshold
        
        # Initialize step handlers (each extends BaseWorkflow)
        self.parsing_handler = QueryParsingHandler(**kwargs)
        self.search_handler = VectorSearchHandler(collection_name=collection_name, **kwargs)
        self.context_handler = ContextProcessingHandler(
            max_context_length=max_context_length,
            min_context_length=min_context_length,
            **kwargs
        )
        self.llm_handler = LLMGenerationHandler(**kwargs)
        
    def define_steps(self) -> List[str]:
        """Define the main workflow steps."""
        return [
            "parse_and_analyze",
            "search_documents", 
            "process_context",
            "generate_response",
            "finalize_response"
        ]
    
    def execute_step(self, step: str, state: QueryState) -> QueryState:
        """
        Execute a single workflow step using appropriate handler.
        
        Args:
            step: Step name to execute
            state: Current query state
            
        Returns:
            Updated query state
        """
        if step == "parse_and_analyze":
            # Use parsing handler's invoke method (inherits from BaseWorkflow)
            state = self.parsing_handler.invoke(state)
            
            # Determine search strategy based on intent
            if state.get("query_intent"):
                state["search_strategy"] = self._determine_search_strategy(
                    state["query_intent"], state["processed_query"]
                )
            
        elif step == "search_documents":
            # Use search handler's invoke method
            state = self.search_handler.invoke(state)
            
        elif step == "process_context":
            # Use context handler's invoke method
            state = self.context_handler.invoke(state)
            
            # Check if context is sufficient, expand search if needed
            if not state.get("context_sufficient", True):
                state = self._expand_search(state)
                # Re-process context after expansion
                state = self.context_handler.invoke(state)
            
        elif step == "generate_response":
            # Use LLM handler's invoke method
            state = self.llm_handler.invoke(state)
            
        elif step == "finalize_response":
            # Finalize response with sources and metadata
            state = self._finalize_response(state)
            
        return state
    
    def validate_state(self, state: QueryState) -> bool:
        """Validate that the state contains required fields for orchestration."""
        return bool(state.get("original_query"))
    
    async def execute_workflow(
        self,
        query: str,
        repositories: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None,
        k: Optional[int] = None,
        **kwargs
    ) -> QueryState:
        """
        Execute the complete workflow.
        
        This method provides the main entry point for query processing
        while maintaining backward compatibility.
        
        Args:
            query: User query string
            repositories: Target repositories filter
            languages: Target languages filter  
            file_types: Target file types filter
            k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Final query state
        """
        # Use existing state creation function
        workflow_id = kwargs.get("workflow_id", f"query-{int(time.time())}")
        state = create_query_state(
            workflow_id=workflow_id,
            original_query=query,
            target_repositories=repositories,
            target_languages=languages,
            target_file_types=file_types,
            retrieval_config={"k": k or self.default_k}
        )
        
        # Use inherited invoke method from BaseWorkflow
        final_state = self.invoke(state)
        
        return final_state

    def _determine_search_strategy(self, query_intent, query: str) -> SearchStrategy:
        """
        Determine optimal search strategy based on query intent.

        This method reuses the exact logic from the original QueryWorkflow
        to maintain consistency.

        Args:
            query_intent: Analyzed query intent
            query: Processed query string

        Returns:
            Optimal search strategy
        """
        # Import here to avoid circular imports
        from src.workflows.workflow_states import QueryIntent

        # Use hybrid search for complex queries
        if len(query.split()) > 10:
            return SearchStrategy.HYBRID

        # Strategy based on intent
        if query_intent == QueryIntent.CODE_SEARCH:
            return SearchStrategy.SEMANTIC
        elif query_intent == QueryIntent.DEBUGGING:
            return SearchStrategy.HYBRID  # Combine semantic and keyword for debugging
        elif query_intent == QueryIntent.DOCUMENTATION:
            return SearchStrategy.KEYWORD  # Documentation often has specific terms
        else:
            return SearchStrategy.SEMANTIC  # Default to semantic

    def _expand_search(self, state: QueryState) -> QueryState:
        """
        Expand search parameters for better context.

        This method reuses the exact logic from the original QueryWorkflow
        to maintain consistency.

        Args:
            state: Current query state

        Returns:
            Updated state with expanded search results
        """
        # Expand search with different strategy and more documents
        expanded_k = min(state["retrieval_config"].get("k", self.default_k) * 2, self.max_k)

        # Create a temporary search handler for expansion
        expanded_search_handler = VectorSearchHandler(collection_name=self.collection_name)

        # Update state for expanded search
        expanded_state = state.copy()
        expanded_state["retrieval_config"]["k"] = expanded_k
        expanded_state["search_strategy"] = SearchStrategy.HYBRID  # Use hybrid for expansion
        expanded_state["search_filters"] = {}  # Remove filters for broader search

        # Perform expanded search
        expanded_state = expanded_search_handler.invoke(expanded_state)

        # Update original state with expanded results
        state["context_documents"] = expanded_state["context_documents"]
        state["document_retrieval"] = expanded_state["document_retrieval"]

        self.logger.info(f"Expanded search retrieved {len(state['context_documents'])} documents")

        return state

    def _finalize_response(self, state: QueryState) -> QueryState:
        """
        Finalize response with sources and metadata.

        Args:
            state: Current query state

        Returns:
            Updated state with finalized response
        """
        # Format final response with sources
        response_text = state["llm_generation"]["generated_response"]
        sources = []

        for doc in state["context_documents"]:
            source = {
                "file_path": doc["metadata"].get("file_path", "unknown"),
                "repository": doc["metadata"].get("repository", "unknown"),
                "line_start": doc["metadata"].get("line_start"),
                "line_end": doc["metadata"].get("line_end"),
            }
            sources.append(source)

        state["response_sources"] = sources

        # Calculate confidence score using PromptManager logic
        confidence_score = self._calculate_response_confidence(state)
        state["response_confidence"] = confidence_score

        # Calculate total processing time
        start_time = state.get("start_time")
        if start_time:
            state["total_query_time"] = time.time() - start_time

        # Mark as completed
        state["status"] = ProcessingStatus.COMPLETED
        state = update_workflow_progress(state, 100.0, "workflow_complete")

        self.logger.info(f"Query workflow completed successfully with confidence score: {confidence_score:.2f}")

        return state

    def _calculate_response_confidence(self, state: QueryState) -> float:
        """
        Calculate response confidence score using PromptManager logic.

        Args:
            state: Current query state

        Returns:
            Confidence score between 0 and 1
        """
        context_documents = state.get("context_documents", [])
        query = state.get("original_query", "")

        if not context_documents:
            return 0.0

        # Basic confidence metrics (based on PromptManager._assess_context_confidence)
        doc_count_score = min(len(context_documents) / 5.0, 1.0)  # More docs = higher confidence
        
        # Content relevance score
        total_content_length = sum(len(doc.get("content", "")) for doc in context_documents)
        content_score = min(total_content_length / 2000.0, 1.0)  # More content = higher confidence
        
        # Metadata quality score
        metadata_quality = 0.0
        for doc in context_documents:
            metadata = doc.get("metadata", {})
            if metadata.get("file_path"):
                metadata_quality += 0.2
            if metadata.get("repository"):
                metadata_quality += 0.1
            if metadata.get("language"):
                metadata_quality += 0.1
            if metadata.get("chunk_type"):
                metadata_quality += 0.1
        
        metadata_quality = min(metadata_quality / len(context_documents), 1.0)
        
        # Combined confidence score
        confidence = (doc_count_score * 0.4 + content_score * 0.4 + metadata_quality * 0.2)
        
        return min(confidence, 1.0)
