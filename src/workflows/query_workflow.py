"""
Refactored LangGraph Query Workflow Implementation.

This module implements a clean, modular query workflow using the orchestrator pattern
while maintaining 100% backward compatibility with the original interface.
"""

from typing import Any, Dict, List, Optional

from langchain.schema import Document

from src.config.settings import settings
from src.llm.llm_factory import LLMFactory
from src.llm.embedding_factory import EmbeddingFactory
from src.vectorstores.store_factory import VectorStoreFactory
from src.workflows.base_workflow import BaseWorkflow
from src.workflows.workflow_states import (
    QueryState,
)


class QueryWorkflowSteps(str):
    """Query workflow step enumeration (maintained for backward compatibility)."""

    PARSE_QUERY = "parse_query"
    VALIDATE_QUERY = "validate_query"
    ANALYZE_QUERY_INTENT = "analyze_query_intent"
    DETERMINE_SEARCH_STRATEGY = "determine_search_strategy"
    VECTOR_SEARCH = "vector_search"
    FILTER_AND_RANK_RESULTS = "filter_and_rank_results"
    CHECK_SUFFICIENT_CONTEXT = "check_sufficient_context"
    EXPAND_SEARCH_PARAMETERS = "expand_search_parameters"
    PREPARE_LLM_CONTEXT = "prepare_llm_context"
    GENERATE_CONTEXTUAL_PROMPT = "generate_contextual_prompt"
    CALL_LLM = "call_llm"
    FORMAT_RESPONSE = "format_response"
    RESPONSE_QUALITY_CHECK = "response_quality_check"
    RETURN_SUCCESS = "return_success"

    # Error handling and fallback states
    HANDLE_RETRIEVAL_ERRORS = "handle_retrieval_errors"
    FALLBACK_SEARCH_STRATEGY = "fallback_search_strategy"
    HANDLE_LLM_ERRORS = "handle_llm_errors"
    RETRY_LLM_CALL = "retry_llm_call"
    RESPONSE_QUALITY_CONTROL = "response_quality_control"
    RETRY_WITH_DIFFERENT_CONTEXT = "retry_with_different_context"


class QueryWorkflow(BaseWorkflow):
    """
    Refactored LangGraph query workflow using modular orchestrator.

    This class now acts as a facade over the modular orchestrator,
    maintaining backward compatibility while using the new architecture.
    
    The workflow handles the complete query processing pipeline:
    1. Parse and validate user query
    2. Analyze query intent and determine search strategy
    3. Perform vector search with filtering and ranking
    4. Check context sufficiency and expand if needed
    5. Prepare LLM context and generate contextual prompt
    6. Call LLM and format response
    7. Perform quality control and return result
    """

    def __init__(
        self,
        vector_store_type: Optional[str] = None,
        collection_name: Optional[str] = None,
        default_k: int = 4,
        max_k: int = 20,
        min_context_length: int = 100,
        max_context_length: int = 8000,
        response_quality_threshold: float = 0.7,
        **kwargs,
    ):
        """
        Initialize query workflow with modular orchestrator.

        Args:
            vector_store_type: Vector store type (overrides env setting)
            collection_name: Collection name (overrides env setting)
            default_k: Default number of documents to retrieve
            max_k: Maximum number of documents to retrieve
            min_context_length: Minimum context length required
            max_context_length: Maximum context length allowed
            response_quality_threshold: Minimum quality score for response
            **kwargs: Additional arguments passed to BaseWorkflow
        """
        super().__init__(**kwargs)

        # Determine collection name using existing logic
        self.collection_name = collection_name or (
            settings.pinecone.collection_name
            if vector_store_type == "pinecone" and settings.pinecone
            else settings.chroma.collection_name
        )
        
        # Store configuration for backward compatibility
        self.vector_store_type = vector_store_type or settings.database_type.value
        self.default_k = default_k
        self.max_k = max_k
        self.min_context_length = min_context_length
        self.max_context_length = max_context_length
        self.response_quality_threshold = response_quality_threshold

        # Initialize the modular orchestrator
        from .query.orchestrator.query_orchestrator import QueryWorkflowOrchestrator
        self.orchestrator = QueryWorkflowOrchestrator(
            collection_name=self.collection_name,
            default_k=default_k,
            max_k=max_k,
            min_context_length=min_context_length,
            max_context_length=max_context_length,
            response_quality_threshold=response_quality_threshold,
            **kwargs
        )

        # Maintain backward compatibility - initialize legacy components
        self.vector_store_factory = VectorStoreFactory()
        self.embedding_factory = EmbeddingFactory()
        self.llm_factory = LLMFactory()
        self._vector_store = None
        self._retriever = None
        self._embeddings = None
        self._llm = None

    async def run(
        self,
        query: str,
        repositories: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None,
        k: Optional[int] = None,
        **kwargs
    ) -> QueryState:
        """
        Execute the query workflow using the modular orchestrator.

        This method maintains the same interface as before but delegates
        to the modular orchestrator for actual processing.

        Args:
            query: User query string
            repositories: Optional repository filter
            languages: Optional language filter
            file_types: Optional file type filter
            k: Optional number of documents to retrieve
            **kwargs: Additional workflow arguments

        Returns:
            Final QueryState
        """
        try:
            # Delegate to the orchestrator for processing
            final_state = await self.orchestrator.execute_workflow(
                query=query,
                repositories=repositories,
                languages=languages,
                file_types=file_types,
                k=k,
                **kwargs
            )

            self.logger.debug(f"Final workflow state: {final_state}")
            
            total_time = final_state.get('total_query_time', 0) or 0
            self.logger.info(f"Query workflow completed in {total_time:.2f}s")
            return final_state

        except Exception as e:
            self.logger.error(f"Query workflow failed: {str(e)}")
            raise

    def define_steps(self) -> List[str]:
        """Define the workflow steps using orchestrator."""
        return self.orchestrator.define_steps()

    def execute_step(self, step: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step using orchestrator."""
        # Convert to QueryState and delegate to orchestrator
        query_state = QueryState(**state)
        result_state = self.orchestrator.execute_step(step, query_state)
        return dict(result_state)

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate workflow state using orchestrator."""
        return self.orchestrator.validate_state(state)
    
    # Backward compatibility methods for legacy components
    def _get_vector_store(self):
        """Get vector store instance (backward compatibility)."""
        return self.orchestrator.search_handler._get_vector_store()
    
    def _get_llm(self):
        """Get LLM instance (backward compatibility)."""
        return self.orchestrator.llm_handler._get_llm()
    
    def _get_embeddings(self):
        """Get embeddings instance (backward compatibility)."""
        if self._embeddings is None:
            self._embeddings = self.embedding_factory.create()
        return self._embeddings
    
    def _get_retriever(self, k: Optional[int] = None, filter_dict: Optional[Dict[str, Any]] = None):
        """Get retriever function (backward compatibility)."""
        vector_store = self._get_vector_store()
        
        def retrieve_documents(query: str) -> List[Document]:
            return vector_store.similarity_search(
                query, 
                k=k or self.default_k, 
                filter=filter_dict or {}
            )
        
        return retrieve_documents


# Helper functions for easy workflow execution
async def execute_query(
    query: str,
    repositories: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute a query workflow and return formatted result.

    Args:
        query: User query string
        repositories: Optional repository filter
        languages: Optional language filter
        **kwargs: Additional workflow arguments

    Returns:
        Dictionary with query result
    """
    workflow = QueryWorkflow(**kwargs)
    state = await workflow.run(
        query=query,
        repositories=repositories,
        languages=languages,
        **kwargs
    )

    return {
        "query": state["original_query"],
        "response": state["llm_generation"]["generated_response"],
        "sources": state.get("response_sources", []),
        "quality_score": state.get("response_quality_score", 0.0),
        "processing_time": state.get("total_query_time", 0.0),
        "documents_retrieved": len(state["document_retrieval"]["retrieved_documents"]),
        "workflow_id": state["workflow_id"],
    }
