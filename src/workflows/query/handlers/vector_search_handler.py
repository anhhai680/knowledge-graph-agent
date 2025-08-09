"""
Vector search handler extending BaseWorkflow.

This module implements vector search operations by extending the existing 
BaseWorkflow infrastructure and using VectorStoreFactory directly.
"""

import time
from typing import List, Dict, Any
from langchain.schema import Document

from src.workflows.base_workflow import BaseWorkflow
from src.workflows.workflow_states import (
    QueryState, 
    SearchStrategy, 
    update_workflow_progress
)


class VectorSearchHandler(BaseWorkflow[QueryState]):
    """
    Handle vector search operations.
    
    Extends BaseWorkflow to leverage existing error handling, retry logic,
    and vector store factory integration while focusing on search concerns.
    """
    
    def __init__(self, collection_name: str = None, **kwargs):
        """Initialize vector search handler."""
        super().__init__(workflow_id="vector-search", **kwargs)
        # Use inherited vector store method from BaseWorkflow
        self.collection_name = collection_name
        self._vector_store = None
        
    def define_steps(self) -> List[str]:
        """Define the vector search workflow steps."""
        return ["extract_filters", "perform_search", "process_results"]
    
    def execute_step(self, step: str, state: QueryState) -> QueryState:
        """
        Execute a single vector search step.
        
        Args:
            step: Step name to execute
            state: Current query state
            
        Returns:
            Updated query state
        """
        if step == "extract_filters":
            # Extract search filters from query
            state["search_filters"] = self._extract_filters_from_query(
                state["processed_query"]
            )
            
        elif step == "perform_search":
            # Perform vector search using existing infrastructure
            search_start = time.time()
            
            # Get search parameters
            k = state.get("retrieval_config", {}).get("k", 4)
            filters = state.get("search_filters", {})
            search_strategy = state.get("search_strategy", SearchStrategy.SEMANTIC)
            
            # Perform search using existing method
            documents = self._perform_vector_search(
                state["processed_query"],
                search_strategy,
                k,
                filters
            )

            self.logger.debug(f"Vector search returned {documents} documents")
            
            search_time = time.time() - search_start
            state["retrieval_time"] = search_time
            
            # Store results in expected format
            state["context_documents"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("file_path", "unknown")
                }
                for doc in documents
            ]
            
            # Update document retrieval state
            state["document_retrieval"]["retrieved_documents"] = state["context_documents"]
            state["document_retrieval"]["retrieval_time"] = search_time
            
        elif step == "process_results":
            # Process and validate search results
            context_documents = state.get("context_documents", [])
            # Defensive programming: ensure context_documents is not None
            if context_documents is None:
                context_documents = []
            num_results = len(context_documents)
            self.logger.info(f"Retrieved {num_results} documents")
            
            # Update progress using existing helper function
            state = update_workflow_progress(state, 50.0, "vector_search_complete")
            
        return state
    
    def validate_state(self, state: QueryState) -> bool:
        """Validate that the state contains required fields for vector search."""
        return bool(state.get("processed_query"))
    
    def _get_vector_store(self):
        """Get or create vector store instance using inherited method."""
        if self._vector_store is None:
            # Use inherited get_vector_store method from BaseWorkflow
            self._vector_store = self.get_vector_store(self.collection_name)
        return self._vector_store

    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """
        Extract search filters from query.

        This method reuses the exact logic from the original QueryWorkflow
        to maintain consistency.

        Args:
            query: Processed query string

        Returns:
            Dictionary of search filters
        """
        filters = {}
        query_lower = query.lower()

        # Extract language filters
        languages = []
        if "python" in query_lower:
            languages.append("python")
        if "javascript" in query_lower or "js" in query_lower:
            languages.append("javascript")
        if "typescript" in query_lower or "ts" in query_lower:
            languages.append("typescript")
        if "java" in query_lower and "javascript" not in query_lower:
            languages.append("java")
        if "c++" in query_lower or "cpp" in query_lower:
            languages.append("cpp")
        if "go" in query_lower and len(query_lower.split()) > 1:  # Avoid matching single "go"
            languages.append("go")

        if languages:
            filters["language"] = {"$in": languages}

        # Extract file type filters
        file_types = []
        if ".py" in query_lower:
            file_types.append(".py")
        if ".js" in query_lower:
            file_types.append(".js")
        if ".ts" in query_lower:
            file_types.append(".ts")
        if ".java" in query_lower:
            file_types.append(".java")
        if ".cpp" in query_lower or ".cc" in query_lower:
            file_types.extend([".cpp", ".cc"])
        if ".go" in query_lower:
            file_types.append(".go")

        if file_types:
            filters["file_extension"] = {"$in": file_types}

        return filters

    def _perform_vector_search(
        self,
        query: str,
        strategy: SearchStrategy,
        k: int,
        filters: Dict[str, Any]
    ) -> List[Document]:
        """
        Perform vector search using the specified strategy.

        This method reuses the exact logic from the original QueryWorkflow
        to maintain consistency.

        Args:
            query: Search query
            strategy: Search strategy to use
            k: Number of documents to retrieve
            filters: Search filters

        Returns:
            List of retrieved documents
        """
        vector_store = self._get_vector_store()

        try:
            if strategy == SearchStrategy.SEMANTIC:
                # Pure semantic search
                documents = vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filters
                )
            elif strategy == SearchStrategy.KEYWORD:
                # Keyword-based search (if supported by vector store)
                # Fall back to semantic search if keyword search not available
                if hasattr(vector_store, 'keyword_search'):
                    documents = vector_store.keyword_search(
                        query=query,
                        k=k,
                        filter=filters
                    )
                else:
                    documents = vector_store.similarity_search(
                        query=query,
                        k=k,
                        filter=filters
                    )
            elif strategy == SearchStrategy.HYBRID:
                # Hybrid search combining semantic and keyword
                if hasattr(vector_store, 'hybrid_search'):
                    documents = vector_store.hybrid_search(
                        query=query,
                        k=k,
                        filter=filters
                    )
                else:
                    # Fall back to semantic search
                    documents = vector_store.similarity_search(
                        query=query,
                        k=k,
                        filter=filters
                    )
            else:
                # Default to semantic search
                documents = vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filters
                )

            self.logger.info(f"Retrieved {len(documents)} documents using {strategy} strategy")
            return documents

        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            # Re-raise to let BaseWorkflow handle retry logic
            raise
