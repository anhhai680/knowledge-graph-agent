"""
Context processing handler extending BaseWorkflow.

This module implements context preparation and formatting by extending the existing 
BaseWorkflow infrastructure.
"""

import time
from typing import List, Tuple
from langchain.schema import Document

from src.workflows.base_workflow import BaseWorkflow
from src.workflows.workflow_states import QueryState, update_workflow_progress


class ContextProcessingHandler(BaseWorkflow[QueryState]):
    """
    Handle context preparation and formatting.
    
    Extends BaseWorkflow to leverage existing error handling, retry logic,
    and progress tracking while focusing on context processing concerns.
    """
    
    def __init__(self, max_context_length: int = 8000, min_context_length: int = 100, **kwargs):
        """Initialize context processing handler."""
        super().__init__(workflow_id="context-processing", **kwargs)
        self.max_context_length = max_context_length
        self.min_context_length = min_context_length
        
    def define_steps(self) -> List[str]:
        """Define the context processing workflow steps."""
        return ["check_sufficiency", "prepare_context", "validate_context"]
    
    def execute_step(self, step: str, state: QueryState) -> QueryState:
        """
        Execute a single context processing step.
        
        Args:
            step: Step name to execute
            state: Current query state
            
        Returns:
            Updated query state
        """
        if step == "check_sufficiency":
            # Check if we have sufficient context
            documents = [
                Document(page_content=doc["content"], metadata=doc["metadata"])
                for doc in state["context_documents"]
            ]
            is_sufficient, context_length = self._check_context_sufficiency(documents)
            state["context_size"] = context_length
            
            if not is_sufficient:
                self.logger.warning(f"Insufficient context: {context_length} chars (min: {self.min_context_length})")
                # This will be handled by the orchestrator to expand search
                state["context_sufficient"] = False
            else:
                state["context_sufficient"] = True
                
        elif step == "prepare_context":
            # Prepare context for LLM
            documents = [
                Document(page_content=doc["content"], metadata=doc["metadata"])
                for doc in state["context_documents"]
            ]
            
            context_start = time.time()
            context = self._prepare_context_for_llm(documents)
            context_time = time.time() - context_start
            
            state["context_preparation_time"] = context_time
            state["context_size"] = len(context)
            
            # Store context in retrieval config for next step
            state["retrieval_config"]["prepared_context"] = context
            
        elif step == "validate_context":
            # Validate the prepared context
            context = state["retrieval_config"].get("prepared_context", "")
            if not context.strip():
                raise ValueError("Prepared context is empty")
                
            self.logger.info(f"Prepared context: {len(context)} characters")
            
            # Update progress using existing helper function
            state = update_workflow_progress(state, 70.0, "context_preparation_complete")
            
        return state
    
    def validate_state(self, state: QueryState) -> bool:
        """Validate that the state contains required fields for context processing."""
        return bool(state.get("context_documents"))
    
    def _check_context_sufficiency(self, documents: List[Document]) -> Tuple[bool, int]:
        """
        Check if retrieved context is sufficient.
        
        This method reuses the exact logic from the original QueryWorkflow
        to maintain consistency.

        Args:
            documents: Retrieved documents

        Returns:
            Tuple of (is_sufficient, total_context_length)
        """
        if not documents:
            return False, 0

        total_length = sum(len(doc.page_content) for doc in documents)
        
        # Check minimum context length
        if total_length < self.min_context_length:
            return False, total_length

        # Check if we have meaningful content
        non_empty_docs = [doc for doc in documents if doc.page_content.strip()]
        if len(non_empty_docs) == 0:
            return False, total_length

        return True, total_length

    def _prepare_context_for_llm(self, documents: List[Document]) -> str:
        """
        Prepare context from documents for LLM.

        This method reuses the exact logic from the original QueryWorkflow
        to maintain consistency.

        Args:
            documents: Retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        context_parts = []
        total_length = 0

        for i, doc in enumerate(documents):
            # Extract metadata for source attribution
            file_path = doc.metadata.get("file_path", "unknown")
            repository = doc.metadata.get("repository", "unknown")
            line_start = doc.metadata.get("line_start", "")
            line_end = doc.metadata.get("line_end", "")

            # Format source info
            line_info = f" (lines {line_start}-{line_end})" if line_start and line_end else ""
            source_info = f"Source {i+1}: {repository}/{file_path}{line_info}"

            # Add context with source
            context_part = f"```\n{source_info}\n{doc.page_content}\n```"

            # Check if adding this would exceed max length
            if total_length + len(context_part) > self.max_context_length:
                break

            context_parts.append(context_part)
            total_length += len(context_part)

        return "\n\n".join(context_parts)
