"""
Query parsing handler extending BaseWorkflow.

This module implements query parsing, validation, and intent analysis
by extending the existing BaseWorkflow infrastructure.
"""

from typing import List
from src.workflows.base_workflow import BaseWorkflow
from src.workflows.workflow_states import QueryState, QueryIntent, update_workflow_progress


class QueryParsingHandler(BaseWorkflow[QueryState]):
    """
    Handle query parsing, validation, and intent analysis.
    
    Extends BaseWorkflow to leverage existing error handling, retry logic,
    and progress tracking while focusing on query processing concerns.
    """
    
    def __init__(self, **kwargs):
        """Initialize query parsing handler."""
        super().__init__(workflow_id="query-parsing", **kwargs)
        # All error handling, retry, progress tracking inherited from BaseWorkflow
    
    def define_steps(self) -> List[str]:
        """Define the query parsing workflow steps."""
        return ["parse_query", "validate_query", "analyze_intent"]
    
    def execute_step(self, step: str, state: QueryState) -> QueryState:
        """
        Execute a single query parsing step.
        
        Args:
            step: Step name to execute
            state: Current query state
            
        Returns:
            Updated query state
        """
        if step == "parse_query":
            # Parse and clean the query
            state["processed_query"] = state["original_query"].strip()
            self.logger.info(f"Parsed query: {state['processed_query']}")
            
        elif step == "validate_query":
            # Validate query is not empty and has reasonable length
            if not state["processed_query"] or len(state["processed_query"]) < 3:
                raise ValueError("Query is too short or empty")
            if len(state["processed_query"]) > 1000:
                raise ValueError("Query is too long (max 1000 characters)")
                
        elif step == "analyze_intent":
            # Analyze query intent using existing enum
            state["query_intent"] = self._determine_query_intent(state["processed_query"])
            self.logger.info(f"Determined query intent: {state['query_intent']}")
        
        return state
    
    def validate_state(self, state: QueryState) -> bool:
        """Validate that the state contains required fields for query parsing."""
        return bool(state.get("original_query"))
    
    def _determine_query_intent(self, query: str) -> QueryIntent:
        """
        Analyze query to determine intent.
        
        This method reuses the exact logic from the original QueryWorkflow
        to maintain consistency and avoid duplication.
        
        Args:
            query: User query string
            
        Returns:
            QueryIntent enum value
        """
        query_lower = query.lower()

        # Code search patterns
        if any(keyword in query_lower for keyword in [
            "function", "method", "class", "variable", "implement", "code",
            "algorithm", "pattern", "design pattern", "how to", "example"
        ]):
            return QueryIntent.CODE_SEARCH

        # Documentation patterns
        elif any(keyword in query_lower for keyword in [
            "document", "readme", "guide", "tutorial", "specification",
            "api doc", "comment", "description"
        ]):
            return QueryIntent.DOCUMENTATION

        # Explanation patterns
        elif any(keyword in query_lower for keyword in [
            "explain", "what is", "what does", "how does", "why",
            "understand", "clarify", "meaning"
        ]):
            return QueryIntent.EXPLANATION

        # Debugging patterns
        elif any(keyword in query_lower for keyword in [
            "error", "bug", "issue", "problem", "fix", "debug",
            "troubleshoot", "exception", "crash"
        ]):
            return QueryIntent.DEBUGGING

        # Architecture patterns
        elif any(keyword in query_lower for keyword in [
            "architecture", "structure", "design", "pattern", "flow",
            "component", "module", "system", "overview"
        ]):
            return QueryIntent.ARCHITECTURE

        # Default to code search
        else:
            return QueryIntent.CODE_SEARCH
