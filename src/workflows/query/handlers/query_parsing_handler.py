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
            original_query = state["original_query"].strip()
            
            # Extract key terms for better search
            simplified_query = self._extract_key_terms(original_query)
            
            # Store both original and simplified queries
            state["processed_query"] = simplified_query
            state["original_processed_query"] = original_query
            
            self.logger.info(f"Original query: {original_query}")
            self.logger.info(f"Simplified query: {simplified_query}")
            
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

    def _extract_key_terms(self, query: str) -> str:
        """
        Extract key terms from complex queries for better vector search.
        
        This method simplifies complex queries by extracting the most relevant
        terms that are likely to match documents in the vector store.
        
        Args:
            query: Original user query
            
        Returns:
            Simplified query with key terms
        """
        query_lower = query.lower()
        
        # Define key term patterns for different query types
        key_terms = []
        
        # Extract repository/service names
        if "car-listing-service" in query_lower:
            key_terms.append("car")
            key_terms.append("listing")
        elif "car-notification-service" in query_lower:
            key_terms.append("car")
            key_terms.append("notification")
        elif "car-order-service" in query_lower:
            key_terms.append("car")
            key_terms.append("order")
        elif "car-web-client" in query_lower:
            key_terms.append("car")
            key_terms.append("web")
        
        # Extract common technical terms
        if "component" in query_lower or "components" in query_lower:
            key_terms.append("class")
            key_terms.append("service")
        if "main" in query_lower:
            key_terms.append("class")
        if "structure" in query_lower:
            key_terms.append("class")
        if "project" in query_lower:
            key_terms.append("class")
        
        # Extract programming language terms
        if "csharp" in query_lower or "c#" in query_lower:
            key_terms.append("class")
        if "dotnet" in query_lower or ".net" in query_lower:
            key_terms.append("class")
        
        # Extract API-related terms
        if "api" in query_lower:
            key_terms.append("controller")
            key_terms.append("service")
        if "endpoint" in query_lower:
            key_terms.append("controller")
        if "controller" in query_lower:
            key_terms.append("controller")
        
        # Extract database-related terms
        if "database" in query_lower or "db" in query_lower:
            key_terms.append("model")
            key_terms.append("entity")
        if "model" in query_lower:
            key_terms.append("class")
        
        # If no specific terms found, extract general terms
        if not key_terms:
            # Extract words that are likely to be in code
            words = query_lower.split()
            for word in words:
                if len(word) > 2 and word not in ["the", "and", "or", "for", "with", "this", "that", "what", "are", "main", "components", "project"]:
                    key_terms.append(word)
        
        # If still no terms, use the original query
        if not key_terms:
            return query
        
        # Return simplified query
        simplified = " ".join(key_terms[:5])  # Limit to 5 terms
        return simplified
