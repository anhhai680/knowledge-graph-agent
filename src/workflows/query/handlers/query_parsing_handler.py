"""
Query parsing handler extending BaseWorkflow.

This module implements query parsing, validation, and intent analysis
by extending the existing BaseWorkflow infrastructure.
"""

from typing import List, Optional
from src.workflows.base_workflow import BaseWorkflow
from src.workflows.workflow_states import QueryState, QueryIntent
from src.config.query_patterns import QueryPatternsConfig, load_query_patterns


class QueryParsingHandler(BaseWorkflow[QueryState]):
    """
    Handle query parsing, validation, and intent analysis.
    
    Extends BaseWorkflow to leverage existing error handling, retry logic,
    and progress tracking while focusing on query processing concerns.
    """
    
    def __init__(self, query_patterns_config: Optional[str] = None, **kwargs):
        """
        Initialize query parsing handler.
        
        Args:
            query_patterns_config: Optional path to query patterns configuration file
            **kwargs: Additional arguments passed to BaseWorkflow
        """
        super().__init__(workflow_id="query-parsing", **kwargs)
        # Load query patterns configuration
        self.patterns_config = load_query_patterns(query_patterns_config)
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
            
            # Store simplified query (original is already in state)
            state["processed_query"] = simplified_query
            
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

        # Explanation patterns (check first to avoid conflicts with code search)
        if any(keyword in query_lower for keyword in [
            "explain", "what is", "what does", "how does", "why",
            "understand", "clarify", "meaning"
        ]):
            return QueryIntent.EXPLANATION

        # Debugging patterns (check before code search for fix/error terms)
        elif any(keyword in query_lower for keyword in [
            "error", "bug", "issue", "problem", "fix", "debug",
            "troubleshoot", "exception", "crash"
        ]):
            return QueryIntent.DEBUGGING

        # Documentation patterns
        elif any(keyword in query_lower for keyword in [
            "document", "readme", "guide", "tutorial", "specification",
            "api doc", "comment", "description"
        ]):
            return QueryIntent.DOCUMENTATION

        # Architecture patterns
        elif any(keyword in query_lower for keyword in [
            "architecture", "structure", "design", "pattern", "flow",
            "component", "module", "system", "overview"
        ]):
            return QueryIntent.ARCHITECTURE

        # Code search patterns (moved to end to avoid conflicts)
        elif any(keyword in query_lower for keyword in [
            "function", "method", "class", "variable", "implement", "code",
            "algorithm", "pattern", "design pattern", "how to", "example"
        ]):
            return QueryIntent.CODE_SEARCH

        # Default to code search
        else:
            return QueryIntent.CODE_SEARCH

    def _extract_key_terms(self, query: str) -> str:
        """
        Extract key terms from complex queries for better vector search.
        
        This method uses configurable patterns to simplify complex queries by 
        extracting the most relevant terms that are likely to match documents 
        in the vector store. Focus on semantic patterns rather than specific names.
        
        Args:
            query: Original user query
            
        Returns:
            Simplified query with key terms
        """
        query_lower = query.lower()
        key_terms = []
        
        # Extract domain-specific terms using configured patterns
        for domain_pattern in self.patterns_config.domain_patterns:
            if any(pattern in query_lower for pattern in domain_pattern.patterns):
                key_terms.extend(domain_pattern.key_terms)
        
        # Extract technical terms using configured patterns
        for tech_pattern in self.patterns_config.technical_patterns:
            if any(pattern in query_lower for pattern in tech_pattern.patterns):
                key_terms.extend(tech_pattern.key_terms)
        
        # Extract programming language terms using configured patterns
        for prog_pattern in self.patterns_config.programming_patterns:
            if any(pattern in query_lower for pattern in prog_pattern.patterns):
                key_terms.extend(prog_pattern.key_terms)
        
        # Extract API-related terms using configured patterns
        for api_pattern in self.patterns_config.api_patterns:
            if any(pattern in query_lower for pattern in api_pattern.patterns):
                key_terms.extend(api_pattern.key_terms)
        
        # Extract database-related terms using configured patterns
        for db_pattern in self.patterns_config.database_patterns:
            if any(pattern in query_lower for pattern in db_pattern.patterns):
                key_terms.extend(db_pattern.key_terms)
        
        # Extract architecture-related terms using configured patterns
        for arch_pattern in self.patterns_config.architecture_patterns:
            if any(pattern in query_lower for pattern in arch_pattern.patterns):
                key_terms.extend(arch_pattern.key_terms)
        
        # If no specific patterns matched, extract general terms
        if not key_terms:
            key_terms = self._extract_general_terms(query_lower)
        
        # If still no terms, use the original query
        if not key_terms:
            return query
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in key_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)
        
        # Return simplified query with configured max terms limit
        simplified = " ".join(unique_terms[:self.patterns_config.max_terms])
        return simplified
    
    def _extract_general_terms(self, query_lower: str) -> List[str]:
        """
        Extract general terms when no specific patterns match.
        
        Args:
            query_lower: Lowercased query string
            
        Returns:
            List of extracted general terms
        """
        words = query_lower.split()
        general_terms = []
        
        for word in words:
            # Check word length and exclusion list
            if (len(word) > self.patterns_config.min_word_length and 
                word not in self.patterns_config.excluded_words):
                general_terms.append(word)
        
        return general_terms
