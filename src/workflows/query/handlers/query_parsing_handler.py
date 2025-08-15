"""
Query parsing handler extending BaseWorkflow.

This module implements query parsing, validation, and intent analysis
by extending the existing BaseWorkflow infrastructure.
"""

from typing import List, Optional
from src.workflows.base_workflow import BaseWorkflow
from src.workflows.workflow_states import QueryState, QueryIntent
from src.config.query_patterns import load_query_patterns


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
            # Analyze query intent using original query (not processed/simplified version)
            original_intent = state.get("query_intent")
            original_query = state.get("original_query", "")
            processed_query = state.get("processed_query", "")
            determined_intent = self._determine_query_intent(original_query)
            
            # Check if this is a Q2-style system relationship visualization query
            is_q2_query = self._is_q2_system_relationship_query(original_query)
            if is_q2_query:
                state["is_q2_system_visualization"] = True
                # Ensure it's marked as architecture intent
                determined_intent = QueryIntent.ARCHITECTURE
            
            state["query_intent"] = determined_intent
            self.logger.debug(f"Intent analysis on original query: '{original_query}' -> {determined_intent}")  
            self.logger.debug(f"Processed query: '{processed_query}'")  
            self.logger.debug(f"State before setting intent: {original_intent}")  
            self.logger.debug(f"State after setting intent: {state.get('query_intent')}")  
            self.logger.debug(f"Intent type: {type(state.get('query_intent'))}")  
            self.logger.debug(f"Q2 system visualization: {is_q2_query}")  

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

        # Event flow patterns (check first as they are specific)
        if self._is_event_flow_query(query):
            return QueryIntent.EVENT_FLOW

        # Explanation patterns (check after event flow to avoid conflicts)
        elif any(keyword in query_lower for keyword in [
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

        # Architecture patterns - enhanced to detect Q2 system relationship visualization
        elif any(keyword in query_lower for keyword in [
            "architecture", "structure", "design", "pattern", "flow",
            "component", "module", "system", "overview", "connected", "connect",
            "relationship", "services", "how", "explain what"
        ]):
            return QueryIntent.ARCHITECTURE

        # Code search patterns (moved to end to avoid conflicts)
        elif any(keyword in query_lower for keyword in [
            "function", "method", "class", "variable", "implement", "code",
            "algorithm", "pattern", "design pattern", "example"
        ]) or "how to " in query_lower:  # More specific "how to" matching
            return QueryIntent.CODE_SEARCH

        # Default to code search
        else:
            return QueryIntent.CODE_SEARCH

    def _is_event_flow_query(self, query: str) -> bool:
        """
        Detect if a query is asking for event flow analysis.
        
        This method checks for event flow patterns like "walk me through",
        "what happens when", etc. that indicate the user wants a step-by-step
        process explanation with sequence diagrams.
        
        Args:
            query: User query string
            
        Returns:
            bool: True if this is an event flow query, False otherwise
        """
        from src.analyzers.event_flow_analyzer import EventFlowAnalyzer
        
        try:
            analyzer = EventFlowAnalyzer()
            return analyzer.is_event_flow_query(query)
        except Exception as e:
            self.logger.error(f"Error in event flow detection: {e}")
            
            # Fallback to simple pattern matching
            query_lower = query.lower().strip()
            event_flow_indicators = [
                "walk me through",
                "walk through", 
                "what happens when",
                "show me the flow",
                "step by step",
                "process flow",
                "sequence of events"
            ]
            
            return any(indicator in query_lower for indicator in event_flow_indicators)

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
    
    def _is_q2_system_relationship_query(self, query: str) -> bool:
        """
        Detect if the query matches the Q2 system relationship visualization pattern.
        
        This method specifically looks for the Q2 question pattern as defined in
        docs/agent-interaction-questions.md:
        "Show me how the four services are connected and explain what I'm looking at."
        
        Args:
            query: User query string
            
        Returns:
            bool: True if this is a Q2-style query, False otherwise
        """
        query_lower = query.lower().strip()
        
        # Specific Q2 pattern matching
        q2_patterns = [
            # Exact Q2 match
            "show me how the four services are connected and explain what i'm looking at",
            "show me how the four services are connected and explain what i am looking at", 
            "show me how the four services are connected",
            # Variations that indicate system relationship visualization
            "how are the services connected",
            "how are the four services connected", 
            "show me how services are connected",
            "explain how the services connect",
            "how do the services work together",
            "show the system architecture",
            "show me the system architecture",
            "explain the system relationships",
            "show service connections",
            "visualize the system",
            "diagram of the system"
        ]
        
        # Check for exact or close matches
        for pattern in q2_patterns:
            if pattern in query_lower:
                return True
        
        # Check for key combination patterns that indicate Q2-style queries
        has_services = any(term in query_lower for term in ["service", "services", "component", "components"])
        has_connection = any(term in query_lower for term in ["connect", "connected", "connection", "relationship", "work together"])
        has_visualization = any(term in query_lower for term in ["show", "explain", "diagram", "visualize", "architecture"])
        
        # If query has all three elements, it's likely a Q2-style query
        if has_services and has_connection and has_visualization:
            return True
            
        return False
