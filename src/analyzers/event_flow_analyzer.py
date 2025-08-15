"""
Event flow analyzer for detecting and parsing event-flow analysis queries.

This module implements event flow query detection and analysis by extracting
workflow components from user queries using LLM-based natural language processing.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum

from src.utils.logging import get_logger


class WorkflowPattern(str, Enum):
    """Workflow pattern enumeration for event flow analysis."""
    
    ORDER_PROCESSING = "order_processing"
    USER_AUTHENTICATION = "user_authentication"
    DATA_PIPELINE = "data_pipeline"
    API_REQUEST_FLOW = "api_request_flow"
    EVENT_DRIVEN = "event_driven"
    GENERIC_WORKFLOW = "generic_workflow"


@dataclass
class EventFlowQuery:
    """Event flow query components extracted from user input."""
    
    entities: List[str]  # Objects/services involved (e.g., "user", "order", "payment")
    actions: List[str]   # Actions/operations (e.g., "places", "processes", "validates")
    workflow: WorkflowPattern  # Detected workflow pattern
    domain: Optional[str] = None  # Business domain (e.g., "e-commerce", "auth")
    intent: Optional[str] = None  # User intent description


class EventFlowAnalyzer:
    """
    Analyzer for event flow queries using pattern matching and keyword extraction.
    
    This class provides methods to detect event flow queries, extract workflow components,
    and classify business logic patterns without requiring LLM calls for basic detection.
    """
    
    def __init__(self):
        """Initialize event flow analyzer."""
        self.logger = get_logger(self.__class__.__name__)
        
        # Define workflow pattern keywords for detection
        self._workflow_patterns = {
            WorkflowPattern.ORDER_PROCESSING: [
                "order", "purchase", "buy", "checkout", "payment", "cart",
                "billing", "invoice", "transaction", "fulfillment"
            ],
            WorkflowPattern.USER_AUTHENTICATION: [
                "login", "signup", "register", "authenticate", "authorize",
                "session", "token", "password", "credentials", "oauth"
            ],
            WorkflowPattern.DATA_PIPELINE: [
                "process", "transform", "etl", "pipeline", "batch", "stream",
                "ingestion", "migration", "sync", "extract", "load"
            ],
            WorkflowPattern.API_REQUEST_FLOW: [
                "api", "request", "response", "endpoint", "service", "call",
                "http", "rest", "graphql", "webhook", "integration"
            ],
            WorkflowPattern.EVENT_DRIVEN: [
                "event", "message", "queue", "publish", "subscribe", "notify",
                "trigger", "handler", "listener", "emit", "dispatch"
            ]
        }
        
        # Define action keywords
        self._action_keywords = [
            "place", "places", "create", "creates", "process", "processes",
            "validate", "validates", "send", "sends", "receive", "receives",
            "update", "updates", "delete", "deletes", "execute", "executes",
            "trigger", "triggers", "handle", "handles", "generate", "generates"
        ]
        
        # Define entity keywords
        self._entity_keywords = [
            "user", "customer", "order", "product", "service", "system",
            "database", "payment", "notification", "email", "message",
            "request", "response", "session", "account", "profile"
        ]
    
    def is_event_flow_query(self, query: str) -> bool:
        """
        Detect if a query is asking for event flow analysis.
        
        Args:
            query: User query string
            
        Returns:
            bool: True if this is an event flow query, False otherwise
        """
        query_lower = query.lower().strip()
        
        # Direct event flow indicators
        event_flow_indicators = [
            "walk me through",
            "walk through",
            "what happens when",
            "show me the flow",
            "step by step",
            "process flow",
            "workflow",
            "sequence of events",
            "how does it work",
            "explain the process",
            "flow of",
            "chain of events"
        ]
        
        # Check for direct indicators
        for indicator in event_flow_indicators:
            if indicator in query_lower:
                self.logger.debug(f"Event flow query detected with indicator: '{indicator}'")
                return True
        
        # Check for action + entity patterns that suggest workflow
        has_action = any(action in query_lower for action in self._action_keywords)
        has_entity = any(entity in query_lower for entity in self._entity_keywords)
        has_workflow_context = any(
            pattern_word in query_lower 
            for patterns in self._workflow_patterns.values() 
            for pattern_word in patterns
        )
        
        # Require at least action + entity or workflow context for detection
        if (has_action and has_entity) or has_workflow_context:
            # Additional check for process-oriented language
            process_indicators = ["when", "how", "what", "process", "flow", "step"]
            if any(indicator in query_lower for indicator in process_indicators):
                self.logger.debug(f"Event flow query detected with pattern matching")
                return True
        
        return False
    
    def parse_query(self, query: str) -> EventFlowQuery:
        """
        Parse user query to extract workflow components.
        
        Args:
            query: User query string
            
        Returns:
            EventFlowQuery: Parsed query components
        """
        query_lower = query.lower()
        
        # Extract entities
        entities = self._extract_entities(query_lower)
        
        # Extract actions
        actions = self._extract_actions(query_lower)
        
        # Detect workflow pattern
        workflow = self._detect_workflow_pattern(query_lower)
        
        # Extract domain and intent
        domain = self._extract_domain(query_lower, workflow)
        intent = self._extract_intent(query)
        
        parsed_query = EventFlowQuery(
            entities=entities,
            actions=actions,
            workflow=workflow,
            domain=domain,
            intent=intent
        )
        
        self.logger.info(f"Parsed event flow query: {parsed_query}")
        return parsed_query
    
    def detect_workflow_pattern(self, query: str) -> WorkflowPattern:
        """
        Detect workflow pattern from query.
        
        Args:
            query: User query string
            
        Returns:
            WorkflowPattern: Detected workflow pattern
        """
        return self._detect_workflow_pattern(query.lower())
    
    def extract_business_logic(self, query: str) -> Dict[str, Any]:
        """
        Extract business logic components from query.
        
        Args:
            query: User query string
            
        Returns:
            Dict containing business logic components
        """
        query_lower = query.lower()
        parsed = self.parse_query(query)
        
        return {
            "entities": parsed.entities,
            "actions": parsed.actions,
            "workflow_pattern": parsed.workflow,
            "domain": parsed.domain,
            "complexity": self._assess_complexity(parsed),
            "key_interactions": self._identify_key_interactions(query_lower),
            "sequence_indicators": self._find_sequence_indicators(query_lower)
        }
    
    def _extract_entities(self, query_lower: str) -> List[str]:
        """Extract entities from query."""
        entities = []
        words = query_lower.split()
        
        # Check for direct entity keywords
        for entity in self._entity_keywords:
            if entity in words:
                entities.append(entity)
        
        # Look for potential entity patterns (nouns that might be services/objects)
        # This is a simple heuristic - could be enhanced with NLP
        potential_entities = []
        for word in words:
            if (len(word) > 3 and 
                word.endswith('er') or word.endswith('or') or 
                word.endswith('service') or word.endswith('system')):
                potential_entities.append(word)
        
        entities.extend(potential_entities[:3])  # Limit to prevent noise
        return list(set(entities))  # Remove duplicates
    
    def _extract_actions(self, query_lower: str) -> List[str]:
        """Extract actions from query."""
        actions = []
        words = query_lower.split()
        
        for action in self._action_keywords:
            if action in words:
                actions.append(action)
        
        return list(set(actions))
    
    def _detect_workflow_pattern(self, query_lower: str) -> WorkflowPattern:
        """Detect the most likely workflow pattern."""
        pattern_scores = {}
        
        for pattern, keywords in self._workflow_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                pattern_scores[pattern] = score
        
        if pattern_scores:
            # Return pattern with highest score
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            self.logger.debug(f"Detected workflow pattern: {best_pattern} (score: {pattern_scores[best_pattern]})")
            return best_pattern
        
        # Default to generic workflow
        return WorkflowPattern.GENERIC_WORKFLOW
    
    def _extract_domain(self, query_lower: str, workflow: WorkflowPattern) -> Optional[str]:
        """Extract business domain from query and workflow pattern."""
        domain_mapping = {
            WorkflowPattern.ORDER_PROCESSING: "e-commerce",
            WorkflowPattern.USER_AUTHENTICATION: "security",
            WorkflowPattern.DATA_PIPELINE: "data-processing",
            WorkflowPattern.API_REQUEST_FLOW: "integration",
            WorkflowPattern.EVENT_DRIVEN: "messaging"
        }
        
        return domain_mapping.get(workflow, "general")
    
    def _extract_intent(self, query: str) -> Optional[str]:
        """Extract user intent description."""
        # Return first 50 characters as intent summary
        return query.strip()[:50] + "..." if len(query) > 50 else query.strip()
    
    def _assess_complexity(self, parsed: EventFlowQuery) -> str:
        """Assess workflow complexity based on parsed components."""
        entity_count = len(parsed.entities)
        action_count = len(parsed.actions)
        
        total_components = entity_count + action_count
        
        if total_components <= 3:
            return "simple"
        elif total_components <= 6:
            return "moderate"
        else:
            return "complex"
    
    def _identify_key_interactions(self, query_lower: str) -> List[str]:
        """Identify key interaction patterns in the query."""
        interactions = []
        
        interaction_patterns = [
            "sends to", "receives from", "calls", "notifies", "triggers",
            "processes", "validates", "creates", "updates", "deletes"
        ]
        
        for pattern in interaction_patterns:
            if pattern in query_lower:
                interactions.append(pattern)
        
        return interactions
    
    def _find_sequence_indicators(self, query_lower: str) -> List[str]:
        """Find words that indicate sequence or order."""
        sequence_words = [
            "first", "then", "next", "after", "before", "finally",
            "initially", "subsequently", "meanwhile", "during"
        ]
        
        found_indicators = []
        for word in sequence_words:
            if word in query_lower:
                found_indicators.append(word)
        
        return found_indicators