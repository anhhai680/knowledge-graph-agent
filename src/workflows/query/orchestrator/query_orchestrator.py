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
    update_workflow_progress,
    QueryIntent
)
from src.workflows.generic_qa_workflow import GenericQAWorkflow
from ..handlers.query_parsing_handler import QueryParsingHandler
from ..handlers.vector_search_handler import VectorSearchHandler
from ..handlers.llm_generation_handler import LLMGenerationHandler
from ..handlers.context_processing_handler import ContextProcessingHandler
from ..handlers.event_flow_handler import EventFlowHandler


class QueryWorkflowOrchestrator(BaseWorkflow[QueryState]):
    """
    Main orchestrator for query workflow execution.
    
    Extends BaseWorkflow to leverage existing error handling, retry logic,
    and progress tracking while composing step handlers for modular execution.
    """
    
    # Confidence Score Calculation Constants
    # These constants control how response confidence is calculated based on
    # retrieved context quality and quantity
    
    # Document count normalization - assumes 5 documents provide good coverage
    # Based on empirical testing showing diminishing returns beyond 5 relevant docs
    CONFIDENCE_OPTIMAL_DOC_COUNT = 5.0
    
    # Content length normalization - assumes 2000 chars provide sufficient context
    # Derived from average code function/class size analysis across multiple repos
    CONFIDENCE_OPTIMAL_CONTENT_LENGTH = 2000.0
    
    # Confidence score component weights (must sum to 1.0)
    # Document count and content quality are primary factors for technical queries
    CONFIDENCE_WEIGHT_DOC_COUNT = 0.4      # Quantity of relevant documents
    CONFIDENCE_WEIGHT_CONTENT = 0.4        # Quality/length of content
    CONFIDENCE_WEIGHT_METADATA = 0.2       # Metadata completeness (file paths, repos, etc.)
    
    # Metadata quality scoring - points awarded for each metadata field present
    # These values reflect the relative importance of different metadata fields
    METADATA_SCORE_FILE_PATH = 0.2    # File path is most important for code context
    METADATA_SCORE_REPOSITORY = 0.1   # Repository context helps with scope understanding
    METADATA_SCORE_LANGUAGE = 0.1     # Programming language aids in interpretation
    METADATA_SCORE_CHUNK_TYPE = 0.1   # Chunk type (function, class, etc.) provides structure info
    
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
        
        # Validate confidence scoring constants at initialization
        self._validate_confidence_constants()
        
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
        self.event_flow_handler = EventFlowHandler(**kwargs)
        self.generic_qa_handler = GenericQAWorkflow(**kwargs)
        
    def _validate_confidence_constants(self) -> None:
        """
        Validate that confidence calculation constants are properly configured.
        
        Raises:
            ValueError: If constants are invalid or inconsistent
        """
        # Ensure weights sum to 1.0 (allowing for small floating point tolerance)
        total_weight = (
            self.CONFIDENCE_WEIGHT_DOC_COUNT + 
            self.CONFIDENCE_WEIGHT_CONTENT + 
            self.CONFIDENCE_WEIGHT_METADATA
        )
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(
                f"Confidence weights must sum to 1.0, got {total_weight:.3f}. "
                f"Current weights: doc_count={self.CONFIDENCE_WEIGHT_DOC_COUNT}, "
                f"content={self.CONFIDENCE_WEIGHT_CONTENT}, "
                f"metadata={self.CONFIDENCE_WEIGHT_METADATA}"
            )
        
        # Ensure all constants are positive
        constants_to_check = [
            ("CONFIDENCE_OPTIMAL_DOC_COUNT", self.CONFIDENCE_OPTIMAL_DOC_COUNT),
            ("CONFIDENCE_OPTIMAL_CONTENT_LENGTH", self.CONFIDENCE_OPTIMAL_CONTENT_LENGTH),
            ("CONFIDENCE_WEIGHT_DOC_COUNT", self.CONFIDENCE_WEIGHT_DOC_COUNT),
            ("CONFIDENCE_WEIGHT_CONTENT", self.CONFIDENCE_WEIGHT_CONTENT),
            ("CONFIDENCE_WEIGHT_METADATA", self.CONFIDENCE_WEIGHT_METADATA),
            ("METADATA_SCORE_FILE_PATH", self.METADATA_SCORE_FILE_PATH),
            ("METADATA_SCORE_REPOSITORY", self.METADATA_SCORE_REPOSITORY),
            ("METADATA_SCORE_LANGUAGE", self.METADATA_SCORE_LANGUAGE),
            ("METADATA_SCORE_CHUNK_TYPE", self.METADATA_SCORE_CHUNK_TYPE),
        ]
        
        for name, value in constants_to_check:
            if value <= 0:
                raise ValueError(f"Constant {name} must be positive, got {value}")
    
    @classmethod
    def get_confidence_constants_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about confidence calculation constants for tuning and documentation.
        
        Returns:
            Dictionary containing constant names, values, descriptions, and tuning guidance
        """
        return {
            "normalization_factors": {
                "CONFIDENCE_OPTIMAL_DOC_COUNT": {
                    "value": cls.CONFIDENCE_OPTIMAL_DOC_COUNT,
                    "description": "Optimal number of documents for confidence calculation",
                    "tuning_guidance": "Increase if queries typically need more documents for good coverage"
                },
                "CONFIDENCE_OPTIMAL_CONTENT_LENGTH": {
                    "value": cls.CONFIDENCE_OPTIMAL_CONTENT_LENGTH,
                    "description": "Optimal content length (chars) for confidence calculation",
                    "tuning_guidance": "Adjust based on average meaningful code chunk size in your domain"
                }
            },
            "component_weights": {
                "CONFIDENCE_WEIGHT_DOC_COUNT": {
                    "value": cls.CONFIDENCE_WEIGHT_DOC_COUNT,
                    "description": "Weight for document count in confidence score",
                    "tuning_guidance": "Increase for domains where quantity of sources matters more"
                },
                "CONFIDENCE_WEIGHT_CONTENT": {
                    "value": cls.CONFIDENCE_WEIGHT_CONTENT,
                    "description": "Weight for content quality/length in confidence score",
                    "tuning_guidance": "Increase for domains where content depth is critical"
                },
                "CONFIDENCE_WEIGHT_METADATA": {
                    "value": cls.CONFIDENCE_WEIGHT_METADATA,
                    "description": "Weight for metadata completeness in confidence score",
                    "tuning_guidance": "Increase if metadata quality significantly impacts answer reliability"
                }
            },
            "metadata_scores": {
                "METADATA_SCORE_FILE_PATH": {
                    "value": cls.METADATA_SCORE_FILE_PATH,
                    "description": "Points awarded for file path metadata presence",
                    "tuning_guidance": "Highest score as file paths are critical for code context"
                },
                "METADATA_SCORE_REPOSITORY": {
                    "value": cls.METADATA_SCORE_REPOSITORY,
                    "description": "Points awarded for repository metadata presence",
                    "tuning_guidance": "Important for understanding code scope and context"
                },
                "METADATA_SCORE_LANGUAGE": {
                    "value": cls.METADATA_SCORE_LANGUAGE,
                    "description": "Points awarded for language metadata presence",
                    "tuning_guidance": "Helps with syntax-specific understanding"
                },
                "METADATA_SCORE_CHUNK_TYPE": {
                    "value": cls.METADATA_SCORE_CHUNK_TYPE,
                    "description": "Points awarded for chunk type metadata presence",
                    "tuning_guidance": "Useful for understanding code structure (function, class, etc.)"
                }
            }
        }
        
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
            self.logger.debug(f"ORCHESTRATOR DEBUG: State before parsing: query_intent={state.get('query_intent')}")
            state = self.parsing_handler.invoke(state)
            self.logger.debug(f"ORCHESTRATOR DEBUG: State after parsing: query_intent={state.get('query_intent')}")

            # Determine search strategy based on intent
            if state.get("query_intent"):
                state["search_strategy"] = self._determine_search_strategy(
                    state["query_intent"], state["processed_query"]
                )
                self.logger.info(f"ORCHESTRATOR: Determined search strategy: {state.get('search_strategy')}")
            else:
                self.logger.warning(f"ORCHESTRATOR: No query_intent found after parsing!")
            
        elif step == "search_documents":
            # Check if this is an event flow query - handle differently
            query_intent = state.get("query_intent")
            if query_intent == QueryIntent.EVENT_FLOW:
                # Use event flow handler for complete processing
                self.logger.info("Using event flow handler for EVENT_FLOW query")
                state = self.event_flow_handler.invoke(state)
                # Mark as processed to skip other steps
                state["metadata"]["event_flow_processed"] = True
            elif query_intent == QueryIntent.GENERIC_QA:
                # Use Generic Q&A handler for complete processing
                self.logger.info("Using Generic Q&A handler for GENERIC_QA query")
                
                # Extract repository identifier from the original query if possible
                original_query = state.get("original_query", "")
                repository_identifier = self._extract_repository_from_query(original_query)
                
                # Create Generic Q&A state with required fields
                generic_qa_state = {
                    "workflow_id": f"generic-qa-{int(time.time())}",
                    "question": original_query,
                    "repository_identifier": repository_identifier or "unknown",
                    "include_code_examples": True,
                    "preferred_template": None,
                    "current_step": "initialize",
                    "metadata": {},
                    "sources": [],
                    "errors": []
                }
                
                # Invoke Generic Q&A workflow
                try:
                    generic_qa_result = self.generic_qa_handler.invoke(generic_qa_state)
                    
                    # Debug logging to understand the result structure
                    self.logger.debug(f"Generic Q&A result keys: {list(generic_qa_result.keys()) if isinstance(generic_qa_result, dict) else type(generic_qa_result)}")
                    if isinstance(generic_qa_result, dict):
                        template_response = generic_qa_result.get('template_response', 'NOT_FOUND')
                        if isinstance(template_response, str):
                            self.logger.debug(f"Generic Q&A result template_response: {template_response[:100]}...")
                        else:
                            self.logger.debug(f"Generic Q&A result template_response type: {type(template_response)}")
                    
                    # Convert Generic Q&A response to Query workflow format
                    state = self._convert_generic_qa_to_query_response(state, generic_qa_result)
                    
                    # Mark as processed to skip other steps
                    state["metadata"]["generic_qa_processed"] = True
                    
                except Exception as e:
                    self.logger.error(f"Generic Q&A processing failed: {e}")
                    # Fallback to regular search
                    state = self.search_handler.invoke(state)
            else:
                # Use standard search handler for other query types
                state = self.search_handler.invoke(state)
            
        elif step == "process_context":
            # Skip context processing for event flow and generic Q&A queries (already handled)
            if state.get("metadata", {}).get("event_flow_processed"):
                self.logger.debug("Skipping context processing for event flow query")
            elif state.get("metadata", {}).get("generic_qa_processed"):
                self.logger.debug("Skipping context processing for generic Q&A query")
            else:
                # Use context handler's invoke method
                state = self.context_handler.invoke(state)
                
                # Check if context is sufficient, expand search if needed
                if not state.get("context_sufficient", True):
                    state = self._expand_search(state)
                    # Re-process context after expansion
                    state = self.context_handler.invoke(state)
            
        elif step == "generate_response":
            # Skip LLM generation for event flow and generic Q&A queries (already handled)
            if state.get("metadata", {}).get("event_flow_processed"):
                self.logger.debug("Skipping LLM generation for event flow query")
            elif state.get("metadata", {}).get("generic_qa_processed"):
                self.logger.debug("Skipping LLM generation for generic Q&A query")
            else:
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

        self.logger.debug(f"ORCHESTRATOR DEBUG: Created initial state with query_intent: {state.get('query_intent')}")
        self.logger.info(f"Starting query workflow with initial state: {state}")    
        
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
        elif query_intent == QueryIntent.EVENT_FLOW:
            return SearchStrategy.HYBRID  # Event flow needs comprehensive search
        elif query_intent == QueryIntent.GENERIC_QA:
            return SearchStrategy.SEMANTIC  # Generic Q&A uses semantic analysis
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
        Calculate response confidence score based on context quality metrics.
        
        This method evaluates the confidence of a generated response by analyzing:
        1. Document count - More relevant documents generally indicate higher confidence
        2. Content quality - Sufficient content length suggests comprehensive context
        3. Metadata completeness - Well-structured metadata improves interpretation
        
        The confidence score ranges from 0.0 (no confidence) to 1.0 (high confidence).

        Args:
            state: Current query state containing context documents and metadata

        Returns:
            Confidence score between 0 and 1, where higher values indicate 
            more reliable responses based on available context
        """
        context_documents = state.get("context_documents", [])
        query = state.get("original_query", "")

        if not context_documents:
            return 0.0

        # Document count score - normalize against optimal document count
        # More documents generally provide better coverage, but with diminishing returns
        doc_count_score = min(
            len(context_documents) / self.CONFIDENCE_OPTIMAL_DOC_COUNT, 
            1.0
        )
        
        # Content relevance score - normalize against optimal content length
        # Sufficient content length indicates comprehensive context coverage
        total_content_length = sum(len(doc.get("content", "")) for doc in context_documents)
        content_score = min(
            total_content_length / self.CONFIDENCE_OPTIMAL_CONTENT_LENGTH, 
            1.0
        )
        
        # Metadata quality score - evaluate completeness of document metadata
        # Better metadata helps with context interpretation and source attribution
        metadata_quality = 0.0
        for doc in context_documents:
            metadata = doc.get("metadata", {})
            if metadata.get("file_path"):
                metadata_quality += self.METADATA_SCORE_FILE_PATH
            if metadata.get("repository"):
                metadata_quality += self.METADATA_SCORE_REPOSITORY
            if metadata.get("language"):
                metadata_quality += self.METADATA_SCORE_LANGUAGE
            if metadata.get("chunk_type"):
                metadata_quality += self.METADATA_SCORE_CHUNK_TYPE
        
        # Normalize metadata quality by document count to get average quality per document
        metadata_quality = min(metadata_quality / len(context_documents), 1.0)
        
        # Weighted combination of confidence components
        # Weights are designed to prioritize content quantity and quality for technical queries
        confidence = (
            doc_count_score * self.CONFIDENCE_WEIGHT_DOC_COUNT + 
            content_score * self.CONFIDENCE_WEIGHT_CONTENT + 
            metadata_quality * self.CONFIDENCE_WEIGHT_METADATA
        )
        
        # Ensure confidence never exceeds 1.0
        return min(confidence, 1.0)

    def _extract_repository_from_query(self, query: str) -> Optional[str]:
        """
        Extract repository identifier from the query text.
        
        Args:
            query: User query string
            
        Returns:
            Repository identifier if found, None otherwise
        """
        query_lower = query.lower()
        
        # Common repository name patterns
        repository_patterns = [
            "car-listing-service", "car-order-service", "car-notification-service", "car-web-client"
        ]
        
        for repo in repository_patterns:
            if repo in query_lower:
                return repo
                
        return None

    def _convert_generic_qa_to_query_response(self, query_state: Dict[str, Any], generic_qa_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Generic Q&A workflow result to Query workflow response format.
        
        Args:
            query_state: Original query workflow state
            generic_qa_result: Result from Generic Q&A workflow
            
        Returns:
            Updated query state with Generic Q&A response converted to query format
        """
        try:
            # Extract the generated response from Generic Q&A result
            # First check for template_response (the actual key used by Generic Q&A workflow)
            template_response = generic_qa_result.get("template_response", "")
            
            # If template_response is a dict, extract the actual response content
            if isinstance(template_response, dict):
                # Try common keys that might contain the response
                response_content = (
                    template_response.get("response", "") or
                    template_response.get("content", "") or
                    template_response.get("answer", "") or
                    template_response.get("text", "") or
                    str(template_response)  # Convert entire dict to string as fallback
                )
            elif isinstance(template_response, str):
                response_content = template_response
            else:
                response_content = str(template_response) if template_response else ""
            
            # If still not found, check other possible keys
            if not response_content:
                response_content = generic_qa_result.get("response", "")
            
            # If not found, check in structured response
            if not response_content:
                structured_response = generic_qa_result.get("structured_response", {})
                response_content = structured_response.get("response", "")
            
            # If not found, check in LLM generation section
            if not response_content:
                llm_generation = generic_qa_result.get("llm_generation", {})
                response_content = llm_generation.get("generated_response", "")
            
            # If still empty, use a fallback message
            if not response_content:
                response_content = "Generic Q&A analysis completed but no response content found."
                self.logger.warning(f"No response content found in Generic Q&A result keys: {list(generic_qa_result.keys())}")
            
            # Update query state with Generic Q&A results
            query_state["results"] = []  # No traditional search results
            query_state["total_results"] = 1  # One structured response
            query_state["generated_response"] = response_content
            query_state["response_type"] = "generated"  # Mark as generated response
            query_state["confidence_score"] = generic_qa_result.get("confidence_score", 0.5)
            query_state["intent"] = "generic_qa"
            query_state["strategy"] = "structured_analysis"
            
            # Add processing metadata
            query_state["processing_time"] = generic_qa_result.get("processing_time_ms", 0) / 1000.0
            query_state["template_used"] = generic_qa_result.get("template_used", "generic_template")
            
            self.logger.info(f"Converted Generic Q&A result to query response format with response length: {len(response_content)}")
            
        except Exception as e:
            self.logger.error(f"Error converting Generic Q&A result: {e}")
            # Fallback to simple response
            query_state["generated_response"] = "Generic Q&A analysis completed but response formatting failed."
            query_state["response_type"] = "generated"
            query_state["confidence_score"] = 0.1
            
        return query_state

    def _format_generic_qa_for_query_response(self, structured_response: Dict[str, Any]) -> str:
        """
        Format Generic Q&A structured response for display in query workflow.
        
        Args:
            structured_response: Structured response from Generic Q&A workflow
            
        Returns:
            Formatted response string
        """
        sections = structured_response.get("sections", {})
        overview = sections.get("overview", {})
        endpoints = sections.get("endpoints", [])
        
        # Build formatted response
        response = []
        
        # Project Overview
        if overview:
            response.append("## Project Analysis")
            response.append(f"**Repository:** {overview.get('repository', 'Unknown')}")
            response.append(f"**Business Domain:** {overview.get('business_domain', 'Not specified')}")
            response.append(f"**Total Endpoints:** {overview.get('total_endpoints', 0)}")
            
            frameworks = overview.get('frameworks', [])
            if frameworks:
                response.append(f"**Frameworks:** {', '.join(frameworks)}")
            
            response.append("")
        
        # API Endpoints
        if endpoints:
            response.append("## API Endpoints")
            for endpoint in endpoints[:10]:  # Show top 10
                method = endpoint.get('method', 'GET')
                path = endpoint.get('path', '')
                description = endpoint.get('description', '')
                response.append(f"- **{method}** `{path}` - {description}")
            
            if len(endpoints) > 10:
                response.append(f"... and {len(endpoints) - 10} more endpoints")
            
            response.append("")
        
        # Authentication
        auth = sections.get("authentication", {})
        if auth and auth.get("methods"):
            response.append("## Authentication")
            methods = auth.get("methods", [])
            response.append(f"**Methods:** {', '.join(methods)}")
            response.append("")
        
        return "\n".join(response)
