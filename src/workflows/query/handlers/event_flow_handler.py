"""
Event flow handler extending BaseWorkflow for event flow analysis queries.

This module implements event flow analysis workflow by integrating event flow analyzer,
code discovery engine, and sequence diagram builder with existing BaseWorkflow infrastructure.
"""

from typing import List, Dict, Any
import time

from src.workflows.base_workflow import BaseWorkflow
from src.workflows.workflow_states import QueryState, QueryIntent, ProcessingStatus
from src.analyzers.event_flow_analyzer import EventFlowAnalyzer, EventFlowQuery
from src.discovery.code_discovery_engine import CodeDiscoveryEngine
from src.diagrams.sequence_diagram_builder import SequenceDiagramBuilder
from src.utils.logging import get_logger


class EventFlowHandler(BaseWorkflow[QueryState]):
    """
    Handler for event flow analysis queries.
    
    Extends BaseWorkflow to leverage existing error handling, retry logic,
    and progress tracking while implementing event flow analysis workflow.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize event flow handler.
        
        Args:
            **kwargs: Additional arguments passed to BaseWorkflow
        """
        super().__init__(workflow_id="event-flow-handler", **kwargs)
        
        # Initialize components
        self.event_analyzer = EventFlowAnalyzer()
        self.diagram_builder = SequenceDiagramBuilder()
        
        # Get settings for configuration
        try:
            from src.config.settings import get_settings
            self.settings = get_settings()
            self.event_flow_config = self.settings.event_flow
        except Exception as e:
            self.logger.warning(f"Could not load event flow settings: {e}")
            # Fallback configuration
            self.event_flow_config = type('EventFlowConfig', (), {
                'enable_event_flow': True,
                'max_sequence_steps': 20,
                'confidence_threshold': 0.7,
                'max_actors': 10,
                'include_code_references': True
            })()
    
    def define_steps(self) -> List[str]:
        """Define the event flow analysis workflow steps."""
        return [
            "validate_event_flow_query",
            "analyze_event_flow",
            "discover_relevant_code", 
            "generate_sequence_diagram",
            "create_explanation",
            "format_response"
        ]
    
    def execute_step(self, step: str, state: QueryState) -> QueryState:
        """
        Execute a single event flow analysis step.
        
        Args:
            step: Step name to execute
            state: Current query state
            
        Returns:
            Updated query state
        """
        if step == "validate_event_flow_query":
            return self._validate_event_flow_query(state)
        elif step == "analyze_event_flow":
            return self._analyze_event_flow(state)
        elif step == "discover_relevant_code":
            return self._discover_relevant_code(state)
        elif step == "generate_sequence_diagram":
            return self._generate_sequence_diagram(state)
        elif step == "create_explanation":
            return self._create_explanation(state)
        elif step == "format_response":
            return self._format_response(state)
        else:
            raise ValueError(f"Unknown step: {step}")
    
    def validate_state(self, state: QueryState) -> bool:
        """Validate that the state contains required fields for event flow analysis."""
        return (
            bool(state.get("original_query")) and
            state.get("query_intent") == QueryIntent.EVENT_FLOW
        )
    
    def _validate_event_flow_query(self, state: QueryState) -> QueryState:
        """Validate that this is indeed an event flow query."""
        original_query = state.get("original_query", "")
        
        # Double-check that this is an event flow query
        if not self.event_analyzer.is_event_flow_query(original_query):
            raise ValueError("Query is not an event flow analysis query")
        
        # Check if event flow analysis is enabled
        if not self.event_flow_config.enable_event_flow:
            raise ValueError("Event flow analysis is disabled in configuration")
        
        self.logger.info(f"Validated event flow query: {original_query}")
        state["metadata"]["event_flow_validated"] = True
        
        return state
    
    def _analyze_event_flow(self, state: QueryState) -> QueryState:
        """Analyze the event flow query to extract workflow components."""
        start_time = time.time()
        
        original_query = state["original_query"]
        
        # Parse the query to extract workflow components
        parsed_workflow = self.event_analyzer.parse_query(original_query)
        
        # Extract business logic components
        business_logic = self.event_analyzer.extract_business_logic(original_query)
        
        # Store analysis results in state
        state["metadata"]["event_flow_analysis"] = {
            "workflow_pattern": parsed_workflow.workflow.value,
            "entities": parsed_workflow.entities,
            "actions": parsed_workflow.actions,
            "domain": parsed_workflow.domain,
            "intent": parsed_workflow.intent,
            "business_logic": business_logic,
            "analysis_time": time.time() - start_time
        }
        
        # Store parsed workflow for next steps
        state["metadata"]["parsed_workflow"] = parsed_workflow
        
        self.logger.info(f"Analyzed event flow: pattern={parsed_workflow.workflow}, entities={parsed_workflow.entities}")
        
        return state
    
    def _discover_relevant_code(self, state: QueryState) -> QueryState:
        """Discover relevant code references using vector store."""
        start_time = time.time()
        
        parsed_workflow = state["metadata"]["parsed_workflow"]
        
        try:
            # Get vector store instance
            vector_store = self.get_vector_store()
            
            # Initialize code discovery engine
            discovery_engine = CodeDiscoveryEngine(vector_store)
            
            # Find relevant code references (now sync)
            code_references = discovery_engine.find_relevant_code(parsed_workflow, max_total_results=15)
            
            # Store code references in state
            state["metadata"]["code_references"] = code_references
            state["metadata"]["code_discovery_time"] = time.time() - start_time
            
            self.logger.info(f"Discovered {len(code_references)} relevant code references")
            
        except Exception as e:
            self.logger.error(f"Error discovering code references: {e}")
            # Continue with empty code references
            state["metadata"]["code_references"] = []
            state["metadata"]["code_discovery_error"] = str(e)
        
        return state
    
    def _generate_sequence_diagram(self, state: QueryState) -> QueryState:
        """Generate Mermaid sequence diagram from workflow and code references."""
        start_time = time.time()
        
        parsed_workflow = state["metadata"]["parsed_workflow"]
        code_references = state["metadata"].get("code_references", [])
        
        try:
            # Configure diagram builder with settings
            diagram_builder = SequenceDiagramBuilder(
                max_actors=self.event_flow_config.max_actors,
                max_steps=self.event_flow_config.max_sequence_steps
            )
            
            # Generate sequence diagram
            diagram = diagram_builder.build_from_workflow(parsed_workflow, code_references)
            
            # Store diagram in state
            state["metadata"]["sequence_diagram"] = diagram
            state["metadata"]["diagram_generation_time"] = time.time() - start_time
            
            self.logger.info("Generated sequence diagram successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating sequence diagram: {e}")
            # Create fallback diagram
            fallback_diagram = f"""```mermaid
sequenceDiagram
    participant User as User
    participant System as System
    
    User->>System: {parsed_workflow.intent or 'Request'}
    System->>System: Process Request
    System->>User: Response
```"""
            state["metadata"]["sequence_diagram"] = fallback_diagram
            state["metadata"]["diagram_generation_error"] = str(e)
        
        return state
    
    def _create_explanation(self, state: QueryState) -> QueryState:
        """Create step-by-step explanation of the event flow."""
        start_time = time.time()
        
        parsed_workflow = state["metadata"]["parsed_workflow"]
        code_references = state["metadata"].get("code_references", [])
        
        # Generate explanation text
        explanation_parts = []
        
        # Introduction
        explanation_parts.append(f"Let me walk you through the {parsed_workflow.workflow.replace('_', ' ')} workflow:\n")
        
        # Steps explanation
        if code_references and self.event_flow_config.include_code_references:
            explanation_parts.append("**Step-by-step process with code references:**\n")
            
            for i, ref in enumerate(code_references[:5], 1):  # Limit to 5 references
                explanation_parts.append(f"{i}. **{ref.method_name}**: Implemented in `{ref.file_path}` ({ref.repository})")
                explanation_parts.append(f"   - Context: {ref.context_type}")
                if ref.content_snippet:
                    explanation_parts.append(f"   - Code: {ref.content_snippet[:100]}...")
                explanation_parts.append("")
        else:
            explanation_parts.append("**Key components involved:**\n")
            for entity in parsed_workflow.entities:
                explanation_parts.append(f"- **{entity.title()}**: Key component in the workflow")
            explanation_parts.append("")
            
            explanation_parts.append("**Main actions:**\n")
            for action in parsed_workflow.actions:
                explanation_parts.append(f"- **{action.title()}**: Core operation in the process")
            explanation_parts.append("")
        
        # Workflow pattern explanation
        workflow_descriptions = {
            "order_processing": "This follows a typical order processing pattern where user actions trigger a series of validations, payments, and confirmations.",
            "user_authentication": "This represents a standard authentication flow with credential validation and session management.",
            "data_pipeline": "This shows a data processing pipeline with transformation and validation steps.",
            "api_request_flow": "This demonstrates an API request/response pattern with processing logic.",
            "event_driven": "This illustrates an event-driven architecture with message passing and handlers.",
            "generic_workflow": "This represents a general workflow pattern with sequential processing steps."
        }
        
        pattern_desc = workflow_descriptions.get(parsed_workflow.workflow.value, "This shows the workflow steps and interactions.")
        explanation_parts.append(f"**Workflow Pattern**: {pattern_desc}")
        
        # Combine explanation
        explanation = "\n".join(explanation_parts)
        
        # Store explanation in state
        state["metadata"]["explanation"] = explanation
        state["metadata"]["explanation_generation_time"] = time.time() - start_time
        
        self.logger.info("Generated event flow explanation")
        
        return state
    
    def _format_response(self, state: QueryState) -> QueryState:
        """Format the final event flow response."""
        diagram = state["metadata"].get("sequence_diagram", "")
        explanation = state["metadata"].get("explanation", "")
        
        # Combine diagram and explanation
        response_parts = [explanation]
        
        if diagram:
            response_parts.append("\n**Sequence Diagram:**\n")
            response_parts.append(diagram)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(state)
        
        # Create final response
        final_response = "\n".join(response_parts)
        
        # Update LLM generation state with our response
        state["llm_generation"]["generated_response"] = final_response
        state["llm_generation"]["status"] = ProcessingStatus.COMPLETED
        state["response_confidence"] = confidence_score
        
        # Store event flow specific metadata
        state["metadata"]["event_flow_response"] = {
            "diagram_included": bool(diagram),
            "explanation_length": len(explanation),
            "confidence_score": confidence_score,
            "code_references_count": len(state["metadata"].get("code_references", [])),
            "workflow_pattern": state["metadata"]["event_flow_analysis"]["workflow_pattern"]
        }
        
        self.logger.info(f"Formatted event flow response (confidence: {confidence_score:.2f})")
        
        return state
    
    def _calculate_confidence_score(self, state: QueryState) -> float:
        """Calculate confidence score for the event flow response."""
        score = 0.5  # Base score
        
        # Increase score based on successful analysis
        if state["metadata"].get("event_flow_analysis"):
            score += 0.2
        
        # Increase score based on code references found
        code_refs = state["metadata"].get("code_references", [])
        if code_refs:
            # Score based on number and relevance of code references
            ref_score = min(len(code_refs) / 10.0, 0.2)  # Max 0.2 for code references
            score += ref_score
        
        # Increase score if diagram was generated successfully
        if state["metadata"].get("sequence_diagram") and "error" not in state["metadata"]:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0