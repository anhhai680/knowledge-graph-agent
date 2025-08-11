"""
Generic Q&A Workflow for processing generic project questions.

This workflow extends BaseWorkflow to provide stateful processing of generic
project questions using repository analysis, template-based response generation,
and structured output formatting.

Follows established LangGraph patterns from BaseWorkflow architecture.
"""

import time
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.workflows.base_workflow import BaseWorkflow, WorkflowStep, WorkflowStatus
from src.workflows.workflow_states import WorkflowState
from src.analyzers.question_classifier import GenericQuestionCategory
from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class GenericQAWorkflow(BaseWorkflow):
    """
    Generic Q&A workflow following established LangGraph patterns.
    
    This workflow processes generic project questions through a series of steps:
    1. Initialize - Set up workflow state and validate input
    2. Classify - Classify question and determine analysis approach
    3. Analyze - Perform repository analysis based on question type
    4. Generate - Generate structured response using templates
    5. Finalize - Format final response and cleanup
    
    Inherits all BaseWorkflow capabilities including error handling, retry logic,
    state persistence, and LangChain Runnable interface.
    """

    def __init__(
        self,
        workflow_id: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        enable_persistence: bool = True,
        **kwargs
    ):
        """
        Initialize Generic Q&A workflow.
        
        Args:
            workflow_id: Optional workflow ID
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            enable_persistence: Enable workflow state persistence
            **kwargs: Additional arguments passed to BaseWorkflow
        """
        # REUSE BaseWorkflow initialization
        super().__init__(
            workflow_id=workflow_id,
            max_retries=max_retries,
            retry_delay=retry_delay,
            enable_persistence=enable_persistence,
            **kwargs
        )
        
        # Override logger to include workflow-specific context
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize workflow-specific components (will be lazy-loaded)
        self._analyzers = {}
        self._template_engine = None
        
        self.logger.info(f"Generic Q&A Workflow initialized with ID: {self.workflow_id}")

    def define_steps(self) -> List[str]:
        """
        Define the workflow steps in execution order.
        
        Returns:
            List of step names in execution order
        """
        return [
            WorkflowStep.INITIALIZE.value,
            "classify_question",
            "analyze_repository",
            "generate_response",
            WorkflowStep.FINALIZE.value
        ]

    def execute_step(self, step: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single workflow step with error handling.
        
        Args:
            step: Step name to execute
            state: Current workflow state
            
        Returns:
            Updated workflow state
            
        Raises:
            ValueError: For invalid step names or state
            RuntimeError: For step execution failures
        """
        self.logger.debug(f"Executing Generic Q&A step: {step}")
        
        # Ensure state has required structure
        if not isinstance(state, dict):
            raise ValueError(f"Invalid state type: {type(state)}")
        
        # Record step execution start time
        step_start_time = time.time()
        
        try:
            # Execute step based on step name
            if step == WorkflowStep.INITIALIZE.value:
                updated_state = self._initialize_step(state)
            elif step == "classify_question":
                updated_state = self._classify_question_step(state)
            elif step == "analyze_repository":
                updated_state = self._analyze_repository_step(state)
            elif step == "generate_response":
                updated_state = self._generate_response_step(state)
            elif step == WorkflowStep.FINALIZE.value:
                updated_state = self._finalize_step(state)
            else:
                raise ValueError(f"Unknown workflow step: {step}")
            
            # Record step completion
            step_duration = time.time() - step_start_time
            self.logger.debug(f"Step {step} completed in {step_duration:.2f}s")
            
            # Update step metadata
            if "metadata" not in updated_state:
                updated_state["metadata"] = {}
            updated_state["metadata"][f"{step}_duration"] = step_duration
            updated_state["current_step"] = step
            
            return updated_state
            
        except Exception as e:
            step_duration = time.time() - step_start_time
            self.logger.error(f"Step {step} failed after {step_duration:.2f}s: {e}")
            
            # Add error to state but let BaseWorkflow handle it
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append({
                "step": step,
                "error": str(e),
                "timestamp": time.time(),
                "duration": step_duration
            })
            
            raise  # Re-raise for BaseWorkflow error handling

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate workflow state structure and required fields.
        
        Args:
            state: Workflow state to validate
            
        Returns:
            True if state is valid, False otherwise
        """
        try:
            # Check basic state structure
            if not isinstance(state, dict):
                self.logger.error("State must be a dictionary")
                return False
            
            # Check required fields
            required_fields = ["question", "workflow_id"]
            for field in required_fields:
                if field not in state:
                    self.logger.error(f"Missing required field: {field}")
                    return False
                if not state[field]:
                    self.logger.error(f"Empty required field: {field}")
                    return False
            
            # Validate question is non-empty string
            question = state.get("question", "")
            if not isinstance(question, str) or not question.strip():
                self.logger.error("Question must be a non-empty string")
                return False
            
            self.logger.debug("Workflow state validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating state: {e}")
            return False

    def _initialize_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize workflow state and validate input.
        
        Args:
            state: Initial workflow state
            
        Returns:
            Initialized workflow state
        """
        self.logger.info(f"Initializing Generic Q&A workflow for question: {state.get('question', '')[:100]}...")
        
        # Set processing start time
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["processing_start"] = time.time()
        
        # Initialize default values for missing fields
        state.setdefault("question_category", "general")
        state.setdefault("classification_confidence", 0.0)
        state.setdefault("keywords_matched", [])
        state.setdefault("context_indicators", [])
        state.setdefault("suggested_analyzers", ["general_analyzer"])
        state.setdefault("repository_context", {})
        state.setdefault("analysis_results", {})
        state.setdefault("template_response", {})
        state.setdefault("confidence_score", 0.0)
        state.setdefault("sources", [])
        state.setdefault("errors", [])
        state.setdefault("preferred_template", "generic_template")
        state.setdefault("include_code_examples", True)
        
        # Validate initialized state
        if not self.validate_state(state):
            raise ValueError("State validation failed after initialization")
        
        self.logger.info("Generic Q&A workflow initialized successfully")
        return state

    def _classify_question_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify question and update analysis strategy.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with classification results
        """
        question = state["question"]
        self.logger.info(f"Classifying question type for: {question[:100]}...")
        
        try:
            # If classification already exists, use it (from agent)
            if (state.get("question_category") and 
                state.get("question_category") != "general" and
                state.get("classification_confidence", 0) > 0):
                self.logger.info(f"Using existing classification: {state['question_category']}")
                return state
            
            # Otherwise, perform classification
            from src.analyzers.question_classifier import QuestionClassifier
            
            classifier = QuestionClassifier()
            # Note: This is a sync method call in an async context
            # In a real implementation, we'd need async support or run in executor
            self.logger.warning("Running async classifier in sync context - consider refactoring")
            
            # For now, use a simple keyword-based classification as fallback
            classification_result = self._simple_classify_question(question)
            
            # Update state with classification
            state["question_category"] = classification_result["category"]
            state["classification_confidence"] = classification_result["confidence"]
            state["keywords_matched"] = classification_result["keywords_matched"]
            state["context_indicators"] = classification_result["context_indicators"]
            state["suggested_analyzers"] = classification_result["suggested_analyzers"]
            
            self.logger.info(f"Question classified as: {state['question_category']} "
                           f"(confidence: {state['classification_confidence']:.2f})")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Question classification failed: {e}")
            # Provide fallback classification
            state["question_category"] = "general"
            state["classification_confidence"] = 0.1
            state["suggested_analyzers"] = ["general_analyzer"]
            return state

    def _simple_classify_question(self, question: str) -> Dict[str, Any]:
        """
        Simple keyword-based question classification (fallback).
        
        Args:
            question: Question to classify
            
        Returns:
            Classification result dictionary
        """
        question_lower = question.lower()
        
        # Simple keyword patterns
        patterns = {
            "business_capability": ["business", "domain", "capability", "functionality", "feature"],
            "architecture": ["architecture", "design", "structure", "component", "pattern"],
            "api_endpoints": ["api", "endpoint", "route", "rest", "service"],
            "data_modeling": ["data", "model", "entity", "database", "schema"],
            "operational": ["deploy", "infrastructure", "docker", "environment"],
            "workflows": ["workflow", "process", "flow", "automation"]
        }
        
        best_category = "general"
        best_score = 0.0
        matched_keywords = []
        
        for category, keywords in patterns.items():
            matches = [kw for kw in keywords if kw in question_lower]
            score = len(matches) / len(keywords) if keywords else 0
            
            if score > best_score:
                best_score = score
                best_category = category
                matched_keywords = matches
        
        return {
            "category": best_category,
            "confidence": min(best_score * 2, 1.0),  # Scale up confidence
            "keywords_matched": matched_keywords,
            "context_indicators": [],
            "suggested_analyzers": [f"{best_category}_analyzer"]
        }

    def _analyze_repository_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze repository context based on question category.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with repository analysis
        """
        repository_identifier = state.get("repository_identifier")
        question_category = state.get("question_category", "general")
        
        self.logger.info(f"Analyzing repository context for category: {question_category}")
        
        try:
            # Get repository context if identifier provided
            if repository_identifier:
                repository_context = self._get_repository_context(repository_identifier)
                state["repository_context"] = repository_context
            else:
                self.logger.info("No repository identifier provided, using generic analysis")
                state["repository_context"] = {"type": "generic", "analysis_limited": True}
            
            # Perform category-specific analysis
            analysis_results = self._perform_category_analysis(
                question_category, 
                state["repository_context"],
                state
            )
            state["analysis_results"] = analysis_results
            
            # Calculate confidence based on analysis quality
            confidence_score = self._calculate_analysis_confidence(analysis_results, state)
            state["confidence_score"] = confidence_score
            
            self.logger.info(f"Repository analysis completed with confidence: {confidence_score:.2f}")
            return state
            
        except Exception as e:
            self.logger.error(f"Repository analysis failed: {e}")
            # Provide fallback analysis
            state["repository_context"] = {"error": str(e), "type": "error"}
            state["analysis_results"] = {"error": "Analysis failed", "fallback": True}
            state["confidence_score"] = 0.1
            return state

    def _get_repository_context(self, repository_identifier: str) -> Dict[str, Any]:
        """
        Get repository context from vector store or analysis.
        
        Args:
            repository_identifier: Repository to analyze
            
        Returns:
            Repository context information
        """
        try:
            # Use vector store factory to get repository information
            vector_store = self.get_vector_store()
            
            # Search for repository-related documents
            # This is a simplified implementation - real version would be more sophisticated
            context = {
                "repository": repository_identifier,
                "type": "analyzed",
                "timestamp": datetime.now().isoformat(),
                "languages": [],  # Would be populated from actual analysis
                "frameworks": [],  # Would be detected from codebase
                "architecture_patterns": [],  # Would be inferred from structure
                "document_count": 0  # Would be actual count
            }
            
            self.logger.debug(f"Retrieved context for repository: {repository_identifier}")
            return context
            
        except Exception as e:
            self.logger.warning(f"Could not get repository context: {e}")
            return {
                "repository": repository_identifier,
                "type": "unknown",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _perform_category_analysis(
        self, 
        category: str, 
        repository_context: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform category-specific analysis.
        
        Args:
            category: Question category
            repository_context: Repository context information
            state: Current workflow state
            
        Returns:
            Analysis results
        """
        question = state.get("question", "")
        
        # Basic analysis based on category
        analysis = {
            "category": category,
            "question": question,
            "repository_context": repository_context,
            "analysis_type": "basic",
            "timestamp": datetime.now().isoformat()
        }
        
        # Category-specific analysis logic
        if category == "business_capability":
            analysis.update(self._analyze_business_capability(repository_context, question))
        elif category == "architecture":
            analysis.update(self._analyze_architecture(repository_context, question))
        elif category == "api_endpoints":
            analysis.update(self._analyze_api_endpoints(repository_context, question))
        elif category == "data_modeling":
            analysis.update(self._analyze_data_modeling(repository_context, question))
        elif category == "operational":
            analysis.update(self._analyze_operational(repository_context, question))
        elif category == "workflows":
            analysis.update(self._analyze_workflows(repository_context, question))
        else:
            analysis.update(self._analyze_general(repository_context, question))
        
        return analysis

    def _analyze_business_capability(self, context: Dict, question: str) -> Dict[str, Any]:
        """Analyze business capability aspects."""
        return {
            "scope": "Business domain and capabilities analysis",
            "core_entities": ["User", "System", "Process"],  # Would be extracted from actual codebase
            "key_capabilities": ["Data Processing", "User Management", "Workflow Automation"],
            "business_value": "Provides automated knowledge management and querying capabilities"
        }

    def _analyze_architecture(self, context: Dict, question: str) -> Dict[str, Any]:
        """Analyze architecture aspects."""
        return {
            "architecture_pattern": "Clean Architecture with FastAPI",
            "layers": ["API Layer", "Business Layer", "Data Layer"],
            "components": ["Agents", "Workflows", "Vector Stores", "LLM Integration"],
            "design_principles": ["Separation of Concerns", "Dependency Injection", "SOLID Principles"]
        }

    def _analyze_api_endpoints(self, context: Dict, question: str) -> Dict[str, Any]:
        """Analyze API endpoints."""
        return {
            "endpoints": [
                {"path": "/api/v1/query", "method": "POST", "purpose": "Process user queries"},
                {"path": "/api/v1/index", "method": "POST", "purpose": "Index repositories"},
                {"path": "/api/v1/repositories", "method": "GET", "purpose": "List indexed repositories"}
            ],
            "authentication": "API Key based",
            "response_format": "JSON with structured data"
        }

    def _analyze_data_modeling(self, context: Dict, question: str) -> Dict[str, Any]:
        """Analyze data modeling aspects."""
        return {
            "data_models": ["Document", "Repository", "QueryResult", "WorkflowState"],
            "relationships": ["One-to-Many: Repository -> Documents"],
            "storage": "Vector database (Chroma/Pinecone) with embeddings",
            "validation": "Pydantic models with type safety"
        }

    def _analyze_operational(self, context: Dict, question: str) -> Dict[str, Any]:
        """Analyze operational aspects."""
        return {
            "deployment": "Docker containerized with FastAPI",
            "dependencies": ["Vector Database", "LLM API", "GitHub API"],
            "monitoring": "Health checks and logging",
            "scaling": "Horizontal scaling via Docker containers"
        }

    def _analyze_workflows(self, context: Dict, question: str) -> Dict[str, Any]:
        """Analyze workflow aspects."""
        return {
            "workflows": ["Indexing Workflow", "Query Workflow", "Generic Q&A Workflow"],
            "orchestration": "LangGraph for stateful workflow management",
            "patterns": ["Producer-Consumer", "Pipeline", "State Machine"],
            "error_handling": "Retry logic with exponential backoff"
        }

    def _analyze_general(self, context: Dict, question: str) -> Dict[str, Any]:
        """Analyze general aspects."""
        return {
            "type": "general_analysis",
            "description": "Knowledge Graph Agent for repository analysis and question answering",
            "key_features": ["Repository Indexing", "Semantic Search", "LLM Integration", "Workflow Management"],
            "technologies": ["Python", "FastAPI", "LangChain", "Vector Databases"]
        }

    def _calculate_analysis_confidence(self, analysis_results: Dict, state: Dict) -> float:
        """
        Calculate confidence score for analysis results.
        
        Args:
            analysis_results: Analysis results
            state: Current workflow state
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_confidence = state.get("classification_confidence", 0.0)
        
        # Boost confidence based on analysis completeness
        analysis_factors = [
            bool(analysis_results.get("scope")),
            bool(analysis_results.get("key_capabilities")),
            bool(analysis_results.get("endpoints")),
            bool(analysis_results.get("architecture_pattern")),
            bool(analysis_results.get("workflows"))
        ]
        
        analysis_completeness = sum(analysis_factors) / len(analysis_factors)
        
        # Combine classification confidence with analysis completeness
        final_confidence = (base_confidence * 0.4) + (analysis_completeness * 0.6)
        
        return min(final_confidence, 1.0)

    def _generate_response_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured response using templates.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with generated response
        """
        self.logger.info("Generating structured response using template engine")
        
        try:
            # Get template engine (lazy initialization)
            template_engine = self._get_template_engine()
            
            # Generate template-based response
            template_response = template_engine.generate_response(
                category=state.get("question_category", "general"),
                question=state.get("question", ""),
                analysis_results=state.get("analysis_results", {}),
                repository_context=state.get("repository_context", {}),
                include_code_examples=state.get("include_code_examples", True),
                preferred_template=state.get("preferred_template", "generic_template")
            )
            
            state["template_response"] = template_response
            
            # Extract sources from analysis
            sources = self._extract_sources(state)
            state["sources"] = sources
            
            self.logger.info("Response generation completed successfully")
            return state
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            # Provide fallback response
            state["template_response"] = self._create_fallback_response(state, str(e))
            state["sources"] = []
            return state

    def _get_template_engine(self):
        """Get template engine instance (lazy initialization)."""
        if self._template_engine is None:
            from src.templates.template_engine import TemplateEngine
            self._template_engine = TemplateEngine()
        return self._template_engine

    def _extract_sources(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract source information from analysis results.
        
        Args:
            state: Current workflow state
            
        Returns:
            List of source references
        """
        sources = []
        
        # Add repository as source if available
        repository_context = state.get("repository_context", {})
        if repository_context.get("repository"):
            sources.append({
                "type": "repository",
                "identifier": repository_context["repository"],
                "context": "Repository analysis"
            })
        
        # Add analysis results as sources
        analysis_results = state.get("analysis_results", {})
        if analysis_results and not analysis_results.get("error"):
            sources.append({
                "type": "analysis",
                "category": state.get("question_category", "general"),
                "context": "Category-specific analysis"
            })
        
        return sources

    def _create_fallback_response(self, state: Dict[str, Any], error: str) -> Dict[str, Any]:
        """
        Create fallback response when template generation fails.
        
        Args:
            state: Current workflow state
            error: Error description
            
        Returns:
            Fallback response structure
        """
        category = state.get("question_category", "general")
        question = state.get("question", "")
        
        return {
            "type": "fallback_response",
            "category": category,
            "question": question,
            "response": f"I can help answer questions about {category} topics, but encountered an issue generating a detailed response. Please try rephrasing your question or contact support.",
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

    def _finalize_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize workflow and prepare final response.
        
        Args:
            state: Current workflow state
            
        Returns:
            Finalized workflow state
        """
        self.logger.info("Finalizing Generic Q&A workflow")
        
        # Set processing end time
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["processing_end"] = time.time()
        
        # Calculate total processing time
        start_time = state["metadata"].get("processing_start")
        end_time = state["metadata"]["processing_end"]
        if start_time and end_time:
            state["metadata"]["total_processing_time"] = end_time - start_time
        
        # Final validation
        if not state.get("template_response"):
            self.logger.warning("No template response generated, creating minimal response")
            state["template_response"] = self._create_fallback_response(state, "No response generated")
        
        # Log completion
        processing_time = state["metadata"].get("total_processing_time", 0)
        question = state.get("question", "")
        category = state.get("question_category", "general")
        confidence = state.get("confidence_score", 0.0)
        
        self.logger.info(f"Generic Q&A workflow completed in {processing_time:.2f}s: "
                        f"question='{question[:50]}...', category={category}, confidence={confidence:.2f}")
        
        return state