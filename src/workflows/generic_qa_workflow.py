"""
Generic Q&A Workflow for LangGraph Stateful Processing.

This module implements the workflow component for Generic Project Q&A processing,
extending BaseWorkflow with template-based response generation and project analysis.
"""

import time
from typing import Any, Dict, List, Optional
from enum import Enum

from src.workflows.base_workflow import BaseWorkflow, WorkflowStep
from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list
from src.agents.generic_qa_agent import QuestionCategory


class GenericQAStep(str, Enum):
    """Workflow steps for Generic Q&A processing."""
    
    INITIALIZE = "initialize"
    CLASSIFY_QUESTION = "classify_question"
    LOAD_TEMPLATE = "load_template"
    ANALYZE_CONTEXT = "analyze_context"
    GENERATE_RESPONSE = "generate_response"
    VALIDATE_RESPONSE = "validate_response"
    FINALIZE = "finalize"


class GenericQAWorkflow(BaseWorkflow):
    """
    Generic Q&A Workflow extending BaseWorkflow with LangGraph stateful processing.
    
    This workflow provides structured processing for Generic Project Q&A with
    template-based response generation, context analysis, and quality validation.
    """
    
    def __init__(
        self,
        workflow_id: Optional[str] = None,
        template_config: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        enable_persistence: bool = True,
    ):
        """
        Initialize Generic Q&A Workflow.
        
        Args:
            workflow_id: Optional workflow ID
            template_config: Template configuration dictionary
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries
            enable_persistence: Enable workflow state persistence
        """
        super().__init__(
            workflow_id=workflow_id,
            max_retries=max_retries,
            retry_delay=retry_delay,
            enable_persistence=enable_persistence,
        )
        
        self.template_config = template_config or {}
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize template cache
        self.template_cache = {}
        self._load_default_templates()
    
    def define_steps(self) -> List[str]:
        """
        Define the workflow steps.
        
        Returns:
            List of step names in execution order
        """
        return [
            GenericQAStep.INITIALIZE,
            GenericQAStep.CLASSIFY_QUESTION,
            GenericQAStep.LOAD_TEMPLATE,
            GenericQAStep.ANALYZE_CONTEXT,
            GenericQAStep.GENERATE_RESPONSE,
            GenericQAStep.VALIDATE_RESPONSE,
            GenericQAStep.FINALIZE,
        ]
    
    def execute_step(self, step: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single workflow step.
        
        Args:
            step: Step name to execute
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        self.logger.debug(f"Executing Generic Q&A step: {step}")
        
        step_handlers = {
            GenericQAStep.INITIALIZE: self._initialize_step,
            GenericQAStep.CLASSIFY_QUESTION: self._classify_question_step,
            GenericQAStep.LOAD_TEMPLATE: self._load_template_step,
            GenericQAStep.ANALYZE_CONTEXT: self._analyze_context_step,
            GenericQAStep.GENERATE_RESPONSE: self._generate_response_step,
            GenericQAStep.VALIDATE_RESPONSE: self._validate_response_step,
            GenericQAStep.FINALIZE: self._finalize_step,
        }
        
        handler = step_handlers.get(step)
        if not handler:
            raise ValueError(f"Unknown workflow step: {step}")
        
        return handler(state)
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate workflow state.
        
        Args:
            state: Workflow state to validate
            
        Returns:
            True if state is valid, False otherwise
        """
        required_fields = ["question", "category"]
        
        for field in required_fields:
            if field not in state:
                self.logger.error(f"Missing required field in state: {field}")
                return False
        
        # Validate question is not empty
        question = state.get("question", "")
        if not isinstance(question, str) or len(question.strip()) == 0:
            self.logger.error("Question field is empty or invalid")
            return False
        
        # Validate category
        category = state.get("category")
        if category:
            try:
                QuestionCategory(category)
            except ValueError:
                self.logger.error(f"Invalid question category: {category}")
                return False
        
        return True
    
    def _initialize_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize workflow state and validate input."""
        self.logger.debug("Initializing Generic Q&A workflow")
        
        # Ensure required fields are present
        if "question" not in state:
            raise ValueError("Question is required for Generic Q&A workflow")
        
        # Initialize workflow metadata
        state.setdefault("workflow_type", "generic_qa")
        state.setdefault("start_time", time.time())
        state.setdefault("analysis_components", [])
        state.setdefault("template_used", None)
        state.setdefault("confidence_score", 0.0)
        state.setdefault("metadata", {})
        
        # Initialize step tracking
        state.setdefault("completed_steps", [])
        state["completed_steps"].append(GenericQAStep.INITIALIZE)
        
        self.logger.info(f"Initialized Generic Q&A workflow for question: {state['question'][:100]}...")
        return state
    
    def _classify_question_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the question into appropriate category."""
        self.logger.debug("Classifying question category")
        
        question = state["question"]
        
        # Use category from state if already provided
        if "category" in state and state["category"]:
            try:
                category = QuestionCategory(state["category"])
                self.logger.debug(f"Using provided category: {category}")
            except ValueError:
                # Invalid category provided, classify automatically
                category = self._classify_question_internal(question)
        else:
            # Classify automatically
            category = self._classify_question_internal(question)
        
        state["category"] = category.value
        state["analysis_components"].append("question_classification")
        state["completed_steps"].append(GenericQAStep.CLASSIFY_QUESTION)
        
        self.logger.info(f"Question classified as: {category}")
        return state
    
    def _load_template_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Load appropriate template for the question category."""
        self.logger.debug("Loading response template")
        
        category = QuestionCategory(state["category"])
        repository_context = state.get("repository_context")
        
        # Determine template type based on context
        template_type = self._determine_template_type(repository_context)
        
        # Load template from cache or configuration
        template = self._get_template(category, template_type)
        
        state["template"] = template
        state["template_type"] = template_type
        state["template_used"] = f"{template_type}_{category.value}"
        state["analysis_components"].append("template_loading")
        state["completed_steps"].append(GenericQAStep.LOAD_TEMPLATE)
        
        self.logger.debug(f"Loaded template: {template_type} for category: {category}")
        return state
    
    def _analyze_context_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze repository context and gather relevant information."""
        self.logger.debug("Analyzing repository context")
        
        repository_context = state.get("repository_context")
        category = QuestionCategory(state["category"])
        
        # Perform context analysis based on category
        context_analysis = self._perform_context_analysis(category, repository_context)
        
        state["context_analysis"] = context_analysis
        state["analysis_components"].extend(context_analysis.get("components_analyzed", []))
        state["completed_steps"].append(GenericQAStep.ANALYZE_CONTEXT)
        
        # Update confidence based on context availability
        context_quality = context_analysis.get("quality_score", 0.0)
        state["confidence_score"] = min(state["confidence_score"] + context_quality, 1.0)
        
        self.logger.debug(f"Context analysis completed with quality score: {context_quality}")
        return state
    
    def _generate_response_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using template and context analysis."""
        self.logger.debug("Generating response")
        
        template = state.get("template", {})
        context_analysis = state.get("context_analysis", {})
        question = state["question"]
        category = QuestionCategory(state["category"])
        
        # Generate response using template and context
        response = self._generate_response_from_template(
            template, context_analysis, question, category
        )
        
        state["answer"] = response
        state["analysis_components"].append("response_generation")
        state["completed_steps"].append(GenericQAStep.GENERATE_RESPONSE)
        
        # Update confidence based on response quality
        response_quality = self._assess_response_quality(response, question)
        state["confidence_score"] = min(state["confidence_score"] + response_quality, 1.0)
        
        self.logger.info(f"Generated response with quality score: {response_quality}")
        return state
    
    def _validate_response_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance the generated response."""
        self.logger.debug("Validating response")
        
        answer = state.get("answer", "")
        question = state["question"]
        category = QuestionCategory(state["category"])
        
        # Perform response validation
        validation_result = self._validate_response_quality(answer, question, category)
        
        # Enhance response if needed
        if validation_result["needs_enhancement"]:
            enhanced_answer = self._enhance_response(answer, validation_result["suggestions"])
            state["answer"] = enhanced_answer
            state["analysis_components"].append("response_enhancement")
        
        state["validation_result"] = validation_result
        state["analysis_components"].append("response_validation")
        state["completed_steps"].append(GenericQAStep.VALIDATE_RESPONSE)
        
        # Final confidence adjustment
        validation_score = validation_result.get("confidence_adjustment", 0.0)
        state["confidence_score"] = max(0.0, min(state["confidence_score"] + validation_score, 1.0))
        
        self.logger.debug(f"Response validation completed with adjustment: {validation_score}")
        return state
    
    def _finalize_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize workflow and prepare final response."""
        self.logger.debug("Finalizing workflow")
        
        # Calculate processing time
        start_time = state.get("start_time", time.time())
        processing_time = time.time() - start_time
        
        # Compile final metadata
        final_metadata = {
            "processing_time": processing_time,
            "analysis_components": state.get("analysis_components", []),
            "template_used": state.get("template_used"),
            "workflow_id": self.workflow_id,
            "completed_steps": state.get("completed_steps", []),
            "validation_result": state.get("validation_result", {}),
        }
        
        state["metadata"] = final_metadata
        state["status"] = "completed"
        state["completed_steps"].append(GenericQAStep.FINALIZE)
        
        self.logger.info(f"Generic Q&A workflow completed in {processing_time:.2f}s")
        return state
    
    def _load_default_templates(self) -> None:
        """Load default templates for different categories and project types."""
        # Default templates for each category
        self.template_cache = {
            "dotnet_clean_architecture": {
                QuestionCategory.BUSINESS_CAPABILITY: {
                    "structure": "## Business Capability Analysis\n\n{analysis}\n\n## Core Entities\n\n{entities}\n\n## Ownership\n\n{ownership}",
                    "components": ["domain_analysis", "entity_extraction", "ownership_mapping"]
                },
                QuestionCategory.API_ENDPOINTS: {
                    "structure": "## API Endpoints\n\n{endpoints}\n\n## Status Codes\n\n{status_codes}\n\n## Patterns\n\n{patterns}",
                    "components": ["endpoint_discovery", "status_code_analysis", "pattern_detection"]
                },
                QuestionCategory.DATA_MODELING: {
                    "structure": "## Data Models\n\n{models}\n\n## Persistence\n\n{persistence}\n\n## Repositories\n\n{repositories}",
                    "components": ["model_analysis", "persistence_patterns", "repository_patterns"]
                },
                QuestionCategory.WORKFLOWS: {
                    "structure": "## Workflows\n\n{workflows}\n\n## Processes\n\n{processes}\n\n## Patterns\n\n{patterns}",
                    "components": ["workflow_analysis", "process_mapping", "pattern_identification"]
                },
                QuestionCategory.ARCHITECTURE: {
                    "structure": "## Architecture\n\n{architecture}\n\n## Layers\n\n{layers}\n\n## Patterns\n\n{patterns}",
                    "components": ["architecture_analysis", "layer_identification", "pattern_detection"]
                }
            },
            "python_fastapi": {
                QuestionCategory.BUSINESS_CAPABILITY: {
                    "structure": "## Business Domain\n\n{analysis}\n\n## Core Models\n\n{models}\n\n## Services\n\n{services}",
                    "components": ["domain_analysis", "model_extraction", "service_mapping"]
                },
                QuestionCategory.API_ENDPOINTS: {
                    "structure": "## FastAPI Endpoints\n\n{endpoints}\n\n## Responses\n\n{responses}\n\n## Validation\n\n{validation}",
                    "components": ["endpoint_discovery", "response_analysis", "validation_patterns"]
                },
                QuestionCategory.DATA_MODELING: {
                    "structure": "## Data Models\n\n{models}\n\n## Database\n\n{database}\n\n## ORM\n\n{orm}",
                    "components": ["model_analysis", "database_patterns", "orm_usage"]
                },
                QuestionCategory.WORKFLOWS: {
                    "structure": "## Application Flows\n\n{flows}\n\n## Business Logic\n\n{logic}\n\n## Dependencies\n\n{dependencies}",
                    "components": ["flow_analysis", "logic_extraction", "dependency_mapping"]
                },
                QuestionCategory.ARCHITECTURE: {
                    "structure": "## FastAPI Architecture\n\n{architecture}\n\n## Structure\n\n{structure}\n\n## Patterns\n\n{patterns}",
                    "components": ["architecture_analysis", "structure_mapping", "pattern_detection"]
                }
            },
            "generic": {
                QuestionCategory.BUSINESS_CAPABILITY: {
                    "structure": "## Business Analysis\n\n{analysis}\n\n## Key Components\n\n{components}",
                    "components": ["general_analysis", "component_identification"]
                },
                QuestionCategory.API_ENDPOINTS: {
                    "structure": "## API Analysis\n\n{analysis}\n\n## Endpoints\n\n{endpoints}",
                    "components": ["api_analysis", "endpoint_discovery"]
                },
                QuestionCategory.DATA_MODELING: {
                    "structure": "## Data Analysis\n\n{analysis}\n\n## Models\n\n{models}",
                    "components": ["data_analysis", "model_identification"]
                },
                QuestionCategory.WORKFLOWS: {
                    "structure": "## Workflow Analysis\n\n{analysis}\n\n## Processes\n\n{processes}",
                    "components": ["workflow_analysis", "process_identification"]
                },
                QuestionCategory.ARCHITECTURE: {
                    "structure": "## Architecture Analysis\n\n{analysis}\n\n## Structure\n\n{structure}",
                    "components": ["architecture_analysis", "structure_identification"]
                }
            }
        }
    
    def _classify_question_internal(self, question: str) -> QuestionCategory:
        """Internal question classification logic."""
        question_lower = question.lower().strip()
        
        classification_patterns = {
            QuestionCategory.BUSINESS_CAPABILITY: [
                "business capability", "domain scope", "core entities", "ownership",
                "bounded context", "business logic", "domain model", "business rules"
            ],
            QuestionCategory.API_ENDPOINTS: [
                "api endpoints", "rest endpoints", "status codes", "pagination",
                "api routes", "http methods", "request response", "api documentation"
            ],
            QuestionCategory.DATA_MODELING: [
                "data modeling", "persistence patterns", "repositories", "transactions",
                "database schema", "data access", "orm", "entity framework"
            ],
            QuestionCategory.WORKFLOWS: [
                "workflows", "business processes", "create flow", "update flow",
                "end-to-end", "process steps", "workflow patterns"
            ],
            QuestionCategory.ARCHITECTURE: [
                "architecture", "layers", "security", "observability", "deployment",
                "clean architecture", "mvc pattern", "microservices", "components"
            ]
        }
        
        category_scores = {}
        for category, patterns in classification_patterns.items():
            score = sum(1 for pattern in patterns if pattern in question_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return QuestionCategory.ARCHITECTURE
    
    def _determine_template_type(self, repository_context: Optional[Dict[str, Any]]) -> str:
        """Determine template type based on repository context."""
        if not repository_context:
            return "generic"
        
        # Detect project type from file patterns or metadata
        file_patterns = repository_context.get("file_patterns", [])
        
        if any(pattern.endswith((".cs", ".csproj", ".sln")) for pattern in file_patterns):
            return "dotnet_clean_architecture"
        elif any(pattern.endswith((".py", "requirements.txt", "pyproject.toml")) for pattern in file_patterns):
            return "python_fastapi"
        elif any(pattern.endswith((".js", ".jsx", ".ts", ".tsx", "package.json")) for pattern in file_patterns):
            return "react_spa"
        
        return "generic"
    
    def _get_template(self, category: QuestionCategory, template_type: str) -> Dict[str, Any]:
        """Get template for category and type."""
        template_set = self.template_cache.get(template_type, self.template_cache["generic"])
        return template_set.get(category, template_set[QuestionCategory.ARCHITECTURE])
    
    def _perform_context_analysis(
        self, 
        category: QuestionCategory, 
        repository_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform context analysis for the given category."""
        # Placeholder for actual context analysis
        # This will be implemented with the project analysis components in Phase 2
        
        analysis = {
            "category": category.value,
            "context_available": repository_context is not None,
            "components_analyzed": [],
            "quality_score": 0.3,  # Base score
            "findings": []
        }
        
        if repository_context:
            analysis["quality_score"] = 0.7
            analysis["components_analyzed"] = ["repository_structure", "file_patterns"]
            analysis["findings"] = ["Repository context available for analysis"]
        
        return analysis
    
    def _generate_response_from_template(
        self,
        template: Dict[str, Any],
        context_analysis: Dict[str, Any],
        question: str,
        category: QuestionCategory
    ) -> str:
        """Generate response using template and context."""
        structure = template.get("structure", "## Analysis\n\n{analysis}")
        
        # Generate content based on category and context
        content_mapping = {
            "analysis": f"Analysis for {category.value} question: {question}",
            "components": "Key components identified in the project",
            "patterns": "Common patterns and best practices",
            "architecture": "System architecture overview",
            "endpoints": "API endpoints and their functionality",
            "models": "Data models and their relationships",
            "workflows": "Business workflows and processes",
            "layers": "Architectural layers and their responsibilities"
        }
        
        # Add context-specific content if available
        if context_analysis.get("findings"):
            content_mapping["analysis"] += f"\n\nContext findings: {', '.join(context_analysis['findings'])}"
        
        # Format template with content
        try:
            response = structure.format(**content_mapping)
        except KeyError as e:
            # Fallback if template has missing placeholders
            response = f"## {category.value.replace('_', ' ').title()} Analysis\n\n{content_mapping['analysis']}"
        
        return response
    
    def _assess_response_quality(self, response: str, question: str) -> float:
        """Assess quality of generated response."""
        quality_score = 0.0
        
        # Basic length check
        if len(response) > 100:
            quality_score += 0.3
        
        # Check for structured content
        if "##" in response:
            quality_score += 0.2
        
        # Check if response addresses question
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        overlap = len(question_words.intersection(response_words))
        if overlap > 0:
            quality_score += min(0.3, overlap * 0.05)
        
        return min(quality_score, 0.8)  # Cap at 0.8 for generated responses
    
    def _validate_response_quality(
        self, 
        answer: str, 
        question: str, 
        category: QuestionCategory
    ) -> Dict[str, Any]:
        """Validate response quality and suggest improvements."""
        validation = {
            "needs_enhancement": False,
            "suggestions": [],
            "confidence_adjustment": 0.0,
            "quality_metrics": {}
        }
        
        # Length validation
        if len(answer) < 50:
            validation["needs_enhancement"] = True
            validation["suggestions"].append("Response is too short")
            validation["confidence_adjustment"] -= 0.2
        
        # Structure validation
        if "##" not in answer:
            validation["suggestions"].append("Add structured headers")
        
        # Category-specific validation
        category_keywords = {
            QuestionCategory.BUSINESS_CAPABILITY: ["business", "domain", "entity"],
            QuestionCategory.API_ENDPOINTS: ["api", "endpoint", "http"],
            QuestionCategory.DATA_MODELING: ["data", "model", "database"],
            QuestionCategory.WORKFLOWS: ["workflow", "process", "flow"],
            QuestionCategory.ARCHITECTURE: ["architecture", "system", "component"]
        }
        
        keywords = category_keywords.get(category, [])
        if keywords and not any(keyword in answer.lower() for keyword in keywords):
            validation["suggestions"].append(f"Include more {category.value} specific terminology")
            validation["confidence_adjustment"] -= 0.1
        
        return validation
    
    def _enhance_response(self, answer: str, suggestions: List[str]) -> str:
        """Enhance response based on validation suggestions."""
        enhanced = answer
        
        if "Response is too short" in suggestions:
            enhanced += f"\n\n**Additional Notes:**\nThis analysis provides a foundation for understanding the {QuestionCategory.ARCHITECTURE.value} aspects. For more detailed information, consider examining specific code files and documentation."
        
        if "Add structured headers" in suggestions and "##" not in enhanced:
            enhanced = f"## Analysis\n\n{enhanced}"
        
        return enhanced