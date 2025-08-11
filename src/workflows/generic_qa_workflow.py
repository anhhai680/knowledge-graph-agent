"""
Generic Q&A Workflow for LangGraph stateful processing.

This module implements a workflow for processing generic project questions
using question classification, project analysis, and template-based responses.
"""

import time
from typing import Any, Dict, List, Optional
from enum import Enum

from src.workflows.base_workflow import BaseWorkflow, WorkflowStep
from src.analyzers.project_analysis import (
    ArchitectureDetector,
    BusinessCapabilityAnalyzer,
    APIEndpointAnalyzer,
    DataModelAnalyzer,
    OperationalAnalyzer,
)
from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class GenericQAStep(str, Enum):
    """Workflow steps for Generic Q&A processing."""
    
    CLASSIFY_QUESTION = "classify_question"
    ANALYZE_PROJECT = "analyze_project"
    GENERATE_RESPONSE = "generate_response"
    VALIDATE_RESPONSE = "validate_response"


class GenericQAWorkflow(BaseWorkflow):
    """
    LangGraph workflow for Generic Q&A processing.
    
    This workflow implements stateful processing for generic project questions
    with question classification, project analysis, and template-based responses.
    """

    def __init__(
        self,
        template_config_path: Optional[str] = None,
        enable_project_analysis: bool = True,
        max_response_length: int = 5000,
        **kwargs,
    ):
        """
        Initialize Generic Q&A Workflow.

        Args:
            template_config_path: Path to template configuration file
            enable_project_analysis: Enable project structure analysis
            max_response_length: Maximum response length in characters
            **kwargs: Additional workflow parameters
        """
        super().__init__(**kwargs)
        
        self.template_config_path = template_config_path
        self.enable_project_analysis = enable_project_analysis
        self.max_response_length = max_response_length
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize project analyzers
        self.architecture_detector = ArchitectureDetector()
        self.business_analyzer = BusinessCapabilityAnalyzer()
        self.api_analyzer = APIEndpointAnalyzer()
        self.data_analyzer = DataModelAnalyzer()
        self.operational_analyzer = OperationalAnalyzer()
        
        # Template configurations (will be loaded from config file)
        self.templates = self._load_templates()
        
        self.logger.info("Initialized GenericQAWorkflow")

    def define_steps(self) -> List[str]:
        """
        Define the workflow steps in execution order.

        Returns:
            List of step names
        """
        return [
            GenericQAStep.CLASSIFY_QUESTION.value,
            GenericQAStep.ANALYZE_PROJECT.value,
            GenericQAStep.GENERATE_RESPONSE.value,
            GenericQAStep.VALIDATE_RESPONSE.value,
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
        self.logger.debug(f"Executing step: {step}")
        
        if step == GenericQAStep.CLASSIFY_QUESTION.value:
            return self._classify_question(state)
        elif step == GenericQAStep.ANALYZE_PROJECT.value:
            return self._analyze_project(state)
        elif step == GenericQAStep.GENERATE_RESPONSE.value:
            return self._generate_response(state)
        elif step == GenericQAStep.VALIDATE_RESPONSE.value:
            return self._validate_response(state)
        else:
            raise ValueError(f"Unknown step: {step}")

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate workflow state.

        Args:
            state: Workflow state to validate

        Returns:
            True if state is valid, False otherwise
        """
        required_fields = ["question"]
        for field in required_fields:
            if field not in state:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        question = state.get("question", "")
        if not isinstance(question, str) or len(question.strip()) == 0:
            self.logger.error("Question must be a non-empty string")
            return False
        
        return True

    def _classify_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify the question into appropriate category.

        Args:
            state: Current workflow state

        Returns:
            Updated state with question classification
        """
        question = state.get("question", "")
        provided_category = state.get("category")
        
        self.logger.debug(f"Classifying question: {question[:100]}")
        
        # If category is already provided, validate and use it
        if provided_category:
            if provided_category in state.get("supported_categories", []):
                detected_category = provided_category
                confidence = 1.0
            else:
                self.logger.warning(f"Invalid category provided: {provided_category}")
                detected_category = self._detect_category(question)
                confidence = 0.8
        else:
            detected_category = self._detect_category(question)
            confidence = 0.9
        
        state.update({
            "detected_category": detected_category,
            "classification_confidence": confidence,
        })
        
        self.logger.info(f"Question classified as: {detected_category} (confidence: {confidence})")
        return state

    def _analyze_project(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze project structure and context.

        Args:
            state: Current workflow state

        Returns:
            Updated state with project analysis
        """
        if not self.enable_project_analysis:
            self.logger.debug("Project analysis disabled, skipping")
            state["project_analysis"] = {}
            return state
        
        project_template = state.get("project_template", "python_fastapi")
        detected_category = state.get("detected_category")
        
        self.logger.debug(f"Analyzing project with template: {project_template}")
        
        # Perform analysis based on category
        analysis = {}
        
        if detected_category == "architecture":
            analysis.update(self.architecture_detector.analyze(template=project_template))
        elif detected_category == "business_capability":
            analysis.update(self.business_analyzer.analyze(template=project_template))
        elif detected_category == "api_endpoints":
            analysis.update(self.api_analyzer.analyze(template=project_template))
        elif detected_category == "data_modeling":
            analysis.update(self.data_analyzer.analyze(template=project_template))
        elif detected_category == "workflows":
            # Use operational analyzer for workflow questions
            analysis.update(self.operational_analyzer.analyze(template=project_template))
        else:
            # General analysis using multiple analyzers
            analysis.update({
                "architecture": self.architecture_detector.analyze(template=project_template),
                "business": self.business_analyzer.analyze(template=project_template),
                "api": self.api_analyzer.analyze(template=project_template),
            })
        
        state["project_analysis"] = analysis
        self.logger.debug(f"Project analysis completed: {safe_len(analysis)} components")
        return state

    def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response using template-based approach.

        Args:
            state: Current workflow state

        Returns:
            Updated state with generated response
        """
        question = state.get("question", "")
        detected_category = state.get("detected_category")
        project_template = state.get("project_template", "python_fastapi")
        project_analysis = state.get("project_analysis", {})
        
        self.logger.debug(f"Generating response for category: {detected_category}")
        
        # Get template for the category and project type
        template_config = self.templates.get(project_template, {})
        category_template = template_config.get(detected_category, {})
        
        if not category_template:
            # Fallback to generic template
            self.logger.warning(f"No template found for {detected_category} in {project_template}")
            answer = self._generate_fallback_response(question, detected_category, project_analysis)
        else:
            answer = self._apply_template(question, category_template, project_analysis)
        
        # Ensure response is within length limits
        if len(answer) > self.max_response_length:
            answer = answer[:self.max_response_length] + "..."
            self.logger.warning(f"Response truncated to {self.max_response_length} characters")
        
        state.update({
            "answer": answer,
            "template_used": project_template,
            "response_length": len(answer),
        })
        
        self.logger.info(f"Generated response ({len(answer)} characters)")
        return state

    def _validate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate generated response.

        Args:
            state: Current workflow state

        Returns:
            Updated state with validation results
        """
        answer = state.get("answer", "")
        detected_category = state.get("detected_category")
        
        self.logger.debug("Validating generated response")
        
        # Basic validation checks
        validation_results = {
            "is_valid": True,
            "issues": [],
            "confidence_score": 1.0,
        }
        
        # Check minimum length
        if len(answer.strip()) < 50:
            validation_results["issues"].append("Response too short")
            validation_results["confidence_score"] -= 0.3
        
        # Check for placeholder text
        placeholder_indicators = ["[TODO]", "[PLACEHOLDER]", "TODO:", "FIXME:"]
        if any(indicator in answer for indicator in placeholder_indicators):
            validation_results["issues"].append("Contains placeholder text")
            validation_results["confidence_score"] -= 0.2
        
        # Category-specific validation
        if detected_category == "api_endpoints" and "endpoint" not in answer.lower():
            validation_results["issues"].append("API endpoint response should mention endpoints")
            validation_results["confidence_score"] -= 0.1
        
        if detected_category == "architecture" and "architecture" not in answer.lower():
            validation_results["issues"].append("Architecture response should mention architecture")
            validation_results["confidence_score"] -= 0.1
        
        # Set overall validity
        validation_results["is_valid"] = validation_results["confidence_score"] > 0.5
        
        state.update({
            "validation_results": validation_results,
            "confidence_score": max(0.0, min(1.0, validation_results["confidence_score"])),
        })
        
        if validation_results["issues"]:
            self.logger.warning(f"Response validation issues: {validation_results['issues']}")
        else:
            self.logger.info("Response validation passed")
        
        return state

    def _detect_category(self, question: str) -> str:
        """
        Detect question category using keyword matching.

        Args:
            question: Question text

        Returns:
            Detected category name
        """
        question_lower = question.lower()
        
        # Category keywords
        category_keywords = {
            "business_capability": [
                "business", "capability", "domain", "entities", "ownership",
                "consumers", "scope", "rules", "invariants"
            ],
            "api_endpoints": [
                "api", "endpoint", "routes", "http", "rest", "status",
                "pagination", "versioning", "response", "request"
            ],
            "data_modeling": [
                "data", "database", "persistence", "repository", "model",
                "transaction", "consistency", "schema", "migration"
            ],
            "workflows": [
                "workflow", "process", "flow", "create", "update", "delete",
                "operation", "step", "sequence", "handle"
            ],
            "architecture": [
                "architecture", "layers", "security", "observability",
                "deployment", "monitoring", "structure", "pattern"
            ]
        }
        
        # Calculate scores for each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or default
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return "architecture"  # Default category

    def _load_templates(self) -> Dict[str, Any]:
        """
        Load template configurations.

        Returns:
            Template configuration dictionary
        """
        # For now, return a basic template structure
        # This would be loaded from JSON files in a real implementation
        return {
            "python_fastapi": {
                "business_capability": {
                    "structure": "Define scope, entities, ownership, and SLAs",
                    "examples": ["Order processing", "User management", "Payment handling"]
                },
                "api_endpoints": {
                    "structure": "REST endpoints with CRUD operations",
                    "patterns": ["GET /items", "POST /items", "PUT /items/{id}"]
                },
                "data_modeling": {
                    "structure": "SQLAlchemy models with Pydantic schemas",
                    "patterns": ["Repository pattern", "Unit of Work", "Database migrations"]
                },
                "workflows": {
                    "structure": "FastAPI dependency injection with async processing",
                    "patterns": ["Request validation", "Business logic", "Response formatting"]
                },
                "architecture": {
                    "structure": "Layered architecture with clean separation",
                    "patterns": ["API layer", "Business layer", "Data layer", "External services"]
                }
            },
            "dotnet_clean_architecture": {
                "business_capability": {
                    "structure": "Domain-driven design with bounded contexts",
                    "examples": ["Aggregates", "Domain services", "Value objects"]
                },
                "api_endpoints": {
                    "structure": "ASP.NET Core Web API with controllers",
                    "patterns": ["RESTful design", "Problem Details", "Swagger/OpenAPI"]
                },
                "data_modeling": {
                    "structure": "Entity Framework Core with domain models",
                    "patterns": ["Repository pattern", "Unit of Work", "Code-first migrations"]
                },
                "workflows": {
                    "structure": "MediatR with CQRS pattern",
                    "patterns": ["Commands", "Queries", "Handlers", "Pipeline behaviors"]
                },
                "architecture": {
                    "structure": "Clean Architecture with dependency inversion",
                    "patterns": ["Domain", "Application", "Infrastructure", "Presentation"]
                }
            },
            "react_spa": {
                "business_capability": {
                    "structure": "Frontend business logic and user workflows",
                    "examples": ["User interfaces", "State management", "User experience"]
                },
                "api_endpoints": {
                    "structure": "HTTP client integration with backend APIs",
                    "patterns": ["Axios/Fetch", "API services", "Error handling"]
                },
                "data_modeling": {
                    "structure": "Frontend data models and state management",
                    "patterns": ["Redux/Context", "Local storage", "Caching"]
                },
                "workflows": {
                    "structure": "User interaction flows and navigation",
                    "patterns": ["React Router", "Form handling", "Async operations"]
                },
                "architecture": {
                    "structure": "Component-based architecture with hooks",
                    "patterns": ["Components", "Hooks", "Services", "Utils"]
                }
            }
        }

    def _apply_template(
        self, 
        question: str, 
        template: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> str:
        """
        Apply template to generate response.

        Args:
            question: Original question
            template: Template configuration
            analysis: Project analysis results

        Returns:
            Generated response
        """
        structure = template.get("structure", "")
        patterns = ensure_list(template.get("patterns", []))
        examples = ensure_list(template.get("examples", []))
        
        # Build response from template
        response_parts = []
        
        if structure:
            response_parts.append(f"**Structure:** {structure}")
        
        if patterns:
            response_parts.append(f"**Common Patterns:**")
            for pattern in patterns[:5]:  # Limit to 5 patterns
                response_parts.append(f"- {pattern}")
        
        if examples:
            response_parts.append(f"**Examples:**")
            for example in examples[:3]:  # Limit to 3 examples
                response_parts.append(f"- {example}")
        
        # Add analysis if available
        if analysis:
            response_parts.append(f"**Analysis:**")
            for key, value in analysis.items():
                if isinstance(value, (str, int, float)):
                    response_parts.append(f"- {key}: {value}")
                elif isinstance(value, list) and safe_len(value) > 0:
                    response_parts.append(f"- {key}: {', '.join(str(v) for v in value[:3])}")
        
        return "\n".join(response_parts)

    def _generate_fallback_response(
        self, 
        question: str, 
        category: str, 
        analysis: Dict[str, Any]
    ) -> str:
        """
        Generate fallback response when no template is available.

        Args:
            question: Original question
            category: Detected category
            analysis: Project analysis results

        Returns:
            Fallback response
        """
        response = f"Based on your question about {category}, here are some general considerations:\n\n"
        
        category_fallbacks = {
            "business_capability": "Consider defining the business scope, core entities, ownership responsibilities, and service level agreements.",
            "api_endpoints": "Design RESTful endpoints with proper HTTP methods, status codes, pagination, and error handling.",
            "data_modeling": "Implement proper data persistence patterns, repositories, transaction management, and data validation.",
            "workflows": "Define clear end-to-end processes with proper error handling, validation, and state management.",
            "architecture": "Organize your system with clear layers, separation of concerns, security measures, and observability."
        }
        
        response += category_fallbacks.get(category, "Consider best practices for your specific use case.")
        
        # Add analysis if available
        if analysis:
            response += f"\n\n**Project Analysis:**\n"
            for key, value in analysis.items():
                if isinstance(value, str):
                    response += f"- {key}: {value}\n"
        
        return response

    def get_available_templates(self) -> List[str]:
        """
        Get list of available project templates.

        Returns:
            List of template names
        """
        return list(self.templates.keys())

    async def analyze_project(self, analysis_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform project structure analysis.

        Args:
            analysis_state: Analysis configuration and parameters

        Returns:
            Analysis results
        """
        try:
            project_path = analysis_state.get("project_path")
            repository_url = analysis_state.get("repository_url")
            template_hint = analysis_state.get("template_hint")
            
            self.logger.info("Performing project structure analysis")
            
            # Use analyzers to detect project characteristics
            results = {
                "detected_template": template_hint or "python_fastapi",
                "architecture_patterns": self.architecture_detector.detect_patterns(
                    project_path=project_path,
                    repository_url=repository_url
                ),
                "business_capabilities": self.business_analyzer.extract_capabilities(
                    project_path=project_path
                ),
                "api_endpoints": self.api_analyzer.discover_endpoints(
                    project_path=project_path
                ),
                "data_models": self.data_analyzer.analyze_models(
                    project_path=project_path
                ),
                "operational_patterns": self.operational_analyzer.analyze_operations(
                    project_path=project_path
                ),
                "confidence": 0.8,
                "analysis_timestamp": time.time()
            }
            
            return {
                "success": True,
                **results
            }
            
        except Exception as e:
            self.logger.error(f"Error in project analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }