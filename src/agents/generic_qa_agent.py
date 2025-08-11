"""
Generic Q&A Agent for Project Architecture Analysis.

This module implements a comprehensive AI Agent for Generic Project Q&A that can answer
structured questions about project architecture, business capabilities, API endpoints,
data modeling, workflows, and operational concerns.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from langchain.schema.runnable import RunnableConfig
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentResponse
from src.workflows.base_workflow import BaseWorkflow
from src.utils.defensive_programming import safe_len, ensure_list
from src.utils.logging import get_logger


class QuestionCategory(str, Enum):
    """Question categories for Generic Project Q&A."""
    
    BUSINESS_CAPABILITY = "business_capability"
    API_ENDPOINTS = "api_endpoints"
    DATA_MODELING = "data_modeling"
    WORKFLOWS = "workflows"
    ARCHITECTURE = "architecture"


@dataclass
class GenericQAResponse:
    """Structured response for Generic Q&A queries."""
    
    category: QuestionCategory
    question: str
    answer: str
    analysis_components: List[str]
    confidence_score: float
    template_used: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GenericQAAgent(BaseAgent):
    """
    Generic Q&A Agent extending BaseAgent with LangChain Runnable interface.
    
    This agent provides comprehensive project analysis and Q&A capabilities
    with template-based response generation and question classification.
    """
    
    def __init__(
        self,
        workflow: Optional[BaseWorkflow] = None,
        default_templates: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize Generic Q&A Agent.
        
        Args:
            workflow: LangGraph workflow instance for stateful processing
            default_templates: Default template configurations
            **kwargs: Additional initialization parameters
        """
        super().__init__(
            workflow=workflow,
            agent_name="GenericQAAgent",
            **kwargs
        )
        
        self.default_templates = default_templates or {}
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize question classification patterns
        self._init_classification_patterns()
    
    def _init_classification_patterns(self) -> None:
        """Initialize question classification patterns for each category."""
        self.classification_patterns = {
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
    
    def classify_question(self, question: str) -> QuestionCategory:
        """
        Classify question into appropriate category.
        
        Args:
            question: User question to classify
            
        Returns:
            QuestionCategory: Detected question category
        """
        question_lower = question.lower().strip()
        
        category_scores = {}
        
        # Score each category based on keyword matches
        for category, patterns in self.classification_patterns.items():
            score = sum(1 for pattern in patterns if pattern in question_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            self.logger.debug(f"Question classified as {best_category} (score: {category_scores[best_category]})")
            return best_category
        
        # Default to architecture if no clear match
        self.logger.debug("No clear category match, defaulting to architecture")
        return QuestionCategory.ARCHITECTURE
    
    def get_supported_categories(self) -> List[Dict[str, Any]]:
        """
        Get list of supported question categories with descriptions.
        
        Returns:
            List of category information dictionaries
        """
        categories = [
            {
                "id": QuestionCategory.BUSINESS_CAPABILITY,
                "name": "Business Capability",
                "description": "Domain scope, core entities, business logic, ownership patterns",
                "examples": [
                    "What business capability does this service own?",
                    "What are the core entities and their relationships?",
                    "Who owns this bounded context?"
                ]
            },
            {
                "id": QuestionCategory.API_ENDPOINTS,
                "name": "API Endpoints", 
                "description": "REST endpoints, status codes, pagination, API patterns",
                "examples": [
                    "What are the main API endpoints?",
                    "How is pagination implemented?",
                    "What status codes are returned?"
                ]
            },
            {
                "id": QuestionCategory.DATA_MODELING,
                "name": "Data Modeling",
                "description": "Persistence patterns, repositories, transactions, data access",
                "examples": [
                    "How is data modeled and persisted?",
                    "What repository patterns are used?",
                    "How are transactions handled?"
                ]
            },
            {
                "id": QuestionCategory.WORKFLOWS,
                "name": "Workflows",
                "description": "End-to-end operations, create/update flows, business processes",
                "examples": [
                    "What's the end-to-end workflow for creating entities?",
                    "How are update operations handled?",
                    "What are the key business processes?"
                ]
            },
            {
                "id": QuestionCategory.ARCHITECTURE,
                "name": "Architecture",
                "description": "Layers, security, observability, deployment, architectural patterns",
                "examples": [
                    "What's the overall architecture?",
                    "How is security implemented?",
                    "What observability patterns are used?"
                ]
            }
        ]
        
        return categories
    
    async def _process_input(
        self, 
        input_data: Union[str, Dict[str, Any]], 
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Process input data and return structured response.
        
        Args:
            input_data: Input data containing question and optional parameters
            config: Optional configuration for processing
            
        Returns:
            Structured response dictionary
        """
        try:
            # Extract question from input
            if isinstance(input_data, str):
                question = input_data
                repository_context = None
                specific_category = None
            else:
                question = input_data.get("question", "")
                repository_context = input_data.get("repository_context")
                specific_category = input_data.get("category")
            
            if not question:
                return {
                    "success": False,
                    "error": "Question is required",
                    "agent": self.agent_name,
                }
            
            self.logger.info(f"Processing Generic Q&A question: {question[:100]}...")
            
            # Classify question if category not specified
            if specific_category:
                try:
                    category = QuestionCategory(specific_category)
                except ValueError:
                    category = self.classify_question(question)
            else:
                category = self.classify_question(question)
            
            # Process through workflow if available
            if self.workflow:
                workflow_state = {
                    "question": question,
                    "category": category.value,
                    "repository_context": repository_context,
                    "agent_type": "generic_qa",
                    "step": "processing"
                }
                
                result_state = self.workflow.invoke(workflow_state)
                
                # Extract response from workflow result
                answer = result_state.get("answer", "Unable to generate answer")
                analysis_components = ensure_list(result_state.get("analysis_components", []))
                confidence_score = result_state.get("confidence_score", 0.0)
                template_used = result_state.get("template_used")
                metadata = result_state.get("metadata", {})
                
            else:
                # Fallback processing without workflow
                answer = self._generate_fallback_answer(question, category)
                analysis_components = [category.value]
                confidence_score = 0.5
                template_used = None
                metadata = {}
            
            # Create structured response
            qa_response = GenericQAResponse(
                category=category,
                question=question,
                answer=answer,
                analysis_components=analysis_components,
                confidence_score=confidence_score,
                template_used=template_used,
                metadata=metadata
            )
            
            return {
                "success": True,
                "response": qa_response,
                "category": category.value,
                "question": question,
                "answer": answer,
                "analysis_components": analysis_components,
                "confidence_score": confidence_score,
                "template_used": template_used,
                "metadata": metadata,
                "agent": self.agent_name,
            }
            
        except Exception as e:
            self.logger.error(f"Error processing Generic Q&A input: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "agent": self.agent_name,
            }
    
    def _validate_input(self, input_data: Any) -> bool:
        """
        Validate input data format and content.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if isinstance(input_data, str):
            return len(input_data.strip()) > 0
        
        if isinstance(input_data, dict):
            question = input_data.get("question", "")
            return isinstance(question, str) and len(question.strip()) > 0
        
        return False
    
    def _generate_fallback_answer(self, question: str, category: QuestionCategory) -> str:
        """
        Generate fallback answer when workflow is not available.
        
        Args:
            question: User question
            category: Classified question category
            
        Returns:
            Fallback answer string
        """
        category_descriptions = {
            QuestionCategory.BUSINESS_CAPABILITY: "business capabilities and domain modeling",
            QuestionCategory.API_ENDPOINTS: "API endpoint design and implementation",
            QuestionCategory.DATA_MODELING: "data modeling and persistence patterns",
            QuestionCategory.WORKFLOWS: "workflow patterns and business processes",
            QuestionCategory.ARCHITECTURE: "architectural patterns and system design"
        }
        
        description = category_descriptions.get(category, "project architecture")
        
        return f"""This question relates to {description}. To provide a comprehensive answer, 
I would need to analyze the project structure and documentation. 

The question falls under the {category.value} category, which typically involves:
- Understanding the current implementation patterns
- Analyzing the codebase structure  
- Identifying best practices and recommendations
- Providing specific examples from the project

Please ensure the project has been indexed and the Generic Q&A workflow is properly configured 
to get detailed, project-specific answers."""
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """
        Get list of available templates for different project types.
        
        Returns:
            List of template information dictionaries
        """
        templates = [
            {
                "id": "dotnet_clean_architecture",
                "name": ".NET Clean Architecture",
                "description": "Template for ASP.NET Core projects using Clean Architecture",
                "categories": [cat.value for cat in QuestionCategory],
                "file_patterns": ["*.cs", "*.csproj", "*.sln"]
            },
            {
                "id": "react_spa",
                "name": "React SPA",
                "description": "Template for React Single Page Applications",
                "categories": [QuestionCategory.ARCHITECTURE.value, QuestionCategory.WORKFLOWS.value],
                "file_patterns": ["*.js", "*.jsx", "*.ts", "*.tsx", "package.json"]
            },
            {
                "id": "python_fastapi",
                "name": "Python FastAPI",
                "description": "Template for Python FastAPI applications",
                "categories": [cat.value for cat in QuestionCategory],
                "file_patterns": ["*.py", "requirements.txt", "pyproject.toml"]
            }
        ]
        
        return templates
    
    async def analyze_project_structure(
        self, 
        repository_path: str,
        analysis_depth: str = "standard"
    ) -> Dict[str, Any]:
        """
        Analyze project structure for Generic Q&A preparation.
        
        Args:
            repository_path: Path to repository for analysis
            analysis_depth: Analysis depth (basic, standard, comprehensive)
            
        Returns:
            Project analysis results
        """
        try:
            self.logger.info(f"Analyzing project structure at: {repository_path}")
            
            # Import analyzers
            from src.analyzers.architecture_detector import ArchitectureDetector
            from src.analyzers.business_capability_analyzer import BusinessCapabilityAnalyzer
            
            analysis = {
                "repository_path": repository_path,
                "analysis_depth": analysis_depth,
                "detected_patterns": [],
                "supported_categories": [cat.value for cat in QuestionCategory],
                "recommended_templates": [],
                "analysis_timestamp": None,
                "readiness_score": 0.0,
                "architecture_analysis": None,
                "business_analysis": None
            }
            
            try:
                # Perform architecture analysis
                arch_detector = ArchitectureDetector()
                arch_analysis = arch_detector.detect_architecture(repository_path)
                
                analysis["architecture_analysis"] = {
                    "primary_pattern": arch_analysis.primary_pattern.value,
                    "secondary_patterns": [p.value for p in arch_analysis.secondary_patterns],
                    "confidence_score": arch_analysis.confidence_score,
                    "detected_layers": arch_analysis.detected_layers,
                    "technologies": arch_analysis.technologies
                }
                
                analysis["detected_patterns"].append(arch_analysis.primary_pattern.value)
                analysis["detected_patterns"].extend([p.value for p in arch_analysis.secondary_patterns])
                
                # Recommend templates based on architecture
                if arch_analysis.primary_pattern.value == "clean_architecture":
                    analysis["recommended_templates"].append("dotnet_clean_architecture")
                elif "python" in arch_analysis.technologies:
                    analysis["recommended_templates"].append("python_fastapi")
                elif any(tech in arch_analysis.technologies for tech in ["javascript", "react", "nodejs"]):
                    analysis["recommended_templates"].append("react_spa")
                else:
                    analysis["recommended_templates"].append("generic")
                
                self.logger.info(f"Architecture analysis completed: {arch_analysis.primary_pattern}")
                
            except Exception as e:
                self.logger.warning(f"Architecture analysis failed: {e}")
            
            try:
                # Perform business capability analysis if depth allows
                if analysis_depth in ["standard", "comprehensive"]:
                    business_analyzer = BusinessCapabilityAnalyzer()
                    business_analysis = business_analyzer.analyze_business_capability(repository_path)
                    
                    analysis["business_analysis"] = {
                        "primary_capability": business_analysis.primary_capability.name,
                        "domain_complexity": business_analysis.domain_complexity.value,
                        "bounded_contexts": business_analysis.bounded_contexts,
                        "confidence_score": business_analysis.confidence_score,
                        "entities_count": len(business_analysis.primary_capability.entities)
                    }
                    
                    self.logger.info(f"Business analysis completed: {business_analysis.primary_capability.name}")
                
            except Exception as e:
                self.logger.warning(f"Business capability analysis failed: {e}")
            
            # Calculate overall readiness score
            readiness_factors = []
            
            if analysis.get("architecture_analysis"):
                readiness_factors.append(analysis["architecture_analysis"]["confidence_score"])
            
            if analysis.get("business_analysis"):
                readiness_factors.append(analysis["business_analysis"]["confidence_score"])
            
            if readiness_factors:
                analysis["readiness_score"] = sum(readiness_factors) / len(readiness_factors)
            else:
                analysis["readiness_score"] = 0.1  # Minimal readiness
            
            analysis["analysis_timestamp"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "analysis": analysis,
                "agent": self.agent_name,
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing project structure: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "agent": self.agent_name,
            }