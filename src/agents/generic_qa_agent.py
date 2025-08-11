"""
Generic Q&A Agent for structured project questions.

This module implements an AI agent that can answer structured questions about
project architecture, business capabilities, API endpoints, data modeling,
workflows, and operational concerns based on configurable templates.
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum

from langchain.schema.runnable import RunnableConfig
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.workflows.generic_qa_workflow import GenericQAWorkflow
from src.utils.defensive_programming import safe_len, ensure_list


class QuestionCategory(str, Enum):
    """Question categories for generic project Q&A."""
    
    BUSINESS_CAPABILITY = "business_capability"
    API_ENDPOINTS = "api_endpoints"
    DATA_MODELING = "data_modeling"
    WORKFLOWS = "workflows"
    ARCHITECTURE = "architecture"


class GenericQAAgent(BaseAgent):
    """
    Generic Q&A Agent for answering structured project questions.
    
    This agent extends BaseAgent with specialized functionality for processing
    generic project questions using template-based responses and project analysis.
    """

    def __init__(
        self,
        workflow: Optional[GenericQAWorkflow] = None,
        default_template: str = "python_fastapi",
        supported_categories: Optional[List[QuestionCategory]] = None,
        **kwargs,
    ):
        """
        Initialize Generic Q&A Agent.

        Args:
            workflow: GenericQAWorkflow instance for stateful processing
            default_template: Default project template to use
            supported_categories: List of supported question categories
            **kwargs: Additional initialization parameters
        """
        # Initialize workflow if not provided
        if workflow is None:
            workflow = GenericQAWorkflow()
        
        super().__init__(
            workflow=workflow,
            agent_name="GenericQAAgent",
            **kwargs
        )
        
        self.default_template = default_template
        self.supported_categories = supported_categories or [
            QuestionCategory.BUSINESS_CAPABILITY,
            QuestionCategory.API_ENDPOINTS,
            QuestionCategory.DATA_MODELING,
            QuestionCategory.WORKFLOWS,
            QuestionCategory.ARCHITECTURE,
        ]
        
        self.logger = logger.bind(agent="GenericQAAgent")
        self.logger.info(f"Initialized GenericQAAgent with template: {default_template}")

    async def _process_input(
        self, 
        input_data: Union[str, Dict[str, Any]], 
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Process input data and return structured response.

        Args:
            input_data: Input data containing question and context
            config: Optional configuration for processing

        Returns:
            Structured response dictionary with answer and metadata
        """
        try:
            # Parse input data
            if isinstance(input_data, str):
                question = input_data
                project_template = self.default_template
                category = None
            else:
                question = input_data.get("question", "")
                project_template = input_data.get("template", self.default_template)
                category = input_data.get("category")
            
            if not question.strip():
                return {
                    "success": False,
                    "error": "Question cannot be empty",
                    "agent": self.agent_name,
                }
            
            self.logger.info(f"Processing question: {question[:100]}...")
            
            # Create workflow state
            workflow_state = {
                "question": question,
                "project_template": project_template,
                "category": category,
                "supported_categories": [cat.value for cat in self.supported_categories],
                "agent_config": {
                    "default_template": self.default_template,
                    "agent_name": self.agent_name,
                }
            }
            
            # Execute workflow
            result_state = self.workflow.invoke(workflow_state)
            
            # Extract results
            answer = result_state.get("answer", "")
            detected_category = result_state.get("detected_category")
            project_analysis = result_state.get("project_analysis", {})
            confidence_score = result_state.get("confidence_score", 0.0)
            template_used = result_state.get("template_used", project_template)
            
            response = {
                "success": True,
                "question": question,
                "answer": answer,
                "category": detected_category,
                "template": template_used,
                "project_analysis": project_analysis,
                "confidence_score": confidence_score,
                "agent": self.agent_name,
                "metadata": {
                    "supported_categories": [cat.value for cat in self.supported_categories],
                    "processing_time": result_state.get("processing_time", 0),
                    "workflow_steps": result_state.get("executed_steps", []),
                }
            }
            
            self.logger.info(f"Successfully processed question with category: {detected_category}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to process question: {str(e)}",
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
            if not isinstance(question, str) or len(question.strip()) == 0:
                return False
            
            # Validate optional template
            template = input_data.get("template")
            if template is not None and not isinstance(template, str):
                return False
            
            # Validate optional category
            category = input_data.get("category")
            if category is not None:
                if isinstance(category, str):
                    try:
                        QuestionCategory(category)
                    except ValueError:
                        self.logger.warning(f"Unknown category: {category}")
                        return False
                else:
                    return False
            
            return True
        
        return False

    def get_supported_categories(self) -> List[str]:
        """
        Get list of supported question categories.

        Returns:
            List of supported category names
        """
        return [cat.value for cat in self.supported_categories]

    def get_available_templates(self) -> List[str]:
        """
        Get list of available project templates.

        Returns:
            List of available template names
        """
        # This will be populated by the workflow from the template configuration
        if hasattr(self.workflow, 'get_available_templates'):
            return self.workflow.get_available_templates()
        return ["python_fastapi", "dotnet_clean_architecture", "react_spa"]

    async def analyze_project_structure(
        self, 
        project_path: Optional[str] = None,
        repository_url: Optional[str] = None,
        template_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze project structure to determine architecture patterns.

        Args:
            project_path: Local path to project directory
            repository_url: URL to project repository
            template_hint: Hint about project template type

        Returns:
            Project analysis results
        """
        try:
            self.logger.info("Analyzing project structure")
            
            analysis_state = {
                "project_path": project_path,
                "repository_url": repository_url,
                "template_hint": template_hint,
                "analysis_type": "structure_analysis"
            }
            
            # Use workflow to perform analysis
            if hasattr(self.workflow, 'analyze_project'):
                result = await self.workflow.analyze_project(analysis_state)
                return result
            else:
                # Fallback implementation
                return {
                    "success": True,
                    "detected_template": template_hint or self.default_template,
                    "architecture_patterns": [],
                    "business_capabilities": [],
                    "api_endpoints": [],
                    "data_models": [],
                    "confidence": 0.5,
                    "message": "Basic analysis completed (full analysis requires workflow implementation)"
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing project structure: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to analyze project: {str(e)}"
            }

    def get_question_examples(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get example questions for supported categories.

        Args:
            category: Specific category to get examples for

        Returns:
            Dictionary of category -> example questions
        """
        examples = {
            QuestionCategory.BUSINESS_CAPABILITY.value: [
                "What business capability does this service own?",
                "What are the core entities in this domain?",
                "Who are the consumers of this service?",
                "What are the key business rules and invariants?"
            ],
            QuestionCategory.API_ENDPOINTS.value: [
                "What are the main API endpoints?",
                "How is pagination implemented?",
                "What status codes are returned?",
                "How is versioning handled?"
            ],
            QuestionCategory.DATA_MODELING.value: [
                "How is data persisted?",
                "What repositories are used?",
                "How is concurrency handled?",
                "What's the transaction strategy?"
            ],
            QuestionCategory.WORKFLOWS.value: [
                "What's the end-to-end workflow for creating entities?",
                "How are updates processed?",
                "What events are published?",
                "How are failures handled?"
            ],
            QuestionCategory.ARCHITECTURE.value: [
                "What's the overall architecture?",
                "How are layers organized?",
                "What security measures are in place?",
                "How is observability implemented?"
            ]
        }
        
        if category:
            return {category: examples.get(category, [])}
        return examples