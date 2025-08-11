"""
Generic Q&A Agent for answering architectural and implementation questions.

This agent extends the BaseAgent architecture to provide structured responses
to common project questions using template-based generation and repository
analysis.

Follows established patterns from BaseAgent and integrates with the existing
LangGraph workflow infrastructure.
"""

from typing import Any, Dict, List, Optional, Union

from langchain.schema.runnable import RunnableConfig

from src.agents.base_agent import BaseAgent
from src.analyzers.question_classifier import QuestionClassifier, QuestionClassificationResult
from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class GenericQAAgent(BaseAgent):
    """
    Generic Q&A agent extending the proven BaseAgent architecture.
    
    This agent specializes in answering generic project questions about
    architecture, business capabilities, APIs, data models, and operations
    using template-based response generation.
    
    Attributes:
        workflow: GenericQAWorkflow instance for stateful processing
        classifier: QuestionClassifier for categorizing questions
        default_template: Default response template to use
        confidence_threshold: Minimum confidence for reliable classification
    
    Example:
        >>> agent = GenericQAAgent()
        >>> result = await agent.ainvoke({"question": "What is the business domain?"})
        >>> print(result["structured_response"])
    """

    def __init__(
        self,
        workflow: Optional[Any] = None,
        agent_name: str = "GenericQAAgent",
        default_template: str = "generic_template",
        confidence_threshold: float = 0.2,
        **kwargs,
    ):
        """
        Initialize Generic Q&A agent.
        
        Args:
            workflow: GenericQAWorkflow instance (will be created if not provided)
            agent_name: Name identifier for the agent
            default_template: Default template name for responses
            confidence_threshold: Minimum confidence for reliable classification
            **kwargs: Additional initialization parameters passed to BaseAgent
            
        Raises:
            ValueError: If required configuration is missing
            RuntimeError: If initialization components fail
        """
        # Import here to avoid circular imports
        from src.workflows.generic_qa_workflow import GenericQAWorkflow
        
        # Create workflow if not provided
        if workflow is None:
            workflow = GenericQAWorkflow()
            
        # REUSE BaseAgent initialization pattern
        super().__init__(
            workflow=workflow,
            agent_name=agent_name,
            **kwargs
        )
        
        # Initialize question classifier
        self.classifier = QuestionClassifier()
        self.default_template = default_template
        self.confidence_threshold = confidence_threshold
        
        # Override logger to include agent-specific context
        self.logger = get_logger(self.__class__.__name__)
        
        self.logger.info(f"Generic Q&A Agent initialized with template: {default_template}")

    async def _process_input(
        self, 
        input_data: Union[str, Dict[str, Any]], 
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Process input data and return structured response.
        
        Args:
            input_data: Input data to process (question string or dict with question)
            config: Optional configuration for processing
            
        Returns:
            Structured response dictionary with classification and analysis
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If processing encounters critical errors
        """
        try:
            self.logger.debug(f"Processing input: {type(input_data)}")
            
            # Extract question from input
            question = self._extract_question(input_data)
            if not question:
                return self._create_error_response("No question provided")
            
            self.logger.info(f"Processing generic question: {question[:100]}...")
            
            # Classify the question
            classification_result = await self.classifier.classify_question(question)
            
            # Check if classification is reliable
            if not self.classifier.is_reliable_classification(classification_result):
                self.logger.warning(f"Low confidence classification: {classification_result.confidence}")
                # Continue with low confidence but note it in response
            
            # Extract additional parameters from input
            repository_identifier = self._extract_repository_identifier(input_data)
            include_code_examples = self._extract_include_code_examples(input_data)
            preferred_template = self._extract_preferred_template(input_data)
            
            # Create workflow state
            workflow_state = self._create_workflow_state(
                question=question,
                classification=classification_result,
                repository_identifier=repository_identifier,
                include_code_examples=include_code_examples,
                preferred_template=preferred_template or self.default_template
            )
            
            # Execute workflow through BaseAgent workflow infrastructure
            self.logger.debug("Executing Generic Q&A workflow")
            result_state = self.workflow.invoke(workflow_state, config)
            
            # Convert workflow result to agent response format
            response = self._format_agent_response(result_state, classification_result)
            
            self.logger.info(f"Generic Q&A processing completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing generic Q&A input: {e}", exc_info=True)
            return self._create_error_response(f"Processing failed: {str(e)}")

    def _validate_input(self, input_data: Any) -> bool:
        """
        Validate input data format and content.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        try:
            # Input can be string (question) or dict with question
            if isinstance(input_data, str):
                return bool(input_data.strip())
            elif isinstance(input_data, dict):
                question = input_data.get("question", "")
                return bool(question and question.strip())
            else:
                self.logger.warning(f"Invalid input type: {type(input_data)}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating input: {e}")
            return False

    def _extract_question(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """
        Extract question from input data.
        
        Args:
            input_data: Input data (string or dict)
            
        Returns:
            Extracted question string
        """
        if isinstance(input_data, str):
            return input_data.strip()
        elif isinstance(input_data, dict):
            return input_data.get("question", "").strip()
        else:
            return ""

    def _extract_repository_identifier(self, input_data: Union[str, Dict[str, Any]]) -> Optional[str]:
        """
        Extract repository identifier from input data.
        
        Args:
            input_data: Input data
            
        Returns:
            Repository identifier if available
        """
        if isinstance(input_data, dict):
            return input_data.get("repository_identifier")
        return None

    def _extract_include_code_examples(self, input_data: Union[str, Dict[str, Any]]) -> bool:
        """
        Extract include_code_examples flag from input data.
        
        Args:
            input_data: Input data
            
        Returns:
            Whether to include code examples (default: True)
        """
        if isinstance(input_data, dict):
            return input_data.get("include_code_examples", True)
        return True

    def _extract_preferred_template(self, input_data: Union[str, Dict[str, Any]]) -> Optional[str]:
        """
        Extract preferred template from input data.
        
        Args:
            input_data: Input data
            
        Returns:
            Preferred template name if specified
        """
        if isinstance(input_data, dict):
            return input_data.get("preferred_template")
        return None

    def _create_workflow_state(
        self,
        question: str,
        classification: QuestionClassificationResult,
        repository_identifier: Optional[str] = None,
        include_code_examples: bool = True,
        preferred_template: str = "generic_template"
    ) -> Dict[str, Any]:
        """
        Create workflow state for processing.
        
        Args:
            question: User question
            classification: Question classification result
            repository_identifier: Repository to analyze
            include_code_examples: Whether to include code examples
            preferred_template: Template to use for response
            
        Returns:
            Workflow state dictionary
        """
        # Import here to avoid circular imports
        from src.workflows.workflow_states import WorkflowStatus
        
        return {
            "workflow_id": f"generic_qa_{hash(question)}",
            "status": WorkflowStatus.PENDING,
            "current_step": "initialize",
            "question": question,
            "question_category": classification.category.value,
            "classification_confidence": classification.confidence,
            "keywords_matched": classification.keywords_matched,
            "context_indicators": classification.context_indicators,
            "suggested_analyzers": classification.suggested_analyzers,
            "repository_identifier": repository_identifier,
            "include_code_examples": include_code_examples,
            "preferred_template": preferred_template,
            "repository_context": {},
            "analysis_results": {},
            "template_response": {},
            "confidence_score": 0.0,
            "sources": [],
            "errors": [],
            "metadata": {
                "agent_name": self.agent_name,
                "processing_start": None,
                "processing_end": None
            }
        }

    def _format_agent_response(
        self, 
        workflow_state: Dict[str, Any], 
        classification: QuestionClassificationResult
    ) -> Dict[str, Any]:
        """
        Format workflow result as agent response.
        
        Args:
            workflow_state: Final workflow state
            classification: Original classification result
            
        Returns:
            Formatted agent response
        """
        try:
            # Extract key results from workflow state
            structured_response = workflow_state.get("template_response", {})
            confidence_score = workflow_state.get("confidence_score", 0.0)
            sources = ensure_list(workflow_state.get("sources", []))
            errors = ensure_list(workflow_state.get("errors", []))
            
            # Calculate processing time if available
            metadata = workflow_state.get("metadata", {})
            processing_time_ms = 0
            if metadata.get("processing_start") and metadata.get("processing_end"):
                processing_time_ms = int(
                    (metadata["processing_end"] - metadata["processing_start"]) * 1000
                )
            
            # Create successful response
            response = {
                "success": True,
                "agent": self.agent_name,
                "question": workflow_state.get("question", ""),
                "question_category": classification.category.value,
                "structured_response": structured_response,
                "confidence_score": confidence_score,
                "sources": sources,
                "processing_time_ms": processing_time_ms,
                "template_used": workflow_state.get("preferred_template", self.default_template),
                "classification": {
                    "category": classification.category.value,
                    "confidence": classification.confidence,
                    "keywords_matched": classification.keywords_matched,
                    "context_indicators": classification.context_indicators,
                    "suggested_analyzers": classification.suggested_analyzers
                }
            }
            
            # Include errors if any occurred (but don't mark as failed)
            if errors:
                response["warnings"] = errors
                self.logger.warning(f"Workflow completed with {safe_len(errors)} warnings")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error formatting agent response: {e}", exc_info=True)
            return self._create_error_response(f"Response formatting failed: {str(e)}")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create standardized error response.
        
        Args:
            error_message: Error description
            
        Returns:
            Standardized error response
        """
        return {
            "success": False,
            "error": error_message,
            "agent": self.agent_name,
            "question": "",
            "question_category": "general",
            "structured_response": {},
            "confidence_score": 0.0,
            "sources": [],
            "processing_time_ms": 0,
            "template_used": self.default_template
        }

    async def get_supported_categories(self) -> List[str]:
        """
        Get list of supported question categories.
        
        Returns:
            List of supported category names
        """
        try:
            categories = self.classifier.get_supported_categories()
            return [category.value for category in categories]
        except Exception as e:
            self.logger.error(f"Error getting supported categories: {e}")
            return ["general"]

    async def get_category_description(self, category: str) -> str:
        """
        Get description for a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Category description
        """
        try:
            from src.analyzers.question_classifier import GenericQuestionCategory
            category_enum = GenericQuestionCategory(category)
            return self.classifier.get_category_description(category_enum)
        except Exception as e:
            self.logger.error(f"Error getting category description: {e}")
            return "Unknown category"

    async def analyze_question_only(self, question: str) -> QuestionClassificationResult:
        """
        Analyze and classify question without full processing.
        
        Args:
            question: Question to analyze
            
        Returns:
            Classification result
        """
        try:
            return await self.classifier.classify_question(question)
        except Exception as e:
            self.logger.error(f"Error analyzing question: {e}")
            from src.analyzers.question_classifier import GenericQuestionCategory
            return QuestionClassificationResult(
                category=GenericQuestionCategory.GENERAL,
                confidence=0.0,
                keywords_matched=[],
                context_indicators=[],
                suggested_analyzers=["general_analyzer"]
            )

    def get_agent_capabilities(self) -> Dict[str, Any]:
        """
        Get agent capabilities and configuration.
        
        Returns:
            Agent capabilities information
        """
        try:
            base_info = self.get_agent_info()
            
            # Add Generic Q&A specific capabilities
            capabilities = {
                **base_info,
                "supported_categories": [],  # Will be populated async
                "default_template": self.default_template,
                "confidence_threshold": self.confidence_threshold,
                "features": {
                    "question_classification": True,
                    "template_based_responses": True,
                    "repository_analysis": True,
                    "code_examples": True,
                    "multiple_analyzers": True
                }
            }
            
            return capabilities
            
        except Exception as e:
            self.logger.error(f"Error getting agent capabilities: {e}")
            return self.get_agent_info()

    async def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate agent configuration and dependencies.
        
        Returns:
            Validation results
        """
        validation_results = {
            "agent_valid": True,
            "workflow_valid": self.workflow is not None,
            "classifier_valid": self.classifier is not None,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Validate workflow
            if not self.workflow:
                validation_results["errors"].append("Workflow not initialized")
                validation_results["agent_valid"] = False
            
            # Validate classifier
            if not self.classifier:
                validation_results["errors"].append("Question classifier not initialized")
                validation_results["agent_valid"] = False
            else:
                # Test classifier with simple question
                test_result = await self.classifier.classify_question("test question")
                if not test_result:
                    validation_results["warnings"].append("Classifier test returned no result")
            
            # Check template configuration
            if not self.default_template:
                validation_results["warnings"].append("No default template configured")
            
            validation_results["overall_status"] = (
                "valid" if validation_results["agent_valid"] and not validation_results["errors"]
                else "invalid"
            )
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            validation_results["errors"].append(f"Validation error: {str(e)}")
            validation_results["agent_valid"] = False
            validation_results["overall_status"] = "invalid"
        
        return validation_results