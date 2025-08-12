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
from src.analyzers.llm_enhanced_analyzer import LLMEnhancedAnalyzer
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
        self._llm_analyzer = None  # LLM-enhanced analyzer
        
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
            Repository context information with actual code data
        """
        try:
            # Use vector store factory to get repository information
            vector_store = self.get_vector_store()
            
            # Search for repository-specific documents
            search_query = f"{repository_identifier} repository structure API endpoints"
            documents = vector_store.similarity_search(search_query, k=50)
            
            # Filter documents for the specific repository with stricter matching
            repo_docs = []
            for doc in documents:
                doc_repo = doc.metadata.get("repository", "").lower()
                doc_source = doc.metadata.get("source", "").lower()
                
                # Strict repository matching
                if (repository_identifier.lower() == doc_repo or 
                    repository_identifier.lower() in doc_repo or
                    f"/{repository_identifier.lower()}/" in doc_source or
                    doc_source.startswith(f"{repository_identifier.lower()}/")):
                    repo_docs.append(doc)
            
            self.logger.info(f"Found {len(repo_docs)} documents for repository '{repository_identifier}'")
            
            if not repo_docs:
                # Try search with just the repository name
                self.logger.warning(f"No specific documents found for {repository_identifier}, trying exact repository search")
                fallback_query = repository_identifier
                fallback_docs = vector_store.similarity_search(fallback_query, k=20)
                
                # Apply stricter filtering on fallback results
                for doc in fallback_docs:
                    doc_repo = doc.metadata.get("repository", "").lower()
                    if repository_identifier.lower() in doc_repo:
                        repo_docs.append(doc)
                
                if not repo_docs:
                    self.logger.error(f"No documents found for repository '{repository_identifier}'")
                    return {
                        "repository": repository_identifier,
                        "type": "not_found",
                        "error": f"No documents found for repository '{repository_identifier}'",
                        "message": f"The repository '{repository_identifier}' has not been indexed yet or does not exist in the vector database.",
                        "suggestion": f"Please ensure the repository '{repository_identifier}' is indexed before asking questions about it."
                    }
            
            # Analyze documents to extract repository context
            context = {
                "repository": repository_identifier,
                "type": "analyzed",
                "timestamp": datetime.now().isoformat(),
                "document_count": len(repo_docs),
                "source_files": [],
                "languages": set(),
                "frameworks": set(),
                "api_endpoints": [],
                "architecture_patterns": set(),
                "key_components": [],
                "readme_content": None,
                "raw_documents": repo_docs  # Store for further analysis
            }
            
            # Extract information from documents
            for doc in repo_docs:
                source = doc.metadata.get("source", "")
                repository = doc.metadata.get("repository", "")
                content = doc.page_content
                
                # Log document information for debugging
                self.logger.debug(f"Processing document: source='{source}', repository='{repository}'")
                
                # Track source files
                context["source_files"].append({
                    "file": source,
                    "repository": repository,
                    "size": len(content)
                })
                
                # Detect languages from file extensions
                if source.endswith(('.cs', '.csproj')):
                    context["languages"].add("C#/.NET")
                elif source.endswith(('.js', '.jsx', '.ts', '.tsx')):
                    context["languages"].add("JavaScript/TypeScript")
                elif source.endswith(('.py', '.ipynb')):
                    context["languages"].add("Python")
                elif source.endswith(('.java', '.kt')):
                    context["languages"].add("Java/Kotlin")
                
                # Detect frameworks from content
                if any(keyword in content.lower() for keyword in ['asp.net', 'webapi', 'controller']):
                    context["frameworks"].add("ASP.NET Web API")
                if any(keyword in content.lower() for keyword in ['react', 'jsx', 'usestate']):
                    context["frameworks"].add("React")
                if any(keyword in content.lower() for keyword in ['fastapi', 'router', 'endpoint']):
                    context["frameworks"].add("FastAPI")
                if any(keyword in content.lower() for keyword in ['mongodb', 'mongoclient']):
                    context["frameworks"].add("MongoDB")
                if any(keyword in content.lower() for keyword in ['rabbitmq', 'message', 'queue']):
                    context["frameworks"].add("RabbitMQ")
                
                # Extract API endpoints
                if "controller" in source.lower() or "routes" in source.lower() or "endpoint" in content.lower():
                    endpoints = self._extract_api_endpoints_from_content(content, source)
                    context["api_endpoints"].extend(endpoints)
                
                # Store README content for overview
                if source.lower().endswith('readme.md'):
                    context["readme_content"] = content
                
                # Detect architecture patterns
                if any(pattern in content.lower() for pattern in ['microservice', 'service', 'api']):
                    context["architecture_patterns"].add("Microservices")
                if any(pattern in content.lower() for pattern in ['mvc', 'controller', 'model']):
                    context["architecture_patterns"].add("MVC")
                if any(pattern in content.lower() for pattern in ['repository', 'interface', 'service']):
                    context["architecture_patterns"].add("Repository Pattern")
            
            # Convert sets to lists for JSON serialization
            context["languages"] = list(context["languages"])
            context["frameworks"] = list(context["frameworks"])
            context["architecture_patterns"] = list(context["architecture_patterns"])
            
            self.logger.debug(f"Retrieved enriched context for repository: {repository_identifier} "
                            f"({len(repo_docs)} documents, {len(context['api_endpoints'])} endpoints)")
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
        """Analyze API endpoints using real repository data."""
        
        # Get API endpoints from repository context
        api_endpoints = context.get("api_endpoints", [])
        repository = context.get("repository", "unknown")
        frameworks = context.get("frameworks", [])
        readme_content = context.get("readme_content", "")
        
        # Base analysis structure
        analysis = {
            "repository": repository,
            "total_endpoints": len(api_endpoints),
            "frameworks_detected": frameworks,
            "endpoints": api_endpoints
        }
        
        if api_endpoints:
            # Group endpoints by method
            methods = {}
            paths = []
            source_files = set()
            
            for endpoint in api_endpoints:
                method = endpoint.get("method", "UNKNOWN")
                path = endpoint.get("path", "")
                source = endpoint.get("source_file", "")
                
                if method not in methods:
                    methods[method] = []
                methods[method].append(endpoint)
                
                if path:
                    paths.append(path)
                if source:
                    source_files.add(source)
            
            analysis.update({
                "methods_summary": {method: len(endpoints) for method, endpoints in methods.items()},
                "unique_paths": len(set(paths)),
                "source_files": list(source_files),
                "api_patterns": self._detect_api_patterns(api_endpoints),
                "detailed_endpoints": self._format_detailed_endpoints(api_endpoints)
            })
            
            # Add authentication info if detected
            auth_info = self._detect_authentication_patterns(context)
            if auth_info:
                analysis["authentication"] = auth_info
            
            # Extract business domain from paths
            business_domain = self._extract_business_domain(paths, repository)
            if business_domain:
                analysis["business_domain"] = business_domain
                
        else:
            # No endpoints found - provide analysis based on available data
            analysis.update({
                "status": "No API endpoints detected",
                "possible_reasons": [
                    "Repository may not contain REST API code",
                    "API endpoints may be defined in files not yet indexed",
                    "Framework may use different endpoint definition patterns"
                ],
                "frameworks_detected": frameworks,
                "suggestions": [
                    "Check for controller files, route definitions, or API documentation",
                    "Look for framework-specific endpoint patterns",
                    "Review README or documentation files for API specifications"
                ]
            })
            
            # Try to extract info from README if available
            if readme_content:
                readme_endpoints = self._extract_readme_api_info(readme_content)
                if readme_endpoints:
                    analysis["readme_endpoints"] = readme_endpoints
        
        return analysis

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
            
            # Format the structured response into human-readable text
            formatted_response = self._format_structured_response(template_response, state)
            
            state["template_response"] = formatted_response
            
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

    def _format_structured_response(self, template_response: Dict[str, Any], state: Dict[str, Any]) -> str:
        """
        Format structured template response into human-readable text.
        
        Args:
            template_response: Structured response from template engine
            state: Current workflow state
            
        Returns:
            Human-readable formatted response
        """
        try:
            category = template_response.get("category", "general")
            
            # Get sections from template response
            sections = template_response.get("sections", {})
            
            if category == "api_endpoints":
                return self._format_api_endpoints_response(sections)
            elif category == "business_capability":
                return self._format_business_capability_response(sections)
            elif category == "architecture":
                return self._format_architecture_response(sections)
            else:
                return self._format_general_response(sections)
            
        except Exception as e:
            self.logger.error(f"Error formatting structured response: {e}")
            return "I was able to analyze the project but encountered an issue formatting the response. The analysis found relevant information but the display formatting failed. Please try rephrasing your question."

    def _format_api_endpoints_response(self, sections: Dict[str, Any]) -> str:
        """Format API endpoints response in a conversational way."""
        lines = []
        
        # Overview section
        overview = sections.get("overview", {})
        if overview:
            repo_name = overview.get('repository', 'this project')
            total_endpoints = overview.get('total_endpoints', 0)
            business_domain = overview.get('business_domain', 'Unknown')
            
            lines.append(f"The {repo_name} project has {total_endpoints} API endpoints and belongs to the {business_domain} domain.")
            
            frameworks = overview.get('frameworks', [])
            if frameworks:
                framework_list = ', '.join(frameworks[:-1]) + f" and {frameworks[-1]}" if len(frameworks) > 1 else frameworks[0]
                lines.append(f"It's built using {framework_list}.")
            
            api_patterns = overview.get('api_patterns', [])
            if api_patterns:
                pattern_list = ', '.join(api_patterns)
                lines.append(f"The API follows {pattern_list} design patterns.")
            lines.append("")
        
        # Endpoints section organized by method
        endpoints = sections.get("endpoints", [])
        if endpoints:
            lines.append("Here are the available endpoints organized by HTTP method:")
            lines.append("")
            
            # Group by method
            method_groups = {}
            for endpoint in endpoints:
                method = endpoint.get("method", "UNKNOWN")
                if method not in method_groups:
                    method_groups[method] = []
                method_groups[method].append(endpoint)
            
            for method, method_endpoints in method_groups.items():
                lines.append(f"{method} Endpoints:")
                
                # Remove duplicates by path
                seen_paths = set()
                unique_endpoints = []
                for endpoint in method_endpoints:
                    path = endpoint.get("path", "")
                    if path not in seen_paths:
                        seen_paths.add(path)
                        unique_endpoints.append(endpoint)
                
                for endpoint in unique_endpoints:
                    path = endpoint.get("path", "")
                    description = endpoint.get("description", "")
                    if path and description:
                        lines.append(f"  • {path} - {description}")
                lines.append("")
        
        # Methods summary
        methods_summary = sections.get("methods_summary", {})
        if methods_summary:
            summary_parts = []
            for method, count in methods_summary.items():
                summary_parts.append(f"{count} {method}")
            lines.append(f"Summary: {', '.join(summary_parts)} endpoints total.")
            lines.append("")
        
        # Authentication
        auth = sections.get("authentication", {})
        if auth:
            auth_methods = auth.get("methods", [])
            if auth_methods:
                auth_list = ', '.join(auth_methods)
                lines.append(f"Authentication: The API uses {auth_list} for security.")
                
                examples = sections.get("examples", {})
                auth_example = examples.get("authentication_example", "")
                if auth_example:
                    lines.append(f"Example: {auth_example.replace('# or', 'or').strip()}")
                lines.append("")
        
        # Request/Response Examples
        examples = sections.get("examples", {})
        if examples:
            request_example = examples.get("request_example", "")
            response_example = examples.get("response_example", "")
            
            if request_example or response_example:
                lines.append("Example usage:")
                if request_example:
                    lines.append(f"Request: {request_example.strip()}")
                if response_example:
                    # Clean up the JSON example
                    clean_response = response_example.replace('\n', ' ').replace('  ', ' ').strip()
                    lines.append(f"Response: {clean_response}")
        
        return "\n".join(lines)

    def _format_business_capability_response(self, sections: Dict[str, Any]) -> str:
        """Format business capability response in a conversational way."""
        lines = []
        
        # Overview section
        overview = sections.get("overview", {})
        if overview:
            status = overview.get('status')
            
            # Handle repository not found case
            if status == "Repository not found in vector database":
                repository = overview.get('repository', 'Unknown Repository')
                message = overview.get('message', 'Repository not found')
                suggestion = overview.get('suggestion', 'Please index the repository first')
                
                lines.append(f"I cannot provide business capability information for '{repository}' because it has not been indexed in the vector database.")
                lines.append("")
                lines.append(message)
                lines.append("")
                lines.append(f"Suggestion: {suggestion}")
                return "\n".join(lines)
            
            # Handle no documents found case
            if status == "No documents found":
                repository = overview.get('repository', 'Unknown Repository')
                message = overview.get('message', 'No documents found')
                suggestion = overview.get('suggestion', 'Repository may need to be indexed')
                
                lines.append(f"I searched for business capability information about '{repository}' but found no relevant documents.")
                lines.append("")
                lines.append(message)
                lines.append("")
                lines.append(f"Suggestion: {suggestion}")
                return "\n".join(lines)
            
            # Normal case with actual repository data
            repository = overview.get('repository', 'this project')
            business_domain = overview.get('business_domain', 'Unknown Domain')
            core_purpose = overview.get('core_purpose', 'Not specified')
            scope = overview.get('scope', 'Not specified')
            
            lines.append(f"Based on my analysis of the '{repository}' repository, here are the business capabilities:")
            lines.append("")
            lines.append(f"Business Domain: {business_domain}")
            lines.append(f"Core Purpose: {core_purpose}")
            lines.append(f"Scope: {scope}")
            lines.append("")
        
        # Core capabilities
        capabilities = sections.get("core_capabilities", [])
        if capabilities:
            lines.append("Main capabilities include:")
            for capability in capabilities:
                lines.append(f"  • {capability}")
            lines.append("")
        
        # Business entities and other info
        entities = sections.get("business_entities", [])
        if entities:
            entity_list = ', '.join(entities)
            lines.append(f"Key business entities: {entity_list}")
        
        # Value proposition
        value_prop = sections.get("value_proposition", "")
        if value_prop:
            lines.append(f"Value proposition: {value_prop}")
        
        # Target users
        target_users = sections.get("target_users", [])
        if target_users:
            user_list = ', '.join(target_users)
            lines.append(f"Target users: {user_list}")
        
        # Key processes
        processes = sections.get("key_processes", [])
        if processes and processes != capabilities:  # Avoid duplication
            lines.append("")
            lines.append("Key processes:")
            for process in processes:
                lines.append(f"  • {process}")
        
        # Analysis summary
        analysis_summary = sections.get("analysis_summary", {})
        if analysis_summary:
            docs_analyzed = analysis_summary.get("documents_analyzed", 0)
            confidence = analysis_summary.get("analysis_confidence", "Unknown")
            
            lines.append("")
            lines.append(f"Analysis Summary: {docs_analyzed} documents analyzed, confidence level: {confidence}")
        
        return "\n".join(lines)

    def _format_architecture_response(self, sections: Dict[str, Any]) -> str:
        """Format architecture response in a conversational way."""
        lines = []
        
        overview = sections.get("overview", {})
        if overview:
            lines.append("Architecture overview:")
            for key, value in overview.items():
                readable_key = key.replace('_', ' ').title()
                if isinstance(value, list):
                    value_str = ', '.join(value)
                    lines.append(f"  • {readable_key}: {value_str}")
                else:
                    lines.append(f"  • {readable_key}: {value}")
        
        return "\n".join(lines)

    def _format_general_response(self, sections: Dict[str, Any]) -> str:
        """Format general response for unknown categories in a conversational way."""
        lines = []
        
        # Check if this is a repository not found case
        overview = sections.get("overview", {})
        if overview:
            status = overview.get('status')
            
            # Handle repository not found case
            if status == "Repository not found in vector database":
                repository = overview.get('repository', 'Unknown Repository')
                message = overview.get('message', 'Repository not found')
                suggestion = overview.get('suggestion', 'Please index the repository first')
                
                lines.append(f"I cannot provide information about '{repository}' because it has not been indexed in the vector database.")
                lines.append("")
                lines.append(message)
                lines.append("")
                lines.append(f"Suggestion: {suggestion}")
                return "\n".join(lines)
            
            # Handle no documents found case
            if status == "No documents found":
                repository = overview.get('repository', 'Unknown Repository')
                message = overview.get('message', 'No documents found')
                suggestion = overview.get('suggestion', 'Repository may need to be indexed')
                
                lines.append(f"I searched for information about '{repository}' but found no relevant documents.")
                lines.append("")
                lines.append(message)
                lines.append("")
                lines.append(f"Suggestion: {suggestion}")
                return "\n".join(lines)
        
        # Normal case - format analysis results
        lines.append("Based on my analysis of the repository, here's what I found:")
        lines.append("")
        
        for section_name, section_content in sections.items():
            readable_name = section_name.replace('_', ' ').title()
            
            if isinstance(section_content, dict):
                lines.append(f"{readable_name}:")
                for key, value in section_content.items():
                    readable_key = key.replace('_', ' ').title()
                    if isinstance(value, list) and value:
                        value_str = ', '.join(str(v) for v in value)
                        lines.append(f"  • {readable_key}: {value_str}")
                    elif value:
                        lines.append(f"  • {readable_key}: {value}")
            elif isinstance(section_content, list) and section_content:
                lines.append(f"{readable_name}:")
                for item in section_content:
                    lines.append(f"  • {item}")
            elif section_content:
                lines.append(f"{readable_name}: {section_content}")
            
            lines.append("")
        
        return "\n".join(lines)

    def _format_metadata_section(self, metadata: Dict[str, Any]) -> str:
        """Format metadata section."""
        lines = ["## Analysis Metadata"]
        
        confidence = metadata.get("analysis_confidence", 0)
        lines.append(f"**Analysis Confidence:** {confidence:.1%}")
        
        repo_type = metadata.get("repository_type", "unknown")
        lines.append(f"**Repository Type:** {repo_type}")
        
        template_version = metadata.get("template_version", "1.0")
        lines.append(f"**Template Version:** {template_version}")
        
        return "\n".join(lines)

    def _extract_sources(self, state: Dict[str, Any]) -> List[str]:
        """
        Extract source information from analysis results.
        
        Args:
            state: Current workflow state
            
        Returns:
            List of source references as strings
        """
        sources = []
        
        # Add repository as source if available
        repository_context = state.get("repository_context", {})
        if repository_context.get("repository"):
            sources.append(f"Repository: {repository_context['repository']}")
        
        # Add analysis results as sources
        analysis_results = state.get("analysis_results", {})
        if analysis_results and not analysis_results.get("error"):
            category = state.get("question_category", "general")
            sources.append(f"Analysis: {category} category")
        
        # Add vector store documents as sources
        if analysis_results.get("documents_analyzed"):
            doc_count = analysis_results.get("documents_analyzed", 0)
            sources.append(f"Vector Store: {doc_count} documents analyzed")
        
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

    def _extract_api_endpoints_from_content(self, content: str, source: str) -> List[Dict[str, str]]:
        """
        Extract API endpoints from source code content.
        
        Args:
            content: Source code content
            source: Source file name
            
        Returns:
            List of endpoint dictionaries
        """
        endpoints = []
        
        try:
            # Extract from ASP.NET Controller patterns
            if source.endswith('.cs') and 'controller' in source.lower():
                # Look for HTTP method attributes like [HttpGet], [HttpPost], etc.
                import re
                
                # Pattern for HTTP method attributes with optional route
                http_patterns = [
                    r'\[Http(Get|Post|Put|Delete|Patch)\s*(?:\("([^"]+)"\))?\]',
                    r'\[Route\s*\("([^"]+)"\)\].*?\[Http(Get|Post|Put|Delete|Patch)\]',
                    r'public.*?(Get|Post|Put|Delete|Patch).*?\((.*?)\)',
                ]
                
                for pattern in http_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        method = match.group(1) if match.group(1) else match.group(2)
                        route = match.group(2) if len(match.groups()) > 1 and match.group(2) else ""
                        
                        if method:
                            endpoints.append({
                                "method": method.upper(),
                                "path": route or f"/{source.replace('Controller.cs', '').replace('Controllers/', '')}",
                                "source_file": source,
                                "description": f"{method.upper()} endpoint in {source}"
                            })
            
            # Extract from README.md API documentation
            elif source.lower().endswith('readme.md'):
                import re
                lines = content.split('\n')
                
                for line in lines:
                    # Look for API endpoint patterns like: - `GET /Car` — Description
                    endpoint_match = re.match(r'[-*]\s*`?(GET|POST|PUT|DELETE|PATCH)\s+([^\s`]+)`?\s*[—-]\s*(.+)', line.strip())
                    if endpoint_match:
                        method, path, description = endpoint_match.groups()
                        endpoints.append({
                            "method": method.upper(),
                            "path": path,
                            "source_file": source,
                            "description": description.strip()
                        })
            
            # Extract from JavaScript/TypeScript API calls
            elif source.endswith(('.js', '.jsx', '.ts', '.tsx')):
                import re
                # Look for fetch calls, axios calls, etc.
                api_call_patterns = [
                    r'fetch\s*\(\s*[\'"`]([^\'"`]+)[\'"`]',
                    r'axios\.(get|post|put|delete|patch)\s*\(\s*[\'"`]([^\'"`]+)[\'"`]',
                    r'(GET|POST|PUT|DELETE|PATCH)\s+[\'"`]([^\'"`]+)[\'"`]'
                ]
                
                for pattern in api_call_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if len(match.groups()) == 2:
                            method, path = match.groups()
                            endpoints.append({
                                "method": method.upper(),
                                "path": path,
                                "source_file": source,
                                "description": f"API call from {source}"
                            })
                        else:
                            path = match.group(1)
                            endpoints.append({
                                "method": "GET",  # Default assumption
                                "path": path,
                                "source_file": source,
                                "description": f"API call from {source}"
                            })
        
        except Exception as e:
            self.logger.warning(f"Error extracting endpoints from {source}: {e}")
        
        return endpoints

    def _detect_api_patterns(self, endpoints: List[Dict[str, str]]) -> List[str]:
        """Detect API design patterns from endpoints."""
        patterns = []
        
        if not endpoints:
            return patterns
            
        # Check for RESTful patterns
        methods = [ep.get("method", "") for ep in endpoints]
        if any(method in methods for method in ["GET", "POST", "PUT", "DELETE"]):
            patterns.append("RESTful API")
        
        # Check for CRUD patterns
        paths = [ep.get("path", "") for ep in endpoints]
        if any("/{id}" in path for path in paths):
            patterns.append("CRUD Operations")
        
        # Check for versioning
        if any("/v1/" in path or "/api/" in path for path in paths):
            patterns.append("API Versioning")
            
        return patterns

    def _format_detailed_endpoints(self, endpoints: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format endpoints with enhanced details."""
        detailed = []
        
        for endpoint in endpoints:
            detailed_endpoint = {
                "method": endpoint.get("method", "UNKNOWN"),
                "path": endpoint.get("path", ""),
                "description": endpoint.get("description", ""),
                "source_file": endpoint.get("source_file", ""),
                "expected_behavior": self._infer_endpoint_behavior(endpoint)
            }
            detailed.append(detailed_endpoint)
            
        return detailed

    def _infer_endpoint_behavior(self, endpoint: Dict[str, str]) -> str:
        """Infer expected behavior from endpoint details."""
        method = endpoint.get("method", "").upper()
        path = endpoint.get("path", "")
        
        if method == "GET":
            if "/{id}" in path:
                return "Retrieve a specific resource by ID"
            else:
                return "Retrieve a list of resources"
        elif method == "POST":
            return "Create a new resource"
        elif method == "PUT":
            return "Update or replace an existing resource"
        elif method == "DELETE":
            return "Delete a resource"
        elif method == "PATCH":
            return "Partially update a resource"
        else:
            return "Endpoint operation not determined"

    def _detect_authentication_patterns(self, context: Dict) -> Dict[str, Any]:
        """Detect authentication patterns from repository context."""
        auth_info = {}
        
        raw_docs = context.get("raw_documents", [])
        for doc in raw_docs:
            content = doc.page_content.lower()
            
            if any(keyword in content for keyword in ["api key", "apikey", "authorization"]):
                auth_info["api_key"] = True
            if any(keyword in content for keyword in ["jwt", "token", "bearer"]):
                auth_info["jwt_token"] = True
            if any(keyword in content for keyword in ["oauth", "openid"]):
                auth_info["oauth"] = True
            if any(keyword in content for keyword in ["basic auth", "username", "password"]):
                auth_info["basic_auth"] = True
                
        return auth_info

    def _extract_business_domain(self, paths: List[str], repository: str) -> str:
        """Extract business domain from API paths and repository name."""
        # Extract from repository name
        if "car" in repository.lower():
            return "Automotive/Car Marketplace"
        elif "order" in repository.lower():
            return "E-commerce/Order Management"
        elif "user" in repository.lower():
            return "User Management"
        elif "notification" in repository.lower():
            return "Communication/Notification"
            
        # Extract from API paths
        path_keywords = set()
        for path in paths:
            path_parts = path.lower().split('/')
            path_keywords.update(part for part in path_parts if part and part != "api" and part != "v1")
        
        if "car" in path_keywords:
            return "Automotive/Car Management"
        elif "order" in path_keywords:
            return "Order Processing"
        elif "user" in path_keywords:
            return "User Management"
        elif "product" in path_keywords:
            return "Product Management"
        else:
            return "General Business Operations"

    def _extract_readme_api_info(self, readme_content: str) -> List[Dict[str, str]]:
        """Extract API information from README content."""
        endpoints = []
        
        import re
        lines = readme_content.split('\n')
        
        for line in lines:
            # Look for API endpoint patterns in README
            endpoint_match = re.match(r'[-*]\s*`?(GET|POST|PUT|DELETE|PATCH)\s+([^\s`]+)`?\s*[—-]\s*(.+)', line.strip())
            if endpoint_match:
                method, path, description = endpoint_match.groups()
                endpoints.append({
                    "method": method.upper(),
                    "path": path,
                    "description": description.strip(),
                    "source": "README documentation"
                })
        
        return endpoints