"""
Question Classification Module for Generic Q&A Agent.

This module extends the existing EventFlowAnalyzer pattern to detect and classify
generic project questions such as business capabilities, architecture decisions,
API endpoints, data modeling, and operational patterns.

Follows the established analyzer patterns from event_flow_analyzer.py.
"""

from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from src.analyzers.event_flow_analyzer import EventFlowAnalyzer
from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class GenericQuestionCategory(str, Enum):
    """Extends existing QueryIntent with generic Q&A categories."""
    BUSINESS_CAPABILITY = "business_capability"
    ARCHITECTURE = "architecture"
    API_ENDPOINTS = "api_endpoints"
    DATA_MODELING = "data_modeling"
    OPERATIONAL = "operational"
    WORKFLOWS = "workflows"
    GENERAL = "general"


@dataclass
class QuestionClassificationResult:
    """
    Result of question classification analysis.
    Similar to EventFlowQuery dataclass in event_flow_analyzer.py.
    """
    category: GenericQuestionCategory
    confidence: float
    keywords_matched: List[str]
    context_indicators: List[str]
    suggested_analyzers: List[str]


class QuestionClassifier:
    """
    Question classifier extending EventFlowAnalyzer pattern detection methodology.
    
    This classifier reuses pattern detection methodology from EventFlowAnalyzer
    to identify generic project questions and route them to appropriate analyzers.
    """

    def __init__(self):
        """Initialize question classifier with reused patterns."""
        self.logger = get_logger(self.__class__.__name__)  # REUSE logging pattern
        self.event_analyzer = EventFlowAnalyzer()  # REUSE existing analyzer
        
        # REUSE pattern detection approach from event_flow_analyzer.py
        self._question_patterns = self._initialize_question_patterns()
        self._context_indicators = self._initialize_context_indicators()

    def _initialize_question_patterns(self) -> Dict[GenericQuestionCategory, List[str]]:
        """
        Initialize question patterns following EventFlowAnalyzer._workflow_patterns approach.
        
        Returns:
            Dictionary mapping categories to keyword patterns
        """
        return {
            GenericQuestionCategory.BUSINESS_CAPABILITY: [
                "business", "domain", "entities", "scope", "capability",
                "requirements", "functional", "use case", "user story",
                "business logic", "business rules", "domain model",
                "purpose", "function", "role", "responsibility", "job",
                "designed for", "used for", "meant for", "what does",
                "what is", "main function", "primary purpose", "core purpose",
                "business purpose", "business value", "value proposition"
            ],
            GenericQuestionCategory.ARCHITECTURE: [
                "architecture", "design", "pattern", "structure", "component",
                "layer", "module", "service", "microservice", "monolith",
                "clean architecture", "mvc", "mvvm", "hexagonal", "onion"
            ],
            GenericQuestionCategory.API_ENDPOINTS: [
                "api", "endpoint", "route", "controller", "rest", "graphql",
                "http", "request", "response", "method", "post", "get",
                "put", "delete", "patch", "webhook", "swagger", "openapi"
            ],
            GenericQuestionCategory.DATA_MODELING: [
                "data", "model", "entity", "schema", "database", "table",
                "field", "relationship", "foreign key", "primary key",
                "migration", "orm", "repository", "dto", "viewmodel"
            ],
            GenericQuestionCategory.OPERATIONAL: [
                "deployment", "infrastructure", "docker", "kubernetes",
                "ci/cd", "pipeline", "environment", "configuration",
                "monitoring", "logging", "metrics", "security", "authentication"
            ],
            GenericQuestionCategory.WORKFLOWS: [
                "workflow", "process", "flow", "step", "pipeline",
                "automation", "task", "job", "schedule", "trigger",
                "state machine", "orchestration", "choreography"
            ]
        }

    def _initialize_context_indicators(self) -> Dict[GenericQuestionCategory, List[str]]:
        """
        Initialize context indicators for enhanced classification.
        
        Returns:
            Dictionary mapping categories to context indicator phrases
        """
        return {
            GenericQuestionCategory.BUSINESS_CAPABILITY: [
                "what does this system do", "business purpose", "core functionality",
                "main features", "business value", "user benefits", "problem solving"
            ],
            GenericQuestionCategory.ARCHITECTURE: [
                "how is structured", "system design", "architectural decisions",
                "technology stack", "design patterns", "system components"
            ],
            GenericQuestionCategory.API_ENDPOINTS: [
                "available endpoints", "api documentation", "how to call",
                "request format", "response format", "api methods"
            ],
            GenericQuestionCategory.DATA_MODELING: [
                "data structure", "database schema", "entity relationships",
                "data flow", "persistence layer", "data access"
            ],
            GenericQuestionCategory.OPERATIONAL: [
                "how to deploy", "infrastructure setup", "environment config",
                "monitoring setup", "security implementation", "operational procedures"
            ],
            GenericQuestionCategory.WORKFLOWS: [
                "process flow", "execution steps", "automation flow",
                "task sequence", "workflow definition", "process automation"
            ]
        }

    async def classify_question(self, question: str) -> QuestionClassificationResult:
        """
        Classify question using pattern matching methodology from EventFlowAnalyzer.
        
        Args:
            question: User question text to classify
            
        Returns:
            Classification result with category and confidence
        """
        if not question or not question.strip():
            self.logger.warning("Empty question provided for classification")
            return QuestionClassificationResult(
                category=GenericQuestionCategory.GENERAL,
                confidence=0.0,
                keywords_matched=[],
                context_indicators=[],
                suggested_analyzers=[]
            )

        try:
            question_lower = question.lower().strip()
            self.logger.debug(f"Classifying question: {question[:100]}...")

            # Score each category using pattern matching
            category_scores = {}
            all_matched_keywords = {}
            all_matched_contexts = {}

            for category, patterns in self._question_patterns.items():
                # Count keyword matches (similar to EventFlowAnalyzer pattern matching)
                matched_keywords = [
                    keyword for keyword in patterns 
                    if keyword.lower() in question_lower
                ]
                keyword_score = safe_len(matched_keywords) / max(safe_len(patterns), 1)

                # Count context indicator matches
                context_patterns = self._context_indicators.get(category, [])
                matched_contexts = [
                    context for context in context_patterns
                    if context.lower() in question_lower
                ]
                context_score = safe_len(matched_contexts) / max(safe_len(context_patterns), 1)

                # Combined score with keyword emphasis
                combined_score = (keyword_score * 0.7) + (context_score * 0.3)
                
                category_scores[category] = combined_score
                all_matched_keywords[category] = matched_keywords
                all_matched_contexts[category] = matched_contexts

                self.logger.debug(f"Category {category.value}: keyword_score={keyword_score:.2f}, "
                                f"context_score={context_score:.2f}, combined={combined_score:.2f}")

            # Find best matching category
            if not category_scores or max(category_scores.values()) == 0:
                self.logger.info("No specific patterns matched, defaulting to GENERAL category")
                return QuestionClassificationResult(
                    category=GenericQuestionCategory.GENERAL,
                    confidence=0.1,
                    keywords_matched=[],
                    context_indicators=[],
                    suggested_analyzers=["general_analyzer"]
                )

            best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            best_score = category_scores[best_category]
            
            # Get suggested analyzers based on category
            suggested_analyzers = self._get_suggested_analyzers(best_category)

            result = QuestionClassificationResult(
                category=best_category,
                confidence=min(best_score, 1.0),  # Cap at 1.0
                keywords_matched=all_matched_keywords[best_category],
                context_indicators=all_matched_contexts[best_category],
                suggested_analyzers=suggested_analyzers
            )

            self.logger.info(f"Question classified as {best_category.value} with confidence {best_score:.2f}")
            return result

        except Exception as e:
            self.logger.error(f"Error classifying question: {e}", exc_info=True)
            # Return safe fallback
            return QuestionClassificationResult(
                category=GenericQuestionCategory.GENERAL,
                confidence=0.0,
                keywords_matched=[],
                context_indicators=[],
                suggested_analyzers=["general_analyzer"]
            )

    def _get_suggested_analyzers(self, category: GenericQuestionCategory) -> List[str]:
        """
        Get suggested analyzer modules for a given category.
        
        Args:
            category: Question category
            
        Returns:
            List of suggested analyzer module names
        """
        analyzer_mapping = {
            GenericQuestionCategory.BUSINESS_CAPABILITY: ["business_capability_analyzer"],
            GenericQuestionCategory.ARCHITECTURE: ["architecture_detector"],
            GenericQuestionCategory.API_ENDPOINTS: ["api_endpoint_analyzer"],
            GenericQuestionCategory.DATA_MODELING: ["data_model_analyzer"],
            GenericQuestionCategory.OPERATIONAL: ["operational_analyzer"],
            GenericQuestionCategory.WORKFLOWS: ["workflow_analyzer"],
            GenericQuestionCategory.GENERAL: ["general_analyzer"]
        }
        
        return analyzer_mapping.get(category, ["general_analyzer"])

    def get_classification_confidence_threshold(self) -> float:
        """
        Get minimum confidence threshold for reliable classification.
        
        Returns:
            Minimum confidence threshold (0.0 to 1.0)
        """
        return 0.2  # Minimum confidence for reliable classification

    def is_reliable_classification(self, result: QuestionClassificationResult) -> bool:
        """
        Check if classification result is reliable enough to use.
        
        Args:
            result: Classification result to evaluate
            
        Returns:
            True if classification is reliable, False otherwise
        """
        return result.confidence >= self.get_classification_confidence_threshold()

    def get_supported_categories(self) -> List[GenericQuestionCategory]:
        """
        Get list of all supported question categories.
        
        Returns:
            List of supported question categories
        """
        return list(GenericQuestionCategory)

    def get_category_description(self, category: GenericQuestionCategory) -> str:
        """
        Get human-readable description for a question category.
        
        Args:
            category: Question category
            
        Returns:
            Description of the category
        """
        descriptions = {
            GenericQuestionCategory.BUSINESS_CAPABILITY: 
                "Questions about business functionality, domain, and capabilities",
            GenericQuestionCategory.ARCHITECTURE: 
                "Questions about system architecture, design patterns, and structure",
            GenericQuestionCategory.API_ENDPOINTS: 
                "Questions about API endpoints, routes, and integration points",
            GenericQuestionCategory.DATA_MODELING: 
                "Questions about data models, database schema, and entities",
            GenericQuestionCategory.OPERATIONAL: 
                "Questions about deployment, infrastructure, and operations",
            GenericQuestionCategory.WORKFLOWS: 
                "Questions about processes, workflows, and automation",
            GenericQuestionCategory.GENERAL: 
                "General questions that don't fit specific categories"
        }
        
        return descriptions.get(category, "Unknown category")

    async def analyze_question_context(self, question: str, repository_context: Optional[Dict] = None) -> Dict:
        """
        Analyze question context with optional repository information.
        
        Args:
            question: Question to analyze
            repository_context: Optional repository context information
            
        Returns:
            Enhanced context analysis
        """
        try:
            classification = await self.classify_question(question)
            
            # Extract additional context if repository information is available
            context_analysis = {
                "classification": classification,
                "question_length": safe_len(question),
                "question_complexity": self._assess_question_complexity(question),
                "repository_context": repository_context or {}
            }
            
            # Add repository-specific insights if available
            if repository_context:
                context_analysis["repository_insights"] = self._analyze_repository_context(
                    classification.category, repository_context
                )
            
            return context_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing question context: {e}", exc_info=True)
            # Return basic classification result on error
            return {
                "classification": QuestionClassificationResult(
                    category=GenericQuestionCategory.BUSINESS_CAPABILITY,
                    confidence=0.0,
                    keywords_matched=[],
                    context_indicators=[],
                    suggested_analyzers=["general_analyzer"]
                ), 
                "error": str(e)
            }

    def _assess_question_complexity(self, question: str) -> str:
        """
        Assess question complexity based on length and structure.
        
        Args:
            question: Question to assess
            
        Returns:
            Complexity level (simple/moderate/complex)
        """
        question_len = safe_len(question)
        word_count = safe_len(question.split()) if question else 0
        
        if question_len < 50 and word_count < 10:
            return "simple"
        elif question_len < 200 and word_count < 30:
            return "moderate"
        else:
            return "complex"

    def _analyze_repository_context(self, category: GenericQuestionCategory, repo_context: Dict) -> Dict:
        """
        Analyze repository context for enhanced classification.
        
        Args:
            category: Classified question category
            repo_context: Repository context information
            
        Returns:
            Repository-specific insights
        """
        insights = {
            "repository_type": repo_context.get("type", "unknown"),
            "languages": ensure_list(repo_context.get("languages", [])),
            "framework_indicators": self._detect_framework_indicators(repo_context),
            "category_relevance": self._assess_category_relevance(category, repo_context)
        }
        
        return insights

    def _detect_framework_indicators(self, repo_context: Dict) -> List[str]:
        """
        Detect framework indicators from repository context.
        
        Args:
            repo_context: Repository context information
            
        Returns:
            List of detected framework indicators
        """
        frameworks = []
        languages = ensure_list(repo_context.get("languages", []))
        
        # Simple framework detection based on languages and common patterns
        if "python" in [lang.lower() for lang in languages]:
            frameworks.extend(["django", "flask", "fastapi"])
        if "javascript" in [lang.lower() for lang in languages]:
            frameworks.extend(["react", "vue", "angular", "node.js"])
        if "csharp" in [lang.lower() for lang in languages]:
            frameworks.extend([".net", "asp.net"])
        
        return frameworks

    def _assess_category_relevance(self, category: GenericQuestionCategory, repo_context: Dict) -> float:
        """
        Assess how relevant the category is for the given repository context.
        
        Args:
            category: Question category
            repo_context: Repository context
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        # This could be enhanced with more sophisticated analysis
        # For now, return a baseline relevance
        return 0.5  # Neutral relevance score