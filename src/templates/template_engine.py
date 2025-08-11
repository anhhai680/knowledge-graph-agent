"""
Template Engine for Generic Q&A Agent.

This module provides template-based response generation following the
appSettings.json configuration pattern from the existing codebase.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class TemplateEngine:
    """
    Template engine following appSettings.json configuration pattern.
    
    This engine loads JSON-based templates and generates structured responses
    for different question categories and architecture types.
    """

    def __init__(self):
        """Initialize template engine with configuration patterns."""
        self.logger = get_logger(self.__class__.__name__)  # REUSE logging pattern
        self.settings = get_settings()  # REUSE configuration system
        self.templates_path = Path(__file__).parent / "templates"
        self._template_cache = {}  # Simple in-memory cache

    def generate_response(
        self,
        category: str,
        question: str,
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any],
        include_code_examples: bool = True,
        preferred_template: str = "generic_template"
    ) -> Dict[str, Any]:
        """
        Generate structured response using template-based approach.
        
        Args:
            category: Question category
            question: Original question
            analysis_results: Analysis results from workflow
            repository_context: Repository context information
            include_code_examples: Whether to include code examples
            preferred_template: Template to use for response
            
        Returns:
            Structured response dictionary
        """
        self.logger.info(f"Generating response for category: {category}")
        
        try:
            # Load appropriate template
            template = self._load_template(category, preferred_template)
            
            # Generate response using template
            response = self._apply_template(
                template=template,
                category=category,
                question=question,
                analysis_results=analysis_results,
                repository_context=repository_context,
                include_code_examples=include_code_examples
            )
            
            self.logger.info(f"Response generated successfully for category: {category}")
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}", exc_info=True)
            return self._create_fallback_response(category, question, str(e))

    def _load_template(self, category: str, preferred_template: str) -> Dict[str, Any]:
        """
        Load template using JSON loading pattern from settings.py.
        
        Args:
            category: Question category
            preferred_template: Preferred template name
            
        Returns:
            Template configuration dictionary
        """
        # Check cache first
        cache_key = f"{category}_{preferred_template}"
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]
        
        # Try to load preferred template
        template_file = self.templates_path / f"{preferred_template}.json"
        
        # Follow same error handling as appSettings.json loading
        try:
            if template_file.exists():
                with open(template_file, 'r', encoding='utf-8') as f:
                    template = json.load(f)
                    self._template_cache[cache_key] = template
                    return template
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load template {template_file}: {e}")
        
        # Fallback to category-specific template
        category_template_file = self.templates_path / f"{category}_template.json"
        try:
            if category_template_file.exists():
                with open(category_template_file, 'r', encoding='utf-8') as f:
                    template = json.load(f)
                    self._template_cache[cache_key] = template
                    return template
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load category template {category_template_file}: {e}")
        
        # Final fallback to generic template
        return self._load_generic_template()

    def _load_generic_template(self) -> Dict[str, Any]:
        """Load generic template as final fallback."""
        try:
            generic_template_file = self.templates_path / "generic_template.json"
            if generic_template_file.exists():
                with open(generic_template_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load generic template: {e}")
        
        # Return hardcoded fallback template
        return self._get_hardcoded_template()

    def _get_hardcoded_template(self) -> Dict[str, Any]:
        """Get hardcoded fallback template."""
        return {
            "template_name": "hardcoded_fallback",
            "response_structure": {
                "overview": "Provides general information about the project",
                "details": "Contains specific analysis based on available information",
                "recommendations": "Suggests areas for improvement or further investigation"
            },
            "sections": ["overview", "details", "recommendations"],
            "include_metadata": True
        }

    def _apply_template(
        self,
        template: Dict[str, Any],
        category: str,
        question: str,
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any],
        include_code_examples: bool
    ) -> Dict[str, Any]:
        """
        Apply template to generate structured response.
        
        Args:
            template: Template configuration
            category: Question category
            question: Original question
            analysis_results: Analysis results
            repository_context: Repository context
            include_code_examples: Whether to include code examples
            
        Returns:
            Generated response dictionary
        """
        try:
            response = {
                "type": "structured_response",
                "category": category,
                "question": question,
                "template_used": template.get("template_name", "unknown"),
                "generated_at": datetime.now().isoformat(),
                "sections": {}
            }
            
            # Apply category-specific response generation
            if category == "business_capability":
                response["sections"] = self._generate_business_capability_response(
                    template, analysis_results, repository_context
                )
            elif category == "architecture":
                response["sections"] = self._generate_architecture_response(
                    template, analysis_results, repository_context, include_code_examples
                )
            elif category == "api_endpoints":
                response["sections"] = self._generate_api_response(
                    template, analysis_results, repository_context, include_code_examples
                )
            elif category == "data_modeling":
                response["sections"] = self._generate_data_modeling_response(
                    template, analysis_results, repository_context
                )
            elif category == "operational":
                response["sections"] = self._generate_operational_response(
                    template, analysis_results, repository_context
                )
            else:
                response["sections"] = self._generate_general_response(
                    template, analysis_results, repository_context
                )
            
            # Add metadata if requested
            if template.get("include_metadata", False):
                response["metadata"] = {
                    "analysis_confidence": analysis_results.get("confidence_score", 0.0),
                    "repository_type": repository_context.get("type", "unknown"),
                    "template_version": template.get("version", "1.0")
                }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Template application failed: {e}")
            return self._create_fallback_response(category, question, str(e))

    def _generate_business_capability_response(
        self,
        template: Dict[str, Any],
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate business capability response."""
        return {
            "overview": {
                "business_domain": analysis_results.get("category", "General Business System"),
                "core_purpose": "Provides business functionality and data management capabilities",
                "scope": "Handles core business operations and user interactions"
            },
            "core_capabilities": [
                "User Management and Authentication",
                "Data Processing and Storage",
                "Business Logic Implementation",
                "API Integration and Communication"
            ],
            "business_entities": analysis_results.get("core_entities", ["User", "System", "Data"]),
            "value_proposition": analysis_results.get("business_value", 
                "Streamlines business operations through automated processes and data management"),
            "target_users": analysis_results.get("target_users", ["End Users", "Administrators"]),
            "key_processes": analysis_results.get("key_capabilities", [
                "Data Input and Validation",
                "Processing and Transformation", 
                "Output Generation and Delivery"
            ])
        }

    def _generate_architecture_response(
        self,
        template: Dict[str, Any],
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any],
        include_code_examples: bool
    ) -> Dict[str, Any]:
        """Generate architecture response."""
        sections = {
            "overview": {
                "architecture_pattern": analysis_results.get("architecture_pattern", "Layered Architecture"),
                "technology_stack": analysis_results.get("technology_stack", ["Python", "FastAPI"]),
                "design_principles": analysis_results.get("design_principles", [
                    "Separation of Concerns",
                    "Dependency Injection",
                    "SOLID Principles"
                ])
            },
            "components": {
                "layers": analysis_results.get("layers", ["API Layer", "Business Layer", "Data Layer"]),
                "main_components": analysis_results.get("components", [
                    "API Controllers",
                    "Business Services", 
                    "Data Access Layer"
                ]),
                "integration_points": ["Database", "External APIs", "Authentication Service"]
            },
            "technical_decisions": {
                "framework_choice": "Selected for performance and developer productivity",
                "database_strategy": "Chosen based on data structure and scalability needs",
                "deployment_approach": "Designed for scalability and maintainability"
            }
        }
        
        if include_code_examples:
            sections["code_examples"] = {
                "api_endpoint": "Example API endpoint implementation",
                "service_layer": "Business logic service example",
                "data_model": "Entity/model definition example"
            }
        
        return sections

    def _generate_api_response(
        self,
        template: Dict[str, Any],
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any],
        include_code_examples: bool
    ) -> Dict[str, Any]:
        """Generate API endpoints response."""
        sections = {
            "overview": {
                "api_type": analysis_results.get("api_type", "REST API"),
                "authentication": analysis_results.get("authentication_methods", ["API Key"]),
                "response_format": "JSON"
            },
            "endpoints": [
                {
                    "path": "/api/v1/health",
                    "method": "GET",
                    "description": "Health check endpoint",
                    "authentication_required": False
                },
                {
                    "path": "/api/v1/data",
                    "method": "POST", 
                    "description": "Data processing endpoint",
                    "authentication_required": True
                }
            ],
            "integration": {
                "rate_limiting": "Configured for API protection",
                "cors": "Enabled for cross-origin requests",
                "documentation": analysis_results.get("documentation_available", False)
            }
        }
        
        if include_code_examples:
            sections["examples"] = {
                "request_example": "Sample API request format",
                "response_example": "Sample API response format",
                "authentication_example": "Authentication header example"
            }
        
        return sections

    def _generate_data_modeling_response(
        self,
        template: Dict[str, Any],
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate data modeling response."""
        return {
            "overview": {
                "database_type": analysis_results.get("database_type", "Relational Database"),
                "orm_framework": analysis_results.get("orm_framework", "SQLAlchemy"),
                "migration_system": analysis_results.get("migration_system", False)
            },
            "data_models": [
                {
                    "name": "User",
                    "fields": ["id", "username", "email", "created_at"],
                    "relationships": ["has_many_posts", "belongs_to_organization"]
                },
                {
                    "name": "Data",
                    "fields": ["id", "content", "metadata", "processed_at"],
                    "relationships": ["belongs_to_user"]
                }
            ],
            "relationships": analysis_results.get("relationships", [
                "User -> Data (One-to-Many)",
                "User -> Organization (Many-to-One)"
            ]),
            "data_access": {
                "repository_pattern": "Used for data access abstraction",
                "caching_strategy": "Implemented for performance optimization",
                "validation": "Model-level validation with business rules"
            }
        }

    def _generate_operational_response(
        self,
        template: Dict[str, Any],
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate operational response."""
        return {
            "deployment": {
                "type": analysis_results.get("deployment_type", "Containerized"),
                "containerization": analysis_results.get("containerization", True),
                "orchestration": "Docker Compose or Kubernetes"
            },
            "infrastructure": {
                "environments": analysis_results.get("environments", ["development", "production"]),
                "ci_cd": analysis_results.get("ci_cd_configured", False),
                "infrastructure_as_code": analysis_results.get("infrastructure_as_code", False)
            },
            "monitoring": {
                "health_checks": "Endpoint-based health monitoring",
                "logging": "Structured logging with log levels",
                "metrics": analysis_results.get("monitoring_setup", False)
            },
            "security": {
                "practices": analysis_results.get("security_practices", ["authentication", "validation"]),
                "secrets_management": "Environment variables and secure storage",
                "access_control": "Role-based access control"
            }
        }

    def _generate_general_response(
        self,
        template: Dict[str, Any],
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate general response."""
        return {
            "overview": {
                "project_type": repository_context.get("type", "Software Application"),
                "primary_purpose": "Provides automated functionality and data management",
                "technology_focus": "Modern software development practices"
            },
            "key_features": [
                "Data Processing and Management",
                "User Interface and Interaction", 
                "Integration Capabilities",
                "Security and Authentication"
            ],
            "technical_highlights": {
                "architecture": "Well-structured and maintainable codebase",
                "scalability": "Designed for growth and performance",
                "maintainability": "Clean code practices and documentation"
            },
            "recommendations": [
                "Continue following established patterns",
                "Maintain comprehensive testing",
                "Keep documentation updated",
                "Monitor performance and security"
            ]
        }

    def _create_fallback_response(self, category: str, question: str, error: str) -> Dict[str, Any]:
        """Create fallback response when template processing fails."""
        return {
            "type": "fallback_response",
            "category": category,
            "question": question,
            "generated_at": datetime.now().isoformat(),
            "sections": {
                "overview": {
                    "message": f"I can help answer questions about {category} topics, but encountered an issue generating a detailed response.",
                    "suggestion": "Please try rephrasing your question or provide more specific details."
                },
                "error_info": {
                    "error": error,
                    "fallback_used": True
                }
            },
            "template_used": "fallback",
            "metadata": {
                "error_occurred": True,
                "fallback_response": True
            }
        }

    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        try:
            templates = []
            if self.templates_path.exists():
                for template_file in self.templates_path.glob("*.json"):
                    templates.append(template_file.stem)
            return templates
        except Exception as e:
            self.logger.error(f"Error getting available templates: {e}")
            return ["generic_template"]

    def validate_template(self, template_name: str) -> Dict[str, Any]:
        """Validate template configuration."""
        validation_result = {
            "valid": False,
            "exists": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            template_file = self.templates_path / f"{template_name}.json"
            validation_result["exists"] = template_file.exists()
            
            if not validation_result["exists"]:
                validation_result["errors"].append(f"Template file {template_name}.json not found")
                return validation_result
            
            # Try to load and validate template
            with open(template_file, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            # Basic validation
            required_fields = ["template_name", "response_structure"]
            for field in required_fields:
                if field not in template_data:
                    validation_result["errors"].append(f"Missing required field: {field}")
            
            if not validation_result["errors"]:
                validation_result["valid"] = True
            
        except json.JSONDecodeError as e:
            validation_result["errors"].append(f"Invalid JSON format: {e}")
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")
        
        return validation_result