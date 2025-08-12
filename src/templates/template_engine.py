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
        """Generate business capability response using actual repository data."""
        
        # Extract repository information
        repository_name = repository_context.get("repository", "Unknown Repository")
        documents = repository_context.get("raw_documents", [])
        frameworks = list(repository_context.get("frameworks", set()))
        languages = list(repository_context.get("languages", set()))
        
        # Analyze actual business capabilities from documents
        business_capabilities = []
        business_entities = []
        api_features = []
        
        # Parse document content for actual business logic
        for doc in documents:
            content = doc.page_content.lower()
            source = doc.metadata.get("source", "")
            
            # Extract business capabilities from code content
            if any(keyword in content for keyword in ["user", "customer", "account"]):
                business_capabilities.append("User Management")
            if any(keyword in content for keyword in ["auth", "login", "token", "jwt"]):
                business_capabilities.append("Authentication & Authorization")
            if any(keyword in content for keyword in ["order", "payment", "transaction"]):
                business_capabilities.append("Transaction Management")
            if any(keyword in content for keyword in ["notification", "email", "sms", "message"]):
                business_capabilities.append("Notification Services")
            if any(keyword in content for keyword in ["search", "filter", "query"]):
                business_capabilities.append("Search & Query Services")
            if any(keyword in content for keyword in ["listing", "catalog", "inventory"]):
                business_capabilities.append("Listing & Catalog Management")
            if any(keyword in content for keyword in ["booking", "reservation", "schedule"]):
                business_capabilities.append("Booking & Scheduling")
            if any(keyword in content for keyword in ["report", "analytics", "dashboard"]):
                business_capabilities.append("Reporting & Analytics")
            
            # Extract business entities
            if "class " in content or "model " in content or "entity " in content:
                lines = content.split('\n')
                for line in lines:
                    if any(keyword in line for keyword in ["class ", "public class", "model "]):
                        words = line.split()
                        for i, word in enumerate(words):
                            if word in ["class", "model"] and i + 1 < len(words):
                                entity_name = words[i + 1].strip(":(){}").replace("(", "").replace(")", "")
                                if entity_name and entity_name.isalpha() and entity_name not in ["controller", "service", "repository"]:
                                    business_entities.append(entity_name.title())
            
            # Extract API functionality
            if any(keyword in content for keyword in ["get", "post", "put", "delete", "api", "endpoint"]):
                if "get" in content and any(term in content for term in ["list", "all", "search"]):
                    api_features.append("Data Retrieval & Listing")
                if "post" in content and any(term in content for term in ["create", "add", "new"]):
                    api_features.append("Data Creation")
                if "put" in content or "patch" in content:
                    api_features.append("Data Updates")
                if "delete" in content:
                    api_features.append("Data Deletion")
        
        # Remove duplicates and clean up
        business_capabilities = list(set(business_capabilities)) or ["General Business Operations"]
        business_entities = list(set(business_entities)) or ["Business Data"]
        api_features = list(set(api_features)) or ["Basic CRUD Operations"]
        
        # Determine business domain based on repository name and capabilities
        business_domain = "General Business System"
        if "car" in repository_name.lower():
            business_domain = "Automotive Services"
        elif "notification" in repository_name.lower():
            business_domain = "Communication Services"  
        elif "listing" in repository_name.lower():
            business_domain = "Catalog & Listing Services"
        elif "user" in repository_name.lower():
            business_domain = "User Management System"
        elif "order" in repository_name.lower():
            business_domain = "Order Management System"
        
        # Determine core purpose based on actual capabilities
        core_purpose = f"Provides {business_domain.lower()} functionality"
        if "notification" in ' '.join(business_capabilities).lower():
            core_purpose += " with notification and communication features"
        if "listing" in ' '.join(business_capabilities).lower():
            core_purpose += " with catalog and listing management"
        if "auth" in ' '.join(business_capabilities).lower():
            core_purpose += " with secure authentication"
        
        return {
            "overview": {
                "repository": repository_name,
                "business_domain": business_domain,
                "core_purpose": core_purpose,
                "scope": f"Handles {', '.join(business_capabilities[:3]).lower()} and related operations",
                "technology_stack": languages + frameworks
            },
            "core_capabilities": business_capabilities,
            "business_entities": business_entities,
            "api_functionality": api_features,
            "value_proposition": f"Streamlines {business_domain.lower()} through automated {', '.join(business_capabilities[:2]).lower()}",
            "target_users": analysis_results.get("target_users", ["End Users", "System Administrators", "API Consumers"]),
            "key_processes": business_capabilities,
            "analysis_summary": {
                "documents_analyzed": len(documents),
                "source_files": len(repository_context.get("source_files", [])),
                "languages_detected": languages,
                "frameworks_detected": frameworks
            }
        }

    def _generate_architecture_response(
        self,
        template: Dict[str, Any],
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any],
        include_code_examples: bool
    ) -> Dict[str, Any]:
        """Generate architecture response using actual repository data."""
        
        # Extract repository information
        repository_name = repository_context.get("repository", "Unknown Repository")
        documents = repository_context.get("raw_documents", [])
        frameworks = list(repository_context.get("frameworks", set()))
        languages = list(repository_context.get("languages", set()))
        source_files = repository_context.get("source_files", [])
        
        # Analyze architecture patterns from actual code
        architecture_patterns = []
        technology_stack = languages + frameworks
        design_principles = []
        layers = []
        components = []
        
        # Parse documents for architecture information
        for doc in documents:
            content = doc.page_content.lower()
            source = doc.metadata.get("source", "")
            
            # Detect architecture patterns
            if any(keyword in content for keyword in ["controller", "service", "repository"]):
                architecture_patterns.append("Layered Architecture")
                layers.extend(["Controller Layer", "Service Layer", "Repository Layer"])
            if "mvc" in content or ("model" in content and "view" in content):
                architecture_patterns.append("Model-View-Controller (MVC)")
            if "microservice" in content or "service" in source:
                architecture_patterns.append("Microservices")
            if any(keyword in content for keyword in ["dependency injection", "di", "ioc"]):
                design_principles.append("Dependency Injection")
                architecture_patterns.append("Dependency Injection Pattern")
            if "solid" in content or any(principle in content for principle in ["single responsibility", "open closed"]):
                design_principles.append("SOLID Principles")
            if "factory" in content:
                architecture_patterns.append("Factory Pattern")
            if "builder" in content:
                architecture_patterns.append("Builder Pattern")
            if "singleton" in content:
                architecture_patterns.append("Singleton Pattern")
            
            # Extract components
            if "controller" in content:
                components.append("API Controllers")
            if "service" in content:
                components.append("Business Services")
            if "repository" in content or "data" in content:
                components.append("Data Access Layer")
            if "middleware" in content:
                components.append("Middleware Components")
            if "auth" in content:
                components.append("Authentication Module")
            if "config" in content:
                components.append("Configuration Management")
        
        # Clean up duplicates
        architecture_patterns = list(set(architecture_patterns)) or ["Modular Architecture"]
        design_principles = list(set(design_principles)) or ["Separation of Concerns", "Clean Code"]
        layers = list(set(layers)) or ["Application Layer", "Business Layer", "Data Layer"]
        components = list(set(components)) or ["Core Components", "Utility Components"]
        
        # Determine integration points based on content
        integration_points = ["Database"]
        for doc in documents:
            content = doc.page_content.lower()
            if any(keyword in content for keyword in ["http", "api", "rest", "endpoint"]):
                integration_points.append("REST APIs")
            if any(keyword in content for keyword in ["auth", "jwt", "token"]):
                integration_points.append("Authentication Service")
            if any(keyword in content for keyword in ["email", "notification", "message"]):
                integration_points.append("Notification Services")
            if any(keyword in content for keyword in ["cache", "redis", "memory"]):
                integration_points.append("Caching Layer")
        
        integration_points = list(set(integration_points))
        
        sections = {
            "overview": {
                "repository": repository_name,
                "architecture_pattern": ", ".join(architecture_patterns),
                "technology_stack": technology_stack,
                "design_principles": design_principles,
                "file_count": len(source_files)
            },
            "components": {
                "layers": layers,
                "main_components": components,
                "integration_points": integration_points
            },
            "technical_decisions": {
                "framework_choice": f"Selected {', '.join(frameworks)} for {', '.join(languages)} development" if frameworks else "Framework selection based on project requirements",
                "architecture_rationale": f"Implements {', '.join(architecture_patterns[:2])} for maintainability and scalability",
                "deployment_approach": "Designed for scalability and maintainability"
            },
            "analysis_summary": {
                "documents_analyzed": len(documents),
                "patterns_detected": architecture_patterns,
                "languages_used": languages
            }
        }
        
        if include_code_examples:
            # Extract actual code examples from documents
            code_examples = {}
            for doc in documents:
                source = doc.metadata.get("source", "")
                content = doc.page_content
                
                if "controller" in source.lower() and len(content) < 500:
                    code_examples["api_endpoint"] = f"From {source}:\n{content[:200]}..."
                elif "service" in source.lower() and len(content) < 500:
                    code_examples["service_layer"] = f"From {source}:\n{content[:200]}..."
                elif any(keyword in source.lower() for keyword in ["model", "entity"]) and len(content) < 500:
                    code_examples["data_model"] = f"From {source}:\n{content[:200]}..."
            
            if not code_examples:
                code_examples = {
                    "note": "Code examples available in the repository structure",
                    "file_locations": [f["file"] for f in source_files[:3]]
                }
            
            sections["code_examples"] = code_examples
        
        return sections

    def _generate_api_response(
        self,
        template: Dict[str, Any],
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any],
        include_code_examples: bool
    ) -> Dict[str, Any]:
        """Generate API endpoints response using real analysis data."""
        
        # Extract real data from analysis results
        repository = analysis_results.get("repository", "unknown")
        total_endpoints = analysis_results.get("total_endpoints", 0)
        frameworks = analysis_results.get("frameworks_detected", [])
        endpoints = analysis_results.get("endpoints", [])
        detailed_endpoints = analysis_results.get("detailed_endpoints", [])
        api_patterns = analysis_results.get("api_patterns", [])
        auth_info = analysis_results.get("authentication", {})
        business_domain = analysis_results.get("business_domain", "")
        
        # Build comprehensive response based on real data
        sections = {
            "overview": {
                "repository": repository,
                "total_endpoints": total_endpoints,
                "business_domain": business_domain,
                "frameworks": frameworks,
                "api_patterns": api_patterns,
                "response_format": "JSON"
            }
        }
        
        if endpoints:
            # Use real endpoints data
            sections["endpoints"] = detailed_endpoints if detailed_endpoints else endpoints
            
            # Group endpoints by method for summary
            method_summary = analysis_results.get("methods_summary", {})
            if method_summary:
                sections["methods_summary"] = method_summary
            
            # Add authentication information if detected
            if auth_info:
                auth_methods = []
                if auth_info.get("api_key"):
                    auth_methods.append("API Key")
                if auth_info.get("jwt_token"):
                    auth_methods.append("JWT Token")
                if auth_info.get("oauth"):
                    auth_methods.append("OAuth")
                if auth_info.get("basic_auth"):
                    auth_methods.append("Basic Authentication")
                
                sections["authentication"] = {
                    "methods": auth_methods,
                    "detected_patterns": auth_info
                }
            
            # Add integration details
            sections["integration"] = {
                "source_files": analysis_results.get("source_files", []),
                "frameworks_detected": frameworks,
                "design_patterns": api_patterns
            }
            
            # Add usage examples based on real endpoints
            if include_code_examples and endpoints:
                sections["examples"] = self._generate_real_api_examples(endpoints, repository)
                
        else:
            # Handle case where no endpoints were found
            sections["no_endpoints_found"] = {
                "status": "No API endpoints detected",
                "analysis": {
                    "possible_reasons": analysis_results.get("possible_reasons", []),
                    "suggestions": analysis_results.get("suggestions", []),
                    "frameworks_detected": frameworks
                }
            }
            
            # Include README endpoints if available
            readme_endpoints = analysis_results.get("readme_endpoints", [])
            if readme_endpoints:
                sections["documentation_endpoints"] = readme_endpoints
        
        return sections

    def _generate_real_api_examples(self, endpoints: List[Dict], repository: str) -> Dict[str, str]:
        """Generate realistic API examples based on actual endpoints."""
        examples = {}
        
        if endpoints:
            # Use first endpoint for example
            first_endpoint = endpoints[0]
            method = first_endpoint.get("method", "GET")
            path = first_endpoint.get("path", "/api/resource")
            
            # Generate request example
            if method == "GET":
                examples["request_example"] = f"GET {path}\nAccept: application/json"
            elif method == "POST":
                examples["request_example"] = f"POST {path}\nContent-Type: application/json\n\n{{\n  \"data\": \"example\"\n}}"
            else:
                examples["request_example"] = f"{method} {path}\nContent-Type: application/json"
            
            # Generate response example
            examples["response_example"] = "{\n  \"success\": true,\n  \"data\": {...},\n  \"timestamp\": \"2025-08-12T09:00:00Z\"\n}"
            
            # Add authentication example if detected
            examples["authentication_example"] = "Authorization: Bearer <your-api-key>\n# or\nX-API-Key: <your-api-key>"
        
        return examples

    def _generate_data_modeling_response(
        self,
        template: Dict[str, Any],
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate data modeling response using actual repository data."""
        
        # Extract repository information
        repository_name = repository_context.get("repository", "Unknown Repository")
        documents = repository_context.get("raw_documents", [])
        languages = list(repository_context.get("languages", set()))
        
        # Analyze data models from actual code
        detected_models = []
        orm_frameworks = []
        database_types = []
        relationships = []
        
        # Parse documents for data modeling information
        for doc in documents:
            content = doc.page_content.lower()
            source = doc.metadata.get("source", "")
            lines = content.split('\n')
            
            # Detect ORM frameworks
            if any(framework in content for framework in ["entity framework", "entityframework", "ef core"]):
                orm_frameworks.append("Entity Framework Core")
            elif any(framework in content for framework in ["sqlalchemy", "sql alchemy"]):
                orm_frameworks.append("SQLAlchemy")
            elif "hibernate" in content:
                orm_frameworks.append("Hibernate")
            elif "mongoose" in content:
                orm_frameworks.append("Mongoose")
            
            # Detect database types
            if any(db in content for db in ["sql server", "sqlserver", "mssql"]):
                database_types.append("SQL Server")
            elif any(db in content for db in ["postgresql", "postgres"]):
                database_types.append("PostgreSQL")
            elif "mysql" in content:
                database_types.append("MySQL")
            elif "mongodb" in content:
                database_types.append("MongoDB")
            elif "sqlite" in content:
                database_types.append("SQLite")
            
            # Extract model/entity definitions
            for line in lines:
                line = line.strip()
                if any(keyword in line for keyword in ["public class", "class ", "public partial class"]):
                    words = line.split()
                    for i, word in enumerate(words):
                        if word == "class" and i + 1 < len(words):
                            model_name = words[i + 1].strip(":(){}")
                            if (model_name and model_name[0].isupper() and 
                                model_name not in ["Program", "Startup", "Controller", "Service"] and
                                not model_name.endswith("Controller")):
                                
                                # Extract fields from the model
                                model_fields = []
                                model_relationships = []
                                
                                # Look ahead for properties/fields
                                doc_lines = doc.page_content.split('\n')
                                start_idx = doc_lines.index(line) if line in doc_lines else 0
                                for j in range(start_idx + 1, min(start_idx + 20, len(doc_lines))):
                                    field_line = doc_lines[j].strip().lower()
                                    if "public " in field_line and any(type_word in field_line for type_word in ["int", "string", "datetime", "bool", "decimal"]):
                                        field_parts = field_line.split()
                                        if len(field_parts) >= 3:
                                            field_name = field_parts[2].replace("{", "").replace(";", "")
                                            model_fields.append(field_name)
                                    
                                    # Detect relationships
                                    if any(rel_word in field_line for rel_word in ["virtual", "icollection", "list<", "fk_"]):
                                        if "icollection" in field_line or "list<" in field_line:
                                            model_relationships.append("One-to-Many")
                                        elif "virtual" in field_line:
                                            model_relationships.append("Navigation Property")
                                
                                detected_models.append({
                                    "name": model_name,
                                    "fields": model_fields[:5] if model_fields else ["id", "created_at"],
                                    "relationships": model_relationships
                                })
            
            # Extract relationships from foreign key patterns
            if any(pattern in content for pattern in ["foreign key", "fk_", "references", "joincolumn"]):
                relationships.append("Foreign Key Relationships Detected")
            if "one-to-many" in content:
                relationships.append("One-to-Many")
            if "many-to-one" in content:
                relationships.append("Many-to-One")
            if "many-to-many" in content:
                relationships.append("Many-to-Many")
        
        # Clean up duplicates
        orm_frameworks = list(set(orm_frameworks))
        database_types = list(set(database_types))
        relationships = list(set(relationships))
        
        # Set defaults if nothing detected
        if not detected_models:
            detected_models = [
                {
                    "name": "Primary Entity",
                    "fields": ["id", "name", "created_at", "updated_at"],
                    "relationships": ["Standard CRUD operations"]
                }
            ]
        
        return {
            "overview": {
                "repository": repository_name,
                "database_type": database_types[0] if database_types else "Database",
                "orm_framework": orm_frameworks[0] if orm_frameworks else "ORM Framework",
                "migration_system": any("migration" in doc.page_content.lower() for doc in documents),
                "languages": languages
            },
            "data_models": detected_models,
            "relationships": relationships if relationships else ["Entity relationships as defined in code"],
            "data_access": {
                "repository_pattern": any("repository" in doc.page_content.lower() for doc in documents),
                "caching_strategy": any("cache" in doc.page_content.lower() for doc in documents),
                "validation": any("validation" in doc.page_content.lower() for doc in documents)
            },
            "database_technologies": {
                "orm_frameworks": orm_frameworks,
                "database_types": database_types,
                "features_detected": []
            },
            "analysis_summary": {
                "documents_analyzed": len(documents),
                "models_detected": len(detected_models),
                "technologies_found": orm_frameworks + database_types
            }
        }

    def _generate_operational_response(
        self,
        template: Dict[str, Any],
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate operational response using actual repository data."""
        
        # Extract repository information
        repository_name = repository_context.get("repository", "Unknown Repository")
        documents = repository_context.get("raw_documents", [])
        source_files = repository_context.get("source_files", [])
        
        # Analyze operational aspects from actual code
        deployment_indicators = []
        infrastructure_setup = []
        monitoring_features = []
        security_practices = []
        ci_cd_features = []
        
        # Parse documents for operational information
        for doc in documents:
            content = doc.page_content.lower()
            source = doc.metadata.get("source", "")
            
            # Detect deployment configuration
            if "docker" in content or "dockerfile" in source.lower():
                deployment_indicators.append("Docker Containerization")
            if "kubernetes" in content or "k8s" in content:
                deployment_indicators.append("Kubernetes Orchestration")
            if "docker-compose" in content or source.endswith("docker-compose.yml"):
                deployment_indicators.append("Docker Compose")
            if any(keyword in content for keyword in ["azure", "aws", "gcp", "cloud"]):
                deployment_indicators.append("Cloud Deployment")
            
            # Detect CI/CD setup
            if any(keyword in source.lower() for keyword in [".github", "pipeline", "build", "deploy"]):
                ci_cd_features.append("CI/CD Pipeline")
            if "github actions" in content or "workflow" in content:
                ci_cd_features.append("GitHub Actions")
            if "jenkins" in content:
                ci_cd_features.append("Jenkins")
            if "azure devops" in content:
                ci_cd_features.append("Azure DevOps")
            
            # Detect monitoring and logging
            if any(keyword in content for keyword in ["log", "logger", "logging"]):
                monitoring_features.append("Application Logging")
            if any(keyword in content for keyword in ["health", "healthcheck", "ping"]):
                monitoring_features.append("Health Checks")
            if any(keyword in content for keyword in ["metric", "monitor", "telemetry"]):
                monitoring_features.append("Performance Monitoring")
            if "swagger" in content or "openapi" in content:
                monitoring_features.append("API Documentation")
            
            # Detect security practices
            if any(keyword in content for keyword in ["auth", "jwt", "token", "oauth"]):
                security_practices.append("Authentication")
            if any(keyword in content for keyword in ["encrypt", "hash", "secure"]):
                security_practices.append("Data Encryption")
            if any(keyword in content for keyword in ["cors", "csrf", "xss"]):
                security_practices.append("Web Security")
            if any(keyword in content for keyword in ["validate", "sanitize", "input validation"]):
                security_practices.append("Input Validation")
            if "https" in content or "ssl" in content or "tls" in content:
                security_practices.append("Secure Communication")
            
            # Detect infrastructure as code
            if any(keyword in source.lower() for keyword in ["terraform", "cloudformation", "bicep"]):
                infrastructure_setup.append("Infrastructure as Code")
            if "appsettings" in source.lower() or "config" in source.lower():
                infrastructure_setup.append("Configuration Management")
        
        # Clean up duplicates
        deployment_indicators = list(set(deployment_indicators))
        ci_cd_features = list(set(ci_cd_features))
        monitoring_features = list(set(monitoring_features))
        security_practices = list(set(security_practices))
        infrastructure_setup = list(set(infrastructure_setup))
        
        # Determine deployment type
        deployment_type = "Standard Application"
        if "Docker Containerization" in deployment_indicators:
            deployment_type = "Containerized Application"
        if "Kubernetes Orchestration" in deployment_indicators:
            deployment_type = "Kubernetes-based Application"
        if "Cloud Deployment" in deployment_indicators:
            deployment_type = "Cloud-native Application"
        
        # Determine environments
        environments = ["development"]
        if any("prod" in f["file"].lower() for f in source_files):
            environments.append("production")
        if any("test" in f["file"].lower() for f in source_files):
            environments.append("testing")
        if any("stage" in f["file"].lower() for f in source_files):
            environments.append("staging")
        
        return {
            "deployment": {
                "repository": repository_name,
                "type": deployment_type,
                "containerization": "Docker Containerization" in deployment_indicators,
                "orchestration": "Kubernetes Orchestration" in deployment_indicators,
                "deployment_features": deployment_indicators
            },
            "infrastructure": {
                "environments": environments,
                "ci_cd": len(ci_cd_features) > 0,
                "ci_cd_tools": ci_cd_features,
                "infrastructure_as_code": "Infrastructure as Code" in infrastructure_setup,
                "infrastructure_features": infrastructure_setup
            },
            "monitoring": {
                "health_checks": "Health Checks" in monitoring_features,
                "logging": "Application Logging" in monitoring_features,
                "metrics": "Performance Monitoring" in monitoring_features,
                "monitoring_features": monitoring_features,
                "api_documentation": "API Documentation" in monitoring_features
            },
            "security": {
                "practices": security_practices,
                "authentication": "Authentication" in security_practices,
                "encryption": "Data Encryption" in security_practices,
                "web_security": "Web Security" in security_practices,
                "secure_communication": "Secure Communication" in security_practices
            },
            "analysis_summary": {
                "documents_analyzed": len(documents),
                "operational_features_detected": len(deployment_indicators + ci_cd_features + monitoring_features + security_practices),
                "deployment_readiness": "High" if deployment_indicators else "Basic"
            }
        }

    def _generate_general_response(
        self,
        template: Dict[str, Any],
        analysis_results: Dict[str, Any],
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate general response using actual repository data."""
        
        # Extract repository information
        repository_name = repository_context.get("repository", "Unknown Repository")
        documents = repository_context.get("raw_documents", [])
        languages = list(repository_context.get("languages", set()))
        frameworks = list(repository_context.get("frameworks", set()))
        source_files = repository_context.get("source_files", [])
        
        # Analyze project characteristics from actual content
        project_features = []
        technical_highlights = []
        
        # Parse documents for general project information
        for doc in documents:
            content = doc.page_content.lower()
            source = doc.metadata.get("source", "")
            
            # Detect key features
            if any(keyword in content for keyword in ["api", "endpoint", "rest", "http"]):
                project_features.append("REST API Services")
            if any(keyword in content for keyword in ["user", "account", "auth"]):
                project_features.append("User Management")
            if any(keyword in content for keyword in ["data", "database", "entity", "model"]):
                project_features.append("Data Management")
            if any(keyword in content for keyword in ["notification", "email", "message"]):
                project_features.append("Communication Services")
            if any(keyword in content for keyword in ["search", "query", "filter"]):
                project_features.append("Search & Query Capabilities")
            if any(keyword in content for keyword in ["security", "encryption", "validation"]):
                project_features.append("Security Features")
            if "test" in source.lower() or "spec" in source.lower():
                project_features.append("Automated Testing")
            
            # Detect technical highlights
            if any(keyword in content for keyword in ["async", "await", "asynchronous"]):
                technical_highlights.append("Asynchronous Processing")
            if any(keyword in content for keyword in ["cache", "memory", "redis"]):
                technical_highlights.append("Performance Optimization")
            if "dependency injection" in content or "di" in content:
                technical_highlights.append("Dependency Injection")
            if any(keyword in content for keyword in ["logging", "logger", "log"]):
                technical_highlights.append("Comprehensive Logging")
            if any(keyword in content for keyword in ["config", "settings", "environment"]):
                technical_highlights.append("Configuration Management")
        
        # Clean up duplicates
        project_features = list(set(project_features))
        technical_highlights = list(set(technical_highlights))
        
        # Determine project type based on analysis
        project_type = "Software Application"
        if any("service" in feature.lower() for feature in project_features):
            project_type = "Service-Oriented Application"
        if "REST API Services" in project_features:
            project_type = "API-Driven Application"
        if "Communication Services" in project_features:
            project_type = "Communication Platform"
        
        # Generate primary purpose based on repository name and features
        primary_purpose = f"Provides {project_type.lower()} functionality"
        if "car" in repository_name.lower():
            primary_purpose = "Automotive service management platform"
        elif "notification" in repository_name.lower():
            primary_purpose = "Communication and notification service platform"
        elif "listing" in repository_name.lower():
            primary_purpose = "Catalog and listing management system"
        
        # Set defaults if nothing detected
        if not project_features:
            project_features = ["Core Application Logic", "Data Processing", "User Interface"]
        if not technical_highlights:
            technical_highlights = ["Clean Code Architecture", "Maintainable Structure"]
        
        return {
            "overview": {
                "repository": repository_name,
                "project_type": project_type,
                "primary_purpose": primary_purpose,
                "technology_focus": f"{', '.join(languages)} development" if languages else "Modern software development",
                "file_count": len(source_files)
            },
            "key_features": project_features,
            "technical_highlights": {
                "architecture": "Structured and maintainable codebase",
                "scalability": "Designed for growth and performance" if technical_highlights else "Standard application structure",
                "maintainability": "Clean code practices" if technical_highlights else "Organized code structure",
                "specific_features": technical_highlights
            },
            "technology_stack": {
                "languages": languages,
                "frameworks": frameworks,
                "total_technologies": len(languages) + len(frameworks)
            },
            "recommendations": [
                "Continue following established patterns",
                "Maintain comprehensive testing" if "Automated Testing" in project_features else "Consider adding automated testing",
                "Keep documentation updated",
                "Monitor performance and security" if "Security Features" in project_features else "Consider enhancing security features"
            ],
            "analysis_summary": {
                "documents_analyzed": len(documents),
                "features_detected": len(project_features),
                "technical_patterns": len(technical_highlights)
            }
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