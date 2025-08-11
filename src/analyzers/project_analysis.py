"""
Project Analysis Components for Generic Q&A.

This module implements various analyzers for detecting project architecture patterns,
business capabilities, API endpoints, data models, and operational concerns.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class BaseAnalyzer(ABC):
    """Base class for project analyzers."""
    
    def __init__(self):
        """Initialize base analyzer."""
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def analyze(self, template: str = "python_fastapi", **kwargs) -> Dict[str, Any]:
        """
        Perform analysis for the given template.
        
        Args:
            template: Project template type
            **kwargs: Additional analysis parameters
            
        Returns:
            Analysis results dictionary
        """
        pass


class ArchitectureDetector(BaseAnalyzer):
    """
    Detector for architecture patterns in projects.
    
    Detects Clean Architecture, MVC, Microservices, and other patterns.
    """
    
    def analyze(self, template: str = "python_fastapi", **kwargs) -> Dict[str, Any]:
        """
        Analyze architecture patterns for the given template.
        
        Args:
            template: Project template type
            **kwargs: Additional analysis parameters
            
        Returns:
            Architecture analysis results
        """
        self.logger.debug(f"Analyzing architecture for template: {template}")
        
        patterns = self._get_template_patterns(template)
        layers = self._get_template_layers(template)
        components = self._get_template_components(template)
        
        return {
            "patterns": patterns,
            "layers": layers,
            "components": components,
            "security_measures": self._get_security_measures(template),
            "observability": self._get_observability_features(template),
        }
    
    def detect_patterns(
        self, 
        project_path: Optional[str] = None,
        repository_url: Optional[str] = None
    ) -> List[str]:
        """
        Detect architecture patterns from project structure.
        
        Args:
            project_path: Local path to project
            repository_url: URL to project repository
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        if project_path and os.path.exists(project_path):
            patterns.extend(self._detect_from_filesystem(project_path))
        
        # Add default patterns based on common structures
        if not patterns:
            patterns = ["layered_architecture", "dependency_injection"]
        
        return patterns
    
    def _get_template_patterns(self, template: str) -> List[str]:
        """Get architecture patterns for template."""
        template_patterns = {
            "python_fastapi": [
                "layered_architecture",
                "dependency_injection",
                "async_processing",
                "api_first"
            ],
            "dotnet_clean_architecture": [
                "clean_architecture",
                "domain_driven_design",
                "cqrs",
                "dependency_inversion"
            ],
            "react_spa": [
                "component_based",
                "unidirectional_data_flow",
                "virtual_dom",
                "single_page_application"
            ]
        }
        return template_patterns.get(template, ["layered_architecture"])
    
    def _get_template_layers(self, template: str) -> List[str]:
        """Get architectural layers for template."""
        template_layers = {
            "python_fastapi": [
                "API Layer (FastAPI routes)",
                "Business Logic Layer (services)",
                "Data Access Layer (repositories)",
                "External Services Layer"
            ],
            "dotnet_clean_architecture": [
                "Presentation Layer (Controllers)",
                "Application Layer (Use Cases)",
                "Domain Layer (Entities)",
                "Infrastructure Layer (Data & External)"
            ],
            "react_spa": [
                "Presentation Layer (Components)",
                "State Management Layer (Redux/Context)",
                "Service Layer (API clients)",
                "Utility Layer (Helpers)"
            ]
        }
        return template_layers.get(template, ["API", "Business", "Data"])
    
    def _get_template_components(self, template: str) -> List[str]:
        """Get key components for template."""
        template_components = {
            "python_fastapi": [
                "FastAPI application",
                "Pydantic models",
                "SQLAlchemy/database models",
                "Dependency injection container",
                "Background tasks",
                "Middleware"
            ],
            "dotnet_clean_architecture": [
                "ASP.NET Core Web API",
                "MediatR handlers",
                "Entity Framework DbContext",
                "Domain entities",
                "Application services",
                "Infrastructure services"
            ],
            "react_spa": [
                "React components",
                "React Router",
                "State management (Redux/Context)",
                "HTTP client (Axios)",
                "UI component library",
                "Build tools (Webpack/Vite)"
            ]
        }
        return template_components.get(template, ["API", "Services", "Models"])
    
    def _get_security_measures(self, template: str) -> List[str]:
        """Get security measures for template."""
        template_security = {
            "python_fastapi": [
                "JWT token authentication",
                "OAuth2 with scopes",
                "CORS handling",
                "Input validation with Pydantic",
                "Rate limiting",
                "HTTPS enforcement"
            ],
            "dotnet_clean_architecture": [
                "JWT Bearer authentication",
                "Authorization policies",
                "Model validation",
                "CORS configuration",
                "Security headers",
                "Data protection"
            ],
            "react_spa": [
                "Token-based authentication",
                "Protected routes",
                "XSS prevention",
                "CSRF protection",
                "Content Security Policy",
                "Secure HTTP headers"
            ]
        }
        return template_security.get(template, ["Authentication", "Authorization"])
    
    def _get_observability_features(self, template: str) -> List[str]:
        """Get observability features for template."""
        template_observability = {
            "python_fastapi": [
                "Structured logging with Loguru",
                "OpenTelemetry tracing",
                "Prometheus metrics",
                "Health check endpoints",
                "Request/response logging",
                "Performance monitoring"
            ],
            "dotnet_clean_architecture": [
                "Structured logging with Serilog",
                "Application Insights",
                "Health checks",
                "Metrics collection",
                "Distributed tracing",
                "Performance counters"
            ],
            "react_spa": [
                "Error boundaries",
                "Performance monitoring",
                "User analytics",
                "Console logging",
                "Network request monitoring",
                "Crash reporting"
            ]
        }
        return template_observability.get(template, ["Logging", "Monitoring"])
    
    def _detect_from_filesystem(self, project_path: str) -> List[str]:
        """Detect patterns from filesystem structure."""
        patterns = []
        
        try:
            # Check for Clean Architecture structure
            if self._has_clean_architecture_structure(project_path):
                patterns.append("clean_architecture")
            
            # Check for MVC structure
            if self._has_mvc_structure(project_path):
                patterns.append("mvc_pattern")
            
            # Check for microservices structure
            if self._has_microservices_structure(project_path):
                patterns.append("microservices")
            
            # Check for layered architecture
            if self._has_layered_structure(project_path):
                patterns.append("layered_architecture")
                
        except Exception as e:
            self.logger.warning(f"Error detecting patterns: {e}")
        
        return patterns
    
    def _has_clean_architecture_structure(self, project_path: str) -> bool:
        """Check for Clean Architecture folder structure."""
        clean_arch_folders = ["Domain", "Application", "Infrastructure", "Presentation"]
        src_path = os.path.join(project_path, "src")
        
        if os.path.exists(src_path):
            existing_folders = [f for f in os.listdir(src_path) 
                             if os.path.isdir(os.path.join(src_path, f))]
            return sum(folder in existing_folders for folder in clean_arch_folders) >= 3
        
        return False
    
    def _has_mvc_structure(self, project_path: str) -> bool:
        """Check for MVC folder structure."""
        mvc_folders = ["Controllers", "Models", "Views"]
        return sum(os.path.exists(os.path.join(project_path, folder)) 
                  for folder in mvc_folders) >= 2
    
    def _has_microservices_structure(self, project_path: str) -> bool:
        """Check for microservices structure."""
        # Look for multiple service directories or docker-compose files
        has_services = os.path.exists(os.path.join(project_path, "services"))
        has_docker_compose = any(f.startswith("docker-compose") 
                               for f in os.listdir(project_path))
        return has_services or has_docker_compose
    
    def _has_layered_structure(self, project_path: str) -> bool:
        """Check for layered architecture structure."""
        layer_indicators = ["api", "services", "repositories", "models", "controllers"]
        src_path = os.path.join(project_path, "src")
        
        if os.path.exists(src_path):
            existing_folders = [f.lower() for f in os.listdir(src_path) 
                             if os.path.isdir(os.path.join(src_path, f))]
            return sum(layer in existing_folders for layer in layer_indicators) >= 2
        
        return False


class BusinessCapabilityAnalyzer(BaseAnalyzer):
    """
    Analyzer for business capabilities and domain entities.
    
    Analyzes business domain scope, core entities, and ownership patterns.
    """
    
    def analyze(self, template: str = "python_fastapi", **kwargs) -> Dict[str, Any]:
        """
        Analyze business capabilities for the given template.
        
        Args:
            template: Project template type
            **kwargs: Additional analysis parameters
            
        Returns:
            Business capability analysis results
        """
        self.logger.debug(f"Analyzing business capabilities for template: {template}")
        
        return {
            "domain_scope": self._get_domain_scope(template),
            "core_entities": self._get_core_entities(template),
            "ownership_model": self._get_ownership_model(template),
            "business_rules": self._get_business_rules(template),
            "sla_requirements": self._get_sla_requirements(template),
        }
    
    def extract_capabilities(self, project_path: Optional[str] = None) -> List[str]:
        """
        Extract business capabilities from project structure.
        
        Args:
            project_path: Local path to project
            
        Returns:
            List of identified capabilities
        """
        capabilities = []
        
        if project_path and os.path.exists(project_path):
            capabilities.extend(self._extract_from_filesystem(project_path))
        
        # Add default capabilities
        if not capabilities:
            capabilities = ["core_business_logic", "data_management", "user_interaction"]
        
        return capabilities
    
    def _get_domain_scope(self, template: str) -> str:
        """Get domain scope description for template."""
        domain_scopes = {
            "python_fastapi": "RESTful API service with specific business functionality",
            "dotnet_clean_architecture": "Enterprise application with well-defined bounded context",
            "react_spa": "Frontend application managing user interactions and workflows"
        }
        return domain_scopes.get(template, "Generic business service")
    
    def _get_core_entities(self, template: str) -> List[str]:
        """Get core entities for template."""
        template_entities = {
            "python_fastapi": [
                "User (ID, email, profile)",
                "Resource (ID, name, metadata)",
                "Request (ID, timestamp, payload)",
                "Response (status, data, errors)"
            ],
            "dotnet_clean_architecture": [
                "Entity (base class with ID)",
                "Aggregate Root (domain boundary)",
                "Value Object (immutable data)",
                "Domain Event (business occurrence)"
            ],
            "react_spa": [
                "User State (authentication, preferences)",
                "Application State (UI, navigation)",
                "Data Models (API responses)",
                "UI Components (reusable elements)"
            ]
        }
        return template_entities.get(template, ["Entity", "User", "Data"])
    
    def _get_ownership_model(self, template: str) -> Dict[str, str]:
        """Get ownership model for template."""
        ownership_models = {
            "python_fastapi": {
                "data_ownership": "Service owns its database schema and data",
                "api_contract": "Service defines and maintains its API contract",
                "business_logic": "Service encapsulates specific business rules",
                "consumers": "Other services, frontend applications, external systems"
            },
            "dotnet_clean_architecture": {
                "domain_ownership": "Domain layer owns business rules and entities",
                "application_ownership": "Application layer owns use cases and workflows",
                "infrastructure_ownership": "Infrastructure layer owns external concerns",
                "consumers": "External systems via well-defined interfaces"
            },
            "react_spa": {
                "ui_ownership": "Application owns user interface and experience",
                "state_ownership": "Application manages client-side state",
                "integration_ownership": "Application handles backend integration",
                "consumers": "End users through web browsers"
            }
        }
        return ownership_models.get(template, {})
    
    def _get_business_rules(self, template: str) -> List[str]:
        """Get business rules for template."""
        template_rules = {
            "python_fastapi": [
                "Input validation using Pydantic models",
                "Business logic validation in service layer",
                "Database constraints for data integrity",
                "API rate limiting for resource protection"
            ],
            "dotnet_clean_architecture": [
                "Domain invariants enforced by entities",
                "Business rules implemented in domain services",
                "Application rules in use case handlers",
                "Validation rules in command/query validators"
            ],
            "react_spa": [
                "Client-side validation for user experience",
                "Form validation rules and error handling",
                "Navigation rules and route protection",
                "State consistency rules for UI components"
            ]
        }
        return template_rules.get(template, ["Validation", "Authorization"])
    
    def _get_sla_requirements(self, template: str) -> Dict[str, str]:
        """Get SLA requirements for template."""
        sla_requirements = {
            "python_fastapi": {
                "response_time": "< 200ms for simple queries, < 2s for complex operations",
                "availability": "99.9% uptime with graceful degradation",
                "throughput": "Handle 1000+ requests per second",
                "consistency": "Strong consistency for critical operations"
            },
            "dotnet_clean_architecture": {
                "response_time": "< 500ms for web requests, < 5s for batch operations",
                "availability": "99.95% uptime with enterprise-grade reliability",
                "throughput": "Support enterprise-level concurrent users",
                "consistency": "ACID compliance for business transactions"
            },
            "react_spa": {
                "response_time": "< 100ms for UI interactions, < 3s for data loading",
                "availability": "Progressive web app with offline capability",
                "throughput": "Smooth performance for concurrent users",
                "consistency": "Optimistic UI updates with conflict resolution"
            }
        }
        return sla_requirements.get(template, {})
    
    def _extract_from_filesystem(self, project_path: str) -> List[str]:
        """Extract capabilities from filesystem structure."""
        capabilities = []
        
        try:
            # Look for domain-specific folders
            domain_folders = self._find_domain_folders(project_path)
            capabilities.extend([f"manage_{folder.lower()}" for folder in domain_folders])
            
            # Look for service patterns
            if self._has_service_pattern(project_path):
                capabilities.append("service_orchestration")
            
            # Look for data patterns
            if self._has_data_access_pattern(project_path):
                capabilities.append("data_management")
                
        except Exception as e:
            self.logger.warning(f"Error extracting capabilities: {e}")
        
        return capabilities
    
    def _find_domain_folders(self, project_path: str) -> List[str]:
        """Find domain-specific folders."""
        domain_indicators = ["users", "orders", "products", "payments", "notifications"]
        found_domains = []
        
        for root, dirs, _ in os.walk(project_path):
            for d in dirs:
                if d.lower() in domain_indicators:
                    found_domains.append(d)
        
        return found_domains
    
    def _has_service_pattern(self, project_path: str) -> bool:
        """Check for service pattern implementation."""
        service_indicators = ["services", "handlers", "managers"]
        return any(os.path.exists(os.path.join(project_path, "src", indicator))
                  for indicator in service_indicators)
    
    def _has_data_access_pattern(self, project_path: str) -> bool:
        """Check for data access pattern implementation."""
        data_indicators = ["repositories", "models", "entities", "database"]
        return any(os.path.exists(os.path.join(project_path, "src", indicator))
                  for indicator in data_indicators)


class APIEndpointAnalyzer(BaseAnalyzer):
    """
    Analyzer for API endpoints and patterns.
    
    Parses API structure, endpoints, status codes, and patterns.
    """
    
    def analyze(self, template: str = "python_fastapi", **kwargs) -> Dict[str, Any]:
        """
        Analyze API endpoints for the given template.
        
        Args:
            template: Project template type
            **kwargs: Additional analysis parameters
            
        Returns:
            API endpoint analysis results
        """
        self.logger.debug(f"Analyzing API endpoints for template: {template}")
        
        return {
            "endpoint_patterns": self._get_endpoint_patterns(template),
            "http_methods": self._get_http_methods(template),
            "status_codes": self._get_status_codes(template),
            "pagination": self._get_pagination_strategy(template),
            "versioning": self._get_versioning_strategy(template),
            "error_handling": self._get_error_handling(template),
        }
    
    def discover_endpoints(self, project_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Discover API endpoints from project structure.
        
        Args:
            project_path: Local path to project
            
        Returns:
            List of discovered endpoints
        """
        endpoints = []
        
        if project_path and os.path.exists(project_path):
            endpoints.extend(self._discover_from_filesystem(project_path))
        
        # Add default endpoint patterns
        if not endpoints:
            endpoints = [
                {"method": "GET", "path": "/health", "description": "Health check"},
                {"method": "GET", "path": "/items", "description": "List items"},
                {"method": "POST", "path": "/items", "description": "Create item"},
                {"method": "GET", "path": "/items/{id}", "description": "Get item"},
                {"method": "PUT", "path": "/items/{id}", "description": "Update item"},
                {"method": "DELETE", "path": "/items/{id}", "description": "Delete item"},
            ]
        
        return endpoints
    
    def _get_endpoint_patterns(self, template: str) -> List[str]:
        """Get endpoint patterns for template."""
        template_patterns = {
            "python_fastapi": [
                "GET /items - List resources with pagination",
                "GET /items/{id} - Retrieve specific resource",
                "POST /items - Create new resource",
                "PUT /items/{id} - Update existing resource",
                "PATCH /items/{id} - Partial update of resource",
                "DELETE /items/{id} - Delete resource",
                "GET /health - Health check endpoint"
            ],
            "dotnet_clean_architecture": [
                "GET /api/v1/entities - List entities",
                "GET /api/v1/entities/{id} - Get entity by ID",
                "POST /api/v1/entities - Create entity",
                "PUT /api/v1/entities/{id} - Update entity",
                "DELETE /api/v1/entities/{id} - Delete entity",
                "GET /health - Health check endpoint"
            ],
            "react_spa": [
                "Integration with backend REST APIs",
                "HTTP client with interceptors",
                "Error handling for API calls",
                "Loading states for async operations",
                "Caching strategies for API responses"
            ]
        }
        return template_patterns.get(template, ["GET", "POST", "PUT", "DELETE"])
    
    def _get_http_methods(self, template: str) -> Dict[str, str]:
        """Get HTTP methods usage for template."""
        http_methods = {
            "GET": "Retrieve data (idempotent, cacheable)",
            "POST": "Create new resources or non-idempotent operations",
            "PUT": "Full resource replacement (idempotent)",
            "PATCH": "Partial resource updates",
            "DELETE": "Remove resources (idempotent)",
            "HEAD": "Retrieve headers only",
            "OPTIONS": "Get allowed methods for CORS"
        }
        
        if template == "react_spa":
            return {k: f"Frontend {v.lower()}" for k, v in http_methods.items()}
        
        return http_methods
    
    def _get_status_codes(self, template: str) -> Dict[str, str]:
        """Get status codes for template."""
        status_codes = {
            "200": "OK - Successful GET, PUT, PATCH",
            "201": "Created - Successful POST with resource creation",
            "204": "No Content - Successful DELETE or PUT without response body",
            "400": "Bad Request - Invalid input or validation errors",
            "401": "Unauthorized - Authentication required",
            "403": "Forbidden - Authorization failed",
            "404": "Not Found - Resource does not exist",
            "409": "Conflict - Resource conflict or constraint violation",
            "422": "Unprocessable Entity - Business rule validation failed",
            "500": "Internal Server Error - Unexpected server error"
        }
        
        if template == "react_spa":
            return {k: f"Handle {v}" for k, v in status_codes.items()}
        
        return status_codes
    
    def _get_pagination_strategy(self, template: str) -> Dict[str, str]:
        """Get pagination strategy for template."""
        pagination_strategies = {
            "python_fastapi": {
                "offset_based": "?page=1&size=20 with total count",
                "cursor_based": "?cursor=abc123&limit=20 for large datasets",
                "link_header": "RFC 5988 Link header for navigation"
            },
            "dotnet_clean_architecture": {
                "page_based": "PagedResult<T> with metadata",
                "odata": "OData $skip and $top parameters",
                "custom_headers": "X-Pagination headers for metadata"
            },
            "react_spa": {
                "infinite_scroll": "Load more data on scroll",
                "page_numbers": "Traditional page-based navigation",
                "load_more": "Button-triggered loading"
            }
        }
        return pagination_strategies.get(template, {})
    
    def _get_versioning_strategy(self, template: str) -> Dict[str, str]:
        """Get versioning strategy for template."""
        versioning_strategies = {
            "python_fastapi": {
                "url_versioning": "/api/v1/items - URL path versioning",
                "header_versioning": "API-Version: v1 header",
                "accept_header": "Accept: application/vnd.api+json;version=1"
            },
            "dotnet_clean_architecture": {
                "url_versioning": "/api/v{version}/controller",
                "query_versioning": "?api-version=1.0",
                "header_versioning": "X-Version header"
            },
            "react_spa": {
                "client_versioning": "Handle different API versions",
                "feature_flags": "Progressive feature rollout",
                "backward_compatibility": "Support multiple API versions"
            }
        }
        return versioning_strategies.get(template, {})
    
    def _get_error_handling(self, template: str) -> Dict[str, str]:
        """Get error handling strategy for template."""
        error_handling = {
            "python_fastapi": {
                "exception_handlers": "FastAPI exception handlers",
                "pydantic_validation": "Automatic request validation",
                "custom_exceptions": "Domain-specific exception classes",
                "error_responses": "Consistent error response format"
            },
            "dotnet_clean_architecture": {
                "global_exception_handler": "Global exception middleware",
                "problem_details": "RFC 7807 Problem Details format",
                "model_validation": "Data annotation validation",
                "custom_exceptions": "Domain and application exceptions"
            },
            "react_spa": {
                "error_boundaries": "React error boundaries",
                "global_error_handler": "Axios interceptors",
                "user_feedback": "User-friendly error messages",
                "retry_logic": "Automatic retry for transient errors"
            }
        }
        return error_handling.get(template, {})
    
    def _discover_from_filesystem(self, project_path: str) -> List[Dict[str, Any]]:
        """Discover endpoints from filesystem structure."""
        endpoints = []
        
        try:
            # Look for route files
            route_files = self._find_route_files(project_path)
            for route_file in route_files:
                endpoints.extend(self._parse_route_file(route_file))
                
        except Exception as e:
            self.logger.warning(f"Error discovering endpoints: {e}")
        
        return endpoints
    
    def _find_route_files(self, project_path: str) -> List[str]:
        """Find route definition files."""
        route_files = []
        route_patterns = [
            "routes.py", "router.py", "controllers.py", "api.py",
            "*Controller.cs", "*Router.js", "*Routes.js"
        ]
        
        for root, _, files in os.walk(project_path):
            for file in files:
                for pattern in route_patterns:
                    if pattern.replace("*", "") in file:
                        route_files.append(os.path.join(root, file))
        
        return route_files
    
    def _parse_route_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse route definitions from file."""
        endpoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Simple regex patterns for common route definitions
            fastapi_pattern = r'@router\.(get|post|put|delete|patch)\("([^"]+)"'
            dotnet_pattern = r'\[Http(Get|Post|Put|Delete|Patch)\("?([^"]*)"?\)\]'
            
            for pattern in [fastapi_pattern, dotnet_pattern]:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    method = match.group(1).upper()
                    path = match.group(2) or "/"
                    endpoints.append({
                        "method": method,
                        "path": path,
                        "file": file_path,
                        "description": f"{method} endpoint at {path}"
                    })
                    
        except Exception as e:
            self.logger.warning(f"Error parsing route file {file_path}: {e}")
        
        return endpoints


class DataModelAnalyzer(BaseAnalyzer):
    """
    Analyzer for data modeling patterns.
    
    Analyzes persistence patterns, repositories, and data modeling approaches.
    """
    
    def analyze(self, template: str = "python_fastapi", **kwargs) -> Dict[str, Any]:
        """
        Analyze data modeling for the given template.
        
        Args:
            template: Project template type
            **kwargs: Additional analysis parameters
            
        Returns:
            Data modeling analysis results
        """
        self.logger.debug(f"Analyzing data modeling for template: {template}")
        
        return {
            "persistence_patterns": self._get_persistence_patterns(template),
            "repository_pattern": self._get_repository_pattern(template),
            "transaction_strategy": self._get_transaction_strategy(template),
            "validation_approach": self._get_validation_approach(template),
            "migration_strategy": self._get_migration_strategy(template),
        }
    
    def analyze_models(self, project_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze data models from project structure.
        
        Args:
            project_path: Local path to project
            
        Returns:
            List of analyzed models
        """
        models = []
        
        if project_path and os.path.exists(project_path):
            models.extend(self._analyze_from_filesystem(project_path))
        
        # Add default model examples
        if not models:
            models = [
                {"name": "BaseEntity", "type": "abstract", "description": "Base class for entities"},
                {"name": "User", "type": "entity", "description": "User entity"},
                {"name": "UserDto", "type": "dto", "description": "User data transfer object"},
            ]
        
        return models
    
    def _get_persistence_patterns(self, template: str) -> Dict[str, str]:
        """Get persistence patterns for template."""
        persistence_patterns = {
            "python_fastapi": {
                "orm": "SQLAlchemy ORM with declarative models",
                "database": "PostgreSQL/MySQL with connection pooling",
                "migrations": "Alembic for database schema migrations",
                "abstractions": "Repository pattern with interface abstractions"
            },
            "dotnet_clean_architecture": {
                "orm": "Entity Framework Core with Code First approach",
                "database": "SQL Server/PostgreSQL with connection resiliency",
                "migrations": "EF Core migrations with automatic deployment",
                "abstractions": "Repository and Unit of Work patterns"
            },
            "react_spa": {
                "storage": "Local Storage and Session Storage",
                "state": "Redux/Context API for application state",
                "caching": "HTTP caching and service worker storage",
                "persistence": "API integration for server-side persistence"
            }
        }
        return persistence_patterns.get(template, {})
    
    def _get_repository_pattern(self, template: str) -> Dict[str, str]:
        """Get repository pattern implementation for template."""
        repository_patterns = {
            "python_fastapi": {
                "interface": "Abstract base repository with generic operations",
                "implementation": "SQLAlchemy-based repository implementations",
                "dependency_injection": "FastAPI dependency injection for repositories",
                "unit_of_work": "Transaction management with Unit of Work pattern"
            },
            "dotnet_clean_architecture": {
                "interface": "IRepository<T> interface in Domain layer",
                "implementation": "EF Core repository in Infrastructure layer",
                "dependency_injection": "Built-in DI container registration",
                "unit_of_work": "IUnitOfWork interface with DbContext as UoW"
            },
            "react_spa": {
                "data_services": "API service classes for data operations",
                "caching": "Query result caching and invalidation",
                "state_management": "Centralized state with actions and reducers",
                "optimistic_updates": "Optimistic UI updates with rollback"
            }
        }
        return repository_patterns.get(template, {})
    
    def _get_transaction_strategy(self, template: str) -> Dict[str, str]:
        """Get transaction strategy for template."""
        transaction_strategies = {
            "python_fastapi": {
                "database_transactions": "SQLAlchemy session-based transactions",
                "distributed_transactions": "Saga pattern for cross-service operations",
                "error_handling": "Automatic rollback on exceptions",
                "isolation_levels": "Configurable isolation levels per operation"
            },
            "dotnet_clean_architecture": {
                "database_transactions": "EF Core transaction management",
                "unit_of_work": "Single transaction per request/command",
                "distributed_transactions": "Outbox pattern for eventual consistency",
                "concurrency": "Optimistic concurrency with row versioning"
            },
            "react_spa": {
                "optimistic_updates": "Update UI immediately, handle conflicts",
                "error_recovery": "Rollback UI state on API errors",
                "batch_operations": "Group related operations together",
                "offline_support": "Queue operations when offline"
            }
        }
        return transaction_strategies.get(template, {})
    
    def _get_validation_approach(self, template: str) -> Dict[str, str]:
        """Get validation approach for template."""
        validation_approaches = {
            "python_fastapi": {
                "input_validation": "Pydantic models for request validation",
                "business_validation": "Domain service validation rules",
                "database_validation": "Database constraints and triggers",
                "error_handling": "Structured validation error responses"
            },
            "dotnet_clean_architecture": {
                "input_validation": "Data annotations and FluentValidation",
                "domain_validation": "Domain entity invariants and rules",
                "application_validation": "Command/query validators",
                "error_handling": "ValidationProblemDetails responses"
            },
            "react_spa": {
                "form_validation": "Client-side form validation libraries",
                "real_time_validation": "As-you-type validation feedback",
                "server_validation": "Handle server validation errors",
                "user_experience": "Accessible error messages and indicators"
            }
        }
        return validation_approaches.get(template, {})
    
    def _get_migration_strategy(self, template: str) -> Dict[str, str]:
        """Get migration strategy for template."""
        migration_strategies = {
            "python_fastapi": {
                "schema_migrations": "Alembic migration scripts",
                "data_migrations": "Custom migration scripts for data changes",
                "deployment": "Automated migrations in CI/CD pipeline",
                "rollback": "Safe rollback procedures for failed migrations"
            },
            "dotnet_clean_architecture": {
                "schema_migrations": "EF Core Add-Migration command",
                "data_migrations": "Custom migration classes for data seeding",
                "deployment": "Database update in deployment pipeline",
                "environments": "Environment-specific migration configurations"
            },
            "react_spa": {
                "version_migrations": "Handle API version changes",
                "data_migrations": "Transform cached data between versions",
                "deployment": "Progressive deployment with feature flags",
                "fallback": "Graceful degradation for unsupported features"
            }
        }
        return migration_strategies.get(template, {})
    
    def _analyze_from_filesystem(self, project_path: str) -> List[Dict[str, Any]]:
        """Analyze models from filesystem structure."""
        models = []
        
        try:
            # Look for model files
            model_files = self._find_model_files(project_path)
            for model_file in model_files:
                models.extend(self._parse_model_file(model_file))
                
        except Exception as e:
            self.logger.warning(f"Error analyzing models: {e}")
        
        return models
    
    def _find_model_files(self, project_path: str) -> List[str]:
        """Find model definition files."""
        model_files = []
        model_patterns = ["models.py", "entities.py", "schemas.py", "Models.cs", "Entities.cs"]
        
        for root, _, files in os.walk(project_path):
            for file in files:
                if any(pattern in file for pattern in model_patterns):
                    model_files.append(os.path.join(root, file))
        
        return model_files
    
    def _parse_model_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse model definitions from file."""
        models = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple regex patterns for model definitions
            python_class_pattern = r'class (\w+)\(.*?\):'
            csharp_class_pattern = r'public class (\w+)'
            
            for pattern in [python_class_pattern, csharp_class_pattern]:
                matches = re.finditer(pattern, content)
                for match in matches:
                    class_name = match.group(1)
                    models.append({
                        "name": class_name,
                        "type": "model",
                        "file": file_path,
                        "description": f"Model class {class_name}"
                    })
                    
        except Exception as e:
            self.logger.warning(f"Error parsing model file {file_path}: {e}")
        
        return models


class OperationalAnalyzer(BaseAnalyzer):
    """
    Analyzer for operational concerns.
    
    Analyzes deployment, monitoring, and operational patterns.
    """
    
    def analyze(self, template: str = "python_fastapi", **kwargs) -> Dict[str, Any]:
        """
        Analyze operational concerns for the given template.
        
        Args:
            template: Project template type
            **kwargs: Additional analysis parameters
            
        Returns:
            Operational analysis results
        """
        self.logger.debug(f"Analyzing operational concerns for template: {template}")
        
        return {
            "deployment_strategy": self._get_deployment_strategy(template),
            "monitoring_approach": self._get_monitoring_approach(template),
            "scaling_strategy": self._get_scaling_strategy(template),
            "security_measures": self._get_security_measures(template),
            "disaster_recovery": self._get_disaster_recovery(template),
        }
    
    def analyze_operations(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze operational patterns from project structure.
        
        Args:
            project_path: Local path to project
            
        Returns:
            Operational analysis results
        """
        operations = {}
        
        if project_path and os.path.exists(project_path):
            operations.update(self._analyze_from_filesystem(project_path))
        
        # Add default operational patterns
        if not operations:
            operations = {
                "containerization": "Docker support detected",
                "health_checks": "Health check endpoints available",
                "logging": "Structured logging implemented"
            }
        
        return operations
    
    def _get_deployment_strategy(self, template: str) -> Dict[str, str]:
        """Get deployment strategy for template."""
        deployment_strategies = {
            "python_fastapi": {
                "containerization": "Docker with multi-stage builds",
                "orchestration": "Kubernetes or Docker Compose",
                "ci_cd": "GitHub Actions or GitLab CI for automated deployment",
                "environments": "Development, staging, and production environments"
            },
            "dotnet_clean_architecture": {
                "containerization": "Docker with ASP.NET Core runtime",
                "cloud_deployment": "Azure App Service or AWS ECS",
                "ci_cd": "Azure DevOps or GitHub Actions",
                "configuration": "Environment-specific appsettings.json"
            },
            "react_spa": {
                "static_hosting": "CDN deployment (Netlify, Vercel, S3)",
                "build_process": "Webpack/Vite build optimization",
                "ci_cd": "Automated build and deployment pipeline",
                "environments": "Environment-specific configuration files"
            }
        }
        return deployment_strategies.get(template, {})
    
    def _get_monitoring_approach(self, template: str) -> Dict[str, str]:
        """Get monitoring approach for template."""
        monitoring_approaches = {
            "python_fastapi": {
                "application_monitoring": "Prometheus metrics with Grafana dashboards",
                "logging": "Structured logging with ELK stack",
                "tracing": "OpenTelemetry distributed tracing",
                "health_checks": "Health check endpoints for load balancers"
            },
            "dotnet_clean_architecture": {
                "application_monitoring": "Application Insights telemetry",
                "logging": "Serilog with structured logging",
                "performance": ".NET performance counters",
                "health_checks": "ASP.NET Core health checks"
            },
            "react_spa": {
                "user_monitoring": "Real User Monitoring (RUM)",
                "error_tracking": "Error boundary reporting",
                "performance": "Core Web Vitals monitoring",
                "analytics": "User behavior analytics"
            }
        }
        return monitoring_approaches.get(template, {})
    
    def _get_scaling_strategy(self, template: str) -> Dict[str, str]:
        """Get scaling strategy for template."""
        scaling_strategies = {
            "python_fastapi": {
                "horizontal_scaling": "Multiple application instances with load balancing",
                "auto_scaling": "Kubernetes HPA based on CPU/memory metrics",
                "database_scaling": "Read replicas and connection pooling",
                "caching": "Redis for application-level caching"
            },
            "dotnet_clean_architecture": {
                "horizontal_scaling": "Azure App Service scale-out",
                "auto_scaling": "Auto-scaling rules based on metrics",
                "database_scaling": "Azure SQL Database scaling",
                "caching": "Azure Redis Cache for distributed caching"
            },
            "react_spa": {
                "cdn_scaling": "Global CDN distribution",
                "lazy_loading": "Code splitting and lazy loading",
                "caching": "Browser caching and service workers",
                "performance": "Bundle optimization and compression"
            }
        }
        return scaling_strategies.get(template, {})
    
    def _get_security_measures(self, template: str) -> Dict[str, str]:
        """Get security measures for template."""
        security_measures = {
            "python_fastapi": {
                "authentication": "JWT token-based authentication",
                "authorization": "Role-based access control (RBAC)",
                "input_validation": "Pydantic model validation",
                "security_headers": "CORS, CSP, and security headers"
            },
            "dotnet_clean_architecture": {
                "authentication": "JWT Bearer or Identity Server integration",
                "authorization": "Policy-based authorization",
                "input_validation": "Model validation and sanitization",
                "security_scanning": "Static analysis and vulnerability scanning"
            },
            "react_spa": {
                "authentication": "Token-based authentication with refresh",
                "xss_protection": "Content Security Policy and sanitization",
                "csrf_protection": "CSRF tokens for state-changing operations",
                "secure_communication": "HTTPS and secure cookie handling"
            }
        }
        return security_measures.get(template, {})
    
    def _get_disaster_recovery(self, template: str) -> Dict[str, str]:
        """Get disaster recovery strategy for template."""
        disaster_recovery = {
            "python_fastapi": {
                "backup_strategy": "Automated database backups with point-in-time recovery",
                "replication": "Multi-region database replication",
                "failover": "Automatic failover with health checks",
                "monitoring": "Alerting for system failures and degradation"
            },
            "dotnet_clean_architecture": {
                "backup_strategy": "Azure Backup for databases and storage",
                "geo_redundancy": "Geo-redundant storage and compute",
                "failover": "Azure Traffic Manager for traffic routing",
                "recovery_testing": "Regular disaster recovery testing"
            },
            "react_spa": {
                "cdn_redundancy": "Multi-CDN setup for availability",
                "fallback_strategies": "Graceful degradation for service failures",
                "offline_support": "Service workers for offline functionality",
                "monitoring": "Uptime monitoring and alerting"
            }
        }
        return disaster_recovery.get(template, {})
    
    def _analyze_from_filesystem(self, project_path: str) -> Dict[str, Any]:
        """Analyze operational patterns from filesystem."""
        operations = {}
        
        try:
            # Check for Docker
            if os.path.exists(os.path.join(project_path, "Dockerfile")):
                operations["containerization"] = "Docker configuration detected"
            
            # Check for CI/CD
            if os.path.exists(os.path.join(project_path, ".github", "workflows")):
                operations["ci_cd"] = "GitHub Actions workflows detected"
            
            # Check for monitoring
            if os.path.exists(os.path.join(project_path, "prometheus.yml")):
                operations["monitoring"] = "Prometheus configuration detected"
            
            # Check for health checks
            health_indicators = ["health.py", "health.cs", "health.js"]
            if any(self._find_files_by_pattern(project_path, pattern) 
                  for pattern in health_indicators):
                operations["health_checks"] = "Health check implementation detected"
                
        except Exception as e:
            self.logger.warning(f"Error analyzing operations: {e}")
        
        return operations
    
    def _find_files_by_pattern(self, project_path: str, pattern: str) -> List[str]:
        """Find files matching pattern."""
        found_files = []
        
        for root, _, files in os.walk(project_path):
            for file in files:
                if pattern in file:
                    found_files.append(os.path.join(root, file))
        
        return found_files