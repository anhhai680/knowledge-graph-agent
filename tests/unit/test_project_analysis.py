"""
Unit tests for Project Analysis Components.

This module tests the various project analyzers including ArchitectureDetector,
BusinessCapabilityAnalyzer, APIEndpointAnalyzer, DataModelAnalyzer, and OperationalAnalyzer.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, mock_open

from src.analyzers.project_analysis import (
    ArchitectureDetector,
    BusinessCapabilityAnalyzer,
    APIEndpointAnalyzer,
    DataModelAnalyzer,
    OperationalAnalyzer,
)


class TestArchitectureDetector:
    """Test cases for ArchitectureDetector."""

    @pytest.fixture
    def detector(self):
        """Create an ArchitectureDetector instance for testing."""
        return ArchitectureDetector()

    def test_analyze_python_fastapi(self, detector):
        """Test architecture analysis for Python FastAPI template."""
        result = detector.analyze(template="python_fastapi")
        
        assert "patterns" in result
        assert "layers" in result
        assert "components" in result
        assert "security_measures" in result
        assert "observability" in result
        
        assert "layered_architecture" in result["patterns"]
        assert "API Layer (FastAPI routes)" in result["layers"]
        assert "FastAPI application" in result["components"]

    def test_analyze_dotnet_clean_architecture(self, detector):
        """Test architecture analysis for .NET Clean Architecture template."""
        result = detector.analyze(template="dotnet_clean_architecture")
        
        assert "clean_architecture" in result["patterns"]
        assert "domain_driven_design" in result["patterns"]
        assert "Presentation Layer (Controllers)" in result["layers"]
        assert "ASP.NET Core Web API" in result["components"]

    def test_analyze_react_spa(self, detector):
        """Test architecture analysis for React SPA template."""
        result = detector.analyze(template="react_spa")
        
        assert "component_based" in result["patterns"]
        assert "Presentation Layer (Components)" in result["layers"]
        assert "React components" in result["components"]

    def test_detect_patterns_from_filesystem(self, detector):
        """Test pattern detection from filesystem structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test directory structure
            src_dir = os.path.join(temp_dir, "src")
            os.makedirs(src_dir)
            
            # Create directories that indicate layered architecture
            os.makedirs(os.path.join(src_dir, "api"))
            os.makedirs(os.path.join(src_dir, "services"))
            os.makedirs(os.path.join(src_dir, "repositories"))
            
            patterns = detector.detect_patterns(project_path=temp_dir)
            
            assert "layered_architecture" in patterns

    def test_detect_clean_architecture_structure(self, detector):
        """Test detection of Clean Architecture structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create Clean Architecture structure
            src_dir = os.path.join(temp_dir, "src")
            os.makedirs(os.path.join(src_dir, "Domain"))
            os.makedirs(os.path.join(src_dir, "Application"))
            os.makedirs(os.path.join(src_dir, "Infrastructure"))
            os.makedirs(os.path.join(src_dir, "Presentation"))
            
            has_clean_arch = detector._has_clean_architecture_structure(temp_dir)
            
            assert has_clean_arch is True

    def test_detect_mvc_structure(self, detector):
        """Test detection of MVC structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create MVC structure
            os.makedirs(os.path.join(temp_dir, "Controllers"))
            os.makedirs(os.path.join(temp_dir, "Models"))
            os.makedirs(os.path.join(temp_dir, "Views"))
            
            has_mvc = detector._has_mvc_structure(temp_dir)
            
            assert has_mvc is True

    def test_detect_microservices_structure(self, detector):
        """Test detection of microservices structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create microservices structure
            os.makedirs(os.path.join(temp_dir, "services"))
            
            # Create docker-compose file
            with open(os.path.join(temp_dir, "docker-compose.yml"), "w") as f:
                f.write("version: '3.8'\n")
            
            has_microservices = detector._has_microservices_structure(temp_dir)
            
            assert has_microservices is True


class TestBusinessCapabilityAnalyzer:
    """Test cases for BusinessCapabilityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a BusinessCapabilityAnalyzer instance for testing."""
        return BusinessCapabilityAnalyzer()

    def test_analyze_python_fastapi(self, analyzer):
        """Test business capability analysis for Python FastAPI template."""
        result = analyzer.analyze(template="python_fastapi")
        
        assert "domain_scope" in result
        assert "core_entities" in result
        assert "ownership_model" in result
        assert "business_rules" in result
        assert "sla_requirements" in result
        
        assert "RESTful API service" in result["domain_scope"]
        assert len(result["core_entities"]) > 0
        assert "data_ownership" in result["ownership_model"]

    def test_analyze_dotnet_clean_architecture(self, analyzer):
        """Test business capability analysis for .NET Clean Architecture template."""
        result = analyzer.analyze(template="dotnet_clean_architecture")
        
        assert "Enterprise application" in result["domain_scope"]
        assert "Entity (base class with ID)" in result["core_entities"]
        assert "domain_ownership" in result["ownership_model"]

    def test_extract_capabilities_from_filesystem(self, analyzer):
        """Test capability extraction from filesystem structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create domain-specific directories
            src_dir = os.path.join(temp_dir, "src")
            os.makedirs(os.path.join(src_dir, "users"))
            os.makedirs(os.path.join(src_dir, "orders"))
            os.makedirs(os.path.join(src_dir, "services"))
            
            capabilities = analyzer.extract_capabilities(project_path=temp_dir)
            
            assert "manage_users" in capabilities
            assert "manage_orders" in capabilities
            assert "service_orchestration" in capabilities

    def test_has_service_pattern(self, analyzer):
        """Test service pattern detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = os.path.join(temp_dir, "src")
            os.makedirs(os.path.join(src_dir, "services"))
            
            has_service = analyzer._has_service_pattern(temp_dir)
            
            assert has_service is True

    def test_has_data_access_pattern(self, analyzer):
        """Test data access pattern detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = os.path.join(temp_dir, "src")
            os.makedirs(os.path.join(src_dir, "repositories"))
            
            has_data_access = analyzer._has_data_access_pattern(temp_dir)
            
            assert has_data_access is True


class TestAPIEndpointAnalyzer:
    """Test cases for APIEndpointAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create an APIEndpointAnalyzer instance for testing."""
        return APIEndpointAnalyzer()

    def test_analyze_python_fastapi(self, analyzer):
        """Test API endpoint analysis for Python FastAPI template."""
        result = analyzer.analyze(template="python_fastapi")
        
        assert "endpoint_patterns" in result
        assert "http_methods" in result
        assert "status_codes" in result
        assert "pagination" in result
        assert "versioning" in result
        assert "error_handling" in result
        
        assert "GET /items - List resources" in result["endpoint_patterns"][0]
        assert "GET" in result["http_methods"]
        assert "200" in result["status_codes"]

    def test_analyze_react_spa(self, analyzer):
        """Test API endpoint analysis for React SPA template."""
        result = analyzer.analyze(template="react_spa")
        
        assert "Integration with backend REST APIs" in result["endpoint_patterns"]
        assert "Frontend" in result["http_methods"]["GET"]

    def test_discover_endpoints_from_filesystem(self, analyzer):
        """Test endpoint discovery from filesystem structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a route file
            route_file = os.path.join(temp_dir, "routes.py")
            route_content = '''
@router.get("/users")
def get_users():
    pass

@router.post("/users")
def create_user():
    pass
'''
            with open(route_file, "w") as f:
                f.write(route_content)
            
            endpoints = analyzer.discover_endpoints(project_path=temp_dir)
            
            # Should find the endpoints from the file
            methods = [ep["method"] for ep in endpoints]
            paths = [ep["path"] for ep in endpoints]
            
            assert "GET" in methods
            assert "POST" in methods
            assert "/users" in paths

    def test_parse_route_file_fastapi(self, analyzer):
        """Test parsing FastAPI route file."""
        content = '''
@router.get("/items")
def get_items():
    pass

@router.post("/items/{item_id}")
def update_item(item_id: int):
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                endpoints = analyzer._parse_route_file(f.name)
                
                assert len(endpoints) == 2
                assert endpoints[0]["method"] == "GET"
                assert endpoints[0]["path"] == "/items"
                assert endpoints[1]["method"] == "POST"
                assert endpoints[1]["path"] == "/items/{item_id}"
            finally:
                os.unlink(f.name)

    def test_parse_route_file_dotnet(self, analyzer):
        """Test parsing .NET route file."""
        content = '''
[HttpGet("users")]
public ActionResult GetUsers()
{
    return Ok();
}

[HttpPost("users/{id}")]
public ActionResult UpdateUser(int id)
{
    return Ok();
}
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cs', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                endpoints = analyzer._parse_route_file(f.name)
                
                assert len(endpoints) == 2
                assert endpoints[0]["method"] == "GET"
                assert endpoints[0]["path"] == "users"
                assert endpoints[1]["method"] == "POST"
                assert endpoints[1]["path"] == "users/{id}"
            finally:
                os.unlink(f.name)


class TestDataModelAnalyzer:
    """Test cases for DataModelAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a DataModelAnalyzer instance for testing."""
        return DataModelAnalyzer()

    def test_analyze_python_fastapi(self, analyzer):
        """Test data model analysis for Python FastAPI template."""
        result = analyzer.analyze(template="python_fastapi")
        
        assert "persistence_patterns" in result
        assert "repository_pattern" in result
        assert "transaction_strategy" in result
        assert "validation_approach" in result
        assert "migration_strategy" in result
        
        assert "SQLAlchemy ORM" in result["persistence_patterns"]["orm"]
        assert "Repository pattern" in result["repository_pattern"]["interface"]

    def test_analyze_models_from_filesystem(self, analyzer):
        """Test model analysis from filesystem structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a model file
            model_file = os.path.join(temp_dir, "models.py")
            model_content = '''
class User(BaseModel):
    id: int
    name: str

class Order(BaseModel):
    id: int
    user_id: int
'''
            with open(model_file, "w") as f:
                f.write(model_content)
            
            models = analyzer.analyze_models(project_path=temp_dir)
            
            model_names = [model["name"] for model in models]
            assert "User" in model_names
            assert "Order" in model_names

    def test_parse_model_file_python(self, analyzer):
        """Test parsing Python model file."""
        content = '''
class UserModel(BaseModel):
    pass

class OrderModel(BaseModel):
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                models = analyzer._parse_model_file(f.name)
                
                assert len(models) == 2
                assert models[0]["name"] == "UserModel"
                assert models[1]["name"] == "OrderModel"
            finally:
                os.unlink(f.name)

    def test_parse_model_file_csharp(self, analyzer):
        """Test parsing C# model file."""
        content = '''
public class UserEntity
{
    public int Id { get; set; }
}

public class OrderEntity
{
    public int Id { get; set; }
}
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cs', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                models = analyzer._parse_model_file(f.name)
                
                assert len(models) == 2
                assert models[0]["name"] == "UserEntity"
                assert models[1]["name"] == "OrderEntity"
            finally:
                os.unlink(f.name)


class TestOperationalAnalyzer:
    """Test cases for OperationalAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create an OperationalAnalyzer instance for testing."""
        return OperationalAnalyzer()

    def test_analyze_python_fastapi(self, analyzer):
        """Test operational analysis for Python FastAPI template."""
        result = analyzer.analyze(template="python_fastapi")
        
        assert "deployment_strategy" in result
        assert "monitoring_approach" in result
        assert "scaling_strategy" in result
        assert "security_measures" in result
        assert "disaster_recovery" in result
        
        assert "Docker with multi-stage builds" in result["deployment_strategy"]["containerization"]
        assert "Prometheus metrics" in result["monitoring_approach"]["application_monitoring"]

    def test_analyze_operations_from_filesystem(self, analyzer):
        """Test operational analysis from filesystem structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create Dockerfile
            with open(os.path.join(temp_dir, "Dockerfile"), "w") as f:
                f.write("FROM python:3.9\n")
            
            # Create GitHub Actions
            actions_dir = os.path.join(temp_dir, ".github", "workflows")
            os.makedirs(actions_dir)
            with open(os.path.join(actions_dir, "ci.yml"), "w") as f:
                f.write("name: CI\n")
            
            # Create health check file
            with open(os.path.join(temp_dir, "health.py"), "w") as f:
                f.write("def health_check(): pass\n")
            
            operations = analyzer.analyze_operations(project_path=temp_dir)
            
            assert "containerization" in operations
            assert "ci_cd" in operations
            assert "health_checks" in operations

    def test_find_files_by_pattern(self, analyzer):
        """Test finding files by pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            with open(os.path.join(temp_dir, "health.py"), "w") as f:
                f.write("# health check\n")
            
            with open(os.path.join(temp_dir, "other.py"), "w") as f:
                f.write("# other file\n")
            
            found_files = analyzer._find_files_by_pattern(temp_dir, "health")
            
            assert len(found_files) == 1
            assert "health.py" in found_files[0]


class TestBaseAnalyzer:
    """Test cases for BaseAnalyzer abstract class."""

    def test_base_analyzer_cannot_be_instantiated(self):
        """Test that BaseAnalyzer cannot be instantiated directly."""
        from src.analyzers.project_analysis import BaseAnalyzer
        
        with pytest.raises(TypeError):
            BaseAnalyzer()

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        detector = ArchitectureDetector()
        
        assert detector.logger is not None
        assert hasattr(detector, 'analyze')

    def test_all_analyzers_implement_analyze(self):
        """Test that all analyzers implement the analyze method."""
        analyzers = [
            ArchitectureDetector(),
            BusinessCapabilityAnalyzer(),
            APIEndpointAnalyzer(),
            DataModelAnalyzer(),
            OperationalAnalyzer(),
        ]
        
        for analyzer in analyzers:
            assert hasattr(analyzer, 'analyze')
            assert callable(analyzer.analyze)
            
            # Test that analyze method returns a dictionary
            result = analyzer.analyze()
            assert isinstance(result, dict)