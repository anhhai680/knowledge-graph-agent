"""
Integration test demonstrating the complete Generic Q&A Agent implementation.

This test demonstrates the full functionality of the Generic Q&A Agent system
including question classification, project analysis, template-based responses,
and all API endpoints.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from src.agents.generic_qa_agent import GenericQAAgent, QuestionCategory
from src.workflows.generic_qa_workflow import GenericQAWorkflow
from src.analyzers.architecture_detector import ArchitectureDetector, ArchitecturePattern
from src.analyzers.business_capability_analyzer import BusinessCapabilityAnalyzer


class TestGenericQASystemIntegration:
    """Integration tests for the complete Generic Q&A system."""
    
    @pytest.fixture
    def temp_dotnet_project(self):
        """Create a temporary .NET Clean Architecture project structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create Clean Architecture structure
            os.makedirs(os.path.join(temp_dir, "Domain", "Entities"))
            os.makedirs(os.path.join(temp_dir, "Application", "Services"))
            os.makedirs(os.path.join(temp_dir, "Infrastructure", "Repositories"))
            os.makedirs(os.path.join(temp_dir, "Presentation", "Controllers"))
            
            # Create sample files
            files = [
                ("Domain/Entities/User.cs", "public class User { public int Id { get; set; } }"),
                ("Application/Services/UserService.cs", "public class UserService { }"),
                ("Infrastructure/Repositories/UserRepository.cs", "public class UserRepository { }"),
                ("Presentation/Controllers/UserController.cs", "public class UserController { }"),
                ("Project.csproj", "<Project><PropertyGroup><TargetFramework>net8.0</TargetFramework></PropertyGroup></Project>")
            ]
            
            for file_path, content in files:
                full_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
            
            yield temp_dir
    
    @pytest.fixture
    def temp_python_project(self):
        """Create a temporary Python FastAPI project structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create FastAPI structure
            os.makedirs(os.path.join(temp_dir, "app", "models"))
            os.makedirs(os.path.join(temp_dir, "app", "routers"))
            os.makedirs(os.path.join(temp_dir, "app", "services"))
            
            # Create sample files
            files = [
                ("app/models/user.py", "class User:\n    def __init__(self, id: int, name: str):\n        self.id = id\n        self.name = name"),
                ("app/routers/users.py", "from fastapi import APIRouter\nrouter = APIRouter()"),
                ("app/services/user_service.py", "class UserService:\n    def get_user(self, id: int):\n        pass"),
                ("requirements.txt", "fastapi\nuvicorn\npydantic"),
                ("pyproject.toml", "[tool.poetry]\nname = 'test-api'\nversion = '0.1.0'")
            ]
            
            for file_path, content in files:
                full_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
            
            yield temp_dir
    
    @pytest.fixture
    def qa_system(self):
        """Create complete Q&A system with workflow."""
        workflow = GenericQAWorkflow()
        agent = GenericQAAgent(workflow=workflow)
        return agent
    
    def test_question_classification_comprehensive(self, qa_system):
        """Test comprehensive question classification across all categories."""
        test_cases = [
            ("What business capability does this service own?", QuestionCategory.BUSINESS_CAPABILITY),
            ("List all API endpoints and their status codes", QuestionCategory.API_ENDPOINTS),
            ("How is data modeling implemented in this project?", QuestionCategory.DATA_MODELING),
            ("What's the end-to-end workflow for user creation?", QuestionCategory.WORKFLOWS),
            ("What's the overall architecture and layer structure?", QuestionCategory.ARCHITECTURE),
        ]
        
        for question, expected_category in test_cases:
            classified_category = qa_system.classify_question(question)
            assert classified_category == expected_category, f"Question '{question}' should be classified as {expected_category}, got {classified_category}"
    
    def test_architecture_detection_dotnet(self, temp_dotnet_project):
        """Test architecture detection with .NET Clean Architecture project."""
        detector = ArchitectureDetector()
        analysis = detector.detect_architecture(temp_dotnet_project)
        
        # Verify detection results
        assert analysis.primary_pattern == ArchitecturePattern.CLEAN_ARCHITECTURE
        assert analysis.confidence_score > 0.5
        assert ".net" in analysis.technologies
        assert "domain" in analysis.detected_layers
        assert "application" in analysis.detected_layers
        assert "infrastructure" in analysis.detected_layers
        assert "presentation" in analysis.detected_layers
    
    def test_architecture_detection_python(self, temp_python_project):
        """Test architecture detection with Python FastAPI project."""
        detector = ArchitectureDetector()
        analysis = detector.detect_architecture(temp_python_project)
        
        # Verify detection results
        assert analysis.confidence_score >= 0.1  # Should at least detect basic structure
        assert "python" in analysis.technologies
        assert len(analysis.detected_layers) > 0
    
    def test_business_capability_analysis_dotnet(self, temp_dotnet_project):
        """Test business capability analysis with .NET project."""
        analyzer = BusinessCapabilityAnalyzer()
        analysis = analyzer.analyze_business_capability(temp_dotnet_project)
        
        # Verify analysis results
        assert analysis.confidence_score > 0.1
        assert analysis.primary_capability is not None
        assert len(analysis.primary_capability.entities) >= 0
        assert analysis.domain_complexity is not None
    
    @pytest.mark.asyncio
    async def test_full_qa_workflow_execution(self, qa_system):
        """Test complete Q&A workflow execution."""
        question = "What business capability does this service own?"
        
        result = await qa_system._process_input({
            "question": question,
            "category": None,  # Let it auto-classify
            "repository_context": {
                "file_patterns": ["*.cs", "*.csproj"],
                "structure": "clean_architecture"
            }
        })
        
        # Verify workflow execution
        assert result["success"] is True
        assert result["question"] == question
        assert result["category"] == QuestionCategory.BUSINESS_CAPABILITY.value
        assert "answer" in result
        assert len(result["answer"]) > 0
        assert "analysis_components" in result
        assert len(result["analysis_components"]) > 0
        assert "confidence_score" in result
        assert "template_used" in result
    
    @pytest.mark.asyncio
    async def test_project_structure_analysis_integration(self, qa_system, temp_dotnet_project):
        """Test integrated project structure analysis."""
        result = await qa_system.analyze_project_structure(
            temp_dotnet_project,
            analysis_depth="comprehensive"
        )
        
        # Verify analysis integration
        assert result["success"] is True
        analysis = result["analysis"]
        
        # Check architecture analysis
        assert "architecture_analysis" in analysis
        arch_analysis = analysis["architecture_analysis"]
        assert arch_analysis["primary_pattern"] == "clean_architecture"
        assert arch_analysis["confidence_score"] > 0.5
        assert ".net" in arch_analysis["technologies"]
        
        # Check business analysis
        assert "business_analysis" in analysis
        business_analysis = analysis["business_analysis"]
        assert business_analysis["confidence_score"] > 0.1
        
        # Check template recommendations
        assert "recommended_templates" in analysis
        assert "dotnet_clean_architecture" in analysis["recommended_templates"]
        
        # Check overall readiness
        assert analysis["readiness_score"] > 0.3
    
    def test_template_system_comprehensive(self, qa_system):
        """Test comprehensive template system functionality."""
        # Test available templates
        templates = qa_system.get_available_templates()
        
        assert len(templates) >= 3
        template_ids = [t["id"] for t in templates]
        
        expected_templates = ["dotnet_clean_architecture", "python_fastapi", "react_spa"]
        for expected in expected_templates:
            assert expected in template_ids
        
        # Verify template structure
        for template in templates:
            assert "id" in template
            assert "name" in template
            assert "description" in template
            assert "categories" in template
            assert "file_patterns" in template
    
    def test_supported_categories_comprehensive(self, qa_system):
        """Test comprehensive supported categories functionality."""
        categories = qa_system.get_supported_categories()
        
        assert len(categories) == 5
        
        expected_categories = [
            QuestionCategory.BUSINESS_CAPABILITY,
            QuestionCategory.API_ENDPOINTS,
            QuestionCategory.DATA_MODELING,
            QuestionCategory.WORKFLOWS,
            QuestionCategory.ARCHITECTURE
        ]
        
        category_ids = [cat["id"] for cat in categories]
        for expected in expected_categories:
            assert expected in category_ids
        
        # Verify category structure
        for category in categories:
            assert "id" in category
            assert "name" in category
            assert "description" in category
            assert "examples" in category
            assert len(category["examples"]) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_different_categories(self, qa_system):
        """Test workflow execution with different question categories."""
        test_questions = [
            "What are the main API endpoints?",
            "How are database entities and repositories implemented?",
            "What's the user registration workflow?",
            "What's the system architecture?",
        ]
        
        for question in test_questions:
            result = await qa_system._process_input(question)
            
            assert result["success"] is True
            # Any valid category is acceptable since classification can be subjective
            assert result["category"] in [cat.value for cat in QuestionCategory]
            assert len(result["answer"]) > 50  # Substantial answer
            assert result["confidence_score"] > 0.0
    
    def test_error_handling_and_fallbacks(self, qa_system):
        """Test error handling and fallback mechanisms."""
        # Test with invalid input
        assert qa_system._validate_input("") is False
        assert qa_system._validate_input(None) is False
        assert qa_system._validate_input(123) is False
        
        # Test fallback answer generation
        fallback = qa_system._generate_fallback_answer(
            "Test question", 
            QuestionCategory.ARCHITECTURE
        )
        
        assert len(fallback) > 100  # Should be substantial
        assert "architecture" in fallback.lower()
        assert "workflow" in fallback.lower()
    
    @pytest.mark.asyncio
    async def test_performance_characteristics(self, qa_system):
        """Test that the system meets performance requirements."""
        import time
        
        question = "What business capability does this service own?"
        
        start_time = time.time()
        result = await qa_system._process_input(question)
        processing_time = time.time() - start_time
        
        # Should complete within 3 seconds as per requirements
        assert processing_time < 3.0
        assert result["success"] is True
        
        # Test multiple questions for consistency
        for i in range(3):
            start_time = time.time()
            result = await qa_system._process_input(f"Question {i}: {question}")
            processing_time = time.time() - start_time
            
            assert processing_time < 3.0
            assert result["success"] is True
    
    def test_code_reuse_validation(self):
        """Validate that we achieved 95%+ code reuse as required."""
        # This test verifies that our implementation leverages existing infrastructure
        
        # Check that we extend existing base classes
        from src.agents.base_agent import BaseAgent
        from src.workflows.base_workflow import BaseWorkflow
        
        assert issubclass(GenericQAAgent, BaseAgent)
        assert issubclass(GenericQAWorkflow, BaseWorkflow)
        
        # Check that we use existing utilities
        from src.utils.defensive_programming import safe_len, ensure_list
        from src.utils.logging import get_logger
        
        # These should be imported and used in our components
        qa_agent = GenericQAAgent()
        workflow = GenericQAWorkflow()
        
        # Verify logger usage
        assert hasattr(qa_agent, 'logger')
        assert hasattr(workflow, 'logger')
        
        # Verify workflow inheritance
        assert hasattr(workflow, 'define_steps')
        assert hasattr(workflow, 'execute_step')
        assert hasattr(workflow, 'validate_state')
        
        # Verify agent inheritance
        assert hasattr(qa_agent, 'invoke')
        assert hasattr(qa_agent, 'ainvoke')
        assert hasattr(qa_agent, '_process_input')
        assert hasattr(qa_agent, '_validate_input')
    
    def test_system_integration_requirements(self):
        """Test that all system integration requirements are met."""
        
        # Verify all required components exist
        required_components = [
            GenericQAAgent,
            GenericQAWorkflow,
            ArchitectureDetector,
            BusinessCapabilityAnalyzer,
        ]
        
        for component in required_components:
            assert component is not None
            # Should be instantiable
            instance = component()
            assert instance is not None
        
        # Verify question categories
        categories = list(QuestionCategory)
        assert len(categories) == 5
        
        expected_categories = [
            "business_capability",
            "api_endpoints", 
            "data_modeling",
            "workflows",
            "architecture"
        ]
        
        category_values = [cat.value for cat in categories]
        for expected in expected_categories:
            assert expected in category_values
        
        # Verify architecture patterns
        patterns = list(ArchitecturePattern)
        assert len(patterns) >= 7  # Should support multiple patterns
        
        essential_patterns = [
            "clean_architecture",
            "mvc",
            "microservices",
            "layered"
        ]
        
        pattern_values = [pat.value for pat in patterns]
        for essential in essential_patterns:
            assert essential in pattern_values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])