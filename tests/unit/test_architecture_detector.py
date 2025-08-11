"""
Unit tests for ArchitectureDetector.

Tests the architecture detection functionality including pattern recognition,
confidence scoring, and technology detection.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.analyzers.architecture_detector import (
    ArchitectureDetector,
    ArchitecturePattern,
    ArchitectureAnalysis
)


class TestArchitectureDetector:
    """Test cases for ArchitectureDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create ArchitectureDetector instance for testing."""
        return ArchitectureDetector()
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_architecture_pattern_values(self):
        """Test architecture pattern enum values."""
        assert ArchitecturePattern.CLEAN_ARCHITECTURE == "clean_architecture"
        assert ArchitecturePattern.MVC == "mvc"
        assert ArchitecturePattern.MICROSERVICES == "microservices"
        assert ArchitecturePattern.LAYERED == "layered"
        assert ArchitecturePattern.EVENT_DRIVEN == "event_driven"
        assert ArchitecturePattern.HEXAGONAL == "hexagonal"
        assert ArchitecturePattern.MODULAR_MONOLITH == "modular_monolith"
        assert ArchitecturePattern.GENERIC == "generic"
    
    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.clean_architecture_signatures is not None
        assert detector.mvc_signatures is not None
        assert detector.microservices_signatures is not None
        assert len(detector.clean_architecture_signatures["folder_patterns"]) > 0
        assert len(detector.mvc_signatures["folder_patterns"]) > 0
    
    def test_analyze_project_structure_empty_dir(self, detector, temp_project_dir):
        """Test project structure analysis with empty directory."""
        structure = detector._analyze_project_structure(temp_project_dir)
        
        assert "folders" in structure
        assert "files" in structure
        assert "technologies" in structure
        assert len(structure["folders"]) == 0
        assert len(structure["files"]) == 0
    
    def test_analyze_project_structure_with_files(self, detector, temp_project_dir):
        """Test project structure analysis with files."""
        # Create test structure
        os.makedirs(os.path.join(temp_project_dir, "domain"))
        os.makedirs(os.path.join(temp_project_dir, "application"))
        
        # Create test files
        open(os.path.join(temp_project_dir, "test.cs"), 'w').close()
        open(os.path.join(temp_project_dir, "requirements.txt"), 'w').close()
        
        structure = detector._analyze_project_structure(temp_project_dir)
        
        assert "domain" in structure["folders"]
        assert "application" in structure["folders"]
        assert "test.cs" in structure["files"]
        assert "requirements.txt" in structure["files"]
        assert ".net" in structure["technologies"]
        assert "python" in structure["technologies"]
    
    def test_detect_file_technologies(self, detector):
        """Test technology detection from file names."""
        technologies = set()
        
        detector._detect_file_technologies("test.cs", technologies)
        assert ".net" in technologies
        
        detector._detect_file_technologies("package.json", technologies)
        assert "nodejs" in technologies
        
        detector._detect_file_technologies("requirements.txt", technologies)
        assert "python" in technologies
        
        detector._detect_file_technologies("dockerfile", technologies)
        assert "docker" in technologies
    
    def test_calculate_pattern_scores_clean_architecture(self, detector):
        """Test pattern scoring for Clean Architecture."""
        project_structure = {
            "folders": ["domain", "application", "infrastructure", "presentation"],
            "files": ["entity.cs", "repository.cs", "controller.cs"],
            "technologies": [".net", "docker"]
        }
        
        scores = detector._calculate_pattern_scores(project_structure)
        
        assert ArchitecturePattern.CLEAN_ARCHITECTURE in scores
        clean_arch_score = scores[ArchitecturePattern.CLEAN_ARCHITECTURE]
        assert clean_arch_score > 0
        
        # Clean architecture should score highly with this structure
        assert clean_arch_score > scores.get(ArchitecturePattern.MVC, 0)
    
    def test_calculate_pattern_scores_mvc(self, detector):
        """Test pattern scoring for MVC."""
        project_structure = {
            "folders": ["models", "views", "controllers"],
            "files": ["usercontroller.cs", "usermodel.cs", "user.view"],
            "technologies": ["asp.net"]
        }
        
        scores = detector._calculate_pattern_scores(project_structure)
        
        assert ArchitecturePattern.MVC in scores
        mvc_score = scores[ArchitecturePattern.MVC]
        assert mvc_score > 0
    
    def test_calculate_pattern_scores_microservices(self, detector):
        """Test pattern scoring for Microservices."""
        project_structure = {
            "folders": ["user-service", "order-service", "gateway"],
            "files": ["service.py", "gateway.py", "client.py"],
            "technologies": ["docker", "kubernetes"]
        }
        
        scores = detector._calculate_pattern_scores(project_structure)
        
        assert ArchitecturePattern.MICROSERVICES in scores
        microservices_score = scores[ArchitecturePattern.MICROSERVICES]
        assert microservices_score > 0
    
    def test_rank_patterns(self, detector):
        """Test pattern ranking."""
        pattern_scores = {
            ArchitecturePattern.CLEAN_ARCHITECTURE: 3.5,
            ArchitecturePattern.MVC: 1.2,
            ArchitecturePattern.MICROSERVICES: 0.8,
            ArchitecturePattern.LAYERED: 2.1
        }
        
        primary, secondary = detector._rank_patterns(pattern_scores)
        
        assert primary == ArchitecturePattern.CLEAN_ARCHITECTURE
        assert ArchitecturePattern.LAYERED in secondary
        assert ArchitecturePattern.MVC in secondary
        assert len(secondary) <= 3
    
    def test_rank_patterns_empty_scores(self, detector):
        """Test pattern ranking with empty scores."""
        pattern_scores = {}
        
        primary, secondary = detector._rank_patterns(pattern_scores)
        
        assert primary == ArchitecturePattern.GENERIC
        assert len(secondary) == 0
    
    def test_rank_patterns_no_significant_scores(self, detector):
        """Test pattern ranking with low scores."""
        pattern_scores = {
            ArchitecturePattern.CLEAN_ARCHITECTURE: 0.5,
            ArchitecturePattern.MVC: 0.2
        }
        
        primary, secondary = detector._rank_patterns(pattern_scores)
        
        assert primary == ArchitecturePattern.GENERIC
        assert len(secondary) == 0
    
    def test_calculate_confidence_high(self, detector):
        """Test confidence calculation with high primary score."""
        pattern_scores = {
            ArchitecturePattern.CLEAN_ARCHITECTURE: 4.0,
            ArchitecturePattern.MVC: 1.0,
            ArchitecturePattern.LAYERED: 0.5
        }
        
        confidence = detector._calculate_confidence(
            pattern_scores, 
            ArchitecturePattern.CLEAN_ARCHITECTURE
        )
        
        assert confidence > 0.7
        assert confidence <= 1.0
    
    def test_calculate_confidence_generic(self, detector):
        """Test confidence calculation for generic pattern."""
        pattern_scores = {}
        
        confidence = detector._calculate_confidence(
            pattern_scores, 
            ArchitecturePattern.GENERIC
        )
        
        assert confidence == 0.1
    
    def test_detect_layers(self, detector):
        """Test layer detection."""
        project_structure = {
            "folders": ["controllers", "services", "domain", "data", "config"],
            "files": [],
            "technologies": []
        }
        
        layers = detector._detect_layers(project_structure)
        
        assert "presentation" in layers
        assert "application" in layers
        assert "domain" in layers
        assert "infrastructure" in layers
        assert "configuration" in layers
    
    def test_detect_technologies(self, detector):
        """Test technology detection."""
        project_structure = {
            "folders": [],
            "files": [],
            "technologies": [".net", "docker", "python"]
        }
        
        technologies = detector._detect_technologies(project_structure)
        
        assert technologies == [".net", "docker", "python"]
    
    def test_extract_characteristics(self, detector):
        """Test characteristic extraction."""
        project_structure = {
            "folders": ["domain", "application", "tests", "config"],
            "files": ["test.cs", "handler.cs"],
            "technologies": []
        }
        
        characteristics = detector._extract_characteristics(
            project_structure, 
            ArchitecturePattern.CLEAN_ARCHITECTURE
        )
        
        assert characteristics["separation_of_concerns"] is True
        assert characteristics["test_coverage"] is True
        assert characteristics["configuration_management"] is True
        assert characteristics["clean_architecture_compliance"] is True
        assert characteristics["dependency_inversion"] is True
    
    def test_extract_characteristics_microservices(self, detector):
        """Test characteristic extraction for microservices."""
        project_structure = {
            "folders": ["user-service", "order-service"],
            "files": [],
            "technologies": []
        }
        
        characteristics = detector._extract_characteristics(
            project_structure, 
            ArchitecturePattern.MICROSERVICES
        )
        
        assert characteristics["service_oriented"] is True
        assert characteristics["distributed_system"] is True
        assert characteristics["scalability_focus"] is True
    
    def test_get_pattern_description(self, detector):
        """Test pattern description retrieval."""
        desc = detector.get_pattern_description(ArchitecturePattern.CLEAN_ARCHITECTURE)
        assert "Clean Architecture" in desc
        assert "Domain" in desc
        
        desc = detector.get_pattern_description(ArchitecturePattern.MVC)
        assert "Model-View-Controller" in desc
        
        desc = detector.get_pattern_description(ArchitecturePattern.MICROSERVICES)
        assert "Distributed architecture" in desc or "microservices" in desc.lower()
    
    def test_get_supported_patterns(self, detector):
        """Test supported patterns listing."""
        patterns = detector.get_supported_patterns()
        
        assert len(patterns) == len(ArchitecturePattern)
        assert all("id" in pattern for pattern in patterns)
        assert all("name" in pattern for pattern in patterns)
        assert all("description" in pattern for pattern in patterns)
        assert all("detection_confidence" in pattern for pattern in patterns)
    
    def test_detect_architecture_nonexistent_path(self, detector):
        """Test architecture detection with nonexistent path."""
        analysis = detector.detect_architecture("/nonexistent/path")
        
        assert analysis.primary_pattern == ArchitecturePattern.GENERIC
        assert analysis.confidence_score == 0.1
        assert len(analysis.detected_layers) == 1
        assert analysis.detected_layers[0] == "unknown"
    
    @patch('src.analyzers.architecture_detector.Path')
    def test_detect_architecture_with_mocked_structure(self, mock_path, detector):
        """Test architecture detection with mocked project structure."""
        # Mock the path to simulate a real project
        mock_project_path = MagicMock()
        mock_path.return_value = mock_project_path
        mock_project_path.exists.return_value = True
        
        # Mock the _analyze_project_structure method
        detector._analyze_project_structure = Mock(return_value={
            "folders": ["domain", "application", "infrastructure", "api"],
            "files": ["entity.cs", "repository.cs", "controller.cs"],
            "technologies": [".net", "docker"]
        })
        
        analysis = detector.detect_architecture("/test/path")
        
        assert analysis.primary_pattern == ArchitecturePattern.CLEAN_ARCHITECTURE
        assert analysis.confidence_score > 0.5
        assert ".net" in analysis.technologies
        assert "docker" in analysis.technologies
        assert len(analysis.detected_layers) > 0
    
    def test_create_fallback_analysis(self, detector):
        """Test fallback analysis creation."""
        analysis = detector._create_fallback_analysis("/test/path")
        
        assert analysis.primary_pattern == ArchitecturePattern.GENERIC
        assert analysis.confidence_score == 0.1
        assert analysis.detected_layers == ["unknown"]
        assert analysis.technologies == []
        assert "unknown_structure" in analysis.characteristics
    
    def test_architecture_analysis_dataclass(self):
        """Test ArchitectureAnalysis dataclass creation."""
        analysis = ArchitectureAnalysis(
            primary_pattern=ArchitecturePattern.CLEAN_ARCHITECTURE,
            secondary_patterns=[ArchitecturePattern.MVC],
            confidence_score=0.85,
            detected_layers=["domain", "application"],
            project_structure={"folders": ["test"]},
            technologies=[".net"],
            characteristics={"test": True}
        )
        
        assert analysis.primary_pattern == ArchitecturePattern.CLEAN_ARCHITECTURE
        assert len(analysis.secondary_patterns) == 1
        assert analysis.confidence_score == 0.85
        assert "domain" in analysis.detected_layers
        assert ".net" in analysis.technologies


if __name__ == "__main__":
    pytest.main([__file__])