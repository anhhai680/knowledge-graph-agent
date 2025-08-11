"""
Architecture Detector for Generic Project Q&A.

This module provides architecture pattern detection for different project types,
including Clean Architecture, MVC, Microservices, and other common patterns.
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class ArchitecturePattern(str, Enum):
    """Architecture pattern enumeration."""
    
    CLEAN_ARCHITECTURE = "clean_architecture"
    MVC = "mvc"
    MICROSERVICES = "microservices"
    LAYERED = "layered"
    EVENT_DRIVEN = "event_driven"
    HEXAGONAL = "hexagonal"
    MODULAR_MONOLITH = "modular_monolith"
    GENERIC = "generic"


@dataclass
class ArchitectureAnalysis:
    """Architecture analysis result."""
    
    primary_pattern: ArchitecturePattern
    secondary_patterns: List[ArchitecturePattern]
    confidence_score: float
    detected_layers: List[str]
    project_structure: Dict[str, List[str]]
    technologies: List[str]
    characteristics: Dict[str, Any]


class ArchitectureDetector:
    """
    Architecture pattern detector for project analysis.
    
    This component analyzes project structure, file patterns, and naming conventions
    to detect architectural patterns and provide insights for Generic Q&A responses.
    """
    
    def __init__(self):
        """Initialize architecture detector."""
        self.logger = get_logger(self.__class__.__name__)
        self._init_pattern_signatures()
    
    def _init_pattern_signatures(self) -> None:
        """Initialize architecture pattern signatures."""
        
        # Clean Architecture signatures
        self.clean_architecture_signatures = {
            "folder_patterns": [
                "domain", "application", "infrastructure", "presentation",
                "core", "api", "web", "entities", "usecases", "interfaces"
            ],
            "file_patterns": [
                "entity", "repository", "service", "usecase", "dto", "mapper",
                "controller", "handler", "command", "query"
            ],
            "technologies": [".net", "spring", "nodejs", "fastapi"],
            "weight": 1.0
        }
        
        # MVC signatures
        self.mvc_signatures = {
            "folder_patterns": [
                "models", "views", "controllers", "mvc", "app", "components"
            ],
            "file_patterns": [
                "controller", "model", "view", "component", "action"
            ],
            "technologies": ["asp.net", "rails", "django", "laravel", "express"],
            "weight": 0.8
        }
        
        # Microservices signatures
        self.microservices_signatures = {
            "folder_patterns": [
                "services", "microservices", "service-", "gateway", "discovery",
                "config-service", "user-service", "order-service"
            ],
            "file_patterns": [
                "service", "gateway", "client", "proxy", "discovery"
            ],
            "technologies": ["docker", "kubernetes", "api-gateway", "service-mesh"],
            "weight": 0.9
        }
        
        # Layered Architecture signatures
        self.layered_signatures = {
            "folder_patterns": [
                "business", "data", "presentation", "service", "dal", "bll",
                "ui", "logic", "persistence", "repository"
            ],
            "file_patterns": [
                "layer", "tier", "business", "data", "service"
            ],
            "technologies": ["enterprise", "n-tier"],
            "weight": 0.7
        }
        
        # Event-driven signatures
        self.event_driven_signatures = {
            "folder_patterns": [
                "events", "handlers", "publishers", "subscribers", "messaging",
                "queue", "eventbus", "sagas", "event-store"
            ],
            "file_patterns": [
                "event", "handler", "publisher", "subscriber", "saga", "command"
            ],
            "technologies": ["event-sourcing", "cqrs", "message-broker"],
            "weight": 0.8
        }
        
        # Hexagonal Architecture signatures
        self.hexagonal_signatures = {
            "folder_patterns": [
                "adapters", "ports", "hexagon", "primary", "secondary",
                "driven", "driving"
            ],
            "file_patterns": [
                "adapter", "port", "primary", "secondary", "driven", "driving"
            ],
            "technologies": ["ports-adapters", "hexagonal"],
            "weight": 0.9
        }
        
        # Modular Monolith signatures
        self.modular_monolith_signatures = {
            "folder_patterns": [
                "modules", "bounded-contexts", "contexts", "features",
                "user-module", "order-module", "payment-module"
            ],
            "file_patterns": [
                "module", "context", "feature", "boundary"
            ],
            "technologies": ["modular", "bounded-context"],
            "weight": 0.8
        }
    
    def detect_architecture(
        self, 
        project_path: str,
        file_patterns: Optional[List[str]] = None
    ) -> ArchitectureAnalysis:
        """
        Detect architecture pattern from project structure.
        
        Args:
            project_path: Path to project directory
            file_patterns: Optional list of file patterns to analyze
            
        Returns:
            ArchitectureAnalysis with detected patterns and metadata
        """
        try:
            self.logger.info(f"Detecting architecture for project: {project_path}")
            
            # Analyze project structure
            project_structure = self._analyze_project_structure(
                project_path, file_patterns
            )
            
            # Detect patterns
            pattern_scores = self._calculate_pattern_scores(project_structure)
            
            # Determine primary and secondary patterns
            primary_pattern, secondary_patterns = self._rank_patterns(pattern_scores)
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(pattern_scores, primary_pattern)
            
            # Detect layers and technologies
            detected_layers = self._detect_layers(project_structure)
            if not detected_layers:  # If no layers detected, indicate unknown
                detected_layers = ["unknown"]
            technologies = self._detect_technologies(project_structure)
            
            # Extract characteristics
            characteristics = self._extract_characteristics(
                project_structure, primary_pattern
            )
            
            analysis = ArchitectureAnalysis(
                primary_pattern=primary_pattern,
                secondary_patterns=secondary_patterns,
                confidence_score=confidence_score,
                detected_layers=detected_layers,
                project_structure=project_structure,
                technologies=technologies,
                characteristics=characteristics
            )
            
            self.logger.info(
                f"Architecture detection completed. Primary pattern: {primary_pattern}, "
                f"Confidence: {confidence_score:.2f}"
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error detecting architecture: {e}")
            # Return generic analysis as fallback
            return self._create_fallback_analysis(project_path)
    
    def _analyze_project_structure(
        self, 
        project_path: str,
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """Analyze project directory structure."""
        try:
            project_root = Path(project_path)
            if not project_root.exists():
                self.logger.warning(f"Project path does not exist: {project_path}")
                return {"folders": [], "files": [], "technologies": []}
            
            folders = []
            files = []
            technologies = set()
            
            # Walk through directory structure (limit depth for performance)
            max_depth = 3
            for path in project_root.rglob("*"):
                # Calculate depth relative to project root
                depth = len(path.relative_to(project_root).parts)
                if depth > max_depth:
                    continue
                
                if path.is_dir():
                    folder_name = path.name.lower()
                    folders.append(folder_name)
                elif path.is_file():
                    file_name = path.name.lower()
                    files.append(file_name)
                    
                    # Detect technologies from file extensions and names
                    self._detect_file_technologies(file_name, technologies)
            
            return {
                "folders": folders,
                "files": files,
                "technologies": list(technologies)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing project structure: {e}")
            return {"folders": [], "files": [], "technologies": []}
    
    def _detect_file_technologies(self, file_name: str, technologies: Set[str]) -> None:
        """Detect technologies from file names and extensions."""
        # Technology detection patterns
        tech_patterns = {
            ".cs": ".net",
            ".csproj": ".net",
            ".sln": ".net",
            ".py": "python",
            "requirements.txt": "python",
            "pyproject.toml": "python",
            ".js": "javascript",
            ".jsx": "react",
            ".ts": "typescript",
            ".tsx": "react",
            "package.json": "nodejs",
            ".java": "java",
            "pom.xml": "maven",
            "build.gradle": "gradle",
            ".php": "php",
            "composer.json": "composer",
            ".rb": "ruby",
            "gemfile": "ruby",
            ".go": "golang",
            "go.mod": "golang",
            "dockerfile": "docker",
            "docker-compose": "docker",
            ".yaml": "kubernetes",
            ".yml": "kubernetes",
            "helm": "helm"
        }
        
        for pattern, tech in tech_patterns.items():
            if pattern in file_name:
                technologies.add(tech)
    
    def _calculate_pattern_scores(
        self, 
        project_structure: Dict[str, List[str]]
    ) -> Dict[ArchitecturePattern, float]:
        """Calculate scores for each architecture pattern."""
        folders = project_structure.get("folders", [])
        files = project_structure.get("files", [])
        technologies = project_structure.get("technologies", [])
        
        pattern_scores = {}
        
        # Score each pattern
        pattern_signatures = {
            ArchitecturePattern.CLEAN_ARCHITECTURE: self.clean_architecture_signatures,
            ArchitecturePattern.MVC: self.mvc_signatures,
            ArchitecturePattern.MICROSERVICES: self.microservices_signatures,
            ArchitecturePattern.LAYERED: self.layered_signatures,
            ArchitecturePattern.EVENT_DRIVEN: self.event_driven_signatures,
            ArchitecturePattern.HEXAGONAL: self.hexagonal_signatures,
            ArchitecturePattern.MODULAR_MONOLITH: self.modular_monolith_signatures,
        }
        
        for pattern, signatures in pattern_signatures.items():
            score = 0.0
            
            # Score folder patterns
            folder_matches = sum(
                1 for folder_pattern in signatures["folder_patterns"]
                if any(folder_pattern in folder for folder in folders)
            )
            score += folder_matches * 0.4
            
            # Score file patterns
            file_matches = sum(
                1 for file_pattern in signatures["file_patterns"]
                if any(file_pattern in file for file in files)
            )
            score += file_matches * 0.3
            
            # Score technology alignment
            tech_matches = sum(
                1 for tech_pattern in signatures["technologies"]
                if any(tech_pattern in tech for tech in technologies)
            )
            score += tech_matches * 0.3
            
            # Apply pattern weight
            final_score = score * signatures["weight"]
            pattern_scores[pattern] = final_score
        
        return pattern_scores
    
    def _rank_patterns(
        self, 
        pattern_scores: Dict[ArchitecturePattern, float]
    ) -> tuple[ArchitecturePattern, List[ArchitecturePattern]]:
        """Rank patterns by score and determine primary/secondary."""
        if not pattern_scores:
            return ArchitecturePattern.GENERIC, []
        
        # Sort patterns by score
        sorted_patterns = sorted(
            pattern_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Primary pattern (highest score) - require minimum threshold
        primary_pattern = (
            sorted_patterns[0][0] if sorted_patterns[0][1] > 1.0 
            else ArchitecturePattern.GENERIC
        )
        
        # Secondary patterns (score > 1.0 and not primary)
        secondary_patterns = [
            pattern for pattern, score in sorted_patterns[1:]
            if score > 1.0
        ]
        
        return primary_pattern, secondary_patterns[:3]  # Limit to top 3 secondary
    
    def _calculate_confidence(
        self, 
        pattern_scores: Dict[ArchitecturePattern, float],
        primary_pattern: ArchitecturePattern
    ) -> float:
        """Calculate confidence score for the detection."""
        if primary_pattern == ArchitecturePattern.GENERIC:
            return 0.1
        
        primary_score = pattern_scores.get(primary_pattern, 0.0)
        total_score = sum(pattern_scores.values())
        
        if total_score == 0:
            return 0.1
        
        # Confidence based on primary pattern dominance
        confidence = primary_score / total_score
        
        # Boost confidence if primary score is high
        if primary_score > 3.0:
            confidence = min(confidence + 0.2, 1.0)
        
        return round(confidence, 2)
    
    def _detect_layers(self, project_structure: Dict[str, List[str]]) -> List[str]:
        """Detect architectural layers from project structure."""
        folders = project_structure.get("folders", [])
        
        layer_patterns = {
            "presentation": ["controllers", "api", "web", "ui", "views", "presentation"],
            "application": ["application", "services", "handlers", "usecases", "business"],
            "domain": ["domain", "entities", "models", "core"],
            "infrastructure": ["infrastructure", "data", "persistence", "repository", "dal"],
            "configuration": ["config", "configuration", "settings"],
            "shared": ["shared", "common", "utilities", "helpers"]
        }
        
        detected_layers = []
        for layer, patterns in layer_patterns.items():
            if any(pattern in folder for pattern in patterns for folder in folders):
                detected_layers.append(layer)
        
        return detected_layers
    
    def _detect_technologies(self, project_structure: Dict[str, List[str]]) -> List[str]:
        """Detect technologies used in the project."""
        return ensure_list(project_structure.get("technologies", []))
    
    def _extract_characteristics(
        self, 
        project_structure: Dict[str, List[str]],
        primary_pattern: ArchitecturePattern
    ) -> Dict[str, Any]:
        """Extract architectural characteristics."""
        folders = project_structure.get("folders", [])
        files = project_structure.get("files", [])
        
        characteristics = {
            "separation_of_concerns": len(folders) > 3,
            "modular_structure": any("module" in folder for folder in folders),
            "test_coverage": any("test" in folder or "test" in file for folder in folders for file in files),
            "configuration_management": any("config" in folder for folder in folders),
            "dependency_injection": False,  # Would need code analysis
            "event_handling": any("event" in folder or "handler" in folder for folder in folders),
            "api_design": any("api" in folder or "controller" in folder for folder in folders),
            "data_access": any("repository" in folder or "data" in folder for folder in folders),
        }
        
        # Pattern-specific characteristics
        if primary_pattern == ArchitecturePattern.CLEAN_ARCHITECTURE:
            characteristics.update({
                "clean_architecture_compliance": True,
                "dependency_inversion": True,
                "testability": True
            })
        elif primary_pattern == ArchitecturePattern.MICROSERVICES:
            characteristics.update({
                "service_oriented": True,
                "distributed_system": True,
                "scalability_focus": True
            })
        
        return characteristics
    
    def _create_fallback_analysis(self, project_path: str) -> ArchitectureAnalysis:
        """Create fallback analysis when detection fails."""
        return ArchitectureAnalysis(
            primary_pattern=ArchitecturePattern.GENERIC,
            secondary_patterns=[],
            confidence_score=0.1,
            detected_layers=["unknown"],
            project_structure={"folders": [], "files": [], "technologies": []},
            technologies=[],
            characteristics={"unknown_structure": True}
        )
    
    def get_pattern_description(self, pattern: ArchitecturePattern) -> str:
        """Get description for an architecture pattern."""
        descriptions = {
            ArchitecturePattern.CLEAN_ARCHITECTURE: "Clean Architecture with Domain, Application, Infrastructure, and Presentation layers",
            ArchitecturePattern.MVC: "Model-View-Controller pattern with clear separation between data, presentation, and control logic",
            ArchitecturePattern.MICROSERVICES: "Distributed architecture with independent, loosely coupled services",
            ArchitecturePattern.LAYERED: "Traditional layered architecture with horizontal separation of concerns",
            ArchitecturePattern.EVENT_DRIVEN: "Event-driven architecture with asynchronous communication and loose coupling",
            ArchitecturePattern.HEXAGONAL: "Ports and Adapters architecture isolating core business logic",
            ArchitecturePattern.MODULAR_MONOLITH: "Monolithic application with modular design and bounded contexts",
            ArchitecturePattern.GENERIC: "Generic or unidentified architecture pattern"
        }
        
        return descriptions.get(pattern, "Unknown architecture pattern")
    
    def get_supported_patterns(self) -> List[Dict[str, Any]]:
        """Get list of supported architecture patterns."""
        patterns = []
        
        for pattern in ArchitecturePattern:
            patterns.append({
                "id": pattern.value,
                "name": pattern.value.replace("_", " ").title(),
                "description": self.get_pattern_description(pattern),
                "detection_confidence": "high" if pattern != ArchitecturePattern.GENERIC else "low"
            })
        
        return patterns