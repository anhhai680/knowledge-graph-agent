"""
Architecture Detector for Generic Q&A Agent.

This module detects and analyzes architecture patterns in project repositories
following the EventFlowAnalyzer pattern detection methodology from the existing
event_flow_analyzer.py module.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class ArchitecturePattern(str, Enum):
    """Architecture pattern enumeration similar to WorkflowPattern in event_flow_analyzer.py."""
    CLEAN_ARCHITECTURE = "clean_architecture"
    MVC = "mvc"
    MICROSERVICES = "microservices"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"
    ONION = "onion"
    MVP = "mvp"
    MVVM = "mvvm"
    EVENT_DRIVEN = "event_driven"
    SERVERLESS = "serverless"
    MONOLITHIC = "monolithic"
    UNKNOWN = "unknown"


@dataclass
class ProjectStructure:
    """
    Project structure information similar to EventFlowQuery dataclass.
    
    Contains detected architecture patterns, technology stack information,
    and structural analysis results.
    """
    architecture_type: ArchitecturePattern
    technology_stack: List[str]
    layers: List[str]
    components: List[str]
    frameworks: List[str]
    confidence_score: float
    structural_indicators: Dict[str, Any]
    directory_structure: List[str]
    key_files: List[str]


class ArchitectureDetector:
    """
    Architecture detector using pattern detection methodology from EventFlowAnalyzer.
    
    This class analyzes repository structure, file patterns, and technology indicators
    to detect architectural patterns and provide structured analysis results.
    """

    def __init__(self):
        """Initialize architecture detector with pattern definitions."""
        self.logger = get_logger(self.__class__.__name__)  # REUSE logging pattern
        
        # REUSE keyword-based pattern detection from EventFlowAnalyzer
        self._architecture_patterns = self._initialize_architecture_patterns()
        self._technology_indicators = self._initialize_technology_indicators()
        self._framework_indicators = self._initialize_framework_indicators()

    def _initialize_architecture_patterns(self) -> Dict[ArchitecturePattern, Dict[str, Any]]:
        """
        Initialize architecture patterns following EventFlowAnalyzer._workflow_patterns approach.
        
        Returns:
            Dictionary mapping architecture patterns to detection criteria
        """
        return {
            ArchitecturePattern.CLEAN_ARCHITECTURE: {
                "directory_patterns": [
                    "domain", "application", "infrastructure", "presentation",
                    "core", "business", "data", "api", "web"
                ],
                "file_patterns": [
                    "entity", "service", "repository", "controller", "dto",
                    "interface", "usecase", "aggregate"
                ],
                "structure_indicators": ["layers", "separation_of_concerns", "dependency_inversion"]
            },
            ArchitecturePattern.MVC: {
                "directory_patterns": ["models", "views", "controllers", "mvc"],
                "file_patterns": ["model", "view", "controller", "action"],
                "structure_indicators": ["model_view_controller", "separation"]
            },
            ArchitecturePattern.MICROSERVICES: {
                "directory_patterns": ["services", "microservices", "service"],
                "file_patterns": ["service", "gateway", "discovery", "config"],
                "structure_indicators": ["distributed", "service_oriented", "api_gateway"]
            },
            ArchitecturePattern.LAYERED: {
                "directory_patterns": ["layers", "tier", "level"],
                "file_patterns": ["layer", "tier"],
                "structure_indicators": ["layered", "hierarchical", "n_tier"]
            },
            ArchitecturePattern.HEXAGONAL: {
                "directory_patterns": ["adapters", "ports", "hexagonal"],
                "file_patterns": ["adapter", "port", "interface"],
                "structure_indicators": ["ports_and_adapters", "hexagonal"]
            },
            ArchitecturePattern.EVENT_DRIVEN: {
                "directory_patterns": ["events", "handlers", "commands", "queries"],
                "file_patterns": ["event", "handler", "command", "query", "saga"],
                "structure_indicators": ["event_sourcing", "cqrs", "message_driven"]
            },
            ArchitecturePattern.SERVERLESS: {
                "directory_patterns": ["functions", "lambda", "serverless"],
                "file_patterns": ["function", "lambda", "handler"],
                "structure_indicators": ["function_as_a_service", "serverless"]
            }
        }

    def _initialize_technology_indicators(self) -> Dict[str, List[str]]:
        """
        Initialize technology stack indicators.
        
        Returns:
            Dictionary mapping technologies to file/pattern indicators
        """
        return {
            "python": [".py", "requirements.txt", "setup.py", "pyproject.toml", "__init__.py"],
            "javascript": [".js", ".jsx", "package.json", "node_modules"],
            "typescript": [".ts", ".tsx", "tsconfig.json"],
            "java": [".java", "pom.xml", "build.gradle", ".jar"],
            "csharp": [".cs", ".csproj", ".sln", ".dll"],
            "golang": [".go", "go.mod", "go.sum"],
            "rust": [".rs", "Cargo.toml", "Cargo.lock"],
            "php": [".php", "composer.json", "vendor"],
            "ruby": [".rb", "Gemfile", "Rakefile"],
            "docker": ["Dockerfile", "docker-compose.yml", ".dockerignore"],
            "kubernetes": ["*.yaml", "*.yml", "kustomization.yaml"]
        }

    def _initialize_framework_indicators(self) -> Dict[str, List[str]]:
        """
        Initialize framework detection patterns.
        
        Returns:
            Dictionary mapping frameworks to detection patterns
        """
        return {
            "fastapi": ["fastapi", "uvicorn", "app.py", "main.py"],
            "django": ["django", "manage.py", "settings.py", "wsgi.py"],
            "flask": ["flask", "app.py", "run.py"],
            "react": ["react", "src/components", "public/index.html"],
            "vue": ["vue", "src/main.js", "vue.config.js"],
            "angular": ["angular", "@angular", "angular.json"],
            "spring": ["spring", "@SpringBootApplication", "application.properties"],
            "dotnet": [".csproj", "Program.cs", "Startup.cs"],
            "express": ["express", "server.js", "app.js"],
            "nextjs": ["next", "pages", "next.config.js"]
        }

    async def analyze_architecture(
        self, 
        repository_path: str, 
        repository_context: Optional[Dict[str, Any]] = None
    ) -> ProjectStructure:
        """
        Analyze repository architecture using pattern detection methodology.
        
        Args:
            repository_path: Path to repository for analysis
            repository_context: Optional context information about repository
            
        Returns:
            Project structure analysis results
        """
        self.logger.info(f"Analyzing architecture for repository: {repository_path}")
        
        try:
            # Analyze directory structure
            directory_analysis = self._analyze_directory_structure(repository_path)
            
            # Detect technology stack
            technology_stack = self._detect_technology_stack(repository_path, directory_analysis)
            
            # Detect frameworks
            frameworks = self._detect_frameworks(repository_path, directory_analysis, technology_stack)
            
            # Detect architecture pattern
            architecture_pattern, confidence = self._detect_architecture_pattern(
                directory_analysis, technology_stack, frameworks
            )
            
            # Analyze layers and components
            layers = self._analyze_layers(directory_analysis, architecture_pattern)
            components = self._analyze_components(directory_analysis, architecture_pattern)
            
            # Create project structure result
            project_structure = ProjectStructure(
                architecture_type=architecture_pattern,
                technology_stack=technology_stack,
                layers=layers,
                components=components,
                frameworks=frameworks,
                confidence_score=confidence,
                structural_indicators=directory_analysis.get("indicators", {}),
                directory_structure=directory_analysis.get("directories", []),
                key_files=directory_analysis.get("key_files", [])
            )
            
            self.logger.info(f"Architecture analysis completed: {architecture_pattern.value} "
                           f"(confidence: {confidence:.2f})")
            
            return project_structure
            
        except Exception as e:
            self.logger.error(f"Architecture analysis failed: {e}", exc_info=True)
            # Return fallback structure
            return self._create_fallback_structure(repository_path, str(e))

    def _analyze_directory_structure(self, repository_path: str) -> Dict[str, Any]:
        """
        Analyze repository directory structure for architectural patterns.
        
        Args:
            repository_path: Path to repository
            
        Returns:
            Directory structure analysis
        """
        try:
            repo_path = Path(repository_path)
            if not repo_path.exists():
                self.logger.warning(f"Repository path does not exist: {repository_path}")
                return {"directories": [], "key_files": [], "indicators": {}}
            
            # Get directory listing
            directories = []
            key_files = []
            
            for item in repo_path.rglob("*"):
                if item.is_dir():
                    # Skip common ignored directories
                    if not any(ignore in item.name for ignore in [".git", "__pycache__", "node_modules", ".venv"]):
                        directories.append(str(item.relative_to(repo_path)))
                elif item.is_file():
                    # Collect key files
                    if self._is_key_file(item.name):
                        key_files.append(str(item.relative_to(repo_path)))
            
            # Analyze structural indicators
            indicators = self._extract_structural_indicators(directories, key_files)
            
            return {
                "directories": directories[:50],  # Limit to prevent excessive data
                "key_files": key_files[:50],
                "indicators": indicators,
                "total_directories": safe_len(directories),
                "total_files": safe_len(key_files)
            }
            
        except Exception as e:
            self.logger.error(f"Directory structure analysis failed: {e}")
            return {"directories": [], "key_files": [], "indicators": {}, "error": str(e)}

    def _is_key_file(self, filename: str) -> bool:
        """
        Check if file is considered a key architectural indicator.
        
        Args:
            filename: Name of file to check
            
        Returns:
            True if file is a key indicator
        """
        key_patterns = [
            "requirements.txt", "package.json", "pom.xml", "build.gradle",
            "Dockerfile", "docker-compose.yml", "main.py", "app.py",
            "server.js", "index.js", "Program.cs", "Startup.cs",
            "settings.py", "config.py", "application.properties"
        ]
        
        return (
            filename.lower() in [p.lower() for p in key_patterns] or
            filename.endswith(('.csproj', '.sln', '.toml', '.yml', '.yaml'))
        )

    def _extract_structural_indicators(
        self, 
        directories: List[str], 
        key_files: List[str]
    ) -> Dict[str, Any]:
        """
        Extract structural indicators from directory and file analysis.
        
        Args:
            directories: List of directory paths
            key_files: List of key file paths
            
        Returns:
            Structural indicators dictionary
        """
        indicators = {
            "has_src_directory": any("src" in d.lower() for d in directories),
            "has_test_directory": any("test" in d.lower() for d in directories),
            "has_docs_directory": any("doc" in d.lower() for d in directories),
            "has_config_files": any(self._is_config_file(f) for f in key_files),
            "layered_structure": self._detect_layered_structure(directories),
            "component_separation": self._detect_component_separation(directories),
            "technology_diversity": self._assess_technology_diversity(key_files)
        }
        
        return indicators

    def _is_config_file(self, filename: str) -> bool:
        """Check if file is a configuration file."""
        config_patterns = ["config", "settings", "properties", "env"]
        return any(pattern in filename.lower() for pattern in config_patterns)

    def _detect_layered_structure(self, directories: List[str]) -> bool:
        """Detect if repository has layered structure."""
        layer_indicators = ["presentation", "business", "data", "service", "controller", "model", "view"]
        return sum(any(indicator in d.lower() for d in directories) for indicator in layer_indicators) >= 2

    def _detect_component_separation(self, directories: List[str]) -> bool:
        """Detect if repository has clear component separation."""
        component_indicators = ["component", "module", "service", "feature"]
        return sum(any(indicator in d.lower() for d in directories) for indicator in component_indicators) >= 1

    def _assess_technology_diversity(self, key_files: List[str]) -> str:
        """Assess technology diversity based on file types."""
        technologies = set()
        for tech, patterns in self._technology_indicators.items():
            if any(any(pattern in f for pattern in patterns) for f in key_files):
                technologies.add(tech)
        
        tech_count = len(technologies)
        if tech_count <= 1:
            return "single"
        elif tech_count <= 3:
            return "moderate"
        else:
            return "diverse"

    def _detect_technology_stack(
        self, 
        repository_path: str, 
        directory_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Detect technology stack from repository analysis.
        
        Args:
            repository_path: Repository path
            directory_analysis: Directory structure analysis
            
        Returns:
            List of detected technologies
        """
        technologies = []
        key_files = directory_analysis.get("key_files", [])
        
        for tech, patterns in self._technology_indicators.items():
            if any(any(pattern in f for pattern in patterns) for f in key_files):
                technologies.append(tech)
        
        return technologies

    def _detect_frameworks(
        self, 
        repository_path: str, 
        directory_analysis: Dict[str, Any], 
        technology_stack: List[str]
    ) -> List[str]:
        """
        Detect frameworks from repository analysis.
        
        Args:
            repository_path: Repository path
            directory_analysis: Directory structure analysis
            technology_stack: Detected technology stack
            
        Returns:
            List of detected frameworks
        """
        frameworks = []
        key_files = directory_analysis.get("key_files", [])
        directories = directory_analysis.get("directories", [])
        
        for framework, patterns in self._framework_indicators.items():
            # Check file patterns
            file_match = any(any(pattern in f for pattern in patterns) for f in key_files)
            # Check directory patterns
            dir_match = any(any(pattern in d for pattern in patterns) for d in directories)
            
            if file_match or dir_match:
                frameworks.append(framework)
        
        return frameworks

    def _detect_architecture_pattern(
        self, 
        directory_analysis: Dict[str, Any], 
        technology_stack: List[str], 
        frameworks: List[str]
    ) -> tuple[ArchitecturePattern, float]:
        """
        Detect architecture pattern using scoring methodology from EventFlowAnalyzer.
        
        Args:
            directory_analysis: Directory structure analysis
            technology_stack: Detected technologies
            frameworks: Detected frameworks
            
        Returns:
            Tuple of (architecture_pattern, confidence_score)
        """
        directories = directory_analysis.get("directories", [])
        key_files = directory_analysis.get("key_files", [])
        
        pattern_scores = {}
        
        for pattern, criteria in self._architecture_patterns.items():
            score = 0.0
            
            # Score directory patterns
            dir_patterns = criteria.get("directory_patterns", [])
            dir_matches = sum(
                1 for pattern_name in dir_patterns
                if any(pattern_name.lower() in d.lower() for d in directories)
            )
            dir_score = dir_matches / max(safe_len(dir_patterns), 1)
            
            # Score file patterns
            file_patterns = criteria.get("file_patterns", [])
            file_matches = sum(
                1 for pattern_name in file_patterns
                if any(pattern_name.lower() in f.lower() for f in key_files)
            )
            file_score = file_matches / max(safe_len(file_patterns), 1)
            
            # Combined score
            combined_score = (dir_score * 0.6) + (file_score * 0.4)
            pattern_scores[pattern] = combined_score
        
        # Find best pattern
        if not pattern_scores or max(pattern_scores.values()) == 0:
            return ArchitecturePattern.UNKNOWN, 0.1
        
        best_pattern = max(pattern_scores, key=pattern_scores.get)
        best_score = pattern_scores[best_pattern]
        
        # Apply confidence adjustments
        confidence = min(best_score, 1.0)
        
        # Boost confidence for strong indicators
        if confidence > 0.5:
            confidence = min(confidence * 1.2, 1.0)
        
        return best_pattern, confidence

    def _analyze_layers(
        self, 
        directory_analysis: Dict[str, Any], 
        architecture_pattern: ArchitecturePattern
    ) -> List[str]:
        """
        Analyze architectural layers based on detected pattern.
        
        Args:
            directory_analysis: Directory structure analysis
            architecture_pattern: Detected architecture pattern
            
        Returns:
            List of identified layers
        """
        directories = directory_analysis.get("directories", [])
        
        # Define layer patterns for different architectures
        layer_patterns = {
            ArchitecturePattern.CLEAN_ARCHITECTURE: [
                "presentation", "application", "domain", "infrastructure"
            ],
            ArchitecturePattern.MVC: ["models", "views", "controllers"],
            ArchitecturePattern.LAYERED: ["presentation", "business", "data"],
            ArchitecturePattern.HEXAGONAL: ["adapters", "ports", "domain"]
        }
        
        detected_layers = []
        patterns = layer_patterns.get(architecture_pattern, [])
        
        for pattern in patterns:
            if any(pattern.lower() in d.lower() for d in directories):
                detected_layers.append(pattern.title())
        
        # Add generic layers if none detected
        if not detected_layers:
            generic_layers = ["API Layer", "Business Layer", "Data Layer"]
            detected_layers = generic_layers
        
        return detected_layers

    def _analyze_components(
        self, 
        directory_analysis: Dict[str, Any], 
        architecture_pattern: ArchitecturePattern
    ) -> List[str]:
        """
        Analyze architectural components based on detected pattern.
        
        Args:
            directory_analysis: Directory structure analysis
            architecture_pattern: Detected architecture pattern
            
        Returns:
            List of identified components
        """
        directories = directory_analysis.get("directories", [])
        key_files = directory_analysis.get("key_files", [])
        
        components = []
        
        # Extract components from directory names
        component_indicators = ["service", "component", "module", "handler", "controller", "manager"]
        
        for directory in directories:
            dir_lower = directory.lower()
            for indicator in component_indicators:
                if indicator in dir_lower:
                    component_name = directory.split("/")[-1].title()  # Get last part of path
                    if component_name not in components:
                        components.append(component_name)
        
        # Add pattern-specific components
        if architecture_pattern == ArchitecturePattern.MICROSERVICES:
            if any("gateway" in f.lower() for f in key_files):
                components.append("API Gateway")
            if any("discovery" in f.lower() for f in key_files):
                components.append("Service Discovery")
        
        # Ensure we have at least some components
        if not components:
            components = ["Core Module", "API Interface", "Data Access"]
        
        return components[:10]  # Limit to prevent excessive data

    def _create_fallback_structure(self, repository_path: str, error: str) -> ProjectStructure:
        """
        Create fallback project structure when analysis fails.
        
        Args:
            repository_path: Repository path
            error: Error description
            
        Returns:
            Fallback project structure
        """
        return ProjectStructure(
            architecture_type=ArchitecturePattern.UNKNOWN,
            technology_stack=[],
            layers=["Unknown Layer"],
            components=["Unknown Component"],
            frameworks=[],
            confidence_score=0.0,
            structural_indicators={"error": error},
            directory_structure=[],
            key_files=[]
        )

    def get_architecture_description(self, pattern: ArchitecturePattern) -> str:
        """
        Get human-readable description of architecture pattern.
        
        Args:
            pattern: Architecture pattern
            
        Returns:
            Description of the pattern
        """
        descriptions = {
            ArchitecturePattern.CLEAN_ARCHITECTURE: 
                "Clean Architecture with clear separation of concerns and dependency inversion",
            ArchitecturePattern.MVC: 
                "Model-View-Controller pattern separating data, presentation, and logic",
            ArchitecturePattern.MICROSERVICES: 
                "Microservices architecture with distributed, independently deployable services",
            ArchitecturePattern.LAYERED: 
                "Layered architecture organizing code into horizontal layers",
            ArchitecturePattern.HEXAGONAL: 
                "Hexagonal (Ports and Adapters) architecture isolating core business logic",
            ArchitecturePattern.EVENT_DRIVEN: 
                "Event-driven architecture using events for communication between components",
            ArchitecturePattern.SERVERLESS: 
                "Serverless architecture using cloud functions and managed services",
            ArchitecturePattern.UNKNOWN: 
                "Architecture pattern could not be determined from repository structure"
        }
        
        return descriptions.get(pattern, "Unknown architecture pattern")

    def get_supported_patterns(self) -> List[ArchitecturePattern]:
        """
        Get list of supported architecture patterns.
        
        Returns:
            List of supported architecture patterns
        """
        return list(ArchitecturePattern)

    async def validate_architecture_analysis(
        self, 
        project_structure: ProjectStructure
    ) -> Dict[str, Any]:
        """
        Validate architecture analysis results.
        
        Args:
            project_structure: Analysis results to validate
            
        Returns:
            Validation results
        """
        validation = {
            "valid": True,
            "confidence_acceptable": project_structure.confidence_score >= 0.3,
            "has_technology_stack": safe_len(project_structure.technology_stack) > 0,
            "has_components": safe_len(project_structure.components) > 0,
            "pattern_recognized": project_structure.architecture_type != ArchitecturePattern.UNKNOWN,
            "warnings": [],
            "recommendations": []
        }
        
        # Add warnings for low confidence
        if project_structure.confidence_score < 0.3:
            validation["warnings"].append(f"Low confidence score: {project_structure.confidence_score:.2f}")
        
        # Add recommendations
        if not validation["has_technology_stack"]:
            validation["recommendations"].append("Consider adding technology detection")
        
        if not validation["pattern_recognized"]:
            validation["recommendations"].append("Repository structure may benefit from clearer architectural organization")
        
        validation["overall_quality"] = (
            "good" if all([validation["confidence_acceptable"], validation["has_technology_stack"], 
                         validation["pattern_recognized"]])
            else "moderate" if validation["confidence_acceptable"]
            else "poor"
        )
        
        return validation