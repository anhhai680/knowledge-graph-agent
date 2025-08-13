"""
Operational Analyzer for Generic Q&A Agent.

This module analyzes operational aspects like deployment, infrastructure, and
DevOps practices in project repositories following the EventFlowAnalyzer pattern.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class DeploymentType(str, Enum):
    """Deployment type enumeration."""
    CONTAINERIZED = "containerized"
    SERVERLESS = "serverless"
    TRADITIONAL = "traditional"
    KUBERNETES = "kubernetes"
    CLOUD_NATIVE = "cloud_native"
    UNKNOWN = "unknown"


@dataclass
class OperationalAnalysisResult:
    """Operational analysis results."""
    deployment_type: DeploymentType
    containerization: bool
    ci_cd_configured: bool
    monitoring_setup: bool
    infrastructure_as_code: bool
    security_practices: List[str]
    environments: List[str]
    confidence_score: float


class OperationalAnalyzer:
    """Operational analyzer using EventFlowAnalyzer patterns."""

    def __init__(self):
        """Initialize operational analyzer."""
        self.logger = get_logger(self.__class__.__name__)
        self._deployment_patterns = self._initialize_deployment_patterns()

    def _initialize_deployment_patterns(self) -> Dict[DeploymentType, List[str]]:
        """Initialize deployment detection patterns."""
        return {
            DeploymentType.CONTAINERIZED: ["dockerfile", "docker-compose", "container"],
            DeploymentType.KUBERNETES: ["kubernetes", "k8s", "helm", "kubectl"],
            DeploymentType.SERVERLESS: ["lambda", "serverless", "azure-functions"],
            DeploymentType.CLOUD_NATIVE: ["terraform", "cloudformation", "aws", "azure", "gcp"]
        }

    async def analyze_operational_aspects(
        self,
        repository_path: str,
        repository_context: Optional[Dict[str, Any]] = None
    ) -> OperationalAnalysisResult:
        """Analyze operational aspects of repository."""
        self.logger.info(f"Analyzing operational aspects for: {repository_path}")
        
        try:
            # Detect deployment type
            deployment_type = self._detect_deployment_type(repository_path)
            
            # Check containerization
            has_containers = self._check_containerization(repository_path)
            
            # Check CI/CD
            has_cicd = self._check_cicd_setup(repository_path)
            
            # Check monitoring
            has_monitoring = self._check_monitoring_setup(repository_path)
            
            # Check Infrastructure as Code
            has_iac = self._check_infrastructure_as_code(repository_path)
            
            # Detect security practices
            security_practices = self._detect_security_practices(repository_path)
            
            # Detect environments
            environments = self._detect_environments(repository_path)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                deployment_type, has_containers, has_cicd, has_monitoring
            )
            
            return OperationalAnalysisResult(
                deployment_type=deployment_type,
                containerization=has_containers,
                ci_cd_configured=has_cicd,
                monitoring_setup=has_monitoring,
                infrastructure_as_code=has_iac,
                security_practices=security_practices,
                environments=environments,
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Operational analysis failed: {e}")
            return self._create_fallback_result()

    def _detect_deployment_type(self, repository_path: str) -> DeploymentType:
        """Detect deployment type from repository files."""
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            for deploy_type, patterns in self._deployment_patterns.items():
                for pattern in patterns:
                    # Check for files matching pattern
                    if list(repo_path.glob(f"**/*{pattern}*")):
                        return deploy_type
            
            return DeploymentType.UNKNOWN
            
        except Exception:
            return DeploymentType.UNKNOWN

    def _check_containerization(self, repository_path: str) -> bool:
        """Check if repository uses containerization."""
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            container_files = ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]
            
            for container_file in container_files:
                if (repo_path / container_file).exists():
                    return True
            
            return False
            
        except Exception:
            return False

    def _check_cicd_setup(self, repository_path: str) -> bool:
        """Check if CI/CD is configured."""
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            cicd_paths = [
                ".github/workflows",
                ".gitlab-ci.yml",
                "azure-pipelines.yml",
                "Jenkinsfile",
                ".circleci"
            ]
            
            for cicd_path in cicd_paths:
                if (repo_path / cicd_path).exists():
                    return True
            
            return False
            
        except Exception:
            return False

    def _check_monitoring_setup(self, repository_path: str) -> bool:
        """Check if monitoring is configured."""
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            monitoring_indicators = ["prometheus", "grafana", "datadog", "newrelic", "logging"]
            
            for file_path in repo_path.rglob("*.py"):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')[:2000]
                    content_lower = content.lower()
                    
                    if any(indicator in content_lower for indicator in monitoring_indicators):
                        return True
                except Exception:
                    continue
            
            return False
            
        except Exception:
            return False

    def _check_infrastructure_as_code(self, repository_path: str) -> bool:
        """Check if Infrastructure as Code is used."""
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            iac_files = ["*.tf", "*.yaml", "*.yml", "cloudformation.json"]
            
            for pattern in iac_files:
                if list(repo_path.glob(f"**/{pattern}")):
                    return True
            
            return False
            
        except Exception:
            return False

    def _detect_security_practices(self, repository_path: str) -> List[str]:
        """Detect security practices in repository."""
        practices = []
        
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            security_patterns = {
                "authentication": ["auth", "jwt", "oauth", "login"],
                "encryption": ["encrypt", "crypto", "ssl", "tls"],
                "validation": ["validate", "sanitize", "security"],
                "secrets_management": ["secret", "env", "vault"],
                "security_scanning": ["security", "scan", "vulnerability"]
            }
            
            for file_path in repo_path.rglob("*.py"):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')[:2000]
                    content_lower = content.lower()
                    
                    for practice, indicators in security_patterns.items():
                        if any(indicator in content_lower for indicator in indicators):
                            if practice not in practices:
                                practices.append(practice)
                except Exception:
                    continue
            
        except Exception:
            pass
        
        return practices

    def _detect_environments(self, repository_path: str) -> List[str]:
        """Detect configured environments."""
        environments = []
        
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            # Check for environment files
            env_patterns = [".env", "config", "settings"]
            env_names = ["development", "staging", "production", "test"]
            
            for pattern in env_patterns:
                for env_file in repo_path.glob(f"**/*{pattern}*"):
                    for env_name in env_names:
                        if env_name in env_file.name.lower():
                            if env_name not in environments:
                                environments.append(env_name)
            
            # Default environments if none found
            if not environments:
                environments = ["development", "production"]
            
        except Exception:
            environments = ["development", "production"]
        
        return environments

    def _calculate_confidence(
        self,
        deployment_type: DeploymentType,
        has_containers: bool,
        has_cicd: bool,
        has_monitoring: bool
    ) -> float:
        """Calculate analysis confidence."""
        base_confidence = 0.2
        
        # Boost for deployment type detection
        deploy_boost = 0.3 if deployment_type != DeploymentType.UNKNOWN else 0
        
        # Boost for operational practices
        practices_boost = 0
        if has_containers:
            practices_boost += 0.15
        if has_cicd:
            practices_boost += 0.15
        if has_monitoring:
            practices_boost += 0.15
        
        return min(base_confidence + deploy_boost + practices_boost, 1.0)

    def _create_fallback_result(self) -> OperationalAnalysisResult:
        """Create fallback result when analysis fails."""
        return OperationalAnalysisResult(
            deployment_type=DeploymentType.UNKNOWN,
            containerization=False,
            ci_cd_configured=False,
            monitoring_setup=False,
            infrastructure_as_code=False,
            security_practices=[],
            environments=["unknown"],
            confidence_score=0.1
        )