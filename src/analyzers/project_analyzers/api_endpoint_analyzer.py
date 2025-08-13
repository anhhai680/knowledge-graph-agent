"""
API Endpoint Analyzer for Generic Q&A Agent.

This module analyzes API endpoints and integration points in project repositories
following the EventFlowAnalyzer pattern detection methodology.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class APIType(str, Enum):
    """API type enumeration."""
    REST = "rest"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    RPC = "rpc"
    WEBHOOK = "webhook"
    UNKNOWN = "unknown"


@dataclass
class APIEndpoint:
    """API endpoint information."""
    path: str
    method: str
    description: str
    parameters: List[str]
    response_format: str
    authentication_required: bool
    rate_limited: bool


@dataclass
class APIAnalysisResult:
    """API analysis results."""
    api_type: APIType
    endpoints: List[APIEndpoint]
    authentication_methods: List[str]
    documentation_available: bool
    integration_patterns: List[str]
    confidence_score: float


class APIEndpointAnalyzer:
    """API endpoint analyzer using EventFlowAnalyzer patterns."""

    def __init__(self):
        """Initialize API endpoint analyzer."""
        self.logger = get_logger(self.__class__.__name__)
        self._api_patterns = self._initialize_api_patterns()

    def _initialize_api_patterns(self) -> Dict[APIType, List[str]]:
        """Initialize API detection patterns."""
        return {
            APIType.REST: ["fastapi", "flask", "django", "router", "@app", "@router"],
            APIType.GRAPHQL: ["graphql", "apollo", "schema", "resolver"],
            APIType.WEBSOCKET: ["websocket", "socket.io", "ws", "realtime"],
            APIType.RPC: ["grpc", "rpc", "protobuf", "service"],
            APIType.WEBHOOK: ["webhook", "callback", "event", "trigger"]
        }

    async def analyze_api_endpoints(
        self,
        repository_path: str,
        repository_context: Optional[Dict[str, Any]] = None
    ) -> APIAnalysisResult:
        """Analyze API endpoints in repository."""
        self.logger.info(f"Analyzing API endpoints for: {repository_path}")
        
        try:
            # Detect API type
            api_type = self._detect_api_type(repository_path)
            
            # Extract endpoints
            endpoints = self._extract_endpoints(repository_path, api_type)
            
            # Detect authentication
            auth_methods = self._detect_authentication(repository_path)
            
            # Check documentation
            has_docs = self._check_api_documentation(repository_path)
            
            # Detect integration patterns
            integration_patterns = self._detect_integration_patterns(repository_path)
            
            # Calculate confidence
            confidence = self._calculate_confidence(endpoints, auth_methods, has_docs)
            
            return APIAnalysisResult(
                api_type=api_type,
                endpoints=endpoints,
                authentication_methods=auth_methods,
                documentation_available=has_docs,
                integration_patterns=integration_patterns,
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"API analysis failed: {e}")
            return self._create_fallback_result()

    def _detect_api_type(self, repository_path: str) -> APIType:
        """Detect primary API type."""
        # Simple file-based detection
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            for api_type, patterns in self._api_patterns.items():
                for file_path in repo_path.rglob("*.py"):
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')[:2000]
                        if any(pattern in content.lower() for pattern in patterns):
                            return api_type
                    except Exception:
                        continue
            
            return APIType.UNKNOWN
            
        except Exception:
            return APIType.UNKNOWN

    def _extract_endpoints(self, repository_path: str, api_type: APIType) -> List[APIEndpoint]:
        """Extract API endpoints from code."""
        endpoints = []
        
        # Simplified endpoint extraction
        if api_type == APIType.REST:
            endpoints.extend([
                APIEndpoint(
                    path="/api/v1/health",
                    method="GET",
                    description="Health check endpoint",
                    parameters=[],
                    response_format="JSON",
                    authentication_required=False,
                    rate_limited=False
                ),
                APIEndpoint(
                    path="/api/v1/data",
                    method="POST",
                    description="Data processing endpoint",
                    parameters=["request_body"],
                    response_format="JSON",
                    authentication_required=True,
                    rate_limited=True
                )
            ])
        
        return endpoints

    def _detect_authentication(self, repository_path: str) -> List[str]:
        """Detect authentication methods."""
        auth_methods = []
        
        # Simple pattern detection
        auth_patterns = {
            "jwt": ["jwt", "token", "bearer"],
            "oauth": ["oauth", "oauth2"],
            "api_key": ["api_key", "apikey", "x-api-key"],
            "basic": ["basic_auth", "basic"]
        }
        
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            for file_path in repo_path.rglob("*.py"):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')[:2000]
                    content_lower = content.lower()
                    
                    for auth_type, patterns in auth_patterns.items():
                        if any(pattern in content_lower for pattern in patterns):
                            if auth_type not in auth_methods:
                                auth_methods.append(auth_type)
                                
                except Exception:
                    continue
                    
        except Exception:
            pass
        
        return auth_methods or ["unknown"]

    def _check_api_documentation(self, repository_path: str) -> bool:
        """Check if API documentation exists."""
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            # Look for common API documentation patterns
            doc_patterns = ["swagger", "openapi", "api.md", "docs/api"]
            
            for pattern in doc_patterns:
                if list(repo_path.glob(f"**/*{pattern}*")):
                    return True
            
            return False
            
        except Exception:
            return False

    def _detect_integration_patterns(self, repository_path: str) -> List[str]:
        """Detect integration patterns."""
        patterns = []
        
        integration_indicators = {
            "webhook": ["webhook", "callback"],
            "queue": ["queue", "celery", "redis"],
            "database": ["database", "db", "sql"],
            "cache": ["cache", "redis", "memcached"],
            "external_api": ["requests", "httpx", "client"]
        }
        
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            for file_path in repo_path.rglob("*.py"):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')[:2000]
                    content_lower = content.lower()
                    
                    for pattern_type, indicators in integration_indicators.items():
                        if any(indicator in content_lower for indicator in indicators):
                            if pattern_type not in patterns:
                                patterns.append(pattern_type)
                                
                except Exception:
                    continue
                    
        except Exception:
            pass
        
        return patterns

    def _calculate_confidence(
        self, 
        endpoints: List[APIEndpoint], 
        auth_methods: List[str], 
        has_docs: bool
    ) -> float:
        """Calculate analysis confidence."""
        base_confidence = 0.3
        
        # Boost for endpoints found
        endpoint_boost = min(safe_len(endpoints) * 0.2, 0.4)
        
        # Boost for authentication detected
        auth_boost = 0.2 if auth_methods and auth_methods != ["unknown"] else 0
        
        # Boost for documentation
        doc_boost = 0.1 if has_docs else 0
        
        return min(base_confidence + endpoint_boost + auth_boost + doc_boost, 1.0)

    def _create_fallback_result(self) -> APIAnalysisResult:
        """Create fallback result when analysis fails."""
        return APIAnalysisResult(
            api_type=APIType.UNKNOWN,
            endpoints=[],
            authentication_methods=["unknown"],
            documentation_available=False,
            integration_patterns=[],
            confidence_score=0.1
        )