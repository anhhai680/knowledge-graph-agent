"""
Business Capability Analyzer for Generic Q&A Agent.

This module analyzes business capabilities and domain models in project repositories
following the EventFlowAnalyzer pattern detection methodology.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from enum import Enum

from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class BusinessDomain(str, Enum):
    """Business domain enumeration for capability analysis."""
    ECOMMERCE = "ecommerce"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    EDUCATION = "education"
    LOGISTICS = "logistics"
    SOCIAL_MEDIA = "social_media"
    CONTENT_MANAGEMENT = "content_management"
    PROJECT_MANAGEMENT = "project_management"
    COMMUNICATION = "communication"
    ANALYTICS = "analytics"
    IOT = "iot"
    GAMING = "gaming"
    UNKNOWN = "unknown"


@dataclass
class BusinessCapability:
    """
    Business capability information similar to EventFlowQuery dataclass.
    
    Represents a specific business capability with its scope, entities,
    and operational characteristics.
    """
    name: str
    description: str
    scope: str
    core_entities: List[str]
    key_operations: List[str]
    business_value: str
    stakeholders: List[str]
    complexity_level: str
    dependencies: List[str]


@dataclass
class BusinessAnalysisResult:
    """Business capability analysis results."""
    domain: BusinessDomain
    capabilities: List[BusinessCapability]
    core_entities: List[str]
    business_processes: List[str]
    value_proposition: str
    target_users: List[str]
    confidence_score: float
    analysis_metadata: Dict[str, Any]


class BusinessCapabilityAnalyzer:
    """
    Business capability analyzer using pattern detection from EventFlowAnalyzer.
    
    This analyzer examines repository structure, documentation, and code patterns
    to identify business capabilities, domain models, and operational scope.
    """

    def __init__(self):
        """Initialize business capability analyzer."""
        self.logger = get_logger(self.__class__.__name__)  # REUSE logging pattern
        
        # REUSE pattern detection approach from EventFlowAnalyzer
        self._domain_patterns = self._initialize_domain_patterns()
        self._capability_indicators = self._initialize_capability_indicators()
        self._entity_patterns = self._initialize_entity_patterns()

    def _initialize_domain_patterns(self) -> Dict[BusinessDomain, List[str]]:
        """
        Initialize business domain patterns following EventFlowAnalyzer pattern methodology.
        
        Returns:
            Dictionary mapping domains to detection patterns
        """
        return {
            BusinessDomain.ECOMMERCE: [
                "product", "order", "cart", "payment", "customer", "inventory",
                "catalog", "checkout", "shipping", "ecommerce", "shop", "store"
            ],
            BusinessDomain.HEALTHCARE: [
                "patient", "doctor", "medical", "appointment", "diagnosis", "treatment",
                "healthcare", "clinical", "hospital", "pharmacy", "prescription"
            ],
            BusinessDomain.FINANCE: [
                "account", "transaction", "payment", "bank", "loan", "credit",
                "finance", "investment", "portfolio", "trading", "currency"
            ],
            BusinessDomain.EDUCATION: [
                "student", "teacher", "course", "lesson", "grade", "assignment",
                "education", "learning", "school", "university", "curriculum"
            ],
            BusinessDomain.LOGISTICS: [
                "shipment", "delivery", "warehouse", "tracking", "logistics",
                "transportation", "freight", "cargo", "supply_chain"
            ],
            BusinessDomain.SOCIAL_MEDIA: [
                "user", "post", "comment", "like", "share", "follow", "feed",
                "social", "community", "message", "notification"
            ],
            BusinessDomain.CONTENT_MANAGEMENT: [
                "content", "article", "blog", "page", "media", "cms",
                "publication", "editor", "workflow", "approval"
            ],
            BusinessDomain.PROJECT_MANAGEMENT: [
                "project", "task", "milestone", "resource", "timeline",
                "planning", "tracking", "team", "collaboration"
            ],
            BusinessDomain.COMMUNICATION: [
                "message", "chat", "call", "meeting", "notification",
                "communication", "conference", "collaboration"
            ],
            BusinessDomain.ANALYTICS: [
                "data", "report", "dashboard", "metrics", "analytics",
                "visualization", "insight", "kpi", "analysis"
            ]
        }

    def _initialize_capability_indicators(self) -> Dict[str, List[str]]:
        """
        Initialize capability detection indicators.
        
        Returns:
            Dictionary mapping capability types to indicators
        """
        return {
            "user_management": [
                "authentication", "authorization", "user", "profile", "account",
                "login", "registration", "password", "session"
            ],
            "data_management": [
                "crud", "database", "storage", "repository", "entity",
                "model", "persistence", "query", "search"
            ],
            "business_logic": [
                "service", "business", "logic", "rule", "validation",
                "processing", "calculation", "workflow"
            ],
            "integration": [
                "api", "integration", "webhook", "gateway", "client",
                "external", "third_party", "connector"
            ],
            "reporting": [
                "report", "analytics", "dashboard", "export", "chart",
                "visualization", "metrics", "statistics"
            ],
            "notification": [
                "notification", "email", "sms", "alert", "messaging",
                "communication", "broadcast"
            ]
        }

    def _initialize_entity_patterns(self) -> List[str]:
        """
        Initialize entity detection patterns.
        
        Returns:
            List of common entity indicators
        """
        return [
            "user", "customer", "client", "account", "profile",
            "product", "item", "service", "order", "transaction",
            "category", "tag", "group", "team", "organization",
            "project", "task", "event", "activity", "process",
            "document", "file", "media", "content", "message",
            "report", "log", "audit", "record", "entry"
        ]

    async def analyze_business_capabilities(
        self,
        repository_path: str,
        repository_context: Optional[Dict[str, Any]] = None
    ) -> BusinessAnalysisResult:
        """
        Analyze business capabilities using pattern detection methodology.
        
        Args:
            repository_path: Path to repository for analysis
            repository_context: Optional context information
            
        Returns:
            Business capability analysis results
        """
        self.logger.info(f"Analyzing business capabilities for repository: {repository_path}")
        
        try:
            # Analyze documentation and README files
            documentation_analysis = self._analyze_documentation(repository_path)
            
            # Analyze code structure for business patterns
            code_analysis = self._analyze_code_patterns(repository_path)
            
            # Detect business domain
            domain, domain_confidence = self._detect_business_domain(
                documentation_analysis, code_analysis
            )
            
            # Extract business capabilities
            capabilities = self._extract_capabilities(
                documentation_analysis, code_analysis, domain
            )
            
            # Extract core entities
            core_entities = self._extract_core_entities(
                documentation_analysis, code_analysis
            )
            
            # Extract business processes
            business_processes = self._extract_business_processes(
                documentation_analysis, code_analysis
            )
            
            # Generate value proposition
            value_proposition = self._generate_value_proposition(
                domain, capabilities, documentation_analysis
            )
            
            # Identify target users
            target_users = self._identify_target_users(
                documentation_analysis, code_analysis, domain
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(
                domain_confidence, capabilities, core_entities
            )
            
            # Create analysis result
            result = BusinessAnalysisResult(
                domain=domain,
                capabilities=capabilities,
                core_entities=core_entities,
                business_processes=business_processes,
                value_proposition=value_proposition,
                target_users=target_users,
                confidence_score=confidence_score,
                analysis_metadata={
                    "documentation_found": documentation_analysis.get("found", False),
                    "code_patterns_detected": safe_len(code_analysis.get("patterns", [])),
                    "entity_extraction_count": safe_len(core_entities),
                    "capability_count": safe_len(capabilities)
                }
            )
            
            self.logger.info(f"Business analysis completed: domain={domain.value}, "
                           f"capabilities={safe_len(capabilities)}, confidence={confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Business capability analysis failed: {e}", exc_info=True)
            return self._create_fallback_analysis(repository_path, str(e))

    def _analyze_documentation(self, repository_path: str) -> Dict[str, Any]:
        """
        Analyze documentation files for business context.
        
        Args:
            repository_path: Repository path
            
        Returns:
            Documentation analysis results
        """
        try:
            from pathlib import Path
            
            repo_path = Path(repository_path)
            documentation = {
                "found": False,
                "readme_content": "",
                "business_keywords": [],
                "feature_descriptions": [],
                "user_stories": []
            }
            
            # Look for README files
            readme_files = list(repo_path.glob("README*")) + list(repo_path.glob("readme*"))
            
            if readme_files:
                documentation["found"] = True
                try:
                    readme_content = readme_files[0].read_text(encoding='utf-8', errors='ignore')
                    documentation["readme_content"] = readme_content[:5000]  # Limit content
                    
                    # Extract business keywords
                    documentation["business_keywords"] = self._extract_business_keywords(readme_content)
                    
                    # Extract feature descriptions
                    documentation["feature_descriptions"] = self._extract_features(readme_content)
                    
                except Exception as e:
                    self.logger.warning(f"Error reading README: {e}")
            
            # Look for additional documentation
            doc_dirs = ["docs", "documentation", "wiki"]
            for doc_dir in doc_dirs:
                doc_path = repo_path / doc_dir
                if doc_path.exists():
                    documentation["has_documentation_dir"] = True
                    break
            
            return documentation
            
        except Exception as e:
            self.logger.error(f"Documentation analysis failed: {e}")
            return {"found": False, "error": str(e)}

    def _extract_business_keywords(self, text: str) -> List[str]:
        """Extract business-related keywords from text."""
        text_lower = text.lower()
        keywords = []
        
        # Check all domain patterns
        for domain, patterns in self._domain_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    keywords.append(pattern)
        
        # Check capability indicators
        for capability, indicators in self._capability_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    keywords.append(indicator)
        
        return list(set(keywords))  # Remove duplicates

    def _extract_features(self, text: str) -> List[str]:
        """Extract feature descriptions from documentation."""
        features = []
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            # Look for feature indicators
            if any(indicator in line_lower for indicator in ['feature', 'capability', 'function']):
                features.append(line.strip()[:200])  # Limit length
        
        return features[:10]  # Limit to 10 features

    def _analyze_code_patterns(self, repository_path: str) -> Dict[str, Any]:
        """
        Analyze code structure for business patterns.
        
        Args:
            repository_path: Repository path
            
        Returns:
            Code pattern analysis results
        """
        try:
            from pathlib import Path
            
            repo_path = Path(repository_path)
            analysis = {
                "patterns": [],
                "models_found": [],
                "services_found": [],
                "controllers_found": [],
                "business_files": []
            }
            
            if not repo_path.exists():
                return analysis
            
            # Analyze file structure for business patterns
            for file_path in repo_path.rglob("*.py"):  # Focus on Python files
                try:
                    relative_path = str(file_path.relative_to(repo_path))
                    filename = file_path.name.lower()
                    
                    # Detect business-related files
                    if any(pattern in filename for pattern in ['model', 'entity', 'domain']):
                        analysis["models_found"].append(relative_path)
                    elif any(pattern in filename for pattern in ['service', 'business', 'logic']):
                        analysis["services_found"].append(relative_path)
                    elif any(pattern in filename for pattern in ['controller', 'handler', 'view']):
                        analysis["controllers_found"].append(relative_path)
                    
                    # Check for business entities in file content
                    if self._contains_business_entities(file_path):
                        analysis["business_files"].append(relative_path)
                        
                except Exception as e:
                    self.logger.debug(f"Error analyzing file {file_path}: {e}")
                    continue
            
            # Identify patterns
            if analysis["models_found"]:
                analysis["patterns"].append("domain_modeling")
            if analysis["services_found"]:
                analysis["patterns"].append("service_layer")
            if analysis["controllers_found"]:
                analysis["patterns"].append("presentation_layer")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Code pattern analysis failed: {e}")
            return {"patterns": [], "error": str(e)}

    def _contains_business_entities(self, file_path: Path) -> bool:
        """Check if file contains business entity definitions."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')[:2000]  # Limit content
            content_lower = content.lower()
            
            # Check for entity patterns
            entity_indicators = ['class ', 'def ', 'model', 'entity']
            business_keywords = []
            
            for patterns in self._domain_patterns.values():
                business_keywords.extend(patterns)
            
            has_entity_structure = any(indicator in content_lower for indicator in entity_indicators)
            has_business_keywords = any(keyword in content_lower for keyword in business_keywords[:20])  # Limit check
            
            return has_entity_structure and has_business_keywords
            
        except Exception:
            return False

    def _detect_business_domain(
        self, 
        documentation_analysis: Dict[str, Any], 
        code_analysis: Dict[str, Any]
    ) -> tuple[BusinessDomain, float]:
        """
        Detect business domain using scoring methodology from EventFlowAnalyzer.
        
        Args:
            documentation_analysis: Documentation analysis results
            code_analysis: Code analysis results
            
        Returns:
            Tuple of (business_domain, confidence_score)
        """
        domain_scores = {}
        
        # Get keywords from documentation and code
        doc_keywords = documentation_analysis.get("business_keywords", [])
        business_files = code_analysis.get("business_files", [])
        
        # Score each domain
        for domain, patterns in self._domain_patterns.items():
            score = 0.0
            
            # Score based on documentation keywords
            doc_matches = sum(1 for keyword in doc_keywords if keyword in patterns)
            doc_score = doc_matches / max(safe_len(patterns), 1)
            
            # Score based on file names
            file_matches = sum(
                1 for file_path in business_files
                if any(pattern in file_path.lower() for pattern in patterns)
            )
            file_score = file_matches / max(safe_len(business_files), 1) if business_files else 0
            
            # Combined score
            combined_score = (doc_score * 0.7) + (file_score * 0.3)
            domain_scores[domain] = combined_score
        
        # Find best domain
        if not domain_scores or max(domain_scores.values()) == 0:
            return BusinessDomain.UNKNOWN, 0.1
        
        best_domain = max(domain_scores, key=domain_scores.get)
        best_score = domain_scores[best_domain]
        
        return best_domain, min(best_score, 1.0)

    def _extract_capabilities(
        self,
        documentation_analysis: Dict[str, Any],
        code_analysis: Dict[str, Any],
        domain: BusinessDomain
    ) -> List[BusinessCapability]:
        """
        Extract business capabilities from analysis results.
        
        Args:
            documentation_analysis: Documentation analysis
            code_analysis: Code analysis
            domain: Detected business domain
            
        Returns:
            List of business capabilities
        """
        capabilities = []
        
        # Get detected patterns and keywords
        patterns = code_analysis.get("patterns", [])
        keywords = documentation_analysis.get("business_keywords", [])
        
        # Map patterns to capabilities
        capability_mapping = {
            "domain_modeling": BusinessCapability(
                name="Domain Modeling",
                description="Define and manage business entities and relationships",
                scope="Core business data structures",
                core_entities=self._extract_core_entities(documentation_analysis, code_analysis)[:5],
                key_operations=["Create", "Read", "Update", "Delete", "Validate"],
                business_value="Ensures data consistency and business rule enforcement",
                stakeholders=["Business Analysts", "Developers", "Data Architects"],
                complexity_level="Medium",
                dependencies=["Database", "Validation Framework"]
            ),
            "service_layer": BusinessCapability(
                name="Business Logic Processing",
                description="Execute business rules and coordinate business operations",
                scope="Business logic implementation",
                core_entities=["Service", "Business Rules", "Workflow"],
                key_operations=["Process", "Validate", "Transform", "Coordinate"],
                business_value="Implements business rules and ensures consistent processing",
                stakeholders=["Business Users", "Developers"],
                complexity_level="High",
                dependencies=["Domain Models", "External Services"]
            ),
            "presentation_layer": BusinessCapability(
                name="User Interface Management",
                description="Provide user interaction and data presentation capabilities",
                scope="User interface and API endpoints",
                core_entities=["Controller", "View", "API"],
                key_operations=["Display", "Input", "Navigate", "Respond"],
                business_value="Enables user interaction and system accessibility",
                stakeholders=["End Users", "Frontend Developers"],
                complexity_level="Medium",
                dependencies=["Business Services", "Authentication"]
            )
        }
        
        # Add capabilities based on detected patterns
        for pattern in patterns:
            if pattern in capability_mapping:
                capabilities.append(capability_mapping[pattern])
        
        # Add domain-specific capabilities
        domain_capabilities = self._get_domain_specific_capabilities(domain)
        capabilities.extend(domain_capabilities)
        
        # Ensure we have at least one capability
        if not capabilities:
            capabilities.append(BusinessCapability(
                name="Core Functionality",
                description="Primary system functionality based on repository analysis",
                scope="General system capabilities",
                core_entities=["System", "User", "Data"],
                key_operations=["Process", "Manage", "Store"],
                business_value="Provides core system functionality",
                stakeholders=["Users", "Administrators"],
                complexity_level="Medium",
                dependencies=["Infrastructure"]
            ))
        
        return capabilities[:5]  # Limit to 5 capabilities

    def _get_domain_specific_capabilities(self, domain: BusinessDomain) -> List[BusinessCapability]:
        """Get capabilities specific to the business domain."""
        domain_capabilities = {
            BusinessDomain.ECOMMERCE: [
                BusinessCapability(
                    name="Product Catalog Management",
                    description="Manage product listings, categories, and inventory",
                    scope="Product information and availability",
                    core_entities=["Product", "Category", "Inventory"],
                    key_operations=["List", "Search", "Filter", "Update Stock"],
                    business_value="Enables customers to find and purchase products",
                    stakeholders=["Customers", "Store Managers"],
                    complexity_level="Medium",
                    dependencies=["Database", "Search Engine"]
                )
            ],
            BusinessDomain.HEALTHCARE: [
                BusinessCapability(
                    name="Patient Management",
                    description="Manage patient information and medical records",
                    scope="Patient data and healthcare delivery",
                    core_entities=["Patient", "Medical Record", "Appointment"],
                    key_operations=["Register", "Schedule", "Diagnose", "Treat"],
                    business_value="Improves healthcare delivery and patient outcomes",
                    stakeholders=["Patients", "Healthcare Providers"],
                    complexity_level="High",
                    dependencies=["EMR System", "Compliance Framework"]
                )
            ]
        }
        
        return domain_capabilities.get(domain, [])

    def _extract_core_entities(
        self,
        documentation_analysis: Dict[str, Any],
        code_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Extract core business entities from analysis.
        
        Args:
            documentation_analysis: Documentation analysis
            code_analysis: Code analysis
            
        Returns:
            List of core entity names
        """
        entities = set()
        
        # Extract from keywords
        keywords = documentation_analysis.get("business_keywords", [])
        for keyword in keywords:
            if keyword in self._entity_patterns:
                entities.add(keyword.title())
        
        # Extract from model files
        model_files = code_analysis.get("models_found", [])
        for model_file in model_files:
            # Extract entity name from filename
            filename = model_file.split("/")[-1].replace(".py", "")
            if any(pattern in filename.lower() for pattern in self._entity_patterns):
                entities.add(filename.title())
        
        # Add common entities if none found
        if not entities:
            entities.update(["User", "System", "Data", "Process"])
        
        return list(entities)[:10]  # Limit to 10 entities

    def _extract_business_processes(
        self,
        documentation_analysis: Dict[str, Any],
        code_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract business processes from analysis."""
        processes = []
        
        # Extract from documentation features
        features = documentation_analysis.get("feature_descriptions", [])
        for feature in features:
            if any(process_word in feature.lower() for process_word in ['process', 'workflow', 'manage']):
                processes.append(feature[:100])  # Limit length
        
        # Add generic processes based on patterns
        patterns = code_analysis.get("patterns", [])
        if "service_layer" in patterns:
            processes.append("Business Logic Processing")
        if "domain_modeling" in patterns:
            processes.append("Data Management")
        if "presentation_layer" in patterns:
            processes.append("User Interaction")
        
        return processes[:5]  # Limit to 5 processes

    def _generate_value_proposition(
        self,
        domain: BusinessDomain,
        capabilities: List[BusinessCapability],
        documentation_analysis: Dict[str, Any]
    ) -> str:
        """Generate value proposition based on analysis."""
        
        # Domain-specific value propositions
        domain_values = {
            BusinessDomain.ECOMMERCE: "Facilitates online commerce with product management and customer engagement",
            BusinessDomain.HEALTHCARE: "Improves healthcare delivery through efficient patient and medical data management",
            BusinessDomain.FINANCE: "Provides secure financial services with transaction processing and account management",
            BusinessDomain.EDUCATION: "Enhances learning outcomes through educational content and student management",
            BusinessDomain.LOGISTICS: "Optimizes supply chain operations with tracking and delivery management"
        }
        
        base_value = domain_values.get(domain, "Provides efficient business operations and data management")
        
        # Enhance with capability information
        if capabilities:
            capability_names = [cap.name for cap in capabilities[:3]]
            enhancement = f" through {', '.join(capability_names)}"
            return base_value + enhancement
        
        return base_value

    def _identify_target_users(
        self,
        documentation_analysis: Dict[str, Any],
        code_analysis: Dict[str, Any],
        domain: BusinessDomain
    ) -> List[str]:
        """Identify target users based on domain and analysis."""
        
        # Domain-specific users
        domain_users = {
            BusinessDomain.ECOMMERCE: ["Customers", "Store Managers", "Administrators"],
            BusinessDomain.HEALTHCARE: ["Patients", "Doctors", "Nurses", "Healthcare Administrators"],
            BusinessDomain.FINANCE: ["Account Holders", "Financial Advisors", "Bank Staff"],
            BusinessDomain.EDUCATION: ["Students", "Teachers", "Administrators", "Parents"],
            BusinessDomain.LOGISTICS: ["Warehouse Staff", "Drivers", "Logistics Coordinators"]
        }
        
        users = domain_users.get(domain, ["End Users", "Administrators", "System Operators"])
        
        # Check for user mentions in documentation
        readme_content = documentation_analysis.get("readme_content", "").lower()
        additional_users = []
        
        user_keywords = ["user", "admin", "customer", "client", "manager", "operator"]
        for keyword in user_keywords:
            if keyword in readme_content and keyword.title() not in users:
                additional_users.append(keyword.title())
        
        return users + additional_users[:3]  # Limit additional users

    def _calculate_confidence(
        self,
        domain_confidence: float,
        capabilities: List[BusinessCapability],
        core_entities: List[str]
    ) -> float:
        """Calculate overall analysis confidence."""
        
        # Base confidence from domain detection
        base_confidence = domain_confidence
        
        # Boost based on capability detection
        capability_boost = min(safe_len(capabilities) * 0.1, 0.3)
        
        # Boost based on entity extraction
        entity_boost = min(safe_len(core_entities) * 0.05, 0.2)
        
        # Combine factors
        total_confidence = base_confidence + capability_boost + entity_boost
        
        return min(total_confidence, 1.0)

    def _create_fallback_analysis(self, repository_path: str, error: str) -> BusinessAnalysisResult:
        """Create fallback analysis when main analysis fails."""
        return BusinessAnalysisResult(
            domain=BusinessDomain.UNKNOWN,
            capabilities=[
                BusinessCapability(
                    name="Unknown Capability",
                    description="Could not determine specific business capabilities",
                    scope="General system functionality",
                    core_entities=["System"],
                    key_operations=["Process"],
                    business_value="Provides system functionality",
                    stakeholders=["Users"],
                    complexity_level="Unknown",
                    dependencies=[]
                )
            ],
            core_entities=["System", "User"],
            business_processes=["General Processing"],
            value_proposition="Provides business functionality",
            target_users=["Users"],
            confidence_score=0.1,
            analysis_metadata={"error": error}
        )

    def get_domain_description(self, domain: BusinessDomain) -> str:
        """Get description of business domain."""
        descriptions = {
            BusinessDomain.ECOMMERCE: "Electronic commerce and online retail operations",
            BusinessDomain.HEALTHCARE: "Healthcare services and medical information management",
            BusinessDomain.FINANCE: "Financial services and transaction processing",
            BusinessDomain.EDUCATION: "Educational services and learning management",
            BusinessDomain.LOGISTICS: "Supply chain and logistics management",
            BusinessDomain.UNKNOWN: "Business domain could not be determined"
        }
        
        return descriptions.get(domain, "Unknown business domain")

    def get_supported_domains(self) -> List[BusinessDomain]:
        """Get list of supported business domains."""
        return list(BusinessDomain)