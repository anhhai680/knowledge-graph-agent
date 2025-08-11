"""
Business Capability Analyzer for Generic Project Q&A.

This module analyzes business domain scope, core entities, and ownership patterns
to provide insights about business capabilities and bounded contexts.
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from pathlib import Path
import re

from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class DomainComplexity(str, Enum):
    """Domain complexity levels."""
    
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


@dataclass
class BusinessEntity:
    """Business entity representation."""
    
    name: str
    type: str  # entity, aggregate, value_object, service
    attributes: List[str]
    relationships: List[str]
    business_rules: List[str]
    confidence_score: float


@dataclass
class BusinessCapability:
    """Business capability representation."""
    
    name: str
    description: str
    scope: str
    entities: List[BusinessEntity]
    services: List[str]
    bounded_context: Optional[str]
    ownership_indicators: List[str]
    complexity: DomainComplexity


@dataclass
class BusinessAnalysis:
    """Business capability analysis result."""
    
    primary_capability: BusinessCapability
    secondary_capabilities: List[BusinessCapability]
    domain_complexity: DomainComplexity
    bounded_contexts: List[str]
    ownership_patterns: Dict[str, List[str]]
    business_rules: List[str]
    integration_points: List[str]
    confidence_score: float


class BusinessCapabilityAnalyzer:
    """
    Business capability analyzer for project analysis.
    
    This component analyzes code structure, naming patterns, and business logic
    to identify business capabilities, domain entities, and ownership patterns.
    """
    
    def __init__(self):
        """Initialize business capability analyzer."""
        self.logger = get_logger(self.__class__.__name__)
        self._init_domain_patterns()
    
    def _init_domain_patterns(self) -> None:
        """Initialize domain analysis patterns."""
        
        # Common business domain keywords
        self.business_domains = {
            "user_management": [
                "user", "account", "profile", "authentication", "authorization",
                "login", "signup", "registration", "permission", "role"
            ],
            "order_management": [
                "order", "purchase", "cart", "checkout", "payment", "billing",
                "invoice", "transaction", "fulfillment", "shipping"
            ],
            "inventory_management": [
                "product", "inventory", "stock", "catalog", "item", "sku",
                "warehouse", "supply", "demand", "replenishment"
            ],
            "customer_management": [
                "customer", "client", "contact", "lead", "prospect", "relationship",
                "support", "ticket", "case", "communication"
            ],
            "financial_management": [
                "payment", "billing", "invoice", "accounting", "finance", "budget",
                "revenue", "cost", "pricing", "subscription", "fee"
            ],
            "content_management": [
                "content", "document", "file", "media", "article", "post",
                "page", "template", "asset", "publication"
            ],
            "notification_management": [
                "notification", "message", "email", "sms", "alert", "reminder",
                "communication", "broadcast", "subscription", "preference"
            ],
            "reporting_analytics": [
                "report", "analytics", "dashboard", "metric", "kpi", "insight",
                "data", "statistics", "visualization", "analysis"
            ]
        }
        
        # Entity type patterns
        self.entity_patterns = {
            "aggregate_root": ["manager", "service", "aggregate", "root"],
            "entity": ["entity", "model", "data", "record"],
            "value_object": ["value", "vo", "dto", "address", "money", "email"],
            "domain_service": ["service", "handler", "processor", "calculator"],
            "repository": ["repository", "repo", "store", "dao"],
            "factory": ["factory", "builder", "creator"]
        }
        
        # Business rule indicators
        self.business_rule_patterns = [
            "validate", "verify", "check", "ensure", "require", "must",
            "cannot", "should", "rule", "constraint", "policy", "business"
        ]
        
        # Ownership indicators
        self.ownership_patterns = {
            "team_ownership": ["team", "squad", "group", "department"],
            "service_ownership": ["service", "microservice", "api", "system"],
            "domain_ownership": ["domain", "bounded", "context", "area"],
            "data_ownership": ["database", "schema", "table", "collection"]
        }
    
    def analyze_business_capability(
        self, 
        project_path: str,
        file_patterns: Optional[List[str]] = None
    ) -> BusinessAnalysis:
        """
        Analyze business capability from project structure and code.
        
        Args:
            project_path: Path to project directory
            file_patterns: Optional list of file patterns to analyze
            
        Returns:
            BusinessAnalysis with capability insights
        """
        try:
            self.logger.info(f"Analyzing business capability for: {project_path}")
            
            # Analyze project structure for business indicators
            business_indicators = self._extract_business_indicators(
                project_path, file_patterns
            )
            
            # Identify entities and their characteristics
            entities = self._identify_business_entities(business_indicators)
            
            # Determine primary business capability
            primary_capability = self._determine_primary_capability(
                business_indicators, entities
            )
            
            # Identify secondary capabilities
            secondary_capabilities = self._identify_secondary_capabilities(
                business_indicators, entities
            )
            
            # Analyze domain complexity
            domain_complexity = self._assess_domain_complexity(
                entities, business_indicators
            )
            
            # Identify bounded contexts
            bounded_contexts = self._identify_bounded_contexts(business_indicators)
            
            # Analyze ownership patterns
            ownership_patterns = self._analyze_ownership_patterns(business_indicators)
            
            # Extract business rules
            business_rules = self._extract_business_rules(business_indicators)
            
            # Identify integration points
            integration_points = self._identify_integration_points(business_indicators)
            
            # Calculate confidence score
            confidence_score = self._calculate_analysis_confidence(
                business_indicators, entities
            )
            
            analysis = BusinessAnalysis(
                primary_capability=primary_capability,
                secondary_capabilities=secondary_capabilities,
                domain_complexity=domain_complexity,
                bounded_contexts=bounded_contexts,
                ownership_patterns=ownership_patterns,
                business_rules=business_rules,
                integration_points=integration_points,
                confidence_score=confidence_score
            )
            
            self.logger.info(
                f"Business analysis completed. Primary capability: {primary_capability.name}, "
                f"Confidence: {confidence_score:.2f}"
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing business capability: {e}")
            return self._create_fallback_analysis()
    
    def _extract_business_indicators(
        self, 
        project_path: str,
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Extract business indicators from project structure."""
        try:
            project_root = Path(project_path)
            if not project_root.exists():
                return self._create_empty_indicators()
            
            indicators = {
                "folder_names": [],
                "file_names": [],
                "class_names": [],
                "method_names": [],
                "domain_keywords": set(),
                "business_terms": set(),
                "technologies": set()
            }
            
            # Analyze directory structure
            for path in project_root.rglob("*"):
                if path.is_dir():
                    folder_name = path.name.lower()
                    indicators["folder_names"].append(folder_name)
                    self._extract_domain_keywords(folder_name, indicators)
                elif path.is_file() and self._is_code_file(path):
                    file_name = path.name.lower()
                    indicators["file_names"].append(file_name)
                    self._extract_domain_keywords(file_name, indicators)
                    
                    # Analyze file content for business terms
                    self._analyze_file_content(path, indicators)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error extracting business indicators: {e}")
            return self._create_empty_indicators()
    
    def _is_code_file(self, file_path: Path) -> bool:
        """Check if file is a code file worth analyzing."""
        code_extensions = {".cs", ".py", ".js", ".ts", ".java", ".php", ".rb", ".go"}
        return file_path.suffix.lower() in code_extensions
    
    def _extract_domain_keywords(
        self, 
        text: str, 
        indicators: Dict[str, Any]
    ) -> None:
        """Extract domain keywords from text."""
        text_lower = text.lower()
        
        # Check against known business domains
        for domain, keywords in self.business_domains.items():
            for keyword in keywords:
                if keyword in text_lower:
                    indicators["domain_keywords"].add(domain)
                    indicators["business_terms"].add(keyword)
    
    def _analyze_file_content(self, file_path: Path, indicators: Dict[str, Any]) -> None:
        """Analyze file content for business terms and patterns."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            content_lower = content.lower()
            
            # Extract class names (basic pattern matching)
            class_matches = re.findall(r'class\s+(\w+)', content, re.IGNORECASE)
            indicators["class_names"].extend([cls.lower() for cls in class_matches])
            
            # Extract method/function names
            method_matches = re.findall(
                r'(?:def|function|public|private|protected)\s+(\w+)', 
                content, 
                re.IGNORECASE
            )
            indicators["method_names"].extend([method.lower() for method in method_matches])
            
            # Look for business terms in content
            for domain, keywords in self.business_domains.items():
                for keyword in keywords:
                    if keyword in content_lower:
                        indicators["domain_keywords"].add(domain)
                        indicators["business_terms"].add(keyword)
            
        except Exception as e:
            self.logger.debug(f"Error analyzing file content {file_path}: {e}")
    
    def _identify_business_entities(
        self, 
        business_indicators: Dict[str, Any]
    ) -> List[BusinessEntity]:
        """Identify business entities from indicators."""
        entities = []
        
        # Combine all potential entity sources
        all_names = (
            business_indicators.get("class_names", []) +
            business_indicators.get("file_names", []) +
            business_indicators.get("folder_names", [])
        )
        
        # Score and classify entities
        entity_candidates = {}
        
        for name in all_names:
            # Skip common non-entity names
            if self._is_likely_entity(name):
                score = self._score_entity_candidate(name, business_indicators)
                if score > 0.3:  # Threshold for entity recognition
                    entity_candidates[name] = score
        
        # Create entity objects for top candidates
        top_candidates = sorted(
            entity_candidates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]  # Limit to top 10 entities
        
        for name, score in top_candidates:
            entity = BusinessEntity(
                name=name,
                type=self._determine_entity_type(name),
                attributes=self._extract_entity_attributes(name, business_indicators),
                relationships=self._identify_entity_relationships(name, business_indicators),
                business_rules=self._extract_entity_business_rules(name, business_indicators),
                confidence_score=score
            )
            entities.append(entity)
        
        return entities
    
    def _is_likely_entity(self, name: str) -> bool:
        """Check if name is likely to be a business entity."""
        # Skip technical/infrastructure terms
        skip_patterns = [
            "controller", "service", "repository", "dto", "dao", "config",
            "util", "helper", "test", "mock", "base", "abstract", "interface"
        ]
        
        return not any(pattern in name.lower() for pattern in skip_patterns)
    
    def _score_entity_candidate(
        self, 
        name: str, 
        business_indicators: Dict[str, Any]
    ) -> float:
        """Score entity candidate based on business relevance."""
        score = 0.0
        
        # Check if name contains business terms
        business_terms = business_indicators.get("business_terms", set())
        for term in business_terms:
            if term in name.lower():
                score += 0.3
        
        # Check if name appears in multiple contexts
        contexts = [
            business_indicators.get("class_names", []),
            business_indicators.get("file_names", []),
            business_indicators.get("folder_names", [])
        ]
        
        appearances = sum(1 for context in contexts if name in context)
        score += appearances * 0.2
        
        # Bonus for entity-like naming patterns
        if name.endswith(("entity", "model", "aggregate")):
            score += 0.2
        
        return min(score, 1.0)
    
    def _determine_entity_type(self, name: str) -> str:
        """Determine the type of business entity."""
        name_lower = name.lower()
        
        for entity_type, patterns in self.entity_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                return entity_type
        
        return "entity"  # Default type
    
    def _extract_entity_attributes(
        self, 
        entity_name: str, 
        business_indicators: Dict[str, Any]
    ) -> List[str]:
        """Extract potential attributes for an entity."""
        # This is a simplified approach - in a real implementation,
        # we would analyze code structure more deeply
        business_terms = business_indicators.get("business_terms", set())
        
        # Return terms that might be related to this entity
        related_terms = [
            term for term in business_terms
            if term != entity_name.lower() and len(term) > 2
        ]
        
        return list(related_terms)[:5]  # Limit to 5 potential attributes
    
    def _identify_entity_relationships(
        self, 
        entity_name: str, 
        business_indicators: Dict[str, Any]
    ) -> List[str]:
        """Identify potential relationships for an entity."""
        # Look for other entities that might be related
        class_names = business_indicators.get("class_names", [])
        
        # Simple heuristic: entities with shared business domain terms
        domain_keywords = business_indicators.get("domain_keywords", set())
        related_entities = []
        
        for class_name in class_names:
            if (class_name != entity_name.lower() and 
                self._is_likely_entity(class_name)):
                # Check if they share domain context
                for domain in domain_keywords:
                    domain_terms = self.business_domains.get(domain, [])
                    if any(term in class_name for term in domain_terms):
                        related_entities.append(class_name)
                        break
        
        return list(set(related_entities))[:3]  # Limit to 3 relationships
    
    def _extract_entity_business_rules(
        self, 
        entity_name: str, 
        business_indicators: Dict[str, Any]
    ) -> List[str]:
        """Extract potential business rules for an entity."""
        method_names = business_indicators.get("method_names", [])
        
        business_rules = []
        for method in method_names:
            if any(pattern in method for pattern in self.business_rule_patterns):
                if entity_name.lower() in method or any(
                    term in method for term in business_indicators.get("business_terms", set())
                ):
                    business_rules.append(f"Rule related to {method}")
        
        return business_rules[:3]  # Limit to 3 rules
    
    def _determine_primary_capability(
        self, 
        business_indicators: Dict[str, Any],
        entities: List[BusinessEntity]
    ) -> BusinessCapability:
        """Determine the primary business capability."""
        domain_keywords = business_indicators.get("domain_keywords", set())
        
        if not domain_keywords:
            return self._create_generic_capability(entities)
        
        # Find the most prominent domain
        domain_scores = {}
        for domain in domain_keywords:
            score = 0
            domain_terms = self.business_domains.get(domain, [])
            
            # Score based on term frequency
            for term in domain_terms:
                if term in business_indicators.get("business_terms", set()):
                    score += 1
            
            domain_scores[domain] = score
        
        primary_domain = max(domain_scores, key=domain_scores.get)
        
        return BusinessCapability(
            name=primary_domain.replace("_", " ").title(),
            description=f"Primary business capability focused on {primary_domain}",
            scope="primary",
            entities=entities[:5],  # Top 5 entities
            services=self._extract_services(business_indicators),
            bounded_context=self._determine_bounded_context(primary_domain),
            ownership_indicators=self._extract_ownership_indicators(business_indicators),
            complexity=self._assess_domain_complexity(entities, business_indicators)
        )
    
    def _identify_secondary_capabilities(
        self, 
        business_indicators: Dict[str, Any],
        entities: List[BusinessEntity]
    ) -> List[BusinessCapability]:
        """Identify secondary business capabilities."""
        domain_keywords = business_indicators.get("domain_keywords", set())
        
        secondary_capabilities = []
        
        # Consider domains with moderate presence
        for domain in domain_keywords:
            if domain != self._get_primary_domain(business_indicators):
                capability = BusinessCapability(
                    name=domain.replace("_", " ").title(),
                    description=f"Secondary capability for {domain}",
                    scope="secondary",
                    entities=[e for e in entities if domain in e.name.lower()][:2],
                    services=[],
                    bounded_context=None,
                    ownership_indicators=[],
                    complexity=DomainComplexity.SIMPLE
                )
                secondary_capabilities.append(capability)
        
        return secondary_capabilities[:3]  # Limit to 3 secondary capabilities
    
    def _assess_domain_complexity(
        self, 
        entities: List[BusinessEntity],
        business_indicators: Dict[str, Any]
    ) -> DomainComplexity:
        """Assess the complexity of the business domain."""
        entity_count = len(entities)
        domain_count = len(business_indicators.get("domain_keywords", set()))
        business_term_count = len(business_indicators.get("business_terms", set()))
        
        complexity_score = entity_count + domain_count + (business_term_count * 0.1)
        
        if complexity_score < 3:
            return DomainComplexity.SIMPLE
        elif complexity_score < 8:
            return DomainComplexity.MODERATE
        elif complexity_score < 15:
            return DomainComplexity.COMPLEX
        else:
            return DomainComplexity.ENTERPRISE
    
    def _identify_bounded_contexts(
        self, 
        business_indicators: Dict[str, Any]
    ) -> List[str]:
        """Identify potential bounded contexts."""
        folder_names = business_indicators.get("folder_names", [])
        domain_keywords = business_indicators.get("domain_keywords", set())
        
        contexts = []
        
        # Look for context indicators in folder structure
        for folder in folder_names:
            if any(keyword in folder for keyword in ["context", "domain", "module", "service"]):
                contexts.append(folder)
        
        # Add domain-based contexts
        for domain in domain_keywords:
            contexts.append(domain.replace("_", " ").title() + " Context")
        
        return list(set(contexts))[:5]  # Limit to 5 contexts
    
    def _analyze_ownership_patterns(
        self, 
        business_indicators: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze ownership patterns in the codebase."""
        ownership = {}
        
        for pattern_type, keywords in self.ownership_patterns.items():
            indicators = []
            
            # Check folder names for ownership indicators
            folder_names = business_indicators.get("folder_names", [])
            for folder in folder_names:
                if any(keyword in folder for keyword in keywords):
                    indicators.append(folder)
            
            if indicators:
                ownership[pattern_type] = indicators[:3]
        
        return ownership
    
    def _extract_business_rules(
        self, 
        business_indicators: Dict[str, Any]
    ) -> List[str]:
        """Extract business rules from method names and patterns."""
        method_names = business_indicators.get("method_names", [])
        
        business_rules = []
        for method in method_names:
            if any(pattern in method for pattern in self.business_rule_patterns):
                business_rules.append(f"Business rule: {method}")
        
        return list(set(business_rules))[:10]  # Limit to 10 rules
    
    def _identify_integration_points(
        self, 
        business_indicators: Dict[str, Any]
    ) -> List[str]:
        """Identify potential integration points."""
        folder_names = business_indicators.get("folder_names", [])
        file_names = business_indicators.get("file_names", [])
        
        integration_keywords = [
            "api", "client", "service", "gateway", "proxy", "adapter",
            "integration", "external", "third-party", "webhook"
        ]
        
        integration_points = []
        
        all_names = folder_names + file_names
        for name in all_names:
            if any(keyword in name.lower() for keyword in integration_keywords):
                integration_points.append(name)
        
        return list(set(integration_points))[:5]  # Limit to 5 integration points
    
    def _calculate_analysis_confidence(
        self, 
        business_indicators: Dict[str, Any],
        entities: List[BusinessEntity]
    ) -> float:
        """Calculate confidence score for the business analysis."""
        factors = {
            "domain_keywords": len(business_indicators.get("domain_keywords", set())) * 0.2,
            "business_terms": len(business_indicators.get("business_terms", set())) * 0.1,
            "entities": len(entities) * 0.15,
            "structure": len(business_indicators.get("folder_names", [])) * 0.05
        }
        
        confidence = sum(factors.values())
        return min(confidence, 1.0)
    
    def _create_empty_indicators(self) -> Dict[str, Any]:
        """Create empty business indicators structure."""
        return {
            "folder_names": [],
            "file_names": [],
            "class_names": [],
            "method_names": [],
            "domain_keywords": set(),
            "business_terms": set(),
            "technologies": set()
        }
    
    def _create_generic_capability(self, entities: List[BusinessEntity]) -> BusinessCapability:
        """Create generic business capability when no specific domain is detected."""
        return BusinessCapability(
            name="Generic Business Logic",
            description="General business capability with unspecified domain",
            scope="primary",
            entities=entities[:3],
            services=[],
            bounded_context=None,
            ownership_indicators=[],
            complexity=DomainComplexity.SIMPLE
        )
    
    def _create_fallback_analysis(self) -> BusinessAnalysis:
        """Create fallback analysis when analysis fails."""
        generic_capability = BusinessCapability(
            name="Unknown Business Capability",
            description="Unable to determine business capability",
            scope="unknown",
            entities=[],
            services=[],
            bounded_context=None,
            ownership_indicators=[],
            complexity=DomainComplexity.SIMPLE
        )
        
        return BusinessAnalysis(
            primary_capability=generic_capability,
            secondary_capabilities=[],
            domain_complexity=DomainComplexity.SIMPLE,
            bounded_contexts=[],
            ownership_patterns={},
            business_rules=[],
            integration_points=[],
            confidence_score=0.1
        )
    
    def _extract_services(self, business_indicators: Dict[str, Any]) -> List[str]:
        """Extract potential services from indicators."""
        class_names = business_indicators.get("class_names", [])
        
        services = [
            name for name in class_names
            if "service" in name.lower() or "handler" in name.lower()
        ]
        
        return services[:5]  # Limit to 5 services
    
    def _determine_bounded_context(self, domain: str) -> Optional[str]:
        """Determine bounded context for a domain."""
        context_mapping = {
            "user_management": "User Management Context",
            "order_management": "Order Management Context",
            "inventory_management": "Inventory Context",
            "customer_management": "Customer Relationship Context",
            "financial_management": "Financial Context",
            "content_management": "Content Management Context",
            "notification_management": "Communication Context",
            "reporting_analytics": "Analytics Context"
        }
        
        return context_mapping.get(domain)
    
    def _extract_ownership_indicators(
        self, 
        business_indicators: Dict[str, Any]
    ) -> List[str]:
        """Extract ownership indicators from the codebase."""
        ownership_indicators = []
        
        folder_names = business_indicators.get("folder_names", [])
        for folder in folder_names:
            if any(pattern in folder for patterns in self.ownership_patterns.values() for pattern in patterns):
                ownership_indicators.append(folder)
        
        return ownership_indicators[:3]
    
    def _get_primary_domain(self, business_indicators: Dict[str, Any]) -> Optional[str]:
        """Get the primary domain from business indicators."""
        domain_keywords = business_indicators.get("domain_keywords", set())
        
        if not domain_keywords:
            return None
        
        # Return the first domain (could be enhanced with better scoring)
        return next(iter(domain_keywords))