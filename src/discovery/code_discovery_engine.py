"""
Code discovery engine for finding relevant code references in event flow analysis.

This module implements code discovery using existing vector store and graph store
infrastructure to find event handlers, service integrations, and data flows.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import asyncio

from src.vectorstores.base_store import BaseStore  
from src.analyzers.event_flow_analyzer import WorkflowPattern, EventFlowQuery
from src.utils.logging import get_logger


@dataclass
class CodeReference:
    """Code reference with metadata for event flow analysis."""
    
    repository: str
    file_path: str
    line_numbers: List[int]
    method_name: str
    context_type: str  # 'controller', 'service', 'handler', 'model', 'component'
    language: str
    content_snippet: str
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CodeDiscoveryEngine:
    """
    Engine for discovering relevant code using vector similarity search.
    
    This class leverages existing vector store infrastructure to find code
    references relevant to event flow analysis queries.
    """
    
    def __init__(self, vector_store: BaseStore):
        """
        Initialize code discovery engine.
        
        Args:
            vector_store: Vector store instance for similarity search
        """
        self.vector_store = vector_store
        self.logger = get_logger(self.__class__.__name__)
        
        # Define search terms for different workflow patterns
        self._pattern_search_terms = {
            WorkflowPattern.ORDER_PROCESSING: [
                "order processing", "place order", "checkout", "payment processing",
                "order validation", "order creation", "billing", "fulfillment"
            ],
            WorkflowPattern.USER_AUTHENTICATION: [
                "user authentication", "login", "signup", "session management",
                "token validation", "password", "authorization", "oauth"
            ],
            WorkflowPattern.DATA_PIPELINE: [
                "data processing", "etl", "data pipeline", "batch processing",
                "data transformation", "data ingestion", "data migration"
            ],
            WorkflowPattern.API_REQUEST_FLOW: [
                "api endpoint", "request handler", "response processing",
                "service integration", "http request", "rest api", "webhook"
            ],
            WorkflowPattern.EVENT_DRIVEN: [
                "event handler", "message processing", "event publishing",
                "event subscription", "queue processing", "notification"
            ],
            WorkflowPattern.GENERIC_WORKFLOW: [
                "business logic", "workflow", "process", "handler", "service"
            ]
        }
        
        # Context type patterns for classification
        self._context_patterns = {
            'controller': ['controller', 'endpoint', 'route', 'api', 'handler'],
            'service': ['service', 'business', 'logic', 'process', 'manager'],
            'handler': ['handler', 'processor', 'listener', 'consumer'],
            'model': ['model', 'entity', 'data', 'schema', 'dto'],
            'component': ['component', 'module', 'utility', 'helper']
        }
    
    def find_event_handlers(self, workflow: WorkflowPattern, max_results: int = 10) -> List[CodeReference]:
        """
        Find event handlers using vector similarity search.
        
        Args:
            workflow: Workflow pattern to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of CodeReference objects for event handlers
        """
        search_terms = self._pattern_search_terms.get(workflow, 
                                                     self._pattern_search_terms[WorkflowPattern.GENERIC_WORKFLOW])
        
        all_references = []
        
        for term in search_terms[:3]:  # Limit search terms to avoid too many calls
            try:
                # Use vector store similarity search (sync method)
                documents = self.vector_store.similarity_search(
                    query=term,
                    k=max_results // len(search_terms[:3]) + 1
                )
                
                # Convert Documents to CodeReference objects
                references = self._convert_documents_to_code_references(documents, term)
                all_references.extend(references)
                
                self.logger.debug(f"Found {len(references)} references for term: {term}")
                
            except Exception as e:
                self.logger.error(f"Error searching for term '{term}': {e}")
                continue
        
        # Remove duplicates and sort by relevance
        unique_references = self._deduplicate_references(all_references)
        sorted_references = sorted(unique_references, key=lambda x: x.relevance_score, reverse=True)
        
        return sorted_references[:max_results]
    
    def discover_service_integrations(self, entities: List[str], max_results: int = 8) -> List[CodeReference]:
        """
        Discover service integrations using entity-based search.
        
        Args:
            entities: List of entities to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of CodeReference objects for service integrations
        """
        all_references = []
        
        for entity in entities:
            # Create search queries for integration patterns
            integration_queries = [
                f"{entity} service integration",
                f"{entity} api call",
                f"{entity} client",
                f"{entity} adapter"
            ]
            
            for query in integration_queries:
                try:
                    results = self.vector_store.similarity_search(
                        query=query,
                        k=3  # Fewer results per query
                    )
                    
                    references = self._convert_documents_to_code_references(results, query)
                    all_references.extend(references)
                    
                except Exception as e:
                    self.logger.error(f"Error searching for integration query '{query}': {e}")
                    continue
        
        # Deduplicate and sort
        unique_references = self._deduplicate_references(all_references)
        sorted_references = sorted(unique_references, key=lambda x: x.relevance_score, reverse=True)
        
        return sorted_references[:max_results]
    
    def map_data_flow(self, workflow_query: EventFlowQuery, max_results: int = 12) -> List[CodeReference]:
        """
        Map data flow using existing vector store capabilities.
        
        Args:
            workflow_query: Parsed event flow query
            max_results: Maximum number of results to return
            
        Returns:
            List of CodeReference objects representing data flow
        """
        # Combine entities and actions for comprehensive search
        search_components = workflow_query.entities + workflow_query.actions
        
        # Create search queries that focus on data flow
        flow_queries = []
        for component in search_components:
            flow_queries.extend([
                f"{component} data flow",
                f"{component} processing",
                f"{component} validation",
                f"{component} transformation"
            ])
        
        all_references = []
        
        # Limit queries to prevent excessive API calls
        for query in flow_queries[:6]:
            try:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=3
                )
                
                references = self._convert_documents_to_code_references(results, query)
                all_references.extend(references)
                
            except Exception as e:
                self.logger.error(f"Error searching for data flow query '{query}': {e}")
                continue
        
        # Deduplicate and sort
        unique_references = self._deduplicate_references(all_references)
        sorted_references = sorted(unique_references, key=lambda x: x.relevance_score, reverse=True)
        
        return sorted_references[:max_results]
    
    def find_relevant_code(self, workflow_query: EventFlowQuery, max_total_results: int = 15) -> List[CodeReference]:
        """
        Find all relevant code for an event flow query.
        
        Args:
            workflow_query: Parsed event flow query
            max_total_results: Maximum total results to return
            
        Returns:
            List of CodeReference objects relevant to the workflow
        """
        # Distribute results across different discovery methods
        handler_results = max_total_results // 3
        integration_results = max_total_results // 3  
        flow_results = max_total_results - handler_results - integration_results
        
        try:
            # Run discovery methods sequentially (since they're now sync)
            handlers = self.find_event_handlers(workflow_query.workflow, handler_results)
            integrations = self.discover_service_integrations(workflow_query.entities, integration_results)
            data_flow = self.map_data_flow(workflow_query, flow_results)
            
            # Combine all references
            all_references = handlers + integrations + data_flow
            
            # Final deduplication and sorting
            unique_references = self._deduplicate_references(all_references)
            sorted_references = sorted(unique_references, key=lambda x: x.relevance_score, reverse=True)
            
            self.logger.info(f"Found {len(sorted_references)} relevant code references for workflow query")
            return sorted_references[:max_total_results]
            
        except Exception as e:
            self.logger.error(f"Error in find_relevant_code: {e}")
            return []
    
    def _convert_documents_to_code_references(self, documents: List[Any], search_term: str) -> List[CodeReference]:
        """Convert vector store search results (Documents) to CodeReference objects."""
        references = []
        
        for doc in documents:
            try:
                # Extract metadata from Document
                metadata = getattr(doc, 'metadata', {})
                content = getattr(doc, 'page_content', '')
                
                # Filter out irrelevant files early
                if not self._is_relevant_code_content(content, metadata):
                    self.logger.debug(f"Skipping irrelevant content from {metadata.get('source', 'unknown')}")
                    continue
                
                # Calculate a basic score (could be enhanced)
                score = 0.8  # Default score since we don't have similarity scores from basic search
                
                # Extract file path and repository
                source = metadata.get('source', '')
                repository = metadata.get('repository', 'unknown')
                language = metadata.get('language', 'unknown')
                
                # Determine context type
                context_type = self._classify_context_type(content, source)
                
                # Extract method name (improved heuristic)
                method_name = self._extract_method_name(content, language)
                
                # Create code reference
                ref = CodeReference(
                    repository=repository,
                    file_path=source,
                    line_numbers=[],  # Could be enhanced to extract line numbers
                    method_name=method_name,
                    context_type=context_type,
                    language=language,
                    content_snippet=content[:200] + "..." if len(content) > 200 else content,
                    relevance_score=score,
                    metadata=metadata
                )
                
                references.append(ref)
                
            except Exception as e:
                self.logger.error(f"Error converting Document to CodeReference: {e}")
                continue
        
        return references
    
    def _is_relevant_code_content(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Filter out irrelevant content that shouldn't be included in event flow analysis.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            bool: True if content is relevant for code analysis, False otherwise
        """
        source = metadata.get('source', '').lower()
        language = metadata.get('language', '').lower()
        content_lower = content.lower()
        
        # Skip configuration files that don't contain business logic
        irrelevant_files = [
            'package.json', 'package-lock.json', 'yarn.lock',
            '.gitignore', '.env', '.env.example',
            'dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
            'readme.md', 'license', 'license.txt',
            'changelog', 'changelog.md', 'changes.md',
            'makefile', 'requirements.txt', 'setup.py',
            'pyproject.toml', 'poetry.lock',
            'tsconfig.json', 'webpack.config.js',
            '.editorconfig', '.eslintrc', '.prettierrc'
        ]
        
        # Check if this is an irrelevant file
        filename = source.split('/')[-1] if '/' in source else source
        if any(irrelevant in filename for irrelevant in irrelevant_files):
            return False
        
        # Skip pure configuration content
        config_indicators = [
            '"dependencies":', '"devDependencies":', '"scripts":',
            'FROM docker', 'RUN apt-get', 'COPY ', 'WORKDIR',
            '# This is a', '## Installation', '## Getting Started',
            'MIT License', 'Copyright (c)', 'Licensed under'
        ]
        
        if any(indicator in content for indicator in config_indicators):
            return False
        
        # Skip very short content that's likely not meaningful code
        if len(content.strip()) < 50:
            return False
        
        # Skip content that's mostly whitespace or comments
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        if len(non_empty_lines) < 2:  # Changed from 3 to 2 - even a single function is meaningful
            return False
        
        # Prefer actual code files
        code_indicators = [
            'def ', 'function ', 'class ', 'public class',
            'private ', 'protected ', 'public ',
            'const ', 'let ', 'var ',
            'import ', 'from ', 'require(',
            'export ', 'module.exports',
            'namespace ', 'interface ', 'enum ',
            'struct ', 'union ', 'typedef'
        ]
        
        has_code_indicators = any(indicator in content for indicator in code_indicators)
        
        # For event flow analysis, prioritize files that seem to contain business logic
        business_logic_indicators = [
            'order', 'payment', 'user', 'auth', 'login',
            'process', 'handle', 'service', 'controller',
            'api', 'endpoint', 'request', 'response',
            'validate', 'create', 'update', 'delete',
            'notification', 'event', 'message', 'queue'
        ]
        
        has_business_logic = any(indicator in content_lower for indicator in business_logic_indicators)
        
        # Accept if it has code indicators OR business logic (to catch both code files and relevant config)
        return has_code_indicators or has_business_logic
    
    def _classify_context_type(self, content: str, file_path: str) -> str:
        """Classify the context type based on content and file path."""
        content_lower = content.lower()
        file_path_lower = file_path.lower()
        
        # Check file path first
        for context_type, patterns in self._context_patterns.items():
            if any(pattern in file_path_lower for pattern in patterns):
                return context_type
        
        # Check content
        for context_type, patterns in self._context_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                return context_type
        
        return 'component'  # Default
    
    def _extract_method_name(self, content: str, language: str) -> str:
        """Extract method name from content (improved heuristic)."""
        lines = content.split('\n')
        language_lower = language.lower()
        
        # Skip non-code files that shouldn't have methods
        if language_lower in ['json', 'yaml', 'yml', 'xml', 'html', 'css', 'markdown', 'md', 'txt']:
            # For configuration files, try to extract a meaningful identifier
            file_type_map = {
                'json': 'config_data',
                'yaml': 'config_data', 
                'yml': 'config_data',
                'xml': 'xml_data',
                'html': 'html_content',
                'css': 'styles',
                'markdown': 'documentation',
                'md': 'documentation',
                'txt': 'text_content'
            }
            return file_type_map.get(language_lower, 'data_file')
        
        # Try to extract meaningful method names for code files
        if language_lower in ['python', 'py']:
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith('def ') and '(' in line_stripped:
                    # Extract function name
                    start = line_stripped.find('def ') + 4
                    end = line_stripped.find('(')
                    if start < end:
                        method_name = line_stripped[start:end].strip()
                        if method_name and method_name.isidentifier():
                            return method_name
                # Also check for class methods
                elif line_stripped.startswith('class '):
                    class_name = line_stripped[6:].split('(')[0].split(':')[0].strip()
                    if class_name and class_name.isidentifier():
                        return f"{class_name}_class"
        
        elif language_lower in ['javascript', 'typescript', 'js', 'ts']:
            for line in lines:
                line_stripped = line.strip()
                # Function declarations
                if 'function ' in line_stripped and '(' in line_stripped:
                    start = line_stripped.find('function ') + 9
                    end = line_stripped.find('(')
                    if start < end:
                        method_name = line_stripped[start:end].strip()
                        if method_name and method_name.replace('_', '').replace('$', '').isalnum():
                            return method_name
                # Arrow functions
                elif '=>' in line_stripped and '=' in line_stripped:
                    # Look for const/let/var functionName = 
                    parts = line_stripped.split('=')[0].strip()
                    if parts.startswith(('const ', 'let ', 'var ')):
                        method_name = parts.split()[-1].strip()
                        if method_name and method_name.replace('_', '').replace('$', '').isalnum():
                            return method_name
        
        elif language_lower in ['csharp', 'c#', 'cs']:
            for line in lines:
                line_stripped = line.strip()
                # Method declarations
                if ('public ' in line_stripped or 'private ' in line_stripped or 'protected ' in line_stripped) and '(' in line_stripped:
                    words = line_stripped.split()
                    for i, word in enumerate(words):
                        if '(' in word:
                            method_name = word.split('(')[0].strip()
                            if method_name and method_name.replace('_', '').isalnum():
                                return method_name
                # Class declarations
                elif line_stripped.startswith('public class ') or line_stripped.startswith('class '):
                    class_name = line_stripped.split('class ')[1].split()[0].strip()
                    if class_name and class_name.replace('_', '').isalnum():
                        return f"{class_name}_class"
        
        elif language_lower in ['java']:
            for line in lines:
                line_stripped = line.strip()
                if ('public ' in line_stripped or 'private ' in line_stripped) and '(' in line_stripped:
                    words = line_stripped.split()
                    for i, word in enumerate(words):
                        if '(' in word:
                            method_name = word.split('(')[0].strip()
                            if method_name and method_name.replace('_', '').isalnum():
                                return method_name
        
        elif language_lower in ['go']:
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith('func ') and '(' in line_stripped:
                    start = line_stripped.find('func ') + 5
                    end = line_stripped.find('(')
                    if start < end:
                        method_name = line_stripped[start:end].strip()
                        if method_name and method_name.replace('_', '').isalnum():
                            return method_name
        
        # If no specific method found, try to extract a meaningful identifier from the content
        # Look for common patterns that might indicate the purpose
        content_lower = content.lower()
        if 'order' in content_lower:
            return 'order_handler'
        elif 'payment' in content_lower:
            return 'payment_handler'
        elif 'user' in content_lower or 'auth' in content_lower:
            return 'user_handler'
        elif 'api' in content_lower or 'endpoint' in content_lower:
            return 'api_handler'
        elif 'service' in content_lower:
            return 'service_handler'
        elif 'controller' in content_lower:
            return 'controller_handler'
        elif 'model' in content_lower or 'entity' in content_lower:
            return 'data_model'
        elif 'config' in content_lower or 'setting' in content_lower:
            return 'configuration'
        
        # Last resort: return a generic but meaningful name
        return 'code_component'
    
    def _deduplicate_references(self, references: List[CodeReference]) -> List[CodeReference]:
        """Remove duplicate code references based on file path and method name."""
        seen = set()
        unique_refs = []
        
        for ref in references:
            # Create a unique key based on file path and method name
            key = (ref.file_path, ref.method_name)
            if key not in seen:
                seen.add(key)
                unique_refs.append(ref)
        
        return unique_refs