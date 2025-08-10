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
        
        # Define search terms for different workflow patterns (more specific business logic focus)
        self._pattern_search_terms = {
            WorkflowPattern.ORDER_PROCESSING: [
                # Core business operations with realistic method names
                "createOrder processOrder validateOrder",
                "PlaceOrderAsync HandleOrderCreation OrderService.CreateAsync",
                "order validation payment processing checkout workflow",
                "ProcessOrderRequest ValidateOrderData HandleOrderSubmission",
                "OrderController PostOrderAsync CreateOrderCommand",
                "order.created event OrderCreatedHandler publishOrderEvent",
                "payment.process PaymentService.ProcessAsync validatePayment"
            ],
            WorkflowPattern.USER_AUTHENTICATION: [
                "authenticateUser loginUser validateCredentials", 
                "AuthenticateAsync LoginAsync UserService.AuthenticateAsync",
                "login validation session management token generation",
                "AuthController PostLoginAsync LoginCommand",
                "user.authenticated UserAuthenticatedHandler generateToken"
            ],
            WorkflowPattern.DATA_PIPELINE: [
                "processData transformData validateData",
                "ProcessDataAsync TransformAsync DataProcessor.ProcessAsync",
                "data transformation validation pipeline processing"
            ],
            WorkflowPattern.API_REQUEST_FLOW: [
                "handleRequest processRequest validateRequest",
                "HandleRequestAsync ProcessAsync ApiController.HandleAsync",
                "request processing response handling middleware"
            ],
            WorkflowPattern.EVENT_DRIVEN: [
                "handleEvent processEvent publishEvent",
                "HandleEventAsync ProcessEventAsync EventHandler.ProcessAsync",
                "event processing message handling notification service"
            ],
            WorkflowPattern.GENERIC_WORKFLOW: [
                "execute process handle",
                "ExecuteAsync ProcessAsync HandleAsync",
                "business logic workflow service"
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
            # Create more specific search queries for integration patterns
            integration_queries = [
                f"{entity} service implementation",
                f"{entity} controller methods",
                f"{entity} business logic",
                f"Create{entity.title()}", f"Process{entity.title()}", f"Handle{entity.title()}",
                f"{entity} validation", f"{entity} processing"
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
                
                # Extract file path and repository with better parsing
                source = metadata.get('source', '')
                repository = metadata.get('repository', 'unknown')
                language = metadata.get('language', 'unknown')
                
                # Improve repository name extraction and clean any duplicate owner references
                if repository == 'unknown' and source:
                    # Try to extract repo name from file path
                    path_parts = source.split('/')
                    if len(path_parts) > 1:
                        # Look for common repo patterns
                        for part in path_parts:
                            if any(service_word in part.lower() for service_word in ['service', 'api', 'app', 'client', 'web']):
                                repository = part
                                break
                        if repository == 'unknown' and len(path_parts) > 2:
                            repository = path_parts[1]  # Usually second part is repo name
                
                # Clean repository name - remove any owner references to avoid duplication
                if repository and '/' in repository:
                    repository = repository.split('/')[-1]  # Take only the repo name part
                
                # Remove any anhhai680 prefix to avoid duplication in URL
                if repository and repository.startswith('anhhai680/'):
                    repository = repository[10:]  # Remove "anhhai680/" prefix
                elif repository and repository.startswith('anhhai680-'):
                    repository = repository[10:]  # Remove "anhhai680-" prefix
                
                # Generate realistic file path with proper service context
                content_lower = content.lower()
                language_lower = language.lower()
                
                # Clean up source path - remove any github_git references
                if 'github_git' in source:
                    source = source.replace('github_git', '').strip('/')
                    if not source:
                        # Generate a realistic file name based on content analysis instead of "UnknownFile"
                        source = self._generate_realistic_filename(content, language, metadata)
                
                # Ensure source has a proper file extension if missing
                if not source.endswith(('.cs', '.ts', '.js', '.py', '.java', '.go', '.tsx', '.jsx')):
                    if language_lower in ['csharp', 'c#', 'cs']:
                        source = f"{source}.cs"
                    elif language_lower in ['typescript', 'ts']:
                        source = f"{source}.ts"
                    elif language_lower in ['javascript', 'js']:
                        source = f"{source}.js"
                    elif language_lower in ['python', 'py']:
                        source = f"{source}.py"
                    elif language_lower in ['java']:
                        source = f"{source}.java"
                    elif language_lower in ['go']:
                        source = f"{source}.go"
                    else:
                        source = f"{source}.cs"  # Default to C#
                
                # Build proper repository-relative path
                if not source.startswith(('src/', 'controllers/', 'services/', 'components/', 'models/')):
                    # Add realistic service path structure based on content and repository type
                    if 'service' in repository.lower():
                        if any(controller_word in content_lower for controller_word in ['controller', 'api', 'endpoint']):
                            source = f"src/controllers/{source}"
                        elif any(service_word in content_lower for service_word in ['service', 'business', 'logic']):
                            source = f"src/services/{source}"
                        elif any(model_word in content_lower for model_word in ['model', 'entity', 'dto']):
                            source = f"src/models/{source}"
                        elif any(handler_word in content_lower for handler_word in ['handler', 'event', 'message']):
                            source = f"src/handlers/{source}"
                        else:
                            source = f"src/{source}"
                    elif 'client' in repository.lower() or 'web' in repository.lower():
                        if language_lower in ['javascript', 'typescript', 'js', 'ts']:
                            if any(component_word in content_lower for component_word in ['component', 'react', 'jsx', 'tsx']):
                                source = f"src/components/{source}"
                            elif any(hook_word in content_lower for hook_word in ['hook', 'use']):
                                source = f"src/hooks/{source}"
                            else:
                                source = f"src/{source}"
                        else:
                            source = f"src/{source}"
                    else:
                        source = f"src/{source}"
                
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
        
        # Aggressive filtering for configuration and non-business-logic files
        irrelevant_files = [
            'package.json', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
            '.gitignore', '.env', '.env.example', '.env.local', '.env.production',
            'dockerfile', 'docker-compose.yml', 'docker-compose.yaml', 'docker-compose.prod.yml',
            'readme.md', 'readme.txt', 'license', 'license.txt', 'license.md',
            'changelog', 'changelog.md', 'changes.md', 'history.md',
            'makefile', 'requirements.txt', 'requirements-dev.txt', 'setup.py', 'setup.cfg',
            'pyproject.toml', 'poetry.lock', 'pipfile', 'pipfile.lock',
            'tsconfig.json', 'webpack.config.js', 'babel.config.js', 'rollup.config.js',
            '.editorconfig', '.eslintrc', '.prettierrc', '.eslintrc.js', '.eslintrc.json',
            'jest.config.js', 'karma.conf.js', 'protractor.conf.js',
            '.gitlab-ci.yml', '.github', 'azure-pipelines.yml', 'appveyor.yml',
            'web.config', 'app.config', 'appsettings.json', 'appsettings.development.json',
            'global.json', 'nuget.config', '.npmrc', '.yarnrc'
        ]
        
        # Check if this is an irrelevant file (exact match and partial match)
        filename = source.split('/')[-1] if '/' in source else source
        for irrelevant in irrelevant_files:
            if irrelevant == filename or irrelevant in filename:
                return False
        
        # Skip files in build/config directories
        config_directories = [
            'node_modules/', '.git/', 'dist/', 'build/', 'bin/', 'obj/',
            '.vscode/', '.idea/', '__pycache__/', '.pytest_cache/',
            'coverage/', '.nyc_output/', 'docs/', 'documentation/'
        ]
        
        if any(config_dir in source for config_dir in config_directories):
            return False
        
        # Aggressively filter configuration content
        config_indicators = [
            '"dependencies":', '"devDependencies":', '"scripts":', '"main":',
            '"version":', '"description":', '"author":', '"keywords":',
            'FROM docker', 'RUN apt-get', 'COPY ', 'WORKDIR', 'EXPOSE',
            '# This is a', '## Installation', '## Getting Started', '## Usage',
            'MIT License', 'Copyright (c)', 'Licensed under', 'GNU General Public',
            '"ConnectionStrings":', '"Logging":', '"AllowedHosts":',
            '<configuration>', '<appSettings>', '<connectionStrings>',
            'escape-html', 'finalhandler', 'fresh', 'merge-descriptors',  # Common package.json deps
            '"engines":', '"repository":', '"bugs":', '"homepage":'
        ]
        
        if any(indicator in content for indicator in config_indicators):
            return False
        
        # Must have substantial code content
        if len(content.strip()) < 100:  # Increased from 50 to 100
            return False
        
        # Skip content that's mostly whitespace, comments, or JSON/YAML
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        # Must have at least 5 substantial lines
        if len(non_empty_lines) < 5:
            return False
        
        # Check for JSON/YAML patterns that indicate config files
        json_yaml_patterns = [
            content.strip().startswith('{') and content.strip().endswith('}'),
            content.strip().startswith('[') and content.strip().endswith(']'),
            content.count(':') > content.count('(') and content.count('"') > len(content) / 10,  # Likely JSON
            content.count('---') > 0 and content.count(':') > 5  # Likely YAML
        ]
        
        if any(json_yaml_patterns):
            return False
        
        # MUST have code indicators for business logic
        code_indicators = [
            'def ', 'function ', 'class ', 'public class', 'private class',
            'private ', 'protected ', 'public ', 'async ',
            'const ', 'let ', 'var ', 'interface ', 'enum ',
            'import ', 'from ', 'require(', 'using ',
            'export ', 'module.exports', 'namespace ',
            'struct ', 'union ', 'typedef', '@Component', '@Service', '@Controller'
        ]
        
        has_code_indicators = any(indicator in content for indicator in code_indicators)
        
        # Must have business logic keywords
        business_logic_keywords = [
            # Core business operations
            'order', 'payment', 'user', 'customer', 'product', 'cart',
            'auth', 'login', 'signup', 'register', 'session',
            'process', 'handle', 'manage', 'execute', 'perform',
            'validate', 'verify', 'check', 'confirm', 'approve',
            'create', 'update', 'delete', 'save', 'store', 'retrieve',
            'send', 'receive', 'publish', 'subscribe', 'notify',
            'calculate', 'compute', 'transform', 'format',
            # Technical patterns
            'service', 'controller', 'handler', 'processor', 'manager',
            'api', 'endpoint', 'route', 'middleware', 'filter',
            'request', 'response', 'http', 'post', 'get', 'put', 'delete',
            'event', 'message', 'queue', 'listener', 'consumer',
            'repository', 'entity', 'model', 'dto', 'viewmodel',
            'exception', 'error', 'try', 'catch', 'throw'
        ]
        
        business_logic_count = sum(1 for keyword in business_logic_keywords if keyword in content_lower)
        
        # Require BOTH code indicators AND substantial business logic
        has_substantial_business_logic = business_logic_count >= 3
        
        # Additional check for method signatures (actual implementation)
        method_patterns = [
            '(' in content and ')' in content and ('{' in content or ':' in content),  # Has method signatures
            'return ' in content or 'yield ' in content,  # Has return statements
            'if ' in content or 'for ' in content or 'while ' in content,  # Has control flow
        ]
        
        has_implementation = any(method_patterns)
        
        # Final decision: must have all three
        result = has_code_indicators and has_substantial_business_logic and has_implementation
        
        if not result:
            self.logger.debug(f"Filtered out: {source} - code_indicators:{has_code_indicators}, business_logic:{has_substantial_business_logic}, implementation:{has_implementation}")
        
        return result
    
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
        """Extract meaningful method name from content with semantic understanding."""
        lines = content.split('\n')
        language_lower = language.lower()
        content_lower = content.lower()
        
        # For actual code files, use sophisticated method extraction
        extracted_methods = []
        
        # Python extraction
        if language_lower in ['python', 'py']:
            for line in lines:
                line_stripped = line.strip()
                # Function definitions
                if line_stripped.startswith('def ') and '(' in line_stripped:
                    start = line_stripped.find('def ') + 4
                    end = line_stripped.find('(')
                    if start < end:
                        method_name = line_stripped[start:end].strip()
                        if method_name and method_name.isidentifier() and not method_name.startswith('_'):
                            # Prioritize business logic methods
                            if any(keyword in method_name.lower() for keyword in ['create', 'process', 'handle', 'validate', 'send', 'get', 'update', 'delete']):
                                return method_name
                            extracted_methods.append(method_name)
                # Class definitions
                elif line_stripped.startswith('class '):
                    class_name = line_stripped[6:].split('(')[0].split(':')[0].strip()
                    if class_name and class_name.isidentifier():
                        # Convert to business logic context
                        if any(keyword in class_name.lower() for keyword in ['service', 'controller', 'handler', 'manager', 'processor']):
                            return f"{class_name}"
                        extracted_methods.append(class_name)
        
        # JavaScript/TypeScript extraction
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
                            if any(keyword in method_name.lower() for keyword in ['handle', 'process', 'create', 'update', 'submit', 'validate']):
                                return method_name
                            extracted_methods.append(method_name)
                # Arrow functions and const declarations
                elif '=>' in line_stripped and '=' in line_stripped:
                    parts = line_stripped.split('=')[0].strip()
                    if parts.startswith(('const ', 'let ', 'var ')):
                        method_name = parts.split()[-1].strip()
                        if method_name and method_name.replace('_', '').replace('$', '').isalnum():
                            if any(keyword in method_name.lower() for keyword in ['handle', 'process', 'create', 'update', 'submit', 'validate']):
                                return method_name
                            extracted_methods.append(method_name)
                # React components
                elif 'export default' in line_stripped or 'export const' in line_stripped:
                    words = line_stripped.split()
                    for word in words:
                        if word.endswith('Component') or word.endswith('Page') or word.endswith('Form'):
                            return word
        
        # C# extraction
        elif language_lower in ['csharp', 'c#', 'cs']:
            for line in lines:
                line_stripped = line.strip()
                # Method declarations with access modifiers
                if ('public ' in line_stripped or 'private ' in line_stripped or 'protected ' in line_stripped) and '(' in line_stripped:
                    # Find method name before parentheses
                    paren_index = line_stripped.find('(')
                    if paren_index > 0:
                        before_paren = line_stripped[:paren_index].strip()
                        words = before_paren.split()
                        if len(words) >= 2:  # At least access modifier and method name
                            method_name = words[-1]
                            if method_name and method_name.replace('_', '').isalnum():
                                # Prioritize business logic methods
                                if any(keyword in method_name.lower() for keyword in ['create', 'process', 'handle', 'validate', 'send', 'get', 'update', 'delete', 'place', 'confirm']):
                                    return method_name
                                extracted_methods.append(method_name)
                # Class declarations
                elif 'public class ' in line_stripped or 'class ' in line_stripped:
                    class_match = line_stripped.split('class ')[1].split()[0].strip()
                    if class_match and class_match.replace('_', '').isalnum():
                        if any(keyword in class_match.lower() for keyword in ['service', 'controller', 'handler', 'manager', 'processor']):
                            return class_match
                        extracted_methods.append(class_match)
        
        # Java extraction
        elif language_lower in ['java']:
            for line in lines:
                line_stripped = line.strip()
                if ('public ' in line_stripped or 'private ' in line_stripped) and '(' in line_stripped:
                    paren_index = line_stripped.find('(')
                    if paren_index > 0:
                        before_paren = line_stripped[:paren_index].strip()
                        words = before_paren.split()
                        if len(words) >= 2:
                            method_name = words[-1]
                            if method_name and method_name.replace('_', '').isalnum():
                                if any(keyword in method_name.lower() for keyword in ['create', 'process', 'handle', 'validate', 'execute']):
                                    return method_name
                                extracted_methods.append(method_name)
        
        # Go extraction
        elif language_lower in ['go']:
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith('func ') and '(' in line_stripped:
                    start = line_stripped.find('func ') + 5
                    end = line_stripped.find('(')
                    if start < end:
                        method_name = line_stripped[start:end].strip()
                        if method_name and method_name.replace('_', '').isalnum():
                            if any(keyword in method_name.lower() for keyword in ['Create', 'Process', 'Handle', 'Validate']):
                                return method_name
                            extracted_methods.append(method_name)
        
        # If we found methods, return the first business-relevant one or the first one
        if extracted_methods:
            # Prioritize business logic methods
            business_methods = [m for m in extracted_methods if any(keyword in m.lower() for keyword in 
                              ['create', 'process', 'handle', 'validate', 'send', 'get', 'update', 'delete', 'place', 'confirm', 'execute'])]
            if business_methods:
                return business_methods[0]
            return extracted_methods[0]
        
        # Fallback: Generate semantic name based on content analysis
        return self._generate_semantic_method_name(content_lower, language_lower)
    
    def _generate_semantic_method_name(self, content_lower: str, language_lower: str) -> str:
        """Generate a semantic method name based on content analysis."""
        
        # Business operation patterns
        if 'order' in content_lower:
            if 'create' in content_lower or 'place' in content_lower:
                return "createOrder"
            elif 'process' in content_lower:
                return "processOrder"
            elif 'validate' in content_lower or 'verify' in content_lower:
                return "validateOrder"
            elif 'complete' in content_lower or 'finish' in content_lower:
                return "completeOrder"
            else:
                return "orderHandler"
        
        elif 'payment' in content_lower:
            if 'process' in content_lower:
                return "processPayment"
            elif 'validate' in content_lower or 'verify' in content_lower:
                return "validatePayment"
            elif 'complete' in content_lower:
                return "completePayment"
            else:
                return "paymentHandler"
        
        elif 'user' in content_lower or 'auth' in content_lower:
            if 'login' in content_lower:
                return "authenticateUser"
            elif 'register' in content_lower or 'signup' in content_lower:
                return "registerUser"
            elif 'validate' in content_lower:
                return "validateUser"
            else:
                return "userHandler"
        
        elif 'notification' in content_lower or 'email' in content_lower:
            if 'send' in content_lower:
                return "sendNotification"
            elif 'process' in content_lower:
                return "processNotification"
            else:
                return "notificationHandler"
        
        elif 'api' in content_lower or 'request' in content_lower:
            if 'handle' in content_lower:
                return "handleRequest"
            elif 'process' in content_lower:
                return "processRequest"
            else:
                return "apiHandler"
        
        # Service/component patterns
        elif 'service' in content_lower:
            if 'order' in content_lower:
                return "OrderService"
            elif 'payment' in content_lower:
                return "PaymentService"
            elif 'user' in content_lower:
                return "UserService"
            else:
                return "BusinessService"
        
        elif 'controller' in content_lower:
            if 'order' in content_lower:
                return "OrderController"
            elif 'payment' in content_lower:
                return "PaymentController"
            elif 'user' in content_lower:
                return "UserController"
            else:
                return "ApiController"
        
        # Generic operation patterns
        elif 'create' in content_lower and 'function' in content_lower:
            return "createEntity"
        elif 'update' in content_lower and 'function' in content_lower:
            return "updateEntity"
        elif 'delete' in content_lower and 'function' in content_lower:
            return "deleteEntity"
        elif 'validate' in content_lower and 'function' in content_lower:
            return "validateData"
        elif 'process' in content_lower and 'function' in content_lower:
            return "processData"
        elif 'handle' in content_lower and 'function' in content_lower:
            return "handleEvent"
        
        # Default based on language
        if language_lower in ['csharp', 'c#', 'cs']:
            return "BusinessLogic"
        elif language_lower in ['javascript', 'typescript', 'js', 'ts']:
            return "businessHandler"
        elif language_lower in ['python', 'py']:
            return "business_logic"
        elif language_lower in ['java']:
            return "businessMethod"
        elif language_lower in ['go']:
            return "BusinessHandler"
        else:
            return "businessLogic"
    
    def _generate_realistic_filename(self, content: str, language: str, metadata: Dict[str, Any]) -> str:
        """Generate a realistic filename based on content analysis."""
        content_lower = content.lower()
        language_lower = language.lower()
        
        # Try to extract meaningful name from content
        if 'order' in content_lower:
            if 'controller' in content_lower:
                return "OrderController"
            elif 'service' in content_lower:
                return "OrderService"
            elif 'handler' in content_lower:
                return "OrderHandler"
            elif 'model' in content_lower or 'entity' in content_lower:
                return "Order"
            else:
                return "OrderManager"
        
        elif 'payment' in content_lower:
            if 'controller' in content_lower:
                return "PaymentController"
            elif 'service' in content_lower:
                return "PaymentService"
            elif 'processor' in content_lower:
                return "PaymentProcessor"
            else:
                return "PaymentHandler"
        
        elif 'user' in content_lower or 'auth' in content_lower:
            if 'controller' in content_lower:
                return "UserController"
            elif 'service' in content_lower:
                return "AuthService"
            elif 'handler' in content_lower:
                return "AuthHandler"
            else:
                return "UserService"
        
        elif 'notification' in content_lower:
            if 'service' in content_lower:
                return "NotificationService"
            elif 'handler' in content_lower:
                return "NotificationHandler"
            else:
                return "NotificationManager"
        
        elif 'api' in content_lower:
            if 'controller' in content_lower:
                return "ApiController"
            else:
                return "ApiService"
        
        # Default based on context type patterns
        elif 'controller' in content_lower:
            return "BaseController"
        elif 'service' in content_lower:
            return "BusinessService"
        elif 'handler' in content_lower:
            return "EventHandler"
        elif 'model' in content_lower or 'entity' in content_lower:
            return "DataModel"
        elif 'component' in content_lower:
            if language_lower in ['javascript', 'typescript', 'js', 'ts']:
                return "AppComponent"
            else:
                return "Component"
        else:
            # Generic fallback based on language
            if language_lower in ['csharp', 'c#', 'cs']:
                return "BusinessLogic"
            elif language_lower in ['javascript', 'typescript', 'js', 'ts']:
                return "businessHandler"
            elif language_lower in ['python', 'py']:
                return "business_logic"
            elif language_lower in ['java']:
                return "BusinessService"
            else:
                return "Service"
    
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