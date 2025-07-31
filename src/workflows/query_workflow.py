"""
LangGraph Query Workflow Implementation.

This module implements a complete stateful query workflow using LangGraph
for adaptive RAG query processing with quality control and fallback mechanisms.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableConfig

from src.config.settings import settings, DatabaseType
from src.llm.llm_factory import LLMFactory
from src.llm.embedding_factory import EmbeddingFactory
from src.vectorstores.store_factory import VectorStoreFactory
from src.workflows.base_workflow import BaseWorkflow
from src.workflows.workflow_states import (
    QueryState,
    ProcessingStatus,
    WorkflowType,
    QueryIntent,
    SearchStrategy,
    create_query_state,
    update_workflow_progress,
    add_workflow_error,
)


class QueryWorkflowSteps(str):
    """Query workflow step enumeration."""

    PARSE_QUERY = "parse_query"
    VALIDATE_QUERY = "validate_query"
    ANALYZE_QUERY_INTENT = "analyze_query_intent"
    DETERMINE_SEARCH_STRATEGY = "determine_search_strategy"
    VECTOR_SEARCH = "vector_search"
    FILTER_AND_RANK_RESULTS = "filter_and_rank_results"
    CHECK_SUFFICIENT_CONTEXT = "check_sufficient_context"
    EXPAND_SEARCH_PARAMETERS = "expand_search_parameters"
    PREPARE_LLM_CONTEXT = "prepare_llm_context"
    GENERATE_CONTEXTUAL_PROMPT = "generate_contextual_prompt"
    CALL_LLM = "call_llm"
    FORMAT_RESPONSE = "format_response"
    RESPONSE_QUALITY_CHECK = "response_quality_check"
    RETURN_SUCCESS = "return_success"

    # Error handling and fallback states
    HANDLE_RETRIEVAL_ERRORS = "handle_retrieval_errors"
    FALLBACK_SEARCH_STRATEGY = "fallback_search_strategy"
    HANDLE_LLM_ERRORS = "handle_llm_errors"
    RETRY_LLM_CALL = "retry_llm_call"
    RESPONSE_QUALITY_CONTROL = "response_quality_control"
    RETRY_WITH_DIFFERENT_CONTEXT = "retry_with_different_context"


class QueryWorkflow(BaseWorkflow):
    """
    LangGraph query workflow for adaptive RAG query processing.

    This workflow handles the complete query processing pipeline:
    1. Parse and validate user query
    2. Analyze query intent and determine search strategy
    3. Perform vector search with filtering and ranking
    4. Check context sufficiency and expand if needed
    5. Prepare LLM context and generate contextual prompt
    6. Call LLM and format response
    7. Perform quality control and return result
    """

    def __init__(
        self,
        vector_store_type: Optional[str] = None,
        collection_name: Optional[str] = None,
        default_k: int = 4,
        max_k: int = 20,
        min_context_length: int = 100,
        max_context_length: int = 8000,
        response_quality_threshold: float = 0.7,
        **kwargs,
    ):
        """
        Initialize query workflow.

        Args:
            vector_store_type: Vector store type (overrides env setting)
            collection_name: Collection name (overrides env setting)
            default_k: Default number of documents to retrieve
            max_k: Maximum number of documents to retrieve
            min_context_length: Minimum context length required
            max_context_length: Maximum context length allowed
            response_quality_threshold: Minimum quality score for response
            **kwargs: Additional arguments passed to BaseWorkflow
        """
        super().__init__(**kwargs)

        self.vector_store_type = vector_store_type or settings.database_type.value
        self.collection_name = collection_name or (
            settings.pinecone.collection_name
            if self.vector_store_type == "pinecone" and settings.pinecone
            else settings.chroma.collection_name
        )
        self.default_k = default_k
        self.max_k = max_k
        self.min_context_length = min_context_length
        self.max_context_length = max_context_length
        self.response_quality_threshold = response_quality_threshold

        # Initialize factories
        self.vector_store_factory = VectorStoreFactory()
        self.embedding_factory = EmbeddingFactory()
        self.llm_factory = LLMFactory()

        # Initialize components
        self._vector_store = None
        self._retriever = None
        self._embeddings = None
        self._llm = None

    def _get_vector_store(self):
        """Get or create vector store instance."""
        if self._vector_store is None:
            # Convert string to DatabaseType enum
            db_type = DatabaseType.CHROMA if self.vector_store_type == "chroma" else DatabaseType.PINECONE
            self._vector_store = self.vector_store_factory.create(
                database_type=db_type,
                collection_name=self.collection_name,
            )
        return self._vector_store

    def _get_retriever(self, k: Optional[int] = None, filter_dict: Optional[Dict[str, Any]] = None):
        """Get vector store and return a simple retrieval function."""
        vector_store = self._get_vector_store()
        
        def retrieve_documents(query: str) -> List[Document]:
            return vector_store.similarity_search(
                query, 
                k=k or self.default_k, 
                filter=filter_dict or {}
            )
        
        return retrieve_documents

    def _get_embeddings(self):
        """Get or create embeddings instance."""
        if self._embeddings is None:
            self._embeddings = self.embedding_factory.create()
        return self._embeddings

    def _get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = self.llm_factory.create()
        return self._llm

    def _determine_query_intent(self, query: str) -> QueryIntent:
        """
        Analyze query to determine intent.

        Args:
            query: User query string

        Returns:
            QueryIntent enum value
        """
        query_lower = query.lower()

        # Code search patterns
        if any(keyword in query_lower for keyword in [
            "function", "method", "class", "variable", "implement", "code",
            "algorithm", "pattern", "design pattern", "how to", "example"
        ]):
            return QueryIntent.CODE_SEARCH

        # Documentation patterns
        elif any(keyword in query_lower for keyword in [
            "document", "readme", "guide", "tutorial", "specification",
            "api doc", "comment", "description"
        ]):
            return QueryIntent.DOCUMENTATION

        # Explanation patterns
        elif any(keyword in query_lower for keyword in [
            "explain", "what is", "what does", "how does", "why",
            "understand", "clarify", "meaning"
        ]):
            return QueryIntent.EXPLANATION

        # Debugging patterns
        elif any(keyword in query_lower for keyword in [
            "error", "bug", "issue", "problem", "fix", "debug",
            "troubleshoot", "exception", "crash"
        ]):
            return QueryIntent.DEBUGGING

        # Architecture patterns
        elif any(keyword in query_lower for keyword in [
            "architecture", "structure", "design", "pattern", "flow",
            "component", "module", "system", "overview"
        ]):
            return QueryIntent.ARCHITECTURE

        # Default to code search
        return QueryIntent.CODE_SEARCH

    def _determine_search_strategy(self, query_intent: QueryIntent, query: str) -> SearchStrategy:
        """
        Determine optimal search strategy based on query intent.

        Args:
            query_intent: Analyzed query intent
            query: Original query string

        Returns:
            SearchStrategy enum value
        """
        # For code search, prefer semantic search
        if query_intent == QueryIntent.CODE_SEARCH:
            return SearchStrategy.SEMANTIC

        # For debugging, use hybrid approach
        elif query_intent == QueryIntent.DEBUGGING:
            return SearchStrategy.HYBRID

        # For architecture questions, use metadata filtering
        elif query_intent == QueryIntent.ARCHITECTURE:
            return SearchStrategy.METADATA_FILTERED

        # Default to semantic search
        return SearchStrategy.SEMANTIC

    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """
        Extract metadata filters from query.

        Args:
            query: User query string

        Returns:
            Dictionary of filters for vector search
        """
        filters = {}
        query_lower = query.lower()

        # Language filters
        if "c#" in query_lower or "csharp" in query_lower or ".net" in query_lower:
            filters["language"] = "csharp"
        elif any(js_term in query_lower for js_term in ["javascript", "js", "react", "tsx", "jsx"]):
            filters["language"] = ["javascript", "typescript"]
        elif "python" in query_lower or ".py" in query_lower:
            filters["language"] = "python"

        # File type filters
        if "test" in query_lower:
            filters["file_path"] = "*test*"
        elif "config" in query_lower:
            filters["file_path"] = "*config*"

        return filters

    def _perform_vector_search(
        self,
        query: str,
        strategy: SearchStrategy,
        k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform vector search based on strategy.

        Args:
            query: Search query
            strategy: Search strategy to use
            k: Number of documents to retrieve
            filters: Metadata filters

        Returns:
            List of retrieved documents
        """
        try:
            retriever = self._get_retriever(k=k, filter_dict=filters)
            
            if strategy == SearchStrategy.SEMANTIC:
                # Standard semantic search
                documents = retriever(query)
                
            elif strategy == SearchStrategy.HYBRID:
                # Combine semantic and keyword search
                # For now, use semantic search with expanded k
                expanded_retriever = self._get_retriever(k=min(k * 2, self.max_k), filter_dict=filters)
                documents = expanded_retriever(query)
                # TODO: Implement actual hybrid search combining semantic + keyword
                
            elif strategy == SearchStrategy.METADATA_FILTERED:
                # Search with heavy metadata filtering
                documents = retriever(query)
                
            else:  # KEYWORD or fallback
                # For now, use semantic search as fallback
                documents = retriever(query)

            self.logger.info(f"Retrieved {len(documents)} documents using {strategy} strategy")
            return documents

        except Exception as e:
            self.logger.error(f"Vector search failed: {str(e)}")
            raise

    def _check_context_sufficiency(self, documents: List[Document]) -> Tuple[bool, int]:
        """
        Check if retrieved context is sufficient.

        Args:
            documents: Retrieved documents

        Returns:
            Tuple of (is_sufficient, total_context_length)
        """
        if not documents:
            return False, 0

        total_length = sum(len(doc.page_content) for doc in documents)
        
        # Check minimum context length
        if total_length < self.min_context_length:
            return False, total_length

        # Check if we have meaningful content
        non_empty_docs = [doc for doc in documents if doc.page_content.strip()]
        if len(non_empty_docs) == 0:
            return False, total_length

        return True, total_length

    def _prepare_context_for_llm(self, documents: List[Document]) -> str:
        """
        Prepare context from documents for LLM.

        Args:
            documents: Retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        context_parts = []
        total_length = 0

        for i, doc in enumerate(documents):
            # Extract metadata for source attribution
            file_path = doc.metadata.get("file_path", "unknown")
            repository = doc.metadata.get("repository", "unknown")
            line_start = doc.metadata.get("line_start", "")
            line_end = doc.metadata.get("line_end", "")
            
            # Format source info
            line_info = f" (lines {line_start}-{line_end})" if line_start and line_end else ""
            source_info = f"Source {i+1}: {repository}/{file_path}{line_info}"
            
            # Add context with source
            context_part = f"```\n{source_info}\n{doc.page_content}\n```"
            
            # Check if adding this would exceed max length
            if total_length + len(context_part) > self.max_context_length:
                break
                
            context_parts.append(context_part)
            total_length += len(context_part)

        return "\n\n".join(context_parts)

    def _generate_system_prompt(self, query_intent: QueryIntent) -> str:
        """
        Generate system prompt based on query intent.

        Args:
            query_intent: Analyzed query intent

        Returns:
            System prompt string
        """
        base_prompt = """You are an expert software engineer and code analyst. You help developers understand codebases by providing accurate, detailed, and contextual information based on the provided code context."""

        intent_specific_prompts = {
            QueryIntent.CODE_SEARCH: """
Focus on:
- Providing specific code examples and implementations
- Explaining how the code works and its purpose
- Identifying relevant patterns and best practices
- Suggesting similar implementations if applicable
""",
            QueryIntent.DOCUMENTATION: """
Focus on:
- Explaining the purpose and functionality clearly
- Providing comprehensive documentation-style responses
- Including usage examples and API information
- Highlighting important configuration details
""",
            QueryIntent.EXPLANATION: """
Focus on:
- Breaking down complex concepts into understandable parts
- Providing step-by-step explanations
- Using analogies and examples where helpful
- Connecting the explanation to the specific codebase context
""",
            QueryIntent.DEBUGGING: """
Focus on:
- Identifying potential issues and their causes
- Suggesting specific debugging approaches
- Providing troubleshooting steps
- Recommending fixes and improvements
""",
            QueryIntent.ARCHITECTURE: """
Focus on:
- Describing system structure and component relationships
- Explaining design patterns and architectural decisions
- Providing high-level overviews and data flow
- Identifying key components and their interactions
"""
        }

        return base_prompt + intent_specific_prompts.get(query_intent, intent_specific_prompts[QueryIntent.CODE_SEARCH])

    def _generate_contextual_prompt(
        self,
        query: str,
        context: str,
        query_intent: QueryIntent
    ) -> str:
        """
        Generate contextual prompt for LLM.

        Args:
            query: User query
            context: Formatted context from retrieved documents
            query_intent: Analyzed query intent

        Returns:
            Complete prompt for LLM
        """
        system_prompt = self._generate_system_prompt(query_intent)
        
        prompt_template = f"""{system_prompt}

Context from codebase:
{context}

User Question: {query}

Please provide a comprehensive answer based on the provided context. Include:
1. Direct answer to the question
2. Relevant code examples from the context
3. Source file references where applicable
4. Additional insights or recommendations

Answer:"""

        return prompt_template

    def _evaluate_response_quality(self, response: str, query: str, context: str) -> float:
        """
        Evaluate response quality using simple heuristics.

        Args:
            response: Generated response
            query: Original query
            context: Retrieved context

        Returns:
            Quality score between 0 and 1
        """
        score = 0.0

        # Basic checks
        if not response or len(response.strip()) < 50:
            return 0.0

        # Length check (not too short, not too long)
        response_length = len(response)
        if 100 <= response_length <= 2000:
            score += 0.3
        elif response_length > 50:
            score += 0.1

        # Context relevance check (simple keyword matching)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        common_words = query_words.intersection(response_words)
        if len(common_words) > 0:
            relevance_ratio = len(common_words) / len(query_words)
            score += min(0.4, relevance_ratio)

        # Structure check (has multiple sentences/paragraphs)
        if '.' in response and len(response.split('.')) > 2:
            score += 0.2

        # Code reference check
        if any(marker in response for marker in ['```', 'Source', 'file', 'line']):
            score += 0.1

        return min(1.0, score)

    async def _process_query_step(self, state: QueryState, step: str) -> QueryState:
        """
        Process a single query workflow step.

        Args:
            state: Current workflow state
            step: Step name to process

        Returns:
            Updated workflow state
        """
        start_time = time.time()
        
        try:
            if step == QueryWorkflowSteps.PARSE_QUERY:
                # Parse and clean the query
                state["processed_query"] = state["original_query"].strip()
                state["status"] = ProcessingStatus.IN_PROGRESS
                self.logger.info(f"Parsed query: {state['processed_query']}")

            elif step == QueryWorkflowSteps.VALIDATE_QUERY:
                # Validate query is not empty and has reasonable length
                if not state["processed_query"] or len(state["processed_query"]) < 3:
                    raise ValueError("Query is too short or empty")
                if len(state["processed_query"]) > 1000:
                    raise ValueError("Query is too long")

            elif step == QueryWorkflowSteps.ANALYZE_QUERY_INTENT:
                # Analyze query intent
                state["query_intent"] = self._determine_query_intent(state["processed_query"])
                self.logger.info(f"Determined query intent: {state['query_intent']}")

            elif step == QueryWorkflowSteps.DETERMINE_SEARCH_STRATEGY:
                # Determine optimal search strategy
                query_intent = state["query_intent"] or QueryIntent.CODE_SEARCH  # Provide default
                state["search_strategy"] = self._determine_search_strategy(
                    query_intent, state["processed_query"]
                )
                self.logger.info(f"Selected search strategy: {state['search_strategy']}")

            elif step == QueryWorkflowSteps.VECTOR_SEARCH:
                # Extract filters and perform search
                filters = self._extract_filters_from_query(state["processed_query"])
                k = state["retrieval_config"].get("k", self.default_k)
                
                # Perform vector search
                retrieval_start = time.time()
                documents = self._perform_vector_search(
                    state["processed_query"],
                    state["search_strategy"],
                    k,
                    filters
                )
                state["retrieval_time"] = time.time() - retrieval_start
                
                # Store results
                state["context_documents"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("file_path", "unknown")
                    }
                    for doc in documents
                ]
                # Update document retrieval state with actual documents
                state["document_retrieval"]["retrieved_documents"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("file_path", "unknown")
                    }
                    for doc in documents
                ]
                # Add retrieval time
                state["document_retrieval"]["retrieval_time"] = state["retrieval_time"]

            elif step == QueryWorkflowSteps.FILTER_AND_RANK_RESULTS:
                # Additional filtering and ranking if needed
                # For now, keep documents as retrieved
                pass

            elif step == QueryWorkflowSteps.CHECK_SUFFICIENT_CONTEXT:
                # Check if we have sufficient context
                documents = [
                    Document(page_content=doc["content"], metadata=doc["metadata"])
                    for doc in state["context_documents"]
                ]
                is_sufficient, context_length = self._check_context_sufficiency(documents)
                state["context_size"] = context_length
                
                if not is_sufficient:
                    # Need to expand search
                    state["current_step"] = QueryWorkflowSteps.EXPAND_SEARCH_PARAMETERS
                    return state

            elif step == QueryWorkflowSteps.EXPAND_SEARCH_PARAMETERS:
                # Expand search parameters for better context
                expanded_k = min(state["retrieval_config"].get("k", self.default_k) * 2, self.max_k)
                
                # Perform expanded search
                retrieval_start = time.time()
                documents = self._perform_vector_search(
                    state["processed_query"],
                    SearchStrategy.HYBRID,  # Use hybrid for expansion
                    expanded_k,
                    {}  # Remove filters for broader search
                )
                if "retrieval_time" not in state:
                    state["retrieval_time"] = 0
                state["retrieval_time"] += time.time() - retrieval_start
                
                # Update context
                state["context_documents"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("file_path", "unknown")
                    }
                    for doc in documents
                ]
                state["document_retrieval"]["retrieved_documents"] = state["context_documents"]

            elif step == QueryWorkflowSteps.PREPARE_LLM_CONTEXT:
                # Prepare context for LLM
                documents = [
                    Document(page_content=doc["content"], metadata=doc["metadata"])
                    for doc in state["context_documents"]
                ]
                context = self._prepare_context_for_llm(documents)
                state["context_preparation_time"] = time.time() - start_time
                
                # Store context in state for next step
                state["context_size"] = len(context)
                # Store in retrieval config for passing to next step
                state["retrieval_config"]["prepared_context"] = context

            elif step == QueryWorkflowSteps.GENERATE_CONTEXTUAL_PROMPT:
                # Generate contextual prompt
                context = state["retrieval_config"].get("prepared_context", "")
                prompt = self._generate_contextual_prompt(
                    state["processed_query"],
                    context,
                    state["query_intent"]
                )
                # Store prompt for LLM call
                state["retrieval_config"]["llm_prompt"] = prompt

            elif step == QueryWorkflowSteps.CALL_LLM:
                # Call LLM for response generation
                llm = self._get_llm()
                prompt = state["retrieval_config"].get("llm_prompt", "")
                
                generation_start = time.time()
                response = llm.invoke(prompt)
                state["generation_time"] = time.time() - generation_start
                
                # Extract response content
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Update LLM generation state
                state["llm_generation"]["status"] = ProcessingStatus.COMPLETED
                state["llm_generation"]["generated_response"] = response_text
                state["llm_generation"]["generation_time"] = state["generation_time"]
                state["llm_generation"]["model_name"] = settings.llm.model
                state["llm_generation"]["temperature"] = settings.app.temperature
                state["llm_generation"]["max_tokens"] = settings.app.max_tokens

            elif step == QueryWorkflowSteps.FORMAT_RESPONSE:
                # Format final response with sources
                response_text = state["llm_generation"]["generated_response"]
                sources = []
                
                for doc in state["context_documents"]:
                    source = {
                        "file_path": doc["metadata"].get("file_path", "unknown"),
                        "repository": doc["metadata"].get("repository", "unknown"),
                        "line_start": doc["metadata"].get("line_start"),
                        "line_end": doc["metadata"].get("line_end"),
                    }
                    sources.append(source)
                
                state["response_sources"] = sources
                # Response is already stored in llm_generation.generated_response

            elif step == QueryWorkflowSteps.RESPONSE_QUALITY_CHECK:
                # Evaluate response quality
                response = state["llm_generation"]["generated_response"]
                context = state["retrieval_config"].get("prepared_context", "")
                
                quality_score = self._evaluate_response_quality(
                    response, state["processed_query"], context
                )
                state["response_quality_score"] = quality_score
                state["response_confidence"] = min(quality_score * 1.2, 1.0)  # Boost confidence slightly
                
                # Check if quality is acceptable
                if quality_score < self.response_quality_threshold:
                    # Trigger quality control retry
                    state["current_step"] = QueryWorkflowSteps.RESPONSE_QUALITY_CONTROL
                    return state

            elif step == QueryWorkflowSteps.RETURN_SUCCESS:
                # Finalize successful response
                state["status"] = ProcessingStatus.COMPLETED
                start_time = state.get("start_time")
                state["total_query_time"] = time.time() - start_time if start_time else None
                self.logger.info(f"Query completed successfully with quality score: {state['response_quality_score']}")

            # Error handling steps
            elif step == QueryWorkflowSteps.HANDLE_RETRIEVAL_ERRORS:
                # Handle retrieval errors
                self.logger.warning("Handling retrieval errors, attempting fallback search")
                state["current_step"] = QueryWorkflowSteps.FALLBACK_SEARCH_STRATEGY
                return state

            elif step == QueryWorkflowSteps.FALLBACK_SEARCH_STRATEGY:
                # Use fallback search strategy
                try:
                    documents = self._perform_vector_search(
                        state["processed_query"],
                        SearchStrategy.SEMANTIC,  # Safe fallback
                        self.default_k,
                        {}  # No filters
                    )
                    state["context_documents"] = [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "source": doc.metadata.get("file_path", "unknown")
                        }
                        for doc in documents
                    ]
                    state["document_retrieval"]["retrieved_documents"] = state["context_documents"]
                    state["document_retrieval"]["status"] = ProcessingStatus.COMPLETED
                except Exception as e:
                    state["document_retrieval"]["status"] = ProcessingStatus.FAILED
                    raise

            elif step == QueryWorkflowSteps.HANDLE_LLM_ERRORS:
                # Handle LLM errors
                self.logger.warning("Handling LLM errors, attempting retry")
                state["current_step"] = QueryWorkflowSteps.RETRY_LLM_CALL
                return state

            elif step == QueryWorkflowSteps.RETRY_LLM_CALL:
                # Retry LLM call with simpler prompt
                try:
                    llm = self._get_llm()
                    simple_prompt = f"Based on the following code context, please answer: {state['processed_query']}\n\nContext: {state['retrieval_config'].get('prepared_context', '')[:2000]}"
                    
                    response = llm.invoke(simple_prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    state["llm_generation"]["generated_response"] = response_text
                    state["llm_generation"]["status"] = ProcessingStatus.COMPLETED
                except Exception as e:
                    state["llm_generation"]["status"] = ProcessingStatus.FAILED
                    raise

            elif step == QueryWorkflowSteps.RESPONSE_QUALITY_CONTROL:
                # Attempt to improve response quality
                self.logger.info("Response quality below threshold, attempting improvement")
                
                # Try with different context or expanded search
                if len(state["document_retrieval"]["retrieved_documents"]) < self.max_k:
                    state["current_step"] = QueryWorkflowSteps.RETRY_WITH_DIFFERENT_CONTEXT
                else:
                    # Accept current response if we can't improve
                    state["status"] = ProcessingStatus.COMPLETED
                    start_time_val = state.get("start_time")
                    state["total_query_time"] = time.time() - start_time_val if start_time_val else None

            elif step == QueryWorkflowSteps.RETRY_WITH_DIFFERENT_CONTEXT:
                # Retry with different context
                try:
                    # Expand search with different strategy
                    documents = self._perform_vector_search(
                        state["processed_query"],
                        SearchStrategy.HYBRID,
                        min(self.max_k, state["document_retrieval"]["retrieved_documents"] * 2),
                        {}
                    )
                    
                    # Update context and retry LLM
                    state["context_documents"] = [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "source": doc.metadata.get("file_path", "unknown")
                        }
                        for doc in documents
                    ]
                    
                    # Prepare new context and call LLM again
                    context = self._prepare_context_for_llm([
                        Document(page_content=doc["content"], metadata=doc["metadata"])
                        for doc in state["context_documents"]
                    ])
                    
                    prompt = self._generate_contextual_prompt(
                        state["processed_query"], context, state["query_intent"]
                    )
                    
                    llm = self._get_llm()
                    response = llm.invoke(prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    state["llm_generation"]["generated_response"] = response_text
                    
                    # Re-evaluate quality
                    quality_score = self._evaluate_response_quality(
                        response_text, state["processed_query"], context
                    )
                    state["response_quality_score"] = quality_score
                    
                except Exception as e:
                    self.logger.warning(f"Retry with different context failed: {str(e)}")
                    # Accept current response
                    pass
                
                # Complete regardless of retry outcome
                state["status"] = ProcessingStatus.COMPLETED
                start_time_val = state.get("start_time")
                state["total_query_time"] = time.time() - start_time_val if start_time_val else None

            # Update step completion
            step_time = time.time() - start_time
            self.logger.info(f"Completed step '{step}' in {step_time:.2f}s")

        except Exception as e:
            # Handle step errors
            error_msg = f"Error in step '{step}': {str(e)}"
            self.logger.error(error_msg)
            
            state = add_workflow_error(state, error_msg, step)
            
            # Route to appropriate error handler
            if step in [QueryWorkflowSteps.VECTOR_SEARCH, QueryWorkflowSteps.EXPAND_SEARCH_PARAMETERS]:
                state["current_step"] = QueryWorkflowSteps.HANDLE_RETRIEVAL_ERRORS
            elif step in [QueryWorkflowSteps.CALL_LLM, QueryWorkflowSteps.RETRY_LLM_CALL]:
                state["current_step"] = QueryWorkflowSteps.HANDLE_LLM_ERRORS
            else:
                # Generic error handling
                state["status"] = ProcessingStatus.FAILED
                raise

        return state

    async def run(
        self,
        query: str,
        repositories: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None,
        k: Optional[int] = None,
        **kwargs
    ) -> QueryState:
        """
        Execute the query workflow.

        Args:
            query: User query string
            repositories: Optional repository filter
            languages: Optional language filter
            file_types: Optional file type filter
            k: Optional number of documents to retrieve
            **kwargs: Additional workflow arguments

        Returns:
            Final QueryState
        """
        # Initialize workflow state
        state = create_query_state(
            workflow_id=kwargs.get("workflow_id"),
            original_query=query,
            target_repositories=repositories,
            target_languages=languages,
            target_file_types=file_types,
            retrieval_config={"k": k or self.default_k}
        )
        
        state["start_time"] = time.time()

        # Define workflow steps
        workflow_steps = [
            QueryWorkflowSteps.PARSE_QUERY,
            QueryWorkflowSteps.VALIDATE_QUERY,
            QueryWorkflowSteps.ANALYZE_QUERY_INTENT,
            QueryWorkflowSteps.DETERMINE_SEARCH_STRATEGY,
            QueryWorkflowSteps.VECTOR_SEARCH,
            QueryWorkflowSteps.FILTER_AND_RANK_RESULTS,
            QueryWorkflowSteps.CHECK_SUFFICIENT_CONTEXT,
            QueryWorkflowSteps.PREPARE_LLM_CONTEXT,
            QueryWorkflowSteps.GENERATE_CONTEXTUAL_PROMPT,
            QueryWorkflowSteps.CALL_LLM,
            QueryWorkflowSteps.FORMAT_RESPONSE,
            QueryWorkflowSteps.RESPONSE_QUALITY_CHECK,
            QueryWorkflowSteps.RETURN_SUCCESS,
        ]

        current_step_index = 0
        max_retries = 3
        retry_count = 0

        try:
            while current_step_index < len(workflow_steps) and state["status"] != ProcessingStatus.COMPLETED:
                step = workflow_steps[current_step_index]
                
                # Update progress
                progress = (current_step_index / len(workflow_steps)) * 100
                state = update_workflow_progress(state, progress, step)
                
                # Process step
                try:
                    state = await self._process_query_step(state, step)
                    
                    # Check if step redirected workflow
                    if "current_step" in state and state["current_step"] != step:
                        # Find the redirected step in workflow
                        try:
                            redirect_index = workflow_steps.index(state["current_step"])
                            current_step_index = redirect_index
                        except ValueError:
                            # Step not in main workflow, handle specially
                            state = await self._process_query_step(state, state["current_step"])
                            del state["current_step"]
                        continue
                    
                    current_step_index += 1
                    retry_count = 0  # Reset retry count on success
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count <= max_retries:
                        self.logger.warning(f"Retrying step {step} (attempt {retry_count}/{max_retries})")
                        time.sleep(2 ** retry_count)  # Exponential backoff
                        continue
                    else:
                        self.logger.error(f"Step {step} failed after {max_retries} retries")
                        raise

            # Ensure workflow is completed
            if state["status"] != ProcessingStatus.COMPLETED:
                state.status = ProcessingStatus.COMPLETED
                state["total_query_time"] = time.time() - state["start_time"]

            self.logger.info(f"Query workflow completed in {state.total_query_time:.2f}s")
            return state

        except Exception as e:
            state["status"] = ProcessingStatus.FAILED
            state = add_workflow_error(state, str(e), state.get("current_step", "unknown"))
            self.logger.error(f"Query workflow failed: {str(e)}")
            raise

    def define_steps(self) -> List[str]:
        """Define the workflow steps."""
        return [
            QueryWorkflowSteps.PARSE_QUERY,
            QueryWorkflowSteps.VALIDATE_QUERY,
            QueryWorkflowSteps.ANALYZE_QUERY_INTENT,
            QueryWorkflowSteps.DETERMINE_SEARCH_STRATEGY,
            QueryWorkflowSteps.VECTOR_SEARCH,
            QueryWorkflowSteps.FILTER_AND_RANK_RESULTS,
            QueryWorkflowSteps.CHECK_SUFFICIENT_CONTEXT,
            QueryWorkflowSteps.PREPARE_LLM_CONTEXT,
            QueryWorkflowSteps.GENERATE_CONTEXTUAL_PROMPT,
            QueryWorkflowSteps.CALL_LLM,
            QueryWorkflowSteps.FORMAT_RESPONSE,
            QueryWorkflowSteps.RESPONSE_QUALITY_CHECK,
            QueryWorkflowSteps.RETURN_SUCCESS,
        ]

    def execute_step(self, step: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        import asyncio
        # Convert to async call
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._process_query_step(state, step))

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate workflow state."""
        required_fields = ["workflow_id", "workflow_type", "status"]
        return all(field in state for field in required_fields)


# Helper functions for easy workflow execution
async def execute_query(
    query: str,
    repositories: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute a query workflow and return formatted result.

    Args:
        query: User query string
        repositories: Optional repository filter
        languages: Optional language filter
        **kwargs: Additional workflow arguments

    Returns:
        Dictionary with query result
    """
    workflow = QueryWorkflow(**kwargs)
    state = await workflow.run(
        query=query,
        repositories=repositories,
        languages=languages,
        **kwargs
    )

    return {
        "query": state["original_query"],
        "response": state["llm_generation"]["generated_response"],
        "sources": state.get("response_sources", []),
        "quality_score": state.get("response_quality_score", 0.0),
        "processing_time": state.get("total_query_time", 0.0),
        "documents_retrieved": len(state["document_retrieval"]["retrieved_documents"]),
        "workflow_id": state["workflow_id"],
    }
