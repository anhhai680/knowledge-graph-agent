"""
RAG Agent Implementation with LangChain RetrievalQA Integration.

This module implements a RAG (Retrieval-Augmented Generation) agent that
integrates with LangGraph query workflow for intelligent document retrieval
and response generation.
"""

import re
from typing import Any, Dict, List, Optional, Union

from langchain.schema import Document
from langchain.schema.runnable import RunnableConfig
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentResponse
from src.workflows.query_workflow import QueryWorkflow
from src.workflows.workflow_states import (
    QueryState,
    QueryIntent,
    SearchStrategy,
    create_query_state,
)
from src.config.settings import settings


class RAGAgent(BaseAgent):
"""
RAG Agent for intelligent document retrieval and query processing.

This module implements a Retrieval-Augmented Generation (RAG) agent that
integrates with the query workflow to provide intelligent document retrieval
and context-aware responses.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from langchain.schema import Document

from src.agents.base_agent import AgentResponse, BaseAgent
from src.utils.logging import get_logger
from src.utils.prompt_manager import PromptManager
from src.workflows.query_workflow import QueryWorkflow
from src.workflows.workflow_states import QueryIntent, QueryState

logger = get_logger(__name__)


class RAGAgent(BaseAgent):
    """
    Retrieval-Augmented Generation agent for intelligent query processing.
    
    This agent combines document retrieval with language model generation
    to provide contextually relevant responses to user queries. It integrates
    with the QueryWorkflow to handle the complete query processing pipeline.
    
    Key Features:
    - Intelligent document retrieval and ranking
    - Context-aware prompt generation with PromptManager
    - Multi-modal query support (code, documentation, debugging)
    - Batch processing capabilities
    - Comprehensive error handling and fallback mechanisms
    
    Attributes:
        workflow: The QueryWorkflow instance for processing queries
        prompt_manager: PromptManager for dynamic prompt generation
        default_top_k: Default number of documents to retrieve
        confidence_threshold: Minimum confidence for query processing
        repository_filter: Optional repository filtering
        language_filter: Optional language filtering
        enable_batch_processing: Whether to support batch operations
    """

    def __init__(
        self,
        workflow: Optional[QueryWorkflow] = None,
        prompt_manager: Optional[PromptManager] = None,
        default_top_k: int = 5,
        confidence_threshold: float = 0.3,
        repository_filter: Optional[List[str]] = None,
        language_filter: Optional[List[str]] = None,
        enable_batch_processing: bool = True,
        **kwargs
    ):
        """
        Initialize the RAG Agent.
        
        Args:
            workflow: QueryWorkflow instance for processing queries
            prompt_manager: PromptManager for dynamic prompt generation
            default_top_k: Default number of documents to retrieve
            confidence_threshold: Minimum confidence threshold
            repository_filter: List of repositories to filter by
            language_filter: List of languages to filter by
            enable_batch_processing: Enable batch processing support
            **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(**kwargs)
        
        self.workflow = workflow or QueryWorkflow()
        self.prompt_manager = prompt_manager or PromptManager()
        self.default_top_k = default_top_k
        self.confidence_threshold = confidence_threshold
        self.repository_filter = repository_filter or []
        self.language_filter = language_filter or []
        self.enable_batch_processing = enable_batch_processing
        
        logger.info("RAGAgent initialized with prompt manager integration")

    def _validate_input(self, input_data: Any) -> bool:
        """
        Validate input data for query processing.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            bool: True if input is valid, False otherwise
        """
        if isinstance(input_data, str):
            return len(input_data.strip()) > 0
        
        if isinstance(input_data, dict):
            required_fields = {"query"}
            return all(field in input_data for field in required_fields)
        
        return False

    async def _process_input(
        self, 
        input_data: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process input data through the RAG pipeline with prompt management.
        
        Args:
            input_data: Query string or dictionary with query parameters
            
        Returns:
            Dict containing the processed response with sources and metadata
        """
        try:
            # Normalize input to dictionary format
            if isinstance(input_data, str):
                query_dict = {
                    "query": input_data,
                    "top_k": self.default_top_k,
                    "repository_filter": self.repository_filter,
                    "language_filter": self.language_filter,
                }
            else:
                query_dict = dict(input_data)
                query_dict.setdefault("top_k", self.default_top_k)
                query_dict.setdefault("repository_filter", self.repository_filter)
                query_dict.setdefault("language_filter", self.language_filter)

            # Extract query parameters
            query_text = query_dict["query"]
            top_k = query_dict.get("top_k", self.default_top_k)
            repo_filter = query_dict.get("repository_filter", [])
            lang_filter = query_dict.get("language_filter", [])
            query_intent = query_dict.get("query_intent")

            # Create initial query state
            query_state = QueryState(
                query=query_text,
                intent=query_intent or QueryIntent.CODE_SEARCH,
                top_k=top_k,
                repository_filter=repo_filter,
                language_filter=lang_filter,
            )

            # Execute the query workflow
            logger.info(f"Processing query: {query_text[:100]}...")
            result = await self.workflow.ainvoke(query_state)

            # Get retrieved documents
            retrieved_docs = result.get("retrieved_documents", [])
            
            # Generate context-aware prompt using PromptManager
            prompt_result = self.prompt_manager.create_query_prompt(
                query=query_text,
                context_documents=retrieved_docs,
                query_intent=query_intent,
                repository_filter=repo_filter,
                language_filter=lang_filter,
                top_k=top_k,
                confidence_threshold=self.confidence_threshold,
            )

            # Check confidence threshold
            confidence_score = prompt_result.get("confidence_score", 0.0)
            if confidence_score < self.confidence_threshold:
                logger.warning(
                    f"Low confidence score ({confidence_score:.2f}) "
                    f"below threshold ({self.confidence_threshold})"
                )

            # Format response with enhanced context
            formatted_response = {
                "answer": result.get("answer", "No answer generated"),
                "sources": self._format_sources(retrieved_docs),
                "confidence": confidence_score,
                "query_intent": query_intent,
                "context_summary": {
                    "documents_found": len(retrieved_docs),
                    "repositories": list(set(
                        doc.metadata.get("repository", "unknown") 
                        for doc in retrieved_docs
                    )),
                    "languages": list(set(
                        doc.metadata.get("language", "unknown") 
                        for doc in retrieved_docs
                    )),
                },
                "prompt_metadata": {
                    "template_type": prompt_result.get("template_type"),
                    "system_prompt_type": prompt_result.get("system_prompt_type"),
                    "confidence_assessment": confidence_score,
                },
                "processing_time": result.get("processing_time", 0),
            }

            logger.info(f"Successfully processed query with {len(retrieved_docs)} sources")
            return formatted_response

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return await self._fallback_processing(input_data)

    async def _fallback_processing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback processing when main pipeline fails.
        
        Args:
            input_data: Original input data
            
        Returns:
            Dict containing fallback response
        """
        query_text = (
            input_data if isinstance(input_data, str) 
            else input_data.get("query", "Unknown query")
        )
        
        # Generate error recovery prompt
        error_prompt = self.prompt_manager._create_error_recovery_prompt(
            query=query_text,
            error_info="Query processing pipeline failed",
        )
        
        return {
            "answer": "I apologize, but I encountered an error processing your query. Please try rephrasing your question or check if the system is properly configured.",
            "sources": [],
            "confidence": 0.1,
            "query_intent": QueryIntent.CODE_SEARCH,
            "context_summary": {
                "documents_found": 0,
                "repositories": [],
                "languages": [],
            },
            "prompt_metadata": error_prompt.get("metadata", {}),
            "processing_time": 0,
            "error": True,
        }

    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format retrieved documents as source references.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of formatted source dictionaries
        """
        sources = []
        for i, doc in enumerate(documents, 1):
            source = {
                "id": i,
                "content": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                "metadata": {
                    "file_path": doc.metadata.get("file_path", "unknown"),
                    "repository": doc.metadata.get("repository", "unknown"),
                    "language": doc.metadata.get("language", "unknown"),
                    "chunk_type": doc.metadata.get("chunk_type", "unknown"),
                    "line_start": doc.metadata.get("line_start"),
                    "line_end": doc.metadata.get("line_end"),
                }
            }
            sources.append(source)
        return sources

    def update_filters(
        self,
        repository_filter: Optional[List[str]] = None,
        language_filter: Optional[List[str]] = None,
    ):
        """
        Update filtering criteria for queries.
        
        Args:
            repository_filter: New repository filter list
            language_filter: New language filter list
        """
        if repository_filter is not None:
            self.repository_filter = repository_filter
            logger.info(f"Updated repository filter: {repository_filter}")
        
        if language_filter is not None:
            self.language_filter = language_filter
            logger.info(f"Updated language filter: {language_filter}")

    async def query_with_context(
        self,
        query: str,
        context_documents: List[Document],
        query_intent: Optional[QueryIntent] = None,
    ) -> Dict[str, Any]:
        """
        Process a query with pre-provided context documents.
        
        Args:
            query: The query string
            context_documents: Pre-retrieved context documents
            query_intent: Optional query intent classification
            
        Returns:
            Dict containing the processed response
        """
        try:
            # Generate prompt with provided context
            prompt_result = self.prompt_manager.create_query_prompt(
                query=query,
                context_documents=context_documents,
                query_intent=query_intent,
                confidence_threshold=self.confidence_threshold,
            )

            # Format response
            return {
                "answer": f"Based on the provided context: {query}",
                "sources": self._format_sources(context_documents),
                "confidence": prompt_result.get("confidence_score", 0.0),
                "query_intent": query_intent,
                "context_summary": {
                    "documents_found": len(context_documents),
                    "repositories": list(set(
                        doc.metadata.get("repository", "unknown") 
                        for doc in context_documents
                    )),
                    "languages": list(set(
                        doc.metadata.get("language", "unknown") 
                        for doc in context_documents
                    )),
                },
                "prompt_metadata": {
                    "template_type": prompt_result.get("template_type"),
                    "confidence_assessment": prompt_result.get("confidence_score", 0.0),
                },
                "processing_time": 0,
            }

        except Exception as e:
            logger.error(f"Error in context query processing: {e}")
            return await self._fallback_processing({"query": query})

    async def batch_query(
        self,
        queries: List[Union[str, Dict[str, Any]]],
        max_concurrent: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries concurrently.
        
        Args:
            queries: List of query strings or dictionaries
            max_concurrent: Maximum number of concurrent queries
            
        Returns:
            List of processed responses
        """
        if not self.enable_batch_processing:
            raise ValueError("Batch processing is disabled for this agent")

        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(query):
            async with semaphore:
                return await self._process_input(query)

        tasks = [process_single(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses  
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing query {i}: {result}")
                processed_results.append(await self._fallback_processing(queries[i]))
            else:
                processed_results.append(result)
        
        return processed_results

    def get_supported_query_types(self) -> List[str]:
        """
        Get list of supported query types.
        
        Returns:
            List of supported query type strings
        """
        return [intent.value for intent in self.prompt_manager.get_supported_intents()]

    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get agent configuration and statistics.
        
        Returns:
            Dict containing agent statistics and configuration
        """
        return {
            "agent_type": "RAGAgent",
            "default_top_k": self.default_top_k,
            "confidence_threshold": self.confidence_threshold,
            "repository_filter": self.repository_filter,
            "language_filter": self.language_filter,
            "batch_processing_enabled": self.enable_batch_processing,
            "supported_query_types": self.get_supported_query_types(),
            "prompt_manager_stats": self.prompt_manager.get_template_statistics(),
        }

import asyncio
from typing import Any, Dict, List, Optional, Union

from langchain.schema import Document

from src.agents.base_agent import AgentResponse, BaseAgent
from src.utils.logging import get_logger
from src.utils.prompt_manager import PromptManager
from src.workflows.query_workflow import QueryWorkflow
from src.workflows.workflow_states import QueryIntent, QueryState

logger = get_logger(__name__)    def __init__(
        self,
        workflow: Optional[QueryWorkflow] = None,
        default_k: int = 4,
        max_k: int = 20,
        repository_filter: Optional[List[str]] = None,
        language_filter: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize RAG agent.

        Args:
            workflow: Query workflow instance
            default_k: Default number of documents to retrieve
            max_k: Maximum number of documents to retrieve
            repository_filter: List of repositories to filter by
            language_filter: List of languages to filter by
            **kwargs: Additional arguments passed to BaseAgent
        """
        # Initialize query workflow if not provided
        if workflow is None:
            workflow = QueryWorkflow(
                default_k=default_k,
                max_k=max_k,
            )

        super().__init__(
            workflow=workflow,
            agent_name="RAGAgent",
            **kwargs,
        )

        self.default_k = default_k
        self.max_k = max_k
        self.repository_filter = repository_filter or []
        self.language_filter = language_filter or []

    def _validate_input(self, input_data: Any) -> bool:
        """
        Validate input data for RAG queries.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise
        """
        if isinstance(input_data, str):
            # Simple string query
            return len(input_data.strip()) > 0

        if isinstance(input_data, dict):
            # Structured query
            if "query" not in input_data:
                return False
            
            query_text = input_data.get("query", "")
            if not isinstance(query_text, str) or len(query_text.strip()) == 0:
                return False

            # Validate optional parameters
            if "k" in input_data:
                k = input_data["k"]
                if not isinstance(k, int) or k <= 0 or k > self.max_k:
                    return False

            if "repository_filter" in input_data:
                repo_filter = input_data["repository_filter"]
                if not isinstance(repo_filter, list):
                    return False

            if "language_filter" in input_data:
                lang_filter = input_data["language_filter"]
                if not isinstance(lang_filter, list):
                    return False

            return True

        return False

    async def _process_input(
        self,
        input_data: Union[str, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """
        Process input through RAG workflow.

        Args:
            input_data: Query input (string or structured dict)
            config: Optional configuration

        Returns:
            Structured response with retrieval results and generated answer
        """
        try:
            # Normalize input to structured format
            if isinstance(input_data, str):
                normalized_input = {
                    "query": input_data,
                    "k": self.default_k,
                    "repository_filter": self.repository_filter,
                    "language_filter": self.language_filter,
                }
            else:
                normalized_input = {
                    "query": input_data["query"],
                    "k": input_data.get("k", self.default_k),
                    "repository_filter": input_data.get("repository_filter", self.repository_filter),
                    "language_filter": input_data.get("language_filter", self.language_filter),
                    "include_metadata": input_data.get("include_metadata", True),
                    "search_strategy": input_data.get("search_strategy"),
                }

            self.logger.info(f"Processing RAG query: {normalized_input['query'][:100]}...")

            # Create initial query state
            query_state = create_query_state(
                workflow_id=f"rag_query_{hash(normalized_input['query'])}",
                original_query=normalized_input["query"],
                top_k=normalized_input["k"],
            )

            # Execute query workflow
            if self.workflow:
                workflow_result = self.workflow.invoke(query_state)
                
                if workflow_result.get("status") == "completed":
                    # Extract response from workflow result
                    response_data = {
                        "answer": workflow_result.get("response", ""),
                        "sources": self._format_sources(workflow_result.get("retrieved_documents", [])),
                        "metadata": {
                            "query_intent": workflow_result.get("query_intent"),
                            "search_strategy": workflow_result.get("search_strategy"),
                            "documents_retrieved": len(workflow_result.get("retrieved_documents", [])),
                            "response_quality_score": workflow_result.get("response_quality_score"),
                            "processing_time": workflow_result.get("processing_time"),
                        },
                    }

                    # Include detailed metadata if requested
                    if normalized_input.get("include_metadata", True):
                        response_data["metadata"].update({
                            "workflow_steps": workflow_result.get("completed_steps", []),
                            "error_count": len(workflow_result.get("errors", [])),
                        })

                    return AgentResponse.success_response(
                        data=response_data,
                        agent_name=self.agent_name,
                    ).to_dict()

                else:
                    # Workflow failed
                    error_message = f"Query workflow failed: {workflow_result.get('error', 'Unknown error')}"
                    self.logger.error(error_message)
                    
                    return AgentResponse.error_response(
                        error=error_message,
                        metadata={
                            "workflow_status": workflow_result.get("status"),
                            "errors": workflow_result.get("errors", []),
                        },
                        agent_name=self.agent_name,
                    ).to_dict()

            else:
                # No workflow available - fallback processing
                return await self._fallback_processing(normalized_input)

        except Exception as e:
            self.logger.error(f"Error processing RAG query: {str(e)}")
            return AgentResponse.error_response(
                error=f"Internal error during query processing: {str(e)}",
                agent_name=self.agent_name,
            ).to_dict()

    async def _fallback_processing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback processing when workflow is not available.

        Args:
            input_data: Normalized input data

        Returns:
            Basic response without full workflow processing
        """
        self.logger.warning("Using fallback processing - workflow not available")
        
        return AgentResponse.error_response(
            error="RAG workflow not available - cannot process query",
            metadata={
                "fallback_mode": True,
                "query": input_data["query"][:100],
            },
            agent_name=self.agent_name,
        ).to_dict()

    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format retrieved documents as source information.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted source information
        """
        sources = []
        
        for i, doc in enumerate(documents):
            source_info = {
                "source_id": i + 1,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
            }

            # Extract key metadata for display
            metadata = doc.metadata
            if "file_path" in metadata:
                source_info["file_path"] = metadata["file_path"]
            if "repository" in metadata:
                source_info["repository"] = metadata["repository"]
            if "language" in metadata:
                source_info["language"] = metadata["language"]
            if "chunk_type" in metadata:
                source_info["chunk_type"] = metadata["chunk_type"]

            sources.append(source_info)

        return sources

    def update_filters(
        self,
        repository_filter: Optional[List[str]] = None,
        language_filter: Optional[List[str]] = None,
    ) -> None:
        """
        Update default filters for the agent.

        Args:
            repository_filter: List of repositories to filter by
            language_filter: List of languages to filter by
        """
        if repository_filter is not None:
            self.repository_filter = repository_filter
            self.logger.info(f"Updated repository filter: {repository_filter}")

        if language_filter is not None:
            self.language_filter = language_filter
            self.logger.info(f"Updated language filter: {language_filter}")

    async def query_with_context(
        self,
        query: str,
        context_repositories: Optional[List[str]] = None,
        context_languages: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query with specific context filters.

        Args:
            query: Query text
            context_repositories: Repositories to search in
            context_languages: Languages to filter by
            k: Number of documents to retrieve

        Returns:
            Query response
        """
        input_data = {
            "query": query,
            "repository_filter": context_repositories or self.repository_filter,
            "language_filter": context_languages or self.language_filter,
            "k": k or self.default_k,
            "include_metadata": True,
        }

        return await self._process_input(input_data)

    async def batch_query(
        self,
        queries: List[str],
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries efficiently.

        Args:
            queries: List of query strings
            shared_context: Shared context for all queries

        Returns:
            List of query responses
        """
        shared_context = shared_context or {}
        
        # Prepare inputs
        inputs = []
        for query in queries:
            input_data = {
                "query": query,
                "repository_filter": shared_context.get("repository_filter", self.repository_filter),
                "language_filter": shared_context.get("language_filter", self.language_filter),
                "k": shared_context.get("k", self.default_k),
                "include_metadata": shared_context.get("include_metadata", False),  # Reduce metadata for batch
            }
            inputs.append(input_data)

        # Process batch
        return await self.abatch(inputs)

    def get_supported_query_types(self) -> List[str]:
        """
        Get list of supported query types.

        Returns:
            List of supported query types
        """
        return [
            "code_search",
            "documentation_search", 
            "concept_explanation",
            "implementation_help",
            "troubleshooting",
        ]

    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get agent usage statistics and configuration.

        Returns:
            Agent statistics and configuration
        """
        base_info = self.get_agent_info()
        
        return {
            **base_info,
            "configuration": {
                "default_k": self.default_k,
                "max_k": self.max_k,
                "repository_filter": self.repository_filter,
                "language_filter": self.language_filter,
            },
            "supported_query_types": self.get_supported_query_types(),
        }
