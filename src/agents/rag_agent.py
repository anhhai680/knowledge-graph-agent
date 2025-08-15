"""
RAG Agent for intelligent document retrieval and query processing.

This module implements a Retrieval-Augmented Generation (RAG) agent that
integrates with the query workflow to provide intelligent document retrieval
and context-aware responses.
"""

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain.schema import Document

from src.agents.base_agent import BaseAgent
from src.utils.logging import get_logger
from src.utils.prompt_manager import PromptManager
from src.workflows.query_workflow import QueryWorkflow
from src.workflows.workflow_states import QueryIntent

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
        super().__init__(
            workflow=workflow,
            agent_name=kwargs.get("name", "RAGAgent"),
        )
        
        # Agent properties
        self.name = kwargs.get("name", "RAGAgent")
        self.description = kwargs.get("description", "Retrieval-Augmented Generation agent for intelligent query processing")
        
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
        input_data: Union[str, Dict[str, Any]],
        config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Process input data through the RAG pipeline with prompt management.
        
        Args:
            input_data: Query string or dictionary with query parameters
            config: Optional configuration overrides
            
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

            # Create initial query state - let workflow determine intent
            # Do NOT pre-set intent, let the query parsing handler analyze it
            query_data = {
                "query": query_text,
                "intent": query_intent,  # Only set if explicitly provided, otherwise None
                "top_k": top_k,
                "repository_filter": repo_filter,
                "language_filter": lang_filter,
            }

            # Execute the query workflow using the proper run method
            # This ensures all workflow steps including intent analysis are executed
            logger.info(f"Processing query: {query_text[:100]}...")
            
            result = await self.workflow.run(
                query=query_text,
                repositories=repo_filter,
                languages=lang_filter,
                k=top_k,
            )

            # Get retrieved documents from QueryState
            retrieved_docs = result.get("document_retrieval", {}).get("retrieved_documents", [])
            
            # Get the actual query intent determined by the workflow
            actual_query_intent = result.get("query_intent")
            
            # Get Q2 system visualization detection state
            is_q2_system_visualization = result.get("is_q2_system_visualization", False)
            
            # Convert dict documents to Document objects for PromptManager compatibility
            document_objects = []
            for doc_dict in retrieved_docs:
                document_objects.append(Document(
                    page_content=doc_dict.get("page_content", doc_dict.get("content", "")),
                    metadata=doc_dict.get("metadata", {})
                ))
            
            # Generate context-aware prompt using PromptManager
            prompt_result = self.prompt_manager.create_query_prompt(
                query=query_text,
                context_documents=document_objects,
                query_intent=actual_query_intent,  # Use the determined intent
                repository_filter=repo_filter,
                language_filter=lang_filter,
                top_k=top_k,
                confidence_threshold=self.confidence_threshold,
                is_q2_system_visualization=is_q2_system_visualization,
            )

            # Check confidence threshold
            confidence_score = prompt_result.get("confidence_score", 0.0)
            if confidence_score < self.confidence_threshold:
                logger.warning(
                    f"Low confidence score ({confidence_score:.2f}) "
                    f"below threshold ({self.confidence_threshold})"
                )

            # For Q2 system visualization queries, use the specialized prompt instead of workflow LLM response
            if is_q2_system_visualization and prompt_result.get("template_type") == "Q2SystemVisualizationTemplate":
                logger.info("RAGAgent: Processing Q2 system visualization query with specialized template")
                
                # Get the specialized Q2 prompt from PromptManager
                q2_prompt = prompt_result.get("prompt")
                if q2_prompt:
                    # Use LLM factory to generate Q2 response
                    from src.llm.llm_factory import LLMFactory
                    llm = LLMFactory.create()
                    
                    # Format the prompt properly for the LLM
                    if hasattr(q2_prompt, 'format_prompt'):
                        formatted_q2_prompt = q2_prompt.format_prompt()
                    else:
                        formatted_q2_prompt = str(q2_prompt)
                    
                    # Generate Q2 response using the specialized prompt
                    q2_response = llm.invoke(formatted_q2_prompt)
                    
                    # Extract response content
                    if hasattr(q2_response, 'content'):
                        q2_answer = q2_response.content
                    else:
                        q2_answer = str(q2_response)
                    
                    logger.info(f"RAGAgent: Generated Q2 response with {len(q2_answer)} characters")
                    
                    # Use Q2 response instead of workflow response
                    generated_answer = q2_answer
                else:
                    logger.warning("RAGAgent: Q2 template detected but no prompt available, using workflow response")
                    generated_answer = result.get("llm_generation", {}).get("generated_response", "No answer generated")
            else:
                # Use standard workflow response for non-Q2 queries
                generated_answer = result.get("llm_generation", {}).get("generated_response", "No answer generated")

            # Format response with enhanced context
            formatted_response = {
                "answer": generated_answer,
                "sources": self._format_sources(retrieved_docs),
                "confidence": confidence_score,
                "query_intent": actual_query_intent,
                "context_summary": {
                    "documents_found": len(retrieved_docs),
                    "repositories": list(set(
                        doc.get("metadata", {}).get("repository", "unknown") 
                        for doc in retrieved_docs
                    )),
                    "languages": list(set(
                        doc.get("metadata", {}).get("language", "unknown") 
                        for doc in retrieved_docs
                    )),
                },
                "prompt_metadata": {
                    "template_type": prompt_result.get("template_type"),
                    "system_prompt_type": prompt_result.get("system_prompt_type"),
                    "confidence_assessment": confidence_score,
                    "is_q2_visualization": is_q2_system_visualization,
                },
                "processing_time": result.get("total_query_time", 0),
            }

            logger.info(f"Successfully processed query with {len(retrieved_docs)} sources")
            return formatted_response

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return await self._fallback_processing(input_data)

    async def _fallback_processing(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
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
            "query_intent": QueryIntent.CODE_SEARCH,  # Default to CODE_SEARCH for fallback
            "context_summary": {
                "documents_found": 0,
                "repositories": [],
                "languages": [],
            },
            "prompt_metadata": error_prompt.get("metadata", {}),
            "processing_time": 0,
            "error": True,
        }

    def _format_sources(self, documents: Union[List[Document], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Format retrieved documents as source references.
        
        Args:
            documents: List of retrieved documents

        Returns:
            List of formatted source dictionaries
        """
        sources = []
        for i, doc in enumerate(documents, 1):
            if isinstance(doc, Document):
                content = doc.page_content
                metadata = doc.metadata
            else:
                content = doc.get("page_content", doc.get("content", ""))
                metadata = doc.get("metadata", {})
            source = {
                "id": i,
                "content": content[:500] + ("..." if len(content) > 500 else ""),
                "metadata": {
                    "file_path": metadata.get("file_path", "unknown"),
                    "repository": metadata.get("repository", "unknown"),
                    "language": metadata.get("language", "unknown"),
                    "chunk_type": metadata.get("chunk_type", "unknown"),
                    "line_start": metadata.get("line_start"),
                    "line_end": metadata.get("line_end"),
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
        queries: Sequence[Union[str, Dict[str, Any]]],
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
