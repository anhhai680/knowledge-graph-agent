"""
LangChain Prompt Manager Integration.

This module provides prompt management functionality with LangChain PromptTemplate
components for dynamic prompt composition and code query optimization.
"""

from typing import Any, Dict, List, Optional, Union

from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from loguru import logger

from src.workflows.workflow_states import QueryIntent
from src.config.settings import settings


class CodeQueryResponse(BaseModel):
    """Structured response model for code queries."""
    
    answer: str
    confidence: float
    sources_used: List[str]
    recommendations: Optional[List[str]] = None


class PromptManager:
    """
    LangChain Prompt Manager for intelligent document retrieval and response generation.
    
    This class manages prompt templates for different query types and handles
    dynamic prompt composition with context injection and fallback strategies.
    """

    def __init__(self):
        """Initialize prompt manager with LangChain templates."""
        self.logger = logger.bind(component="PromptManager")
        self.output_parser = PydanticOutputParser(pydantic_object=CodeQueryResponse)
        
        # Initialize prompt templates
        self._initialize_system_prompts()
        self._initialize_query_templates()
        self._initialize_fallback_templates()

    def _initialize_system_prompts(self) -> None:
        """Initialize system prompt templates for different contexts."""
        
        # Base system prompt for code queries
        self.base_system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert software engineer and code analysis assistant with deep knowledge of software architecture, design patterns, and best practices across multiple programming languages including .NET (C#), JavaScript/TypeScript, React, Python, and others.

Your primary role is to analyze code repositories and provide accurate, helpful responses about:
- Code structure, functionality, and implementation details
- Architecture patterns and design decisions
- Bug identification and troubleshooting guidance
- Best practices and code improvement suggestions
- API usage and integration patterns
- Documentation and feature explanations

**Response Guidelines:**
1. **Accuracy**: Base responses strictly on the provided code context
2. **Clarity**: Use clear, technical language appropriate for developers
3. **Context**: Reference specific files, functions, or code sections when relevant
4. **Practicality**: Provide actionable insights and concrete examples
5. **Completeness**: Address the full scope of the query when possible

**Source Attribution:**
Always cite the specific files and code sections you reference in your analysis.

Current query context: {query_intent}
Repository focus: {repository_filter}
Language focus: {language_filter}
""")

        # Specialized system prompts by query intent
        self.intent_system_prompts = {
            QueryIntent.CODE_SEARCH: SystemMessagePromptTemplate.from_template("""
You are a code search specialist. Focus on finding and explaining specific code implementations, functions, classes, or patterns within the codebase. Provide detailed analysis of how the code works and its relationships to other components.

**Search Focus:**
- Function implementations and usage patterns
- Class definitions and inheritance relationships
- Variable usage and data flow
- Code patterns and architectural decisions

Repository context: {repository_filter}
Language context: {language_filter}
"""),

            QueryIntent.DOCUMENTATION: SystemMessagePromptTemplate.from_template("""
You are a documentation specialist. Focus on explaining features, APIs, configuration options, and usage patterns based on the codebase and documentation files.

**Documentation Focus:**
- Feature explanations and usage guides
- API endpoint documentation and examples
- Configuration options and settings
- Setup and deployment instructions

Repository context: {repository_filter}
Language context: {language_filter}
"""),

            QueryIntent.EXPLANATION: SystemMessagePromptTemplate.from_template("""
You are a code explanation specialist. Focus on breaking down complex code concepts, algorithms, and architectural patterns into clear, understandable explanations.

**Explanation Focus:**
- Code logic and algorithm explanations
- Architecture and design pattern analysis
- Data flow and system interactions
- Complex concept simplification

Repository context: {repository_filter}
Language context: {language_filter}
"""),

            QueryIntent.DEBUGGING: SystemMessagePromptTemplate.from_template("""
You are a debugging specialist. Focus on identifying potential issues, analyzing error patterns, and providing troubleshooting guidance based on the codebase.

**Debugging Focus:**
- Error pattern identification
- Common pitfall analysis
- Troubleshooting steps and solutions
- Code quality and potential improvements

Repository context: {repository_filter}
Language context: {language_filter}
"""),

            QueryIntent.ARCHITECTURE: SystemMessagePromptTemplate.from_template("""
You are a software architecture specialist. Focus on analyzing system design, component relationships, and architectural patterns within the codebase.

**Architecture Focus:**
- System design and component relationships
- Architectural pattern identification
- Scalability and maintainability analysis
- Integration patterns and dependencies

Repository context: {repository_filter}
Language context: {language_filter}
"""),
        }

    def _initialize_query_templates(self) -> None:
        """Initialize query-specific prompt templates."""
        
        # Main query template with context injection
        self.main_query_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", """
Based on the following code and documentation context, please answer this question:

**Question:** {query}

**Relevant Code Context:**
{context_documents}

**Additional Parameters:**
- Query Intent: {query_intent}
- Repository Filter: {repository_filter}
- Language Filter: {language_filter}
- Top-K Results: {top_k}

Please provide a comprehensive response that addresses the question using the provided context. Include specific references to files and code sections when relevant.

{format_instructions}
"""),
        ])

        # High-confidence template for well-supported answers
        self.high_confidence_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", """
I have substantial relevant context for this question. Please provide a detailed, confident response:

**Question:** {query}

**Strong Context Available:**
{context_documents}

Focus on providing specific, actionable insights with high confidence. Reference the exact code sections and files that support your response.

{format_instructions}
"""),
        ])

        # Low-confidence template for limited context
        self.low_confidence_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", """
I have limited context for this question. Please provide the best possible response while being clear about limitations:

**Question:** {query}

**Available Context (Limited):**
{context_documents}

Please provide what insights you can based on the available context, and clearly indicate any limitations or areas where additional context would be helpful.

{format_instructions}
"""),
        ])
        
        # Q2 System Relationship Visualization Template
        self.q2_system_visualization_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", """
I need to provide a system relationship visualization response that includes both a Mermaid diagram and detailed explanation with code references.

**Question:** {query}

**Context Available:**
{context_documents}

Please respond in the following format:

Let me show you how these services work together:

```mermaid
graph TB
    subgraph "Frontend Layer"
        WC[car-web-client<br/>React + TypeScript<br/>User Interface]
    end
    
    subgraph "API Gateway"
        AGW[Load Balancer<br/>Rate Limiting<br/>Authentication]
    end
    
    subgraph "Microservices"
        CLS[car-listing-service<br/>.NET 8 Web API<br/>Inventory Management]
        OS[car-order-service<br/>.NET 8 Web API<br/>Order Processing]
        NS[car-notification-service<br/>.NET 8 Web API<br/>Event Notifications]
    end
    
    subgraph "Data Layer"
        CLSDB[(PostgreSQL<br/>Car Catalog)]
        ODB[(PostgreSQL<br/>Orders & Payments)]
        NDB[(MongoDB<br/>Notifications)]
    end
    
    subgraph "Message Infrastructure"
        RMQ[RabbitMQ<br/>Event Broker]
    end
    
    %% Frontend Communication
    WC -->|HTTPS REST| AGW
    AGW --> CLS
    AGW --> OS
    WC -->|WebSocket Connect| NS
    NS -->|WebSocket Updates| WC
    
    %% Inter-Service Communication
    OS -->|HTTP| CLS
    
    %% Event-Driven Communication
    CLS -->|Events| RMQ
    OS -->|Events| RMQ
    RMQ -->|Events| NS
    
    %% Data Persistence
    CLS --> CLSDB
    OS --> ODB
    NS --> NDB
```

Here's how these connections are implemented:

**Frontend to Backend Communication:**
- **React API calls**: Reference specific files like `car-web-client/src/hooks/useCars.ts` with line numbers for fetching car data
- **WebSocket connection**: Reference files like `car-web-client/src/hooks/useNotifications.ts` for real-time updates

**Inter-Service HTTP Communication:**
- **Car verification**: Reference specific service files with methods and line numbers
- **Status updates**: Reference integration service files with specific methods

**Event-Driven Communication:**
- **Event publishing**: Reference event publisher files with specific methods and line numbers
- **Event consumption**: Reference event handler files with specific methods

Provide a conversational explanation that explains what the user is seeing in terms of a modern microservices architecture, using the exact format and style from the expected Q2 response pattern.

Make sure to:
1. Include specific file paths and line number references where available from the context
2. Use conversational language explaining the architecture
3. Focus on the four main services and their connections
4. Explain both the obvious and interesting patterns

{format_instructions}
"""),
        ])

    def _initialize_fallback_templates(self) -> None:
        """Initialize fallback templates for edge cases."""
        
        # No context template
        self.no_context_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful software engineering assistant."),
            ("human", """
I don't have specific code context for this question, but I can provide general software engineering guidance:

**Question:** {query}

Please provide general guidance and best practices related to this question. Indicate that this is general advice and recommend accessing specific code context for more detailed insights.
"""),
        ])

        # Error recovery template
        self.error_recovery_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful software engineering assistant focused on error recovery."),
            ("human", """
There was an issue processing the original query context. Please provide a helpful response based on the question:

**Question:** {query}

**Available Information:**
{available_info}

Provide the best guidance possible while acknowledging the processing limitations.
"""),
        ])

    def create_query_prompt(
        self,
        query: str,
        context_documents: List[Document],
        query_intent: Optional[QueryIntent] = None,
        repository_filter: Optional[List[str]] = None,
        language_filter: Optional[List[str]] = None,
        top_k: int = 4,
        confidence_threshold: float = 0.7,
        is_q2_system_visualization: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a dynamic prompt for query processing.

        Args:
            query: User query text
            context_documents: Retrieved context documents
            query_intent: Detected query intent
            repository_filter: Repository filter list
            language_filter: Language filter list
            top_k: Number of top results
            confidence_threshold: Confidence threshold for template selection
            is_q2_system_visualization: Whether this is a Q2 system visualization query

        Returns:
            Dictionary containing prompt and metadata
        """
        try:
            # Check if this is a Q2 system relationship visualization query
            if is_q2_system_visualization:
                # Use specialized Q2 template regardless of confidence
                system_prompt = self.intent_system_prompts.get(
                    QueryIntent.ARCHITECTURE, 
                    self.base_system_prompt
                ).format(
                    repository_filter=repository_filter or ["all"],
                    language_filter=language_filter or ["all"],
                )
                
                context_text = self._format_context_documents(context_documents)
                
                template = self.q2_system_visualization_template
                prompt_values = {
                    "system_prompt": system_prompt,
                    "query": query,
                    "context_documents": context_text,
                    "format_instructions": self.output_parser.get_format_instructions(),
                }
                
                # Format the prompt
                formatted_prompt = template.format_prompt(**prompt_values)
                
                return {
                    "prompt": formatted_prompt,
                    "template_type": "Q2SystemVisualizationTemplate",
                    "confidence_score": 1.0,  # Always high confidence for Q2
                    "context_documents_count": len(context_documents),
                    "system_prompt_type": "q2_architecture",
                    "metadata": {
                        "query_intent": query_intent,
                        "repository_filter": repository_filter,
                        "language_filter": language_filter,
                        "top_k": top_k,
                        "is_q2_visualization": True,
                    },
                }
            
            # Continue with normal processing for non-Q2 queries
            # Determine system prompt based on intent
            if query_intent and query_intent in self.intent_system_prompts:
                system_prompt = self.intent_system_prompts[query_intent].format(
                    repository_filter=repository_filter or ["all"],
                    language_filter=language_filter or ["all"],
                )
            else:
                system_prompt = self.base_system_prompt.format(
                    query_intent=query_intent or "general",
                    repository_filter=repository_filter or ["all"],
                    language_filter=language_filter or ["all"],
                )

            # Format context documents
            context_text = self._format_context_documents(context_documents)
            
            # Determine confidence level and select template
            confidence_score = self._assess_context_confidence(context_documents, query)
            
            if len(context_documents) == 0:
                # No context available
                template = self.no_context_template
                prompt_values = {
                    "query": query,
                }
            elif confidence_score >= confidence_threshold:
                # High confidence
                template = self.high_confidence_template
                prompt_values = {
                    "system_prompt": system_prompt,
                    "query": query,
                    "context_documents": context_text,
                    "format_instructions": self.output_parser.get_format_instructions(),
                }
            elif confidence_score >= 0.3:
                # Low confidence but some context
                template = self.low_confidence_template
                prompt_values = {
                    "system_prompt": system_prompt,
                    "query": query,
                    "context_documents": context_text,
                    "format_instructions": self.output_parser.get_format_instructions(),
                }
            else:
                # Very low confidence, use main template
                template = self.main_query_template
                prompt_values = {
                    "system_prompt": system_prompt,
                    "query": query,
                    "context_documents": context_text,
                    "query_intent": query_intent or "general",
                    "repository_filter": repository_filter or ["all"],
                    "language_filter": language_filter or ["all"],
                    "top_k": top_k,
                    "format_instructions": self.output_parser.get_format_instructions(),
                }

            # Format the prompt
            formatted_prompt = template.format_prompt(**prompt_values)

            return {
                "prompt": formatted_prompt,
                "template_type": template.__class__.__name__,
                "confidence_score": confidence_score,
                "context_documents_count": len(context_documents),
                "system_prompt_type": query_intent.value if query_intent else "base",
                "metadata": {
                    "query_intent": query_intent,
                    "repository_filter": repository_filter,
                    "language_filter": language_filter,
                    "top_k": top_k,
                },
            }

        except Exception as e:
            self.logger.error(f"Error creating query prompt: {str(e)}")
            return self._create_error_recovery_prompt(query, str(e))

    def _format_context_documents(self, documents: List[Document]) -> str:
        """
        Format context documents for prompt injection.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context documents found."

        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            
            # Extract key metadata
            file_path = metadata.get("file_path", "Unknown file")
            repository = metadata.get("repository", "Unknown repository")
            language = metadata.get("language", "Unknown language")
            chunk_type = metadata.get("chunk_type", "code")
            
            # Format document
            doc_header = f"**Source {i}: {file_path}**"
            doc_meta = f"Repository: {repository} | Language: {language} | Type: {chunk_type}"
            doc_content = doc.page_content
            
            # Limit content length to prevent token overflow
            if len(doc_content) > 1500:
                doc_content = doc_content[:1500] + "... [truncated]"
            
            formatted_doc = f"{doc_header}\n{doc_meta}\n\n```{language.lower()}\n{doc_content}\n```\n"
            formatted_docs.append(formatted_doc)

        return "\n".join(formatted_docs)

    def _assess_context_confidence(self, documents: List[Document], query: str) -> float:
        """
        Assess confidence level based on context quality.

        Args:
            documents: Retrieved context documents
            query: Original query

        Returns:
            Confidence score between 0 and 1
        """
        if not documents:
            return 0.0

        # Basic confidence metrics
        doc_count_score = min(len(documents) / 5.0, 1.0)  # More docs = higher confidence
        
        # Content relevance score (simplified)
        total_content_length = sum(len(doc.page_content) for doc in documents)
        content_score = min(total_content_length / 2000.0, 1.0)  # More content = higher confidence
        
        # Metadata quality score
        metadata_quality = 0.0
        for doc in documents:
            metadata = doc.metadata
            if metadata.get("file_path"):
                metadata_quality += 0.2
            if metadata.get("repository"):
                metadata_quality += 0.1
            if metadata.get("language"):
                metadata_quality += 0.1
            if metadata.get("chunk_type"):
                metadata_quality += 0.1
        
        metadata_quality = min(metadata_quality / len(documents), 1.0)
        
        # Combined confidence score
        confidence = (doc_count_score * 0.4 + content_score * 0.4 + metadata_quality * 0.2)
        
        return min(confidence, 1.0)

    def _create_error_recovery_prompt(self, query: str, error_info: str) -> Dict[str, Any]:
        """
        Create error recovery prompt when main prompt creation fails.

        Args:
            query: Original query
            error_info: Error information

        Returns:
            Error recovery prompt dictionary
        """
        try:
            formatted_prompt = self.error_recovery_template.format_prompt(
                query=query,
                available_info=f"Error during prompt creation: {error_info}",
            )

            return {
                "prompt": formatted_prompt,
                "template_type": "error_recovery",
                "confidence_score": 0.1,
                "context_documents_count": 0,
                "system_prompt_type": "error_recovery",
                "metadata": {
                    "error": error_info,
                    "recovery_mode": True,
                },
            }

        except Exception as e:
            self.logger.error(f"Error in error recovery prompt creation: {str(e)}")
            # Final fallback - return basic structure
            return {
                "prompt": f"Query: {query}\nNote: Error in prompt processing.",
                "template_type": "basic_fallback",
                "confidence_score": 0.0,
                "context_documents_count": 0,
                "system_prompt_type": "fallback",
                "metadata": {"error": str(e), "fallback_mode": True},
            }

    def create_response_formatting_prompt(
        self,
        raw_response: str,
        source_documents: List[Document],
        query_intent: Optional[QueryIntent] = None,
    ) -> Dict[str, Any]:
        """
        Create prompt for response formatting and source citation.

        Args:
            raw_response: Raw LLM response
            source_documents: Source documents used
            query_intent: Query intent for formatting context

        Returns:
            Formatting prompt dictionary
        """
        formatting_template = ChatPromptTemplate.from_messages([
            ("system", """
You are a response formatting specialist. Your job is to take a raw response and format it with proper source citations and structure.

Focus on:
1. Clear, well-structured formatting
2. Proper source citations with file references
3. Code snippet formatting when relevant
4. Professional presentation suitable for developers
"""),
            ("human", """
Please format this response with proper source citations:

**Original Response:**
{raw_response}

**Source Documents:**
{source_documents}

**Query Intent:** {query_intent}

Format the response with:
- Clear section headers where appropriate
- Proper source citations [Source: filename.ext]
- Code formatting for any code snippets
- Professional developer-friendly structure

{format_instructions}
"""),
        ])

        try:
            context_text = self._format_context_documents(source_documents)
            
            formatted_prompt = formatting_template.format_prompt(
                raw_response=raw_response,
                source_documents=context_text,
                query_intent=query_intent.value if query_intent else "general",
                format_instructions=self.output_parser.get_format_instructions(),
            )

            return {
                "prompt": formatted_prompt,
                "template_type": "response_formatting",
                "source_count": len(source_documents),
                "metadata": {
                    "query_intent": query_intent,
                    "formatting_mode": True,
                },
            }

        except Exception as e:
            self.logger.error(f"Error creating formatting prompt: {str(e)}")
            return {
                "prompt": f"Format this response: {raw_response}",
                "template_type": "basic_formatting",
                "source_count": 0,
                "metadata": {"error": str(e)},
            }

    def get_supported_intents(self) -> List[QueryIntent]:
        """Get list of supported query intents."""
        return list(self.intent_system_prompts.keys())

    def get_template_statistics(self) -> Dict[str, Any]:
        """Get statistics about available templates."""
        return {
            "system_prompts": len(self.intent_system_prompts) + 1,  # +1 for base
            "query_templates": 3,  # main, high_confidence, low_confidence
            "fallback_templates": 2,  # no_context, error_recovery
            "supported_intents": [intent.value for intent in self.get_supported_intents()],
            "output_parser": self.output_parser.__class__.__name__,
        }
