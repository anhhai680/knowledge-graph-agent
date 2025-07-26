"""
Complete OpenAI SDK-based chatbot agent for Knowledge Graph Agent.

This module implements a comprehensive chatbot with chat completion, function calling,
batching, and message management capabilities.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import openai
from openai import AsyncOpenAI
from loguru import logger

from src.config.settings import settings
from src.llm.prompt_templates import PromptTemplates
from src.vectorstores.store_factory import VectorStoreFactory
from src.llm.embedding_factory import EmbeddingFactory


class MessageRole(str, Enum):
    """Message roles for conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class Message:
    """Message structure for conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "role": self.role.value,
            "content": self.content
        }
        if self.name:
            result["name"] = self.name
        if self.function_call:
            result["function_call"] = self.function_call
        return result


@dataclass
class Conversation:
    """Conversation structure."""
    id: str
    user_id: str
    messages: List[Message]
    context: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get recent messages for context."""
        return self.messages[-count:] if len(self.messages) > count else self.messages


class ChatbotAgent:
    """
    Complete OpenAI SDK-based chatbot agent.
    
    This class implements a comprehensive chatbot with:
    - Chat completion with streaming
    - Function calling capabilities
    - Message batching for efficiency
    - Conversation management
    - RAG integration for knowledge retrieval
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the chatbot agent.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
        """
        self.api_key = api_key or settings.openai.api_key
        self.model = model or "GPT-4o-mini"  # Hardcode model name for compatibility
        self.temperature = temperature or settings.openai.temperature
        self.max_tokens = max_tokens or settings.openai.max_tokens
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=settings.llm_api_base_url,
            default_headers={"User-Agent": "Knowledge-Graph-Agent"}
        )
        
        # Initialize vector store and embeddings
        self.vector_store = VectorStoreFactory.create()
        self.embeddings = EmbeddingFactory.create()
        
        # Conversation storage
        self.conversations: Dict[str, Conversation] = {}
        
        # Function definitions for function calling
        self.available_functions = self._get_available_functions()
        
        logger.info(f"Initialized ChatbotAgent with model: {self.model}")
    
    def _get_available_functions(self) -> List[Dict[str, Any]]:
        """Get available functions for function calling."""
        return [
            {
                "name": "search_codebase",
                "description": "Search for code, functions, or classes in the indexed repositories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for codebase"
                        },
                        "file_type": {
                            "type": "string",
                            "description": "Type of file to search (optional)",
                            "enum": ["python", "javascript", "typescript", "csharp", "java"]
                        },
                        "repository": {
                            "type": "string",
                            "description": "Specific repository to search (optional)"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_architecture_overview",
                "description": "Get an overview of the system architecture",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repository": {
                            "type": "string",
                            "description": "Repository to analyze"
                        },
                        "detail_level": {
                            "type": "string",
                            "description": "Level of detail",
                            "enum": ["high", "medium", "low"]
                        }
                    },
                    "required": ["repository"]
                }
            },
            {
                "name": "analyze_dependencies",
                "description": "Analyze dependencies between components",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "component": {
                            "type": "string",
                            "description": "Component to analyze"
                        },
                        "repository": {
                            "type": "string",
                            "description": "Repository containing the component"
                        }
                    },
                    "required": ["component", "repository"]
                }
            }
        ]
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        function_call: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Perform chat completion with OpenAI.
        
        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response
            function_call: Function calling mode
            
        Returns:
            Response string or streaming response
        """
        try:
            # Prepare the request
            request_data = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Add function calling if requested
            if function_call:
                request_data["functions"] = self.available_functions
                request_data["function_call"] = function_call
            
            if stream:
                return await self._stream_completion(request_data)
            else:
                return await self._complete_chat(request_data)
                
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    async def _complete_chat(self, request_data: Dict[str, Any]) -> str:
        """Complete chat without streaming."""
        response = await self.client.chat.completions.create(**request_data)
        
        # Handle function calls
        if response.choices[0].message.function_call:
            return await self._handle_function_call(response.choices[0].message.function_call)
        
        return response.choices[0].message.content or ""
    
    async def _stream_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stream chat completion."""
        stream = await self.client.chat.completions.create(**request_data, stream=True)
        
        collected_messages = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                collected_messages.append(chunk.choices[0].delta.content)
        
        return {
            "content": "".join(collected_messages),
            "stream": True
        }
    
    async def _handle_function_call(self, function_call: Dict[str, Any]) -> str:
        """Handle function calls from the LLM."""
        function_name = function_call.get("name", "")
        arguments = json.loads(function_call.get("arguments", "{}"))
        
        logger.info(f"Handling function call: {function_name} with args: {arguments}")
        
        # Execute the function
        if function_name == "search_codebase":
            return await self._search_codebase(**arguments)
        elif function_name == "get_architecture_overview":
            return await self._get_architecture_overview(**arguments)
        elif function_name == "analyze_dependencies":
            return await self._analyze_dependencies(**arguments)
        else:
            return f"Unknown function: {function_name}"
    
    async def _search_codebase(self, query: str, **kwargs) -> str:
        """Search the codebase for relevant information."""
        try:
            # Search vector store
            results = self.vector_store.similarity_search(query, k=5)
            
            if not results:
                return "No relevant code found for your query."
            
            # Format results
            response = f"Found {len(results)} relevant code snippets:\n\n"
            for i, doc in enumerate(results, 1):
                response += f"**Result {i}:**\n"
                response += f"File: {doc.metadata.get('file_path', 'Unknown')}\n"
                response += f"Content:\n```\n{doc.page_content[:500]}...\n```\n\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error searching codebase: {str(e)}")
            return f"Error searching codebase: {str(e)}"
    
    async def _get_architecture_overview(self, repository: str, **kwargs) -> str:
        """Get architecture overview of a repository."""
        try:
            # Search for architecture-related documents
            query = f"architecture overview system design {repository}"
            results = self.vector_store.similarity_search(query, k=3)
            
            if not results:
                return f"No architecture information found for {repository}."
            
            response = f"**Architecture Overview for {repository}:**\n\n"
            for doc in results:
                response += f"{doc.page_content}\n\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting architecture overview: {str(e)}")
            return f"Error getting architecture overview: {str(e)}"
    
    async def _analyze_dependencies(self, component: str, repository: str, **kwargs) -> str:
        """Analyze dependencies of a component."""
        try:
            query = f"dependencies imports {component} {repository}"
            results = self.vector_store.similarity_search(query, k=3)
            
            if not results:
                return f"No dependency information found for {component} in {repository}."
            
            response = f"**Dependency Analysis for {component} in {repository}:**\n\n"
            for doc in results:
                response += f"{doc.page_content}\n\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing dependencies: {str(e)}")
            return f"Error analyzing dependencies: {str(e)}"
    
    async def process_message(
        self,
        user_id: str,
        message: str,
        conversation_id: Optional[str] = None,
        use_functions: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user message and return a response.
        
        Args:
            user_id: User identifier
            message: User message
            conversation_id: Conversation ID (optional)
            use_functions: Whether to enable function calling
            
        Returns:
            Response dictionary with message and metadata
        """
        start_time = time.time()
        
        # Get or create conversation
        if conversation_id and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
        else:
            conversation_id = f"conv_{int(time.time())}"
            conversation = Conversation(
                id=conversation_id,
                user_id=user_id,
                messages=[],
                context={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.conversations[conversation_id] = conversation
        
        # Add user message
        user_msg = Message(
            role=MessageRole.USER,
            content=message,
            timestamp=datetime.now()
        )
        conversation.add_message(user_msg)
        
        # Prepare messages for LLM
        messages = []
        
        # Add system message
        system_msg = Message(
            role=MessageRole.SYSTEM,
            content=PromptTemplates.SYSTEM_PROMPT
        )
        messages.append(system_msg.to_dict())
        
        # Add conversation history
        for msg in conversation.get_recent_messages(10):
            messages.append(msg.to_dict())
        
        # Get response
        if use_functions:
            response = await self.chat_completion(
                messages=messages,
                function_call="auto"
            )
        else:
            response = await self.chat_completion(messages=messages)
        
        # Add assistant response
        assistant_msg = Message(
            role=MessageRole.ASSISTANT,
            content=response,
            timestamp=datetime.now()
        )
        conversation.add_message(assistant_msg)
        
        processing_time = time.time() - start_time
        
        return {
            "conversation_id": conversation_id,
            "response": response,
            "processing_time": processing_time,
            "message_count": len(conversation.messages)
        }
    
    async def batch_process_messages(
        self,
        messages: List[Tuple[str, str, Optional[str]]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple messages in batch.
        
        Args:
            messages: List of (user_id, message, conversation_id) tuples
            
        Returns:
            List of response dictionaries
        """
        tasks = []
        for user_id, message, conversation_id in messages:
            task = self.process_message(user_id, message, conversation_id)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def get_conversation_history(self, conversation_id: str) -> Optional[List[Message]]:
        """Get conversation history."""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id].messages
        return None
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        total_conversations = len(self.conversations)
        total_messages = sum(len(conv.messages) for conv in self.conversations.values())
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "active_conversations": len([c for c in self.conversations.values() 
                                      if (datetime.now() - c.updated_at).seconds < 3600])
        } 