"""
OpenAI provider module for the Knowledge Graph Agent.

This module provides integration with OpenAI for LLM and embeddings.
"""

from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.messages import BaseMessage
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config.settings import settings


class OpenAIProvider:
    """
    OpenAI provider class for LLM and embeddings.

    This class provides methods for interacting with OpenAI API
    with error handling and retry logic.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        callbacks: Optional[list[BaseCallbackHandler]] = None,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            model_name: Name of the model to use (default: from settings)
            temperature: Temperature parameter (default: from settings)
            max_tokens: Maximum tokens (default: from settings)
            callbacks: List of callback handlers (default: None)
        """
        # Get parameters from settings if not provided
        self.model_name = model_name or settings.openai.model
        self.temperature = temperature or settings.openai.temperature
        self.max_tokens = max_tokens or settings.openai.max_tokens
        self.callbacks = callbacks

        # Import here to avoid circular imports
        from src.llm.llm_factory import LLMFactory

        # Create LLM instance
        self.llm = LLMFactory.create(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            callbacks=self.callbacks,
        )

        logger.debug(f"Initialized OpenAI provider with model {self.model_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception)),
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            chat_history: Optional chat history

        Returns:
            Generated response text
        """
        # Create messages for the LLM
        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt),
        ]

        # Add chat history if provided
        if chat_history:
            for message in chat_history:
                if message["role"] == "user":
                    messages.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    messages.append(AIMessage(content=message["content"]))

        # Add the current user prompt
        messages.append(HumanMessage(content=user_prompt))

        try:
            # Call the LLM
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception)),
    )
    def generate_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a response from the LLM synchronously.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            chat_history: Optional chat history

        Returns:
            Generated response text
        """
        # Create messages for the LLM
        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt),
        ]

        # Add chat history if provided
        if chat_history:
            for message in chat_history:
                if message["role"] == "user":
                    messages.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    messages.append(AIMessage(content=message["content"]))

        # Add the current user prompt
        messages.append(HumanMessage(content=user_prompt))

        try:
            # Call the LLM
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    @staticmethod
    def format_context_for_prompt(context: List[Dict[str, Any]]) -> str:
        """
        Format context for inclusion in a prompt.

        Args:
            context: List of context dictionaries

        Returns:
            Formatted context string
        """
        formatted_context = []

        for i, item in enumerate(context):
            content = item.get("content", "")
            metadata = item.get("metadata", {})

            # Extract metadata fields
            source = metadata.get("source", "")
            file_path = metadata.get("file_path", "")
            language = metadata.get("language", "")

            # Format source information
            source_info = f"[{i+1}] "
            if source and file_path:
                source_info += f"{source}: {file_path}"
            elif file_path:
                source_info += file_path
            elif source:
                source_info += source
            else:
                source_info += "Unknown source"

            if language:
                source_info += f" (Language: {language})"

            # Format content with source information
            formatted_item = f"{source_info}\n{'-' * 40}\n{content}\n{'-' * 40}\n"
            formatted_context.append(formatted_item)

        return "\n".join(formatted_context)
