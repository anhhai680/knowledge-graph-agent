"""
LLM factory module for the Knowledge Graph Agent.

This module provides a factory pattern for creating LLM instances.
"""

from typing import Dict, Any, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.schema.runnable import RunnableConfig
from loguru import logger

from src.config.settings import LLMProvider, settings


class LLMFactory:
    """
    Factory class for creating LLM instances.

    This class provides methods for creating different types of LLM instances
    based on the application configuration.
    """

    @staticmethod
    def create(
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        callbacks: Optional[list[BaseCallbackHandler]] = None,
        streaming: bool = False,
        **kwargs,
    ) -> BaseChatModel:
        """
        Create an LLM instance based on the provider specified in settings.

        Args:
            model_name: Name of the model to use (default: from settings)
            temperature: Temperature parameter (default: from settings)
            max_tokens: Maximum tokens (default: from settings)
            callbacks: List of callback handlers (default: None)
            streaming: Whether to use streaming mode (default: False)
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            LangChain BaseChatModel instance

        Raises:
            ValueError: If the LLM provider is not supported
        """
        # Get parameters from settings if not provided
        model_name = model_name or settings.openai.model
        temperature = temperature or settings.openai.temperature
        max_tokens = max_tokens or settings.openai.max_tokens

        # Create LLM based on provider
        if settings.llm_provider == LLMProvider.OPENAI:
            return LLMFactory._create_openai_llm(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbacks,
                streaming=streaming,
                **kwargs,
            )
        elif settings.llm_provider == LLMProvider.OLLAMA:
            return LLMFactory._create_ollama_llm(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbacks,
                streaming=streaming,
                **kwargs,
            )
        else:
            error_message = f"Unsupported LLM provider: {settings.llm_provider}"
            logger.error(error_message)
            raise ValueError(error_message)

    @staticmethod
    def _create_openai_llm(
        model_name: str,
        temperature: float,
        max_tokens: int,
        callbacks: Optional[list[BaseCallbackHandler]] = None,
        streaming: bool = False,
        **kwargs,
    ) -> BaseChatModel:
        """
        Create an OpenAI LLM instance.

        Args:
            model_name: Name of the model to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            callbacks: List of callback handlers
            streaming: Whether to use streaming mode
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            LangChain OpenAI chat model instance
        """
        from langchain_openai import ChatOpenAI

        logger.debug(f"Creating OpenAI LLM with model {model_name}")

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            callbacks=callbacks,
            openai_api_key=settings.openai.api_key,
            base_url=settings.llm_api_base_url if settings.llm_api_base_url else None,
            **kwargs,
        )

    @staticmethod
    def _create_ollama_llm(
        model_name: str,
        temperature: float,
        max_tokens: int,
        callbacks: Optional[list[BaseCallbackHandler]] = None,
        streaming: bool = False,
        **kwargs,
    ) -> BaseChatModel:
        """
        Create an Ollama LLM instance.

        Args:
            model_name: Name of the model to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            callbacks: List of callback handlers
            streaming: Whether to use streaming mode
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            LangChain Ollama chat model instance
        """
        from langchain_community.chat_models import ChatOllama

        logger.debug(f"Creating Ollama LLM with model {model_name}")

        return ChatOllama(
            model=model_name,
            temperature=temperature,
            num_ctx=max_tokens,
            streaming=streaming,
            callbacks=callbacks,
            base_url=settings.llm_api_base_url,
            **kwargs,
        )

    @staticmethod
    def get_default_config(
        callbacks: Optional[list[BaseCallbackHandler]] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RunnableConfig:
        """
        Get default runnable configuration for LLM calls.

        Args:
            callbacks: List of callback handlers
            tags: Tags for tracking
            metadata: Additional metadata

        Returns:
            LangChain RunnableConfig object
        """
        config: Dict[str, Any] = {}

        if callbacks:
            config["callbacks"] = callbacks

        if tags:
            config["tags"] = tags

        if metadata:
            config["metadata"] = metadata

        return config
