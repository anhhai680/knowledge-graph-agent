"""
Demo settings for testing without real API keys.
"""

import os
from typing import Dict, Any

# Demo environment variables
DEMO_ENV_VARS = {
    "APP_ENV": "development",
    "LOG_LEVEL": "INFO",
    "HOST": "0.0.0.0",
    "PORT": "8000",
    "LLM_PROVIDER": "openai",
    "OPENAI_API_KEY": "demo_key_for_testing",
    "OPENAI_MODEL": "gpt-4o-mini",
    "OPENAI_TEMPERATURE": "0.7",
    "OPENAI_MAX_TOKENS": "4000",
    "DATABASE_TYPE": "chroma",
    "CHROMA_HOST": "localhost",
    "CHROMA_PORT": "8001",
    "CHROMA_COLLECTION_NAME": "knowledge-base-graph",
    "GITHUB_TOKEN": "demo_github_token",
    "GITHUB_FILE_EXTENSIONS": ".py,.js,.ts,.cs,.java,.md,.json,.yml,.yaml,.txt",
    "EMBEDDING_PROVIDER": "openai",
    "EMBEDDING_MODEL": "text-embedding-ada-002",
    "EMBEDDING_BATCH_SIZE": "50",
    "CHUNK_SIZE": "1000",
    "CHUNK_OVERLAP": "200",
    "WORKFLOW_STATE_PERSISTENCE": "true",
    "WORKFLOW_RETRY_ATTEMPTS": "3",
    "WORKFLOW_TIMEOUT_SECONDS": "3600"
}

def setup_demo_environment():
    """Set up demo environment variables."""
    for key, value in DEMO_ENV_VARS.items():
        os.environ[key] = value

def get_demo_settings() -> Dict[str, Any]:
    """Get demo settings dictionary."""
    return DEMO_ENV_VARS.copy() 