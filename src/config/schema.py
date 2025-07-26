"""
Data schema definitions for Knowledge Graph Agent.

This module defines the data structures and schemas used throughout the application.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Types of documents that can be processed."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    README = "readme"
    TEST = "test"


class LanguageType(str, Enum):
    """Programming languages supported."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CSHARP = "csharp"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    UNKNOWN = "unknown"


class QueryType(str, Enum):
    """Types of queries that can be processed."""
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    DEPENDENCY = "dependency"
    USAGE = "usage"
    DOCUMENTATION = "documentation"
    GENERAL = "general"


class MockDataSchema(BaseModel):
    """Schema for mock data used in testing and development."""
    
    class RepositoryMock(BaseModel):
        """Mock repository data."""
        owner: str = Field(..., description="Repository owner")
        name: str = Field(..., description="Repository name")
        description: str = Field(..., description="Repository description")
        language: LanguageType = Field(..., description="Primary language")
        stars: int = Field(0, description="Number of stars")
        forks: int = Field(0, description="Number of forks")
        last_updated: datetime = Field(..., description="Last update time")
        
    class DocumentMock(BaseModel):
        """Mock document data."""
        id: str = Field(..., description="Document ID")
        content: str = Field(..., description="Document content")
        file_path: str = Field(..., description="File path")
        language: LanguageType = Field(..., description="Programming language")
        doc_type: DocumentType = Field(..., description="Document type")
        repository: str = Field(..., description="Repository name")
        chunk_index: int = Field(0, description="Chunk index")
        metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
        
    class QueryMock(BaseModel):
        """Mock query data."""
        id: str = Field(..., description="Query ID")
        question: str = Field(..., description="User question")
        query_type: QueryType = Field(..., description="Type of query")
        context: Dict[str, Any] = Field(default_factory=dict, description="Query context")
        expected_response: str = Field(..., description="Expected response")
        
    class ConversationMock(BaseModel):
        """Mock conversation data."""
        id: str = Field(..., description="Conversation ID")
        user_id: str = Field(..., description="User ID")
        messages: List[Dict[str, Any]] = Field(..., description="Conversation messages")
        context: Dict[str, Any] = Field(default_factory=dict, description="Conversation context")
        created_at: datetime = Field(..., description="Creation time")
        updated_at: datetime = Field(..., description="Last update time")


# Mock data examples
MOCK_REPOSITORIES = [
    {
        "owner": "example-org",
        "name": "microservice-architecture",
        "description": "A comprehensive microservices architecture example with Docker and Kubernetes",
        "language": LanguageType.PYTHON,
        "stars": 150,
        "forks": 25,
        "last_updated": datetime.now()
    },
    {
        "owner": "example-org", 
        "name": "react-frontend",
        "description": "Modern React frontend with TypeScript and Material-UI",
        "language": LanguageType.TYPESCRIPT,
        "stars": 89,
        "forks": 12,
        "last_updated": datetime.now()
    }
]

MOCK_DOCUMENTS = [
    {
        "id": "doc_001",
        "content": "class UserService:\n    def __init__(self):\n        self.db = Database()\n    \n    def get_user(self, user_id: str):\n        return self.db.query('SELECT * FROM users WHERE id = ?', user_id)",
        "file_path": "src/services/user_service.py",
        "language": LanguageType.PYTHON,
        "doc_type": DocumentType.CODE,
        "repository": "microservice-architecture",
        "chunk_index": 0,
        "metadata": {
            "class_name": "UserService",
            "methods": ["get_user"],
            "dependencies": ["Database"]
        }
    }
]

MOCK_QUERIES = [
    {
        "id": "query_001",
        "question": "How does the UserService handle user authentication?",
        "query_type": QueryType.IMPLEMENTATION,
        "context": {"repository": "microservice-architecture"},
        "expected_response": "The UserService class provides user authentication through the get_user method..."
    }
] 