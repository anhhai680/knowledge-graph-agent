"""
Test conversation logs for Knowledge Graph Agent.

This module contains comprehensive tests for multi-turn dialogue with context.
"""

import asyncio
import json
import pytest
from datetime import datetime
from typing import Dict, List, Any

from src.agents.chatbot_agent import ChatbotAgent, Message, MessageRole, Conversation


class TestConversationLogs:
    """Test conversation logs with multi-turn dialogue."""
    
    @pytest.fixture
    async def chatbot_agent(self):
        """Create a chatbot agent for testing."""
        return ChatbotAgent(
            api_key="test_key",
            model="gpt-4o-mini",
            temperature=0.7
        )
    
    @pytest.fixture
    def sample_conversation_logs(self):
        """Sample conversation logs for testing."""
        return [
            {
                "conversation_id": "conv_001",
                "user_id": "user_001",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the architecture of the microservice?",
                        "timestamp": "2024-01-15T10:00:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "Based on the codebase analysis, this microservice follows a layered architecture pattern:\n\n1. **API Layer**: FastAPI endpoints handle HTTP requests\n2. **Service Layer**: Business logic in service classes\n3. **Data Layer**: Database operations through repositories\n4. **Domain Layer**: Core business entities and models\n\nKey architectural decisions:\n- Dependency injection for loose coupling\n- Repository pattern for data access\n- DTOs for API request/response handling\n- Middleware for cross-cutting concerns",
                        "timestamp": "2024-01-15T10:00:05Z"
                    },
                    {
                        "role": "user",
                        "content": "Can you show me the UserService implementation?",
                        "timestamp": "2024-01-15T10:01:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "Here's the UserService implementation:\n\n```python\nclass UserService:\n    def __init__(self, db: Database):\n        self.db = db\n    \n    def get_user(self, user_id: str) -> User:\n        return self.db.query('SELECT * FROM users WHERE id = ?', user_id)\n    \n    def create_user(self, user_data: UserCreate) -> User:\n        # Implementation details...\n        pass\n```\n\nThe service uses dependency injection and follows the repository pattern for data access.",
                        "timestamp": "2024-01-15T10:01:08Z"
                    },
                    {
                        "role": "user",
                        "content": "What are its dependencies?",
                        "timestamp": "2024-01-15T10:02:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "The UserService has the following dependencies:\n\n**Direct Dependencies:**\n- `Database`: For data access operations\n- `UserCreate`: Pydantic model for user creation\n- `User`: Domain model for user entity\n\n**External Dependencies:**\n- `PostgreSQL`: Primary database\n- `Redis`: Caching layer (optional)\n\n**Development Dependencies:**\n- `pytest`: Testing framework\n- `mypy`: Type checking\n\nThe service is designed with loose coupling through dependency injection.",
                        "timestamp": "2024-01-15T10:02:05Z"
                    }
                ],
                "context": {
                    "repository": "microservice-architecture",
                    "focus_area": "user_management",
                    "discussion_topic": "service_architecture"
                }
            },
            {
                "conversation_id": "conv_002",
                "user_id": "user_002",
                "messages": [
                    {
                        "role": "user",
                        "content": "How do I add authentication to the API?",
                        "timestamp": "2024-01-15T11:00:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "To add authentication to the API, you can implement JWT-based authentication:\n\n```python\nfrom fastapi import Depends, HTTPException\nfrom fastapi.security import HTTPBearer\n\nsecurity = HTTPBearer()\n\ndef get_current_user(token: str = Depends(security)):\n    # Validate JWT token\n    # Return user information\n    pass\n\n@app.get('/protected')\ndef protected_route(current_user = Depends(get_current_user)):\n    return {'message': 'Authenticated'}\n```\n\nYou'll also need to:\n1. Install `python-jose` for JWT handling\n2. Set up secret keys\n3. Create login endpoints\n4. Add middleware for token validation",
                        "timestamp": "2024-01-15T11:00:08Z"
                    },
                    {
                        "role": "user",
                        "content": "What about role-based access control?",
                        "timestamp": "2024-01-15T11:01:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "For role-based access control (RBAC), you can extend the authentication system:\n\n```python\nfrom enum import Enum\n\nclass UserRole(Enum):\n    ADMIN = 'admin'\n    USER = 'user'\n    GUEST = 'guest'\n\ndef require_role(required_role: UserRole):\n    def role_checker(current_user = Depends(get_current_user)):\n        if current_user.role != required_role:\n            raise HTTPException(403, 'Insufficient permissions')\n        return current_user\n    return role_checker\n\n@app.get('/admin-only')\ndef admin_route(user = Depends(require_role(UserRole.ADMIN))):\n    return {'message': 'Admin access'}\n```\n\nThis provides fine-grained access control based on user roles.",
                        "timestamp": "2024-01-15T11:01:10Z"
                    }
                ],
                "context": {
                    "repository": "microservice-architecture",
                    "focus_area": "security",
                    "discussion_topic": "authentication_authorization"
                }
            }
        ]
    
    async def test_multi_turn_dialogue_context_preservation(self, chatbot_agent):
        """Test that context is preserved across multiple turns."""
        conversation_id = "test_conv_001"
        
        # First turn
        response1 = await chatbot_agent.process_message(
            user_id="test_user",
            message="What is the architecture of the microservice?",
            conversation_id=conversation_id
        )
        
        assert response1["conversation_id"] == conversation_id
        assert "architecture" in response1["response"].lower()
        
        # Second turn - should reference previous context
        response2 = await chatbot_agent.process_message(
            user_id="test_user",
            message="Can you show me the UserService implementation?",
            conversation_id=conversation_id
        )
        
        assert response2["conversation_id"] == conversation_id
        assert response2["message_count"] > response1["message_count"]
        
        # Third turn - should maintain context about the service
        response3 = await chatbot_agent.process_message(
            user_id="test_user",
            message="What are its dependencies?",
            conversation_id=conversation_id
        )
        
        assert response3["conversation_id"] == conversation_id
        # Should reference UserService from previous context
        assert "dependencies" in response3["response"].lower()
    
    async def test_conversation_history_retrieval(self, chatbot_agent):
        """Test conversation history retrieval."""
        conversation_id = "test_conv_002"
        
        # Add multiple messages
        await chatbot_agent.process_message(
            user_id="test_user",
            message="Hello, I need help with the codebase.",
            conversation_id=conversation_id
        )
        
        await chatbot_agent.process_message(
            user_id="test_user",
            message="What's the main architecture?",
            conversation_id=conversation_id
        )
        
        # Retrieve conversation history
        history = chatbot_agent.get_conversation_history(conversation_id)
        
        assert history is not None
        assert len(history) >= 4  # 2 user messages + 2 assistant responses
        assert all(isinstance(msg, Message) for msg in history)
        
        # Check message roles
        user_messages = [msg for msg in history if msg.role == MessageRole.USER]
        assistant_messages = [msg for msg in history if msg.role == MessageRole.ASSISTANT]
        
        assert len(user_messages) >= 2
        assert len(assistant_messages) >= 2
    
    async def test_context_aware_responses(self, chatbot_agent):
        """Test that responses are context-aware."""
        conversation_id = "test_conv_003"
        
        # Start with architecture question
        await chatbot_agent.process_message(
            user_id="test_user",
            message="Explain the microservice architecture.",
            conversation_id=conversation_id
        )
        
        # Follow up with specific component question
        response = await chatbot_agent.process_message(
            user_id="test_user",
            message="How does the authentication work in this architecture?",
            conversation_id=conversation_id
        )
        
        # Response should reference both authentication and architecture context
        assert "authentication" in response["response"].lower()
        assert "architecture" in response["response"].lower()
    
    async def test_function_calling_in_conversation(self, chatbot_agent):
        """Test function calling within conversation context."""
        conversation_id = "test_conv_004"
        
        # Ask a question that should trigger function calling
        response = await chatbot_agent.process_message(
            user_id="test_user",
            message="Search for UserService in the codebase",
            conversation_id=conversation_id,
            use_functions=True
        )
        
        assert response["conversation_id"] == conversation_id
        # Should either call function or provide relevant response
        assert "UserService" in response["response"] or "search" in response["response"].lower()
    
    async def test_conversation_statistics(self, chatbot_agent):
        """Test conversation statistics tracking."""
        # Create multiple conversations
        await chatbot_agent.process_message("user1", "Hello", "conv1")
        await chatbot_agent.process_message("user2", "Hi there", "conv2")
        await chatbot_agent.process_message("user1", "How are you?", "conv1")
        
        stats = chatbot_agent.get_conversation_stats()
        
        assert stats["total_conversations"] >= 2
        assert stats["total_messages"] >= 6  # 3 user messages + 3 assistant responses
        assert stats["active_conversations"] >= 2
    
    async def test_conversation_clearing(self, chatbot_agent):
        """Test conversation clearing functionality."""
        conversation_id = "test_conv_005"
        
        # Add a message
        await chatbot_agent.process_message(
            user_id="test_user",
            message="Test message",
            conversation_id=conversation_id
        )
        
        # Verify conversation exists
        history = chatbot_agent.get_conversation_history(conversation_id)
        assert history is not None
        
        # Clear conversation
        cleared = chatbot_agent.clear_conversation(conversation_id)
        assert cleared is True
        
        # Verify conversation is gone
        history_after = chatbot_agent.get_conversation_history(conversation_id)
        assert history_after is None
    
    async def test_batch_message_processing(self, chatbot_agent):
        """Test batch processing of multiple messages."""
        messages = [
            ("user1", "What is the architecture?", "conv_batch1"),
            ("user2", "How does authentication work?", "conv_batch2"),
            ("user3", "Show me the UserService", "conv_batch3")
        ]
        
        responses = await chatbot_agent.batch_process_messages(messages)
        
        assert len(responses) == 3
        for response in responses:
            assert "conversation_id" in response
            assert "response" in response
            assert "processing_time" in response
    
    def test_conversation_data_structure(self):
        """Test conversation data structure."""
        conversation = Conversation(
            id="test_conv",
            user_id="test_user",
            messages=[],
            context={"topic": "testing"},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Test message addition
        message = Message(
            role=MessageRole.USER,
            content="Test message",
            timestamp=datetime.now()
        )
        
        conversation.add_message(message)
        assert len(conversation.messages) == 1
        assert conversation.messages[0].content == "Test message"
        
        # Test recent messages
        recent = conversation.get_recent_messages(5)
        assert len(recent) == 1
    
    async def test_error_handling_in_conversation(self, chatbot_agent):
        """Test error handling in conversation."""
        # Test with invalid API key (should handle gracefully)
        chatbot_agent.api_key = "invalid_key"
        
        response = await chatbot_agent.process_message(
            user_id="test_user",
            message="Test message",
            conversation_id="error_test"
        )
        
        # Should return error message instead of crashing
        assert "error" in response["response"].lower() or "sorry" in response["response"].lower()


# Integration test for conversation logs
class TestConversationLogsIntegration:
    """Integration tests for conversation logs."""
    
    async def test_full_conversation_flow(self):
        """Test a complete conversation flow."""
        agent = ChatbotAgent(
            api_key="test_key",
            model="gpt-4o-mini"
        )
        
        conversation_id = "integration_test"
        
        # Simulate a real conversation flow
        messages = [
            "What is the overall architecture of this system?",
            "Can you explain the authentication flow?",
            "What are the main dependencies?",
            "How do I add a new service?",
            "What testing strategies are used?"
        ]
        
        responses = []
        for message in messages:
            response = await agent.process_message(
                user_id="integration_user",
                message=message,
                conversation_id=conversation_id
            )
            responses.append(response)
        
        # Verify all responses were generated
        assert len(responses) == len(messages)
        
        # Verify conversation history
        history = agent.get_conversation_history(conversation_id)
        assert len(history) == len(messages) * 2  # User + Assistant messages
        
        # Verify context preservation
        for i, response in enumerate(responses):
            assert response["conversation_id"] == conversation_id
            assert response["message_count"] == (i + 1) * 2  # Each turn adds 2 messages


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 