"""
Prompt templates for Knowledge Graph Agent.

This module contains well-crafted prompt templates using few-shot and chain-of-thought techniques.
"""

from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage, AIMessage


class PromptTemplates:
    """Collection of prompt templates for different use cases."""
    
    # System prompts
    SYSTEM_PROMPT = """You are a Knowledge Graph Agent, an AI assistant specialized in analyzing and explaining codebases from GitHub repositories. You have access to indexed code, documentation, and configuration files.

Your capabilities include:
- Understanding code architecture and patterns
- Explaining implementation details and dependencies
- Providing code examples and best practices
- Analyzing relationships between different components
- Answering questions about specific files or functions

Always provide accurate, helpful responses based on the indexed knowledge base. If you're unsure about something, acknowledge the limitation rather than guessing."""

    # Few-shot examples for different query types
    FEW_SHOT_EXAMPLES = {
        "architecture": [
            {
                "question": "What is the overall architecture of this microservice?",
                "answer": "Based on the codebase analysis, this microservice follows a layered architecture pattern:\n\n1. **API Layer**: FastAPI endpoints handle HTTP requests\n2. **Service Layer**: Business logic in service classes\n3. **Data Layer**: Database operations through repositories\n4. **Domain Layer**: Core business entities and models\n\nKey architectural decisions:\n- Dependency injection for loose coupling\n- Repository pattern for data access\n- DTOs for API request/response handling\n- Middleware for cross-cutting concerns"
            }
        ],
        "implementation": [
            {
                "question": "How does the UserService handle authentication?",
                "answer": "The UserService implements authentication through the following pattern:\n\n```python\nclass UserService:\n    def __init__(self, db: Database):\n        self.db = db\n    \n    def authenticate_user(self, credentials: UserCredentials) -> AuthResult:\n        # 1. Validate input credentials\n        # 2. Hash password and compare\n        # 3. Generate JWT token\n        # 4. Return authentication result\n```\n\nKey implementation details:\n- Uses bcrypt for password hashing\n- JWT tokens for session management\n- Input validation with Pydantic models\n- Error handling for invalid credentials"
            }
        ],
        "dependency": [
            {
                "question": "What are the dependencies of the OrderService?",
                "answer": "The OrderService has the following dependencies:\n\n**Direct Dependencies:**\n- `PaymentService`: For processing payments\n- `InventoryService`: For stock validation\n- `NotificationService`: For order confirmations\n\n**External Dependencies:**\n- `PostgreSQL`: Primary database\n- `Redis`: Caching layer\n- `RabbitMQ`: Message queue for async operations\n\n**Development Dependencies:**\n- `pytest`: Testing framework\n- `mypy`: Type checking\n- `black`: Code formatting"
            }
        ]
    }

    # Chain-of-thought reasoning templates
    COT_TEMPLATES = {
        "code_analysis": """Let me analyze this code step by step:

1. **First, I'll identify the main components:**
   - What classes and functions are present?
   - What is the primary purpose of this code?

2. **Then, I'll examine the relationships:**
   - How do different components interact?
   - What dependencies exist between them?

3. **Next, I'll analyze the patterns:**
   - What design patterns are being used?
   - How is the code structured?

4. **Finally, I'll provide insights:**
   - What are the key implementation details?
   - What are the potential improvements?

Based on this analysis, here's what I found:""",

        "architecture_review": """Let me review the architecture systematically:

1. **System Overview:**
   - What is the main purpose of this system?
   - What are the core components?

2. **Data Flow:**
   - How does data move through the system?
   - What are the entry and exit points?

3. **Technology Stack:**
   - What technologies are being used?
   - How are they integrated?

4. **Scalability & Performance:**
   - How does the system handle load?
   - What are the bottlenecks?

5. **Security & Reliability:**
   - What security measures are in place?
   - How is error handling implemented?

Based on this review, here's the architecture analysis:"""
    }

    @staticmethod
    def create_rag_prompt(query: str, context: List[str], query_type: str = "general") -> str:
        """Create a RAG prompt with few-shot examples and chain-of-thought reasoning."""
        
        # Get relevant few-shot examples
        examples = PromptTemplates.FEW_SHOT_EXAMPLES.get(query_type, [])
        example_text = ""
        
        if examples:
            example_text = "\n\n**Relevant Examples:**\n"
            for i, example in enumerate(examples[:2], 1):  # Limit to 2 examples
                example_text += f"\nExample {i}:\nQ: {example['question']}\nA: {example['answer']}\n"
        
        # Get chain-of-thought template
        cot_template = PromptTemplates.COT_TEMPLATES.get("code_analysis", "")
        
        # Combine context
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
        
        prompt = f"""{PromptTemplates.SYSTEM_PROMPT}

**User Question:** {query}

**Available Context:**
{context_text}

{cot_template}

{example_text}

**Your Response:**"""
        
        return prompt

    @staticmethod
    def create_conversation_prompt(messages: List[Dict[str, str]], current_query: str) -> str:
        """Create a conversation-aware prompt."""
        
        conversation_history = ""
        for msg in messages[-5:]:  # Last 5 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conversation_history += f"\n{role.title()}: {content}"
        
        prompt = f"""{PromptTemplates.SYSTEM_PROMPT}

**Conversation History:**
{conversation_history}

**Current Question:** {current_query}

**Your Response:**"""
        
        return prompt

    @staticmethod
    def create_function_calling_prompt(query: str, available_functions: List[Dict[str, Any]]) -> str:
        """Create a prompt for function calling scenarios."""
        
        functions_text = ""
        for func in available_functions:
            functions_text += f"\n- {func['name']}: {func['description']}"
        
        prompt = f"""{PromptTemplates.SYSTEM_PROMPT}

**User Question:** {query}

**Available Functions:**
{functions_text}

**Instructions:**
1. Analyze the user's question
2. Determine which function(s) would be most helpful
3. Provide a clear explanation of your reasoning
4. If function calling is needed, specify which functions and parameters

**Your Response:**"""
        
        return prompt


# LangChain prompt templates
class LangChainTemplates:
    """LangChain-specific prompt templates."""
    
    @staticmethod
    def get_rag_template() -> PromptTemplate:
        """Get RAG prompt template for LangChain."""
        return PromptTemplate(
            input_variables=["query", "context"],
            template="""You are a Knowledge Graph Agent. Answer the following question based on the provided context.

Context: {context}

Question: {query}

Answer:"""
        )
    
    @staticmethod
    def get_conversation_template() -> ChatPromptTemplate:
        """Get conversation template for LangChain."""
        return ChatPromptTemplate.from_messages([
            ("system", PromptTemplates.SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
    
    @staticmethod
    def get_function_calling_template() -> ChatPromptTemplate:
        """Get function calling template for LangChain."""
        return ChatPromptTemplate.from_messages([
            ("system", PromptTemplates.SYSTEM_PROMPT),
            ("human", "Available functions: {functions}\n\nUser question: {query}\n\nDetermine which functions to call and why.")
        ]) 