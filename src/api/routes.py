from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio

from src.agents.chatbot_agent import ChatbotAgent
from src.config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Graph Agent API",
    description="API for the Knowledge Graph Agent, providing endpoints to interact with the knowledge graph.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot agent
chatbot_agent = ChatbotAgent()

# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    user_id: str
    message: str
    conversation_id: Optional[str] = None
    use_functions: bool = True

class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    processing_time: float
    message_count: int

class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: List[Dict[str, Any]]
    total_messages: int

class ConversationStatsResponse(BaseModel):
    total_conversations: int
    total_messages: int
    active_conversations: int

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float

# API Routes

@app.get("/")
def index():
    return {"message": "Welcome to the Knowledge Graph Agent API!"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=0.0  # TODO: Implement uptime tracking
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message and return a response.
    
    This endpoint handles:
    - Single message processing
    - Conversation context preservation
    - Function calling for code analysis
    - Multi-turn dialogue support
    """
    try:
        response = await chatbot_agent.process_message(
            user_id=request.user_id,
            message=request.message,
            conversation_id=request.conversation_id,
            use_functions=request.use_functions
        )
        
        return ChatResponse(**response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/chat/batch")
async def batch_chat_endpoint(messages: List[ChatRequest]):
    """
    Process multiple chat messages in batch.
    
    This endpoint is useful for:
    - Bulk processing of messages
    - Testing multiple scenarios
    - Performance optimization
    """
    try:
        # Convert to batch format
        batch_messages = [
            (msg.user_id, msg.message, msg.conversation_id)
            for msg in messages
        ]
        
        responses = await chatbot_agent.batch_process_messages(batch_messages)
        
        return {
            "responses": responses,
            "total_processed": len(responses)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch messages: {str(e)}")

@app.get("/conversations/{conversation_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(conversation_id: str):
    """
    Get conversation history for a specific conversation.
    
    Returns:
    - All messages in the conversation
    - Message metadata (timestamps, roles)
    - Total message count
    """
    try:
        history = chatbot_agent.get_conversation_history(conversation_id)
        
        if history is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Convert messages to dictionary format
        messages = []
        for msg in history:
            message_dict = {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
            }
            if msg.name:
                message_dict["name"] = msg.name
            if msg.function_call:
                message_dict["function_call"] = msg.function_call
            
            messages.append(message_dict)
        
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=messages,
            total_messages=len(messages)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation history: {str(e)}")

@app.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """
    Clear a specific conversation.
    
    This removes all messages and context for the conversation.
    """
    try:
        cleared = chatbot_agent.clear_conversation(conversation_id)
        
        if not cleared:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"message": f"Conversation {conversation_id} cleared successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing conversation: {str(e)}")

@app.get("/conversations/stats", response_model=ConversationStatsResponse)
async def get_conversation_stats():
    """
    Get conversation statistics.
    
    Returns:
    - Total number of conversations
    - Total number of messages
    - Number of active conversations
    """
    try:
        stats = chatbot_agent.get_conversation_stats()
        return ConversationStatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation stats: {str(e)}")

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Stream chat responses in real-time.
    
    This endpoint provides streaming responses for:
    - Real-time interaction
    - Long-running queries
    - Progress feedback
    """
    try:
        # For now, return a simple streaming response
        # TODO: Implement actual streaming with Server-Sent Events
        response = await chatbot_agent.process_message(
            user_id=request.user_id,
            message=request.message,
            conversation_id=request.conversation_id,
            use_functions=request.use_functions
        )
        
        return {
            "stream": True,
            "data": response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in streaming chat: {str(e)}")

@app.get("/functions")
async def get_available_functions():
    """
    Get available functions for function calling.
    
    Returns the list of functions that can be called by the chatbot.
    """
    try:
        return {
            "functions": chatbot_agent.available_functions,
            "total_functions": len(chatbot_agent.available_functions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving functions: {str(e)}")

@app.post("/index/repository")
async def index_repository(background_tasks: BackgroundTasks, repository_url: str):
    """
    Index a new repository.
    
    This endpoint triggers the indexing process for a new repository.
    The indexing runs in the background to avoid blocking the API.
    """
    try:
        # TODO: Implement repository indexing
        background_tasks.add_task(index_repository_task, repository_url)
        
        return {
            "message": f"Repository indexing started for {repository_url}",
            "status": "indexing",
            "repository_url": repository_url
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting repository indexing: {str(e)}")

async def index_repository_task(repository_url: str):
    """
    Background task for repository indexing.
    
    This function runs in the background to avoid blocking the API.
    """
    try:
        # TODO: Implement actual repository indexing
        print(f"Indexing repository: {repository_url}")
        # Simulate indexing process
        await asyncio.sleep(5)
        print(f"Repository indexed: {repository_url}")
        
    except Exception as e:
        print(f"Error indexing repository {repository_url}: {str(e)}")

@app.get("/search")
async def search_codebase(query: str, limit: int = 5):
    """
    Search the codebase for relevant information.
    
    This endpoint provides direct access to the vector search functionality.
    """
    try:
        # Use the chatbot's search functionality
        results = await chatbot_agent._search_codebase(query)
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results) if isinstance(results, list) else 1
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching codebase: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Resource not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("Knowledge Graph Agent API starting up...")
    # TODO: Add initialization logic

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    print("Knowledge Graph Agent API shutting down...")
    # TODO: Add cleanup logic