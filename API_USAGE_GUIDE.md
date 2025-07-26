# 🚀 Knowledge Graph Agent - API Usage Guide

Hướng dẫn sử dụng tất cả các API endpoints của Knowledge Graph Agent.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Testing Tools](#testing-tools)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### 1. Start the Server

```bash
# Start ChromaDB
docker-compose up -d chroma

# Start the API server
python main.py
```

### 2. Test Basic Health

```bash
curl http://localhost:8000/health
```

### 3. Run Quick Demo

```bash
python quick_demo.py
```

## 📡 API Endpoints

### 🔍 Basic Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

### 💬 Chat Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Single chat message |
| `/chat/batch` | POST | Batch chat messages |
| `/chat/stream` | POST | Streaming chat |

### 📊 Conversation Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/conversations/stats` | GET | Get conversation statistics |
| `/conversations/{id}/history` | GET | Get conversation history |

### 🔧 Advanced Features

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/functions` | GET | Get available functions |
| `/search` | GET | Search codebase |

## 💡 Usage Examples

### 1. Single Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, what can you do?",
    "user_id": "user123"
  }'
```

**Response:**
```json
{
  "conversation_id": "conv_1234567890",
  "response": "Hello! I can help you analyze and understand codebases...",
  "processing_time": 1.23,
  "message_count": 2
}
```

### 2. Batch Chat

```bash
curl -X POST http://localhost:8000/chat/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"message": "What is this project about?", "user_id": "user1"},
    {"message": "Explain the architecture", "user_id": "user2"},
    {"message": "What are the main features?", "user_id": "user3"}
  ]'
```

**Response:**
```json
{
  "responses": [
    {
      "conversation_id": "conv_1234567890",
      "response": "This project is a Knowledge Graph Agent...",
      "processing_time": 1.23,
      "message_count": 2
    }
  ],
  "total_processed": 3
}
```

### 3. Multi-turn Conversation

```bash
# First message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, I want to understand this codebase",
    "user_id": "user123"
  }'

# Follow-up message (using conversation_id from first response)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Can you explain the main components?",
    "user_id": "user123",
    "conversation_id": "conv_1234567890"
  }'
```

### 4. Search Codebase

```bash
curl "http://localhost:8000/search?query=chatbot&limit=5"
```

**Response:**
```json
{
  "query": "chatbot",
  "results": [
    {
      "content": "Chatbot agent implementation...",
      "metadata": {
        "file_path": "src/agents/chatbot_agent.py",
        "line_number": 10
      }
    }
  ],
  "total_results": 1
}
```

### 5. Get Available Functions

```bash
curl http://localhost:8000/functions
```

**Response:**
```json
{
  "functions": [
    {
      "name": "search_codebase",
      "description": "Search for code, functions, or classes in the indexed repositories",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Search query for codebase"
          }
        },
        "required": ["query"]
      }
    }
  ],
  "total_functions": 3
}
```

### 6. Conversation Statistics

```bash
curl http://localhost:8000/conversations/stats
```

**Response:**
```json
{
  "total_conversations": 5,
  "total_messages": 12,
  "active_conversations": 3
}
```

## 🛠️ Testing Tools

### 1. Quick Demo

```bash
python quick_demo.py
```

Chạy demo nhanh tất cả các API endpoints.

### 2. Full Test Suite

```bash
python api_test_suite.py
```

Chạy test suite đầy đủ với async testing.

### 3. Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# Welcome message
curl http://localhost:8000/

# Single chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "user_id": "test"}'

# Batch chat
curl -X POST http://localhost:8000/chat/batch \
  -H "Content-Type: application/json" \
  -d '[{"message": "Test", "user_id": "user1"}]'

# Functions
curl http://localhost:8000/functions

# Search
curl "http://localhost:8000/search?query=test&limit=5"
```

## 🔧 Configuration

### Environment Variables

Tạo file `.env` với các cấu hình:

```env
# Server Configuration
HOST=127.0.0.1
PORT=8000
APP_ENV=development
LOG_LEVEL=INFO

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
LLM_API_BASE_URL=https://your-api-base-url
OPENAI_MODEL=GPT-4o-mini

# Database Configuration
DATABASE_TYPE=chroma
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_COLLECTION_NAME=knowledge-base-graph

# GitHub Configuration
GITHUB_TOKEN=your_github_token
GITHUB_FILE_EXTENSIONS=["py","js","ts","cs","java","md","json","yml","yaml","txt"]
```

### Docker Setup

```bash
# Start ChromaDB
docker-compose up -d chroma

# Check if ChromaDB is running
docker ps
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Server Not Starting

```bash
# Check if port is in use
lsof -i :8000

# Kill existing process
pkill -f "python main.py"

# Start server
python main.py
```

#### 2. ChromaDB Connection Error

```bash
# Start ChromaDB
docker-compose up -d chroma

# Check ChromaDB logs
docker-compose logs chroma
```

#### 3. OpenAI API Errors

- Kiểm tra API key trong file `.env`
- Kiểm tra base URL và model name
- Đảm bảo API key có quyền truy cập model

#### 4. Module Import Errors

```bash
# Install dependencies
pip install -r requirements.txt

# Install missing packages
pip install httpx requests
```

### Debug Mode

```bash
# Set debug logging
export LOG_LEVEL=DEBUG

# Start server with debug
python main.py
```

## 📚 Advanced Usage

### 1. Function Calling

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Search for chatbot related code",
    "user_id": "user123"
  }'
```

### 2. Streaming Responses

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain the architecture",
    "user_id": "user123"
  }'
```

### 3. Custom Headers

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-custom-key" \
  -d '{"message": "Hello", "user_id": "user123"}'
```

## 🎯 Performance Tips

1. **Batch Processing**: Sử dụng `/chat/batch` cho nhiều messages
2. **Connection Pooling**: Sử dụng persistent connections
3. **Caching**: Cache responses cho repeated queries
4. **Async Processing**: Sử dụng async clients cho high throughput

## 📞 Support

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Logs**: Check console output for detailed logs

---

**🎉 Happy Coding with Knowledge Graph Agent!** 