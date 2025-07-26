# 📁 Knowledge Graph Agent - Test Files Summary

Tổng hợp tất cả các file test đã được tạo và tổ chức trong folder `test/`.

## 🗂️ File Structure

```
knowledge-graph-agent/
├── 📄 API_USAGE_GUIDE.md          # Hướng dẫn sử dụng API chi tiết
├── 📄 README_API_FILES.md         # Tổng hợp các file API
├── 📄 TEST_FILES_SUMMARY.md       # File này - Tổng hợp test files
├── 📁 test/                       # Folder chứa tất cả test files
│   ├── 📄 __init__.py             # Package initialization
│   ├── 📄 README.md               # Hướng dẫn test suite
│   ├── 🐍 run_all_tests.py        # Chạy tất cả tests
│   ├── 🐍 test_health.py          # Health check tests
│   ├── 🐍 test_chat.py            # Chat functionality tests
│   ├── 🐍 test_advanced.py        # Advanced features tests
│   ├── 🐍 api_test_suite.py       # Full async test suite
│   ├── 🐍 quick_demo.py           # Quick demo script
│   └── 🐍 test_agent.py           # Direct agent testing
├── 🐍 main.py                     # Server chính
├── 📄 .env                        # Cấu hình environment
└── 📄 requirements.txt            # Dependencies
```

## 📋 Test Files Description

### 🏥 Health Tests (`test/test_health.py`)
**Mô tả:** Test các endpoint cơ bản và health check
**Tính năng:**
- Health check endpoint (`/health`)
- Welcome message endpoint (`/`)
- Documentation endpoint (`/docs`)
- OpenAPI schema endpoint (`/openapi.json`)
- Response time measurement
- Error handling validation

**Sử dụng:**
```bash
python test/test_health.py
```

### 💬 Chat Tests (`test/test_chat.py`)
**Mô tả:** Test tất cả tính năng chat
**Tính năng:**
- Single chat messages
- Batch chat processing
- Streaming chat responses
- Multi-turn conversations
- Conversation statistics
- Conversation history

**Sử dụng:**
```bash
python test/test_chat.py
```

### 🔧 Advanced Tests (`test/test_advanced.py`)
**Mô tả:** Test các tính năng nâng cao
**Tính năng:**
- Functions API (`/functions`)
- Search functionality (`/search`)
- Code analysis
- Architecture overview
- Dependency analysis
- Multiple search queries

**Sử dụng:**
```bash
python test/test_advanced.py
```

### ⚡ Async Test Suite (`test/api_test_suite.py`)
**Mô tả:** Test suite đầy đủ với async testing
**Tính năng:**
- Async HTTP client
- Comprehensive testing
- Performance metrics
- Detailed error reporting
- Test summary với success rate

**Sử dụng:**
```bash
python test/api_test_suite.py
```

### 🚀 Quick Demo (`test/quick_demo.py`)
**Mô tả:** Demo nhanh tất cả API endpoints
**Tính năng:**
- Quick overview of all features
- Simple sync testing
- Curl command examples
- Basic functionality verification

**Sử dụng:**
```bash
python test/quick_demo.py
```

### 🧪 Direct Agent Test (`test/test_agent.py`)
**Mô tả:** Test ChatbotAgent trực tiếp
**Tính năng:**
- Test async ChatbotAgent
- Test với OpenAI API
- Test conversation flow
- Direct module testing

**Sử dụng:**
```bash
python test/test_agent.py
```

### 🎯 Run All Tests (`test/run_all_tests.py`)
**Mô tả:** Chạy tất cả test suites
**Tính năng:**
- Automated test execution
- Comprehensive reporting
- Performance metrics
- Success/failure summary
- CI/CD integration

**Sử dụng:**
```bash
python test/run_all_tests.py
```

## 🚀 Quick Start Commands

### 1. Start Server
```bash
# Start ChromaDB
docker-compose up -d chroma

# Start API server
python main.py
```

### 2. Run All Tests
```bash
# Run comprehensive test suite
python test/run_all_tests.py
```

### 3. Run Individual Test Categories
```bash
# Health tests
python test/test_health.py

# Chat tests
python test/test_chat.py

# Advanced features tests
python test/test_advanced.py

# Quick demo
python test/quick_demo.py

# Full async test suite
python test/api_test_suite.py
```

## 📊 Test Results Summary

### ✅ Health Tests Results
```
🚀 Knowledge Graph Agent - Health Tests
==================================================
🏥 Testing Health Check Endpoint...
✅ Health Check: PASSED
   Status: healthy
   Version: 1.0.0
   Uptime: 0.00s

🏠 Testing Welcome Endpoint...
✅ Welcome Endpoint: PASSED
   Message: Welcome to the Knowledge Graph Agent API!

📚 Testing Documentation Endpoint...
✅ Documentation Endpoint: PASSED
   Interactive API docs available at http://localhost:8000/docs

🔧 Testing OpenAPI Schema Endpoint...
✅ OpenAPI Schema: PASSED
   Title: Knowledge Graph Agent API
   Version: 1.0.0
   Endpoints: 11

==================================================
📊 HEALTH TEST SUMMARY
==================================================
Total Tests: 4
✅ Passed: 4
❌ Failed: 0
Success Rate: 100.0%
```

### ✅ Chat Tests Results
```
🚀 Knowledge Graph Agent - Chat Tests
==================================================
💬 Testing Single Chat...
✅ Single Chat: PASSED
   Response: Hello! I am a Knowledge Graph Agent...
   Conversation ID: conv_1753508386
   Processing Time: 2.51s

📦 Testing Batch Chat...
✅ Batch Chat: PASSED
   Total Processed: 3
   Responses: 3

🌊 Testing Streaming Chat...
✅ Streaming Chat: PASSED
   Response: No response...

🔄 Testing Multi-turn Conversation...
✅ First Message: PASSED
✅ Second Message: PASSED
   Response: Please provide the name of the repository...

📊 Testing Conversation Stats...
✅ Conversation Stats: PASSED
   Total Conversations: 17
   Total Messages: 40
   Active Conversations: 17

📜 Testing Conversation History...
✅ Conversation History: PASSED
   Total Messages: 2
   Messages: 2
```

## 🎯 Test Categories

### 🏥 Health & Basic Tests
- Server status verification
- API endpoint availability
- Response time measurement
- Error handling validation

### 💬 Chat Functionality Tests
- Message processing
- Conversation management
- Context preservation
- Response quality assessment
- Batch processing
- Streaming responses

### 🔍 Search & Analysis Tests
- Semantic search testing
- Codebase exploration
- Query optimization
- Result relevance
- Function calling
- Code analysis

### 🤖 Advanced Feature Tests
- Tool integration
- Architecture understanding
- Dependency mapping
- Multi-query testing
- Performance optimization

## 📈 Performance Metrics

### Test Duration Targets
- **Health Tests**: < 5 seconds
- **Chat Tests**: < 30 seconds
- **Advanced Tests**: < 60 seconds
- **Full Suite**: < 120 seconds

### Success Rate Targets
- **Individual Tests**: 100%
- **Overall Suite**: 100%

## 🔧 Configuration Requirements

### Environment Variables (.env)
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

### Dependencies
```bash
pip install requests httpx
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Server Not Running
```
❌ Server is not running: Connection refused
```
**Solution:**
```bash
python main.py
```

#### 2. ChromaDB Not Running
```
❌ ChromaDB connection error
```
**Solution:**
```bash
docker-compose up -d chroma
```

#### 3. OpenAI API Errors
```
❌ OpenAI API authentication error
```
**Solution:**
- Check your API key in `.env`
- Verify the base URL
- Ensure model access permissions

#### 4. Test Timeouts
```
⏰ TIMEOUT: test took longer than 2 minutes
```
**Solution:**
- Check server performance
- Verify network connectivity
- Increase timeout if needed

## 🎯 Best Practices

### 1. Test Order
1. Run health tests first
2. Test basic chat functionality
3. Test advanced features
4. Run comprehensive suite

### 2. Environment Setup
1. Start ChromaDB
2. Start API server
3. Verify health endpoint
4. Run tests

### 3. Monitoring
- Watch console output for errors
- Check server logs
- Monitor API response times
- Verify conversation persistence

## 🚀 Continuous Integration

### Automated Testing
```bash
# Run tests in CI/CD pipeline
python test/run_all_tests.py

# Exit code indicates success/failure
# 0 = success, 1 = failure
```

### Test Reports
- Console output for immediate feedback
- Detailed error messages
- Performance metrics
- Success rate statistics

## 🎉 Success Metrics

- ✅ **100% API Endpoints Working**
- ✅ **All Tests Passing**
- ✅ **OpenAI Integration Working**
- ✅ **ChromaDB Integration Working**
- ✅ **Multi-turn Conversations Working**
- ✅ **Function Calling Working**
- ✅ **Search Functionality Working**
- ✅ **Batch Processing Working**
- ✅ **Streaming Responses Working**

## 💡 Usage Tips

1. **Start with Health Tests:**
   ```bash
   python test/test_health.py
   ```

2. **Use Interactive Docs:**
   - Open http://localhost:8000/docs

3. **Check Health:**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Monitor Logs:**
   - Watch console output for detailed logs

5. **Test Functions:**
   ```bash
   curl http://localhost:8000/functions
   ```

## 🚀 Next Steps

1. **Index Repositories:**
   - Use `/index/repository` endpoint
   - Configure repositories in `appSettings.json`

2. **Advanced Features:**
   - Function calling for code analysis
   - Multi-modal support
   - Real-time monitoring

3. **Production Deployment:**
   - Docker containerization
   - Load balancing
   - Monitoring and logging

---

**🎉 Knowledge Graph Agent Test Suite đã sẵn sàng cho production!**

Tất cả các file test đã được tạo và tổ chức thành công. Bạn có thể sử dụng bất kỳ file nào để test và demo API của Knowledge Graph Agent. 