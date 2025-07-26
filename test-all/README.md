# 🧪 Test Suite for Knowledge Graph Agent

This directory contains comprehensive test suites for the Knowledge Graph Agent API.

## 📁 File Structure

```
test/
├── __init__.py              # Package initialization
├── README.md               # This file
├── run_all_tests.py        # Run all tests
├── test_health.py          # Health check tests
├── test_chat.py            # Chat functionality tests
├── test_advanced.py        # Advanced features tests
├── api_test_suite.py       # Full async test suite
├── quick_demo.py           # Quick demo script
└── test_agent.py           # Direct agent testing
```

## 🚀 Quick Start

### 1. Start the Server
```bash
# Start ChromaDB
docker-compose up -d chroma

# Start the API server
python main.py
```

### 2. Run All Tests
```bash
python test/run_all_tests.py
```

### 3. Run Individual Tests
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

## 📋 Test Categories

### 🏥 Health Tests (`test_health.py`)
- Health check endpoint
- Welcome message endpoint
- Documentation endpoint
- OpenAPI schema endpoint

### 💬 Chat Tests (`test_chat.py`)
- Single chat messages
- Batch chat processing
- Streaming chat responses
- Multi-turn conversations
- Conversation statistics
- Conversation history

### 🔧 Advanced Tests (`test_advanced.py`)
- Functions API
- Search functionality
- Code analysis
- Architecture overview
- Dependency analysis
- Multiple search queries

### 🚀 Quick Demo (`quick_demo.py`)
- Quick overview of all features
- Simple sync testing
- Curl command examples

### ⚡ Async Test Suite (`api_test_suite.py`)
- Comprehensive async testing
- Performance metrics
- Detailed error reporting

## 🎯 Test Features

### ✅ Health Monitoring
- Server status verification
- API endpoint availability
- Response time measurement
- Error handling validation

### 💬 Chat Functionality
- Message processing
- Conversation management
- Context preservation
- Response quality assessment

### 🔍 Search Capabilities
- Semantic search testing
- Codebase exploration
- Query optimization
- Result relevance

### 🤖 Function Calling
- Tool integration
- Code analysis
- Architecture understanding
- Dependency mapping

## 📊 Test Results

### Expected Output
```
🚀 Knowledge Graph Agent - Comprehensive Test Suite
================================================================================
🏥 Checking if server is running...
✅ Server is running!

📋 Found 6 test files:
   - test_advanced.py
   - test_chat.py
   - test_health.py
   - api_test_suite.py
   - quick_demo.py
   - test_agent.py

================================================================================
🧪 Running: test/test_health.py
================================================================================
🚀 Knowledge Graph Agent - Health Tests
==================================================
🏥 Testing Health Check Endpoint...
✅ Health Check: PASSED
   Status: healthy
   Version: 1.0.0
   Uptime: 0.00s

🏠 Testing Welcome Endpoint...
✅ Welcome Endpoint: PASSED
   Message: Welcome to Knowledge Graph Agent API

📚 Testing Documentation Endpoint...
✅ Documentation Endpoint: PASSED
   Interactive API docs available at http://localhost:8000/docs

🔧 Testing OpenAPI Schema Endpoint...
✅ OpenAPI Schema: PASSED
   Title: Knowledge Graph Agent API
   Version: 1.0.0
   Endpoints: 9

==================================================
📊 HEALTH TEST SUMMARY
==================================================
Total Tests: 4
✅ Passed: 4
❌ Failed: 0
Success Rate: 100.0%

================================================================================
📊 COMPREHENSIVE TEST SUMMARY
================================================================================
Total Test Files: 6
✅ Passed: 6
❌ Failed: 0
Success Rate: 100.0%
Total Duration: 45.23s

✅ Passed Tests:
  - test/test_advanced.py (Duration: 12.34s)
  - test/test_chat.py (Duration: 8.76s)
  - test/test_health.py (Duration: 2.45s)
  - test/api_test_suite.py (Duration: 15.67s)
  - test/quick_demo.py (Duration: 3.21s)
  - test/test_agent.py (Duration: 2.80s)

💡 Recommendations:
   🎉 All tests passed! The Knowledge Graph Agent is working perfectly.

🎉 All 6 tests passed!
```

## 🔧 Configuration

### Environment Variables
Make sure your `.env` file is properly configured:
```env
OPENAI_API_KEY=your_openai_api_key
LLM_API_BASE_URL=https://your-api-base-url
OPENAI_MODEL=GPT-4o-mini
CHROMA_HOST=localhost
CHROMA_PORT=8001
```

### Dependencies
Install required packages:
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

## 📈 Performance Metrics

### Test Duration Targets
- **Health Tests**: < 5 seconds
- **Chat Tests**: < 30 seconds
- **Advanced Tests**: < 60 seconds
- **Full Suite**: < 120 seconds

### Success Rate Targets
- **Individual Tests**: 100%
- **Overall Suite**: 100%

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

---

**🎉 Happy Testing with Knowledge Graph Agent!**

All tests are designed to ensure the API works correctly and provides reliable service to users. 