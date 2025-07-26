#!/usr/bin/env python3
"""
Knowledge Graph Agent - API Test Suite

File này tổng hợp và test tất cả các API endpoints của Knowledge Graph Agent.
Sử dụng để demo và kiểm tra tính năng của hệ thống.
"""

import asyncio
import json
import time
from typing import Dict, List, Any
import httpx
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

class KnowledgeGraphAgentTester:
    """Test suite cho Knowledge Graph Agent API."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def log_test(self, test_name: str, success: bool, response: Any, duration: float):
        """Log kết quả test."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({duration:.2f}s)")
        if not success:
            print(f"   Error: {response}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "response": response,
            "duration": duration
        })
    
    async def test_health_check(self):
        """Test health check endpoint."""
        start_time = time.time()
        try:
            response = await self.client.get(f"{self.base_url}/health")
            success = response.status_code == 200
            await self.log_test("Health Check", success, response.json(), time.time() - start_time)
        except Exception as e:
            await self.log_test("Health Check", False, str(e), time.time() - start_time)
    
    async def test_welcome_message(self):
        """Test welcome message endpoint."""
        start_time = time.time()
        try:
            response = await self.client.get(f"{self.base_url}/")
            success = response.status_code == 200
            await self.log_test("Welcome Message", success, response.json(), time.time() - start_time)
        except Exception as e:
            await self.log_test("Welcome Message", False, str(e), time.time() - start_time)
    
    async def test_single_chat(self):
        """Test single chat endpoint."""
        start_time = time.time()
        try:
            payload = {
                "message": "Hello! What can you do? Explain your capabilities.",
                "user_id": "test_user_001"
            }
            response = await self.client.post(f"{self.base_url}/chat", json=payload, headers=HEADERS)
            success = response.status_code == 200
            result = response.json()
            await self.log_test("Single Chat", success, result.get("response", "No response"), time.time() - start_time)
            return result.get("conversation_id")
        except Exception as e:
            await self.log_test("Single Chat", False, str(e), time.time() - start_time)
            return None
    
    async def test_batch_chat(self):
        """Test batch chat endpoint."""
        start_time = time.time()
        try:
            payload = [
                {"message": "What is this project about?", "user_id": "user1"},
                {"message": "Explain the architecture", "user_id": "user2"},
                {"message": "What are the main features?", "user_id": "user3"}
            ]
            response = await self.client.post(f"{self.base_url}/chat/batch", json=payload, headers=HEADERS)
            success = response.status_code == 200
            result = response.json()
            await self.log_test("Batch Chat", success, f"Processed {result.get('total_processed', 0)} messages", time.time() - start_time)
        except Exception as e:
            await self.log_test("Batch Chat", False, str(e), time.time() - start_time)
    
    async def test_streaming_chat(self):
        """Test streaming chat endpoint."""
        start_time = time.time()
        try:
            payload = {
                "message": "Hello, test streaming response",
                "user_id": "stream_test_user"
            }
            response = await self.client.post(f"{self.base_url}/chat/stream", json=payload, headers=HEADERS)
            success = response.status_code == 200
            result = response.json()
            await self.log_test("Streaming Chat", success, result.get("response", "No response"), time.time() - start_time)
        except Exception as e:
            await self.log_test("Streaming Chat", False, str(e), time.time() - start_time)
    
    async def test_conversation_stats(self):
        """Test conversation statistics endpoint."""
        start_time = time.time()
        try:
            response = await self.client.get(f"{self.base_url}/conversations/stats")
            success = response.status_code == 200
            result = response.json()
            await self.log_test("Conversation Stats", success, result, time.time() - start_time)
        except Exception as e:
            await self.log_test("Conversation Stats", False, str(e), time.time() - start_time)
    
    async def test_conversation_history(self, conversation_id: str):
        """Test conversation history endpoint."""
        start_time = time.time()
        try:
            response = await self.client.get(f"{self.base_url}/conversations/{conversation_id}/history")
            success = response.status_code == 200
            result = response.json()
            await self.log_test("Conversation History", success, f"Found {result.get('total_messages', 0)} messages", time.time() - start_time)
        except Exception as e:
            await self.log_test("Conversation History", False, str(e), time.time() - start_time)
    
    async def test_functions_endpoint(self):
        """Test functions endpoint."""
        start_time = time.time()
        try:
            response = await self.client.get(f"{self.base_url}/functions")
            success = response.status_code == 200
            result = response.json()
            await self.log_test("Functions API", success, f"Available: {result.get('total_functions', 0)} functions", time.time() - start_time)
        except Exception as e:
            await self.log_test("Functions API", False, str(e), time.time() - start_time)
    
    async def test_search_endpoint(self):
        """Test search endpoint."""
        start_time = time.time()
        try:
            params = {"query": "chatbot agent", "limit": 5}
            response = await self.client.get(f"{self.base_url}/search", params=params)
            success = response.status_code == 200
            result = response.json()
            await self.log_test("Search API", success, f"Found {result.get('total_results', 0)} results", time.time() - start_time)
        except Exception as e:
            await self.log_test("Search API", False, str(e), time.time() - start_time)
    
    async def test_advanced_chat_with_functions(self):
        """Test advanced chat with function calling."""
        start_time = time.time()
        try:
            payload = {
                "message": "Search for chatbot related code in the codebase",
                "user_id": "function_test_user"
            }
            response = await self.client.post(f"{self.base_url}/chat", json=payload, headers=HEADERS)
            success = response.status_code == 200
            result = response.json()
            await self.log_test("Advanced Chat (Functions)", success, result.get("response", "No response"), time.time() - start_time)
        except Exception as e:
            await self.log_test("Advanced Chat (Functions)", False, str(e), time.time() - start_time)
    
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        start_time = time.time()
        try:
            # First message
            payload1 = {
                "message": "Hello, I want to understand this codebase",
                "user_id": "multi_turn_user"
            }
            response1 = await self.client.post(f"{self.base_url}/chat", json=payload1, headers=HEADERS)
            result1 = response1.json()
            conversation_id = result1.get("conversation_id")
            
            # Second message (follow-up)
            payload2 = {
                "message": "Can you explain the main components?",
                "user_id": "multi_turn_user",
                "conversation_id": conversation_id
            }
            response2 = await self.client.post(f"{self.base_url}/chat", json=payload2, headers=HEADERS)
            success = response2.status_code == 200
            result2 = response2.json()
            
            await self.log_test("Multi-turn Conversation", success, result2.get("response", "No response"), time.time() - start_time)
        except Exception as e:
            await self.log_test("Multi-turn Conversation", False, str(e), time.time() - start_time)
    
    async def run_all_tests(self):
        """Chạy tất cả các test."""
        print("🚀 Knowledge Graph Agent - API Test Suite")
        print("=" * 50)
        
        # Basic endpoints
        await self.test_health_check()
        await self.test_welcome_message()
        
        # Chat endpoints
        conversation_id = await self.test_single_chat()
        await self.test_batch_chat()
        await self.test_streaming_chat()
        
        # Conversation management
        await self.test_conversation_stats()
        if conversation_id:
            await self.test_conversation_history(conversation_id)
        
        # Advanced features
        await self.test_functions_endpoint()
        await self.test_search_endpoint()
        await self.test_advanced_chat_with_functions()
        await self.test_multi_turn_conversation()
        
        # Summary
        await self.print_summary()
    
    async def print_summary(self):
        """In kết quả tổng hợp."""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['response']}")
        
        print("\n🎉 Knowledge Graph Agent API Test Complete!")


async def demo_usage_examples():
    """Demo các ví dụ sử dụng API."""
    print("\n" + "=" * 50)
    print("📚 USAGE EXAMPLES")
    print("=" * 50)
    
    examples = [
        {
            "name": "Single Chat",
            "method": "POST",
            "endpoint": "/chat",
            "payload": {
                "message": "Hello, what can you do?",
                "user_id": "user123"
            }
        },
        {
            "name": "Batch Chat",
            "method": "POST", 
            "endpoint": "/chat/batch",
            "payload": [
                {"message": "Question 1", "user_id": "user1"},
                {"message": "Question 2", "user_id": "user2"}
            ]
        },
        {
            "name": "Streaming Chat",
            "method": "POST",
            "endpoint": "/chat/stream", 
            "payload": {
                "message": "Stream this response",
                "user_id": "stream_user"
            }
        },
        {
            "name": "Search Codebase",
            "method": "GET",
            "endpoint": "/search?query=function&limit=5",
            "payload": None
        },
        {
            "name": "Get Functions",
            "method": "GET",
            "endpoint": "/functions",
            "payload": None
        },
        {
            "name": "Conversation Stats",
            "method": "GET", 
            "endpoint": "/conversations/stats",
            "payload": None
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   Method: {example['method']}")
        print(f"   Endpoint: {example['endpoint']}")
        if example['payload']:
            print(f"   Payload: {json.dumps(example['payload'], indent=2)}")
        print()


async def main():
    """Main function."""
    print("🔧 Starting Knowledge Graph Agent API Test Suite...")
    
    # Demo usage examples
    await demo_usage_examples()
    
    # Run all tests
    async with KnowledgeGraphAgentTester() as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 