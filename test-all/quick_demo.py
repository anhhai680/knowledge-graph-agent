#!/usr/bin/env python3
"""
Knowledge Graph Agent - Quick Demo

File demo nhanh để test các API chính của Knowledge Graph Agent.
Sử dụng để demo nhanh và kiểm tra tính năng.
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_health():
    """Test health check."""
    print("🏥 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health Check: OK")
            print(f"   Status: {response.json()}")
        else:
            print("❌ Health Check: Failed")
    except Exception as e:
        print(f"❌ Health Check Error: {e}")

def test_single_chat():
    """Test single chat."""
    print("\n💬 Testing Single Chat...")
    try:
        payload = {
            "message": "Hello! What can you do? Explain your capabilities.",
            "user_id": "demo_user"
        }
        response = requests.post(f"{BASE_URL}/chat", json=payload, headers=HEADERS)
        if response.status_code == 200:
            result = response.json()
            print("✅ Single Chat: OK")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
            return result.get("conversation_id")
        else:
            print("❌ Single Chat: Failed")
    except Exception as e:
        print(f"❌ Single Chat Error: {e}")
    return None

def test_batch_chat():
    """Test batch chat."""
    print("\n📦 Testing Batch Chat...")
    try:
        payload = [
            {"message": "What is this project about?", "user_id": "user1"},
            {"message": "Explain the architecture", "user_id": "user2"},
            {"message": "What are the main features?", "user_id": "user3"}
        ]
        response = requests.post(f"{BASE_URL}/chat/batch", json=payload, headers=HEADERS)
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch Chat: OK")
            print(f"   Processed: {result.get('total_processed', 0)} messages")
        else:
            print("❌ Batch Chat: Failed")
    except Exception as e:
        print(f"❌ Batch Chat Error: {e}")

def test_conversation_stats():
    """Test conversation stats."""
    print("\n📊 Testing Conversation Stats...")
    try:
        response = requests.get(f"{BASE_URL}/conversations/stats")
        if response.status_code == 200:
            result = response.json()
            print("✅ Conversation Stats: OK")
            print(f"   Total Conversations: {result.get('total_conversations', 0)}")
            print(f"   Total Messages: {result.get('total_messages', 0)}")
            print(f"   Active Conversations: {result.get('active_conversations', 0)}")
        else:
            print("❌ Conversation Stats: Failed")
    except Exception as e:
        print(f"❌ Conversation Stats Error: {e}")

def test_functions():
    """Test functions endpoint."""
    print("\n🔧 Testing Functions API...")
    try:
        response = requests.get(f"{BASE_URL}/functions")
        if response.status_code == 200:
            result = response.json()
            print("✅ Functions API: OK")
            print(f"   Available Functions: {result.get('total_functions', 0)}")
            for func in result.get('functions', []):
                print(f"   - {func.get('name', 'Unknown')}: {func.get('description', 'No description')}")
        else:
            print("❌ Functions API: Failed")
    except Exception as e:
        print(f"❌ Functions API Error: {e}")

def test_search():
    """Test search endpoint."""
    print("\n🔍 Testing Search API...")
    try:
        params = {"query": "chatbot agent", "limit": 5}
        response = requests.get(f"{BASE_URL}/search", params=params)
        if response.status_code == 200:
            result = response.json()
            print("✅ Search API: OK")
            print(f"   Query: {result.get('query', 'Unknown')}")
            print(f"   Total Results: {result.get('total_results', 0)}")
        else:
            print("❌ Search API: Failed")
    except Exception as e:
        print(f"❌ Search API Error: {e}")

def test_multi_turn_conversation():
    """Test multi-turn conversation."""
    print("\n🔄 Testing Multi-turn Conversation...")
    try:
        # First message
        payload1 = {
            "message": "Hello, I want to understand this codebase",
            "user_id": "multi_turn_user"
        }
        response1 = requests.post(f"{BASE_URL}/chat", json=payload1, headers=HEADERS)
        if response1.status_code == 200:
            result1 = response1.json()
            conversation_id = result1.get("conversation_id")
            print("✅ First Message: OK")
            
            # Second message (follow-up)
            payload2 = {
                "message": "Can you explain the main components?",
                "user_id": "multi_turn_user",
                "conversation_id": conversation_id
            }
            response2 = requests.post(f"{BASE_URL}/chat", json=payload2, headers=HEADERS)
            if response2.status_code == 200:
                result2 = response2.json()
                print("✅ Second Message: OK")
                print(f"   Response: {result2.get('response', 'No response')[:100]}...")
            else:
                print("❌ Second Message: Failed")
        else:
            print("❌ First Message: Failed")
    except Exception as e:
        print(f"❌ Multi-turn Conversation Error: {e}")

def demo_curl_commands():
    """Show curl commands for manual testing."""
    print("\n" + "=" * 60)
    print("📋 CURL COMMANDS FOR MANUAL TESTING")
    print("=" * 60)
    
    commands = [
        {
            "name": "Health Check",
            "command": f"curl -s {BASE_URL}/health"
        },
        {
            "name": "Welcome Message", 
            "command": f"curl -s {BASE_URL}/"
        },
        {
            "name": "Single Chat",
            "command": f"""curl -s -X POST {BASE_URL}/chat \\
  -H "Content-Type: application/json" \\
  -d '{{"message": "Hello, what can you do?", "user_id": "test_user"}}'"""
        },
        {
            "name": "Batch Chat",
            "command": f"""curl -s -X POST {BASE_URL}/chat/batch \\
  -H "Content-Type: application/json" \\
  -d '[{{"message": "Question 1", "user_id": "user1"}}, {{"message": "Question 2", "user_id": "user2"}}]'"""
        },
        {
            "name": "Conversation Stats",
            "command": f"curl -s {BASE_URL}/conversations/stats"
        },
        {
            "name": "Functions API",
            "command": f"curl -s {BASE_URL}/functions"
        },
        {
            "name": "Search API",
            "command": f"curl -s '{BASE_URL}/search?query=chatbot&limit=5'"
        }
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"\n{i}. {cmd['name']}")
        print(f"   {cmd['command']}")

def main():
    """Main demo function."""
    print("🚀 Knowledge Graph Agent - Quick Demo")
    print("=" * 50)
    
    # Run all tests
    test_health()
    conversation_id = test_single_chat()
    test_batch_chat()
    test_conversation_stats()
    test_functions()
    test_search()
    test_multi_turn_conversation()
    
    # Show curl commands
    demo_curl_commands()
    
    print("\n" + "=" * 50)
    print("🎉 Demo Complete!")
    print("=" * 50)
    print("💡 Tips:")
    print("   - Open http://localhost:8000/docs for interactive API docs")
    print("   - Use the curl commands above for manual testing")
    print("   - Check logs for detailed information")

if __name__ == "__main__":
    main() 