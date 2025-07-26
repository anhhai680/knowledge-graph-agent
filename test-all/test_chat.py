#!/usr/bin/env python3
"""
Chat Functionality Test for Knowledge Graph Agent.

Test script to verify chat endpoints and conversation management.
"""

import requests
import json
import time
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_single_chat():
    """Test single chat endpoint."""
    print("💬 Testing Single Chat...")
    
    try:
        payload = {
            "message": "Hello! What can you do? Explain your capabilities.",
            "user_id": "test_user_001"
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=payload, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single Chat: PASSED")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
            print(f"   Conversation ID: {result.get('conversation_id', 'None')}")
            print(f"   Processing Time: {result.get('processing_time', 0):.2f}s")
            return result.get("conversation_id")
        else:
            print(f"❌ Single Chat: FAILED (Status: {response.status_code})")
            print(f"   Error: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Single Chat: FAILED (Connection Error)")
        return None
    except Exception as e:
        print(f"❌ Single Chat: FAILED (Error: {e})")
        return None

def test_batch_chat():
    """Test batch chat endpoint."""
    print("\n📦 Testing Batch Chat...")
    
    try:
        payload = [
            {"message": "What is this project about?", "user_id": "user1"},
            {"message": "Explain the architecture", "user_id": "user2"},
            {"message": "What are the main features?", "user_id": "user3"}
        ]
        
        response = requests.post(f"{BASE_URL}/chat/batch", json=payload, headers=HEADERS, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch Chat: PASSED")
            print(f"   Total Processed: {result.get('total_processed', 0)}")
            print(f"   Responses: {len(result.get('responses', []))}")
            return True
        else:
            print(f"❌ Batch Chat: FAILED (Status: {response.status_code})")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Batch Chat: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Batch Chat: FAILED (Error: {e})")
        return False

def test_streaming_chat():
    """Test streaming chat endpoint."""
    print("\n🌊 Testing Streaming Chat...")
    
    try:
        payload = {
            "message": "Hello, test streaming response",
            "user_id": "stream_test_user"
        }
        
        response = requests.post(f"{BASE_URL}/chat/stream", json=payload, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Streaming Chat: PASSED")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Streaming Chat: FAILED (Status: {response.status_code})")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Streaming Chat: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Streaming Chat: FAILED (Error: {e})")
        return False

def test_multi_turn_conversation():
    """Test multi-turn conversation."""
    print("\n🔄 Testing Multi-turn Conversation...")
    
    try:
        # First message
        payload1 = {
            "message": "Hello, I want to understand this codebase",
            "user_id": "multi_turn_user"
        }
        
        response1 = requests.post(f"{BASE_URL}/chat", json=payload1, headers=HEADERS, timeout=30)
        
        if response1.status_code == 200:
            result1 = response1.json()
            conversation_id = result1.get("conversation_id")
            print("✅ First Message: PASSED")
            
            # Second message (follow-up)
            payload2 = {
                "message": "Can you explain the main components?",
                "user_id": "multi_turn_user",
                "conversation_id": conversation_id
            }
            
            response2 = requests.post(f"{BASE_URL}/chat", json=payload2, headers=HEADERS, timeout=30)
            
            if response2.status_code == 200:
                result2 = response2.json()
                print("✅ Second Message: PASSED")
                print(f"   Response: {result2.get('response', 'No response')[:100]}...")
                return True
            else:
                print(f"❌ Second Message: FAILED (Status: {response2.status_code})")
                return False
        else:
            print(f"❌ First Message: FAILED (Status: {response1.status_code})")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Multi-turn Conversation: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Multi-turn Conversation: FAILED (Error: {e})")
        return False

def test_conversation_stats():
    """Test conversation statistics."""
    print("\n📊 Testing Conversation Stats...")
    
    try:
        response = requests.get(f"{BASE_URL}/conversations/stats", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Conversation Stats: PASSED")
            print(f"   Total Conversations: {result.get('total_conversations', 0)}")
            print(f"   Total Messages: {result.get('total_messages', 0)}")
            print(f"   Active Conversations: {result.get('active_conversations', 0)}")
            return True
        else:
            print(f"❌ Conversation Stats: FAILED (Status: {response.status_code})")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Conversation Stats: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Conversation Stats: FAILED (Error: {e})")
        return False

def test_conversation_history(conversation_id: str):
    """Test conversation history."""
    print(f"\n📜 Testing Conversation History (ID: {conversation_id})...")
    
    try:
        response = requests.get(f"{BASE_URL}/conversations/{conversation_id}/history", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Conversation History: PASSED")
            print(f"   Total Messages: {result.get('total_messages', 0)}")
            print(f"   Messages: {len(result.get('messages', []))}")
            return True
        else:
            print(f"❌ Conversation History: FAILED (Status: {response.status_code})")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Conversation History: FAILED (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ Conversation History: FAILED (Error: {e})")
        return False

def run_chat_tests():
    """Run all chat-related tests."""
    print("🚀 Knowledge Graph Agent - Chat Tests")
    print("=" * 50)
    
    # Run tests
    conversation_id = test_single_chat()
    test_batch_chat()
    test_streaming_chat()
    test_multi_turn_conversation()
    test_conversation_stats()
    
    if conversation_id:
        test_conversation_history(conversation_id)
    
    print("\n" + "=" * 50)
    print("💬 Chat Tests Complete!")
    print("=" * 50)
    print("💡 Tips:")
    print("   - Check conversation stats for usage metrics")
    print("   - Use conversation_id for multi-turn chats")
    print("   - Batch processing for multiple messages")

if __name__ == "__main__":
    run_chat_tests() 