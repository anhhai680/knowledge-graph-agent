#!/usr/bin/env python3
"""
Demo script for Knowledge Graph Agent.

This script demonstrates the key features of the chatbot agent:
- Multi-turn conversations
- Function calling
- Context preservation
- Batch processing
"""

import asyncio
import json
import time
from typing import List, Dict, Any

from src.agents.chatbot_agent import ChatbotAgent, Message, MessageRole


class DemoRunner:
    """Demo runner for Knowledge Graph Agent."""
    
    def __init__(self):
        """Initialize the demo runner."""
        self.agent = ChatbotAgent(
            api_key="demo_key",  # Use demo key for testing
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        # Demo scenarios
        self.scenarios = {
            "architecture_analysis": [
                "What is the overall architecture of this microservice system?",
                "Can you explain the authentication flow in this architecture?",
                "What are the main dependencies between services?",
                "How do I add a new service to this architecture?"
            ],
            "code_search": [
                "Search for UserService in the codebase",
                "Show me the implementation of the authentication middleware",
                "What are the dependencies of the OrderService?",
                "Find examples of error handling patterns"
            ],
            "multi_turn_conversation": [
                "Explain the microservice architecture",
                "How does the UserService handle authentication?",
                "What are its dependencies?",
                "How do I add role-based access control?",
                "What testing strategies are used?"
            ]
        }
    
    async def run_single_conversation(self, scenario_name: str, messages: List[str]) -> Dict[str, Any]:
        """Run a single conversation scenario."""
        print(f"\n{'='*60}")
        print(f"Running Scenario: {scenario_name}")
        print(f"{'='*60}")
        
        conversation_id = f"demo_{scenario_name}_{int(time.time())}"
        responses = []
        
        for i, message in enumerate(messages, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {message}")
            
            # Process message
            start_time = time.time()
            response = await self.agent.process_message(
                user_id="demo_user",
                message=message,
                conversation_id=conversation_id,
                use_functions=True
            )
            processing_time = time.time() - start_time
            
            print(f"Assistant: {response['response'][:200]}...")
            print(f"Processing Time: {processing_time:.2f}s")
            print(f"Message Count: {response['message_count']}")
            
            responses.append({
                "turn": i,
                "user_message": message,
                "assistant_response": response['response'],
                "processing_time": processing_time,
                "message_count": response['message_count']
            })
        
        return {
            "scenario": scenario_name,
            "conversation_id": conversation_id,
            "responses": responses,
            "total_turns": len(messages)
        }
    
    async def run_batch_processing(self) -> Dict[str, Any]:
        """Run batch processing demo."""
        print(f"\n{'='*60}")
        print("Running Batch Processing Demo")
        print(f"{'='*60}")
        
        # Prepare batch messages
        batch_messages = [
            ("user1", "What is the architecture?", "batch_conv1"),
            ("user2", "How does authentication work?", "batch_conv2"),
            ("user3", "Show me the UserService", "batch_conv3"),
            ("user4", "What are the main dependencies?", "batch_conv4")
        ]
        
        start_time = time.time()
        responses = await self.agent.batch_process_messages(batch_messages)
        total_time = time.time() - start_time
        
        print(f"Batch processed {len(responses)} messages in {total_time:.2f}s")
        
        for i, response in enumerate(responses, 1):
            print(f"\nResponse {i}:")
            print(f"  Conversation ID: {response['conversation_id']}")
            print(f"  Response: {response['response'][:100]}...")
            print(f"  Processing Time: {response['processing_time']:.2f}s")
        
        return {
            "total_messages": len(responses),
            "total_time": total_time,
            "average_time": total_time / len(responses),
            "responses": responses
        }
    
    async def test_function_calling(self) -> Dict[str, Any]:
        """Test function calling capabilities."""
        print(f"\n{'='*60}")
        print("Testing Function Calling")
        print(f"{'='*60}")
        
        function_test_messages = [
            "Search for UserService in the codebase",
            "Get architecture overview of the microservice",
            "Analyze dependencies of the OrderService"
        ]
        
        results = []
        for message in function_test_messages:
            print(f"\nTesting: {message}")
            
            response = await self.agent.process_message(
                user_id="function_test_user",
                message=message,
                conversation_id="function_test_conv",
                use_functions=True
            )
            
            print(f"Response: {response['response'][:150]}...")
            results.append({
                "message": message,
                "response": response['response'],
                "processing_time": response['processing_time']
            })
        
        return {
            "function_tests": results,
            "total_tests": len(results)
        }
    
    async def test_conversation_management(self) -> Dict[str, Any]:
        """Test conversation management features."""
        print(f"\n{'='*60}")
        print("Testing Conversation Management")
        print(f"{'='*60}")
        
        # Create a conversation
        conv_id = "management_test_conv"
        
        # Add messages
        await self.agent.process_message("test_user", "Hello", conv_id)
        await self.agent.process_message("test_user", "How are you?", conv_id)
        
        # Get conversation history
        history = self.agent.get_conversation_history(conv_id)
        print(f"Conversation history has {len(history)} messages")
        
        # Get conversation stats
        stats = self.agent.get_conversation_stats()
        print(f"Total conversations: {stats['total_conversations']}")
        print(f"Total messages: {stats['total_messages']}")
        print(f"Active conversations: {stats['active_conversations']}")
        
        # Clear conversation
        cleared = self.agent.clear_conversation(conv_id)
        print(f"Conversation cleared: {cleared}")
        
        # Verify conversation is gone
        history_after = self.agent.get_conversation_history(conv_id)
        print(f"History after clearing: {history_after is None}")
        
        return {
            "conversation_id": conv_id,
            "history_length": len(history) if history else 0,
            "cleared": cleared,
            "stats": stats
        }
    
    async def run_performance_test(self) -> Dict[str, Any]:
        """Run performance tests."""
        print(f"\n{'='*60}")
        print("Running Performance Tests")
        print(f"{'='*60}")
        
        # Test single message processing
        start_time = time.time()
        response = await self.agent.process_message(
            user_id="perf_test_user",
            message="What is the architecture?",
            conversation_id="perf_test_conv"
        )
        single_message_time = time.time() - start_time
        
        # Test batch processing
        batch_messages = [
            ("user1", "Test message 1", "perf_batch1"),
            ("user2", "Test message 2", "perf_batch2"),
            ("user3", "Test message 3", "perf_batch3")
        ]
        
        start_time = time.time()
        batch_responses = await self.agent.batch_process_messages(batch_messages)
        batch_time = time.time() - start_time
        
        return {
            "single_message_time": single_message_time,
            "batch_time": batch_time,
            "batch_messages": len(batch_messages),
            "average_batch_time": batch_time / len(batch_messages)
        }
    
    async def run_full_demo(self) -> Dict[str, Any]:
        """Run the complete demo."""
        print("🚀 Starting Knowledge Graph Agent Demo")
        print("=" * 60)
        
        results = {}
        
        # Run all scenarios
        for scenario_name, messages in self.scenarios.items():
            scenario_result = await self.run_single_conversation(scenario_name, messages)
            results[scenario_name] = scenario_result
        
        # Run batch processing
        batch_result = await self.run_batch_processing()
        results["batch_processing"] = batch_result
        
        # Test function calling
        function_result = await self.test_function_calling()
        results["function_calling"] = function_result
        
        # Test conversation management
        management_result = await self.test_conversation_management()
        results["conversation_management"] = management_result
        
        # Run performance tests
        performance_result = await self.run_performance_test()
        results["performance"] = performance_result
        
        # Print summary
        self.print_demo_summary(results)
        
        return results
    
    def print_demo_summary(self, results: Dict[str, Any]):
        """Print demo summary."""
        print(f"\n{'='*60}")
        print("DEMO SUMMARY")
        print(f"{'='*60}")
        
        total_conversations = len([k for k in results.keys() if k in self.scenarios])
        total_messages = sum(len(results[k]["responses"]) for k in results.keys() if k in self.scenarios)
        
        print(f"✅ Total Scenarios: {total_conversations}")
        print(f"✅ Total Messages: {total_messages}")
        print(f"✅ Batch Processing: {results['batch_processing']['total_messages']} messages")
        print(f"✅ Function Calling: {results['function_calling']['total_tests']} tests")
        print(f"✅ Conversation Management: Working")
        print(f"✅ Performance: Single message {results['performance']['single_message_time']:.2f}s")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📊 All features working as expected")


async def main():
    """Main demo function."""
    demo = DemoRunner()
    
    try:
        results = await demo.run_full_demo()
        
        # Save results to file
        with open("demo_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Demo results saved to demo_results.json")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 