import asyncio
from src.agents.chatbot_agent import ChatbotAgent

async def test_agent():
    agent = ChatbotAgent()
    print('Testing ChatbotAgent...')
    response = await agent.chat_completion([{"role": "user", "content": "Hello"}])
    print('Response:', response)

if __name__ == "__main__":
    asyncio.run(test_agent()) 