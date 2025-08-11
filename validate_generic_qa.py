#!/usr/bin/env python3
"""
Simple validation script for Generic Q&A Agent.

This script demonstrates the basic functionality of the Generic Q&A Agent
by testing question classification and response generation.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.analyzers.question_classifier import QuestionClassifier
from src.agents.generic_qa_agent import GenericQAAgent
from src.templates.template_engine import TemplateEngine


async def test_question_classification():
    """Test question classification functionality."""
    print("🔍 Testing Question Classification...")
    
    classifier = QuestionClassifier()
    
    test_questions = [
        "What is the business domain of this project?",
        "What architecture pattern is used?", 
        "What API endpoints are available?",
        "How is the data modeled?",
        "How is this deployed?"
    ]
    
    for question in test_questions:
        try:
            result = await classifier.classify_question(question)
            print(f"  ✅ '{question}' → {result.category.value} (confidence: {result.confidence:.2f})")
        except Exception as e:
            print(f"  ❌ '{question}' → Error: {e}")
    
    print()


async def test_template_engine():
    """Test template engine functionality."""
    print("📋 Testing Template Engine...")
    
    template_engine = TemplateEngine()
    
    try:
        # Test getting available templates
        templates = template_engine.get_available_templates()
        print(f"  ✅ Available templates: {templates}")
        
        # Test generating a response
        response = template_engine.generate_response(
            category="business_capability",
            question="What is the business domain?",
            analysis_results={"confidence_score": 0.8},
            repository_context={"type": "python"},
            include_code_examples=True
        )
        print(f"  ✅ Generated response type: {response.get('type')}")
        print(f"  ✅ Response sections: {list(response.get('sections', {}).keys())}")
        
    except Exception as e:
        print(f"  ❌ Template engine error: {e}")
    
    print()


async def test_generic_qa_agent():
    """Test Generic Q&A Agent end-to-end functionality."""
    print("🤖 Testing Generic Q&A Agent...")
    
    try:
        agent = GenericQAAgent()
        print(f"  ✅ Agent initialized: {agent.agent_name}")
        
        # Test agent capabilities
        capabilities = agent.get_agent_capabilities()
        print(f"  ✅ Agent type: {capabilities.get('agent_type')}")
        
        # Test question analysis
        test_question = "What is the architecture of this system?"
        classification = await agent.analyze_question_only(test_question)
        print(f"  ✅ Question classified as: {classification.category.value}")
        
        # Test full processing
        result = await agent.ainvoke({
            "question": test_question,
            "repository_identifier": "test-repo",
            "include_code_examples": True
        })
        
        if result.get("success"):
            print(f"  ✅ Full processing successful")
            print(f"  ✅ Confidence score: {result.get('confidence_score', 0):.2f}")
            print(f"  ✅ Template used: {result.get('template_used')}")
        else:
            print(f"  ❌ Processing failed: {result.get('error')}")
            
    except Exception as e:
        print(f"  ❌ Agent error: {e}")
    
    print()


async def main():
    """Run all validation tests."""
    print("🚀 Generic Q&A Agent Validation Script")
    print("=" * 50)
    
    await test_question_classification()
    await test_template_engine()
    await test_generic_qa_agent()
    
    print("✅ Validation completed!")


if __name__ == "__main__":
    asyncio.run(main())