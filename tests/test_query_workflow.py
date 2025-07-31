"""
Test for query workflow implementation.
"""

import asyncio
from src.workflows.query_workflow import QueryWorkflow, execute_query


async def test_basic_workflow():
    """Test basic query workflow creation."""
    workflow = QueryWorkflow()
    print("Query workflow created successfully")
    
    # Test simple query
    result = await execute_query("What is this project about?")
    print(f"Query result: {result}")

    assert isinstance(result, dict)
    assert "response" in result
    assert "sources" in result
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0


if __name__ == "__main__":
    asyncio.run(test_basic_workflow())
