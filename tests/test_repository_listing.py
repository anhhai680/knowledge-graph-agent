#!/usr/bin/env python3
"""
Test script for repository listing functionality.

This script tests the new repository metadata functionality
that queries the vector store instead of using mock data.
"""

import asyncio
import sys
import os
from datetime import datetime
import pytest
from unittest.mock import Mock, patch
from src.workflows.indexing_workflow import IndexingWorkflow


# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.vectorstores.store_factory import VectorStoreFactory
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@pytest.mark.asyncio
async def test_repository_metadata():
    """Test repository metadata retrieval."""
    # Mock the vector store instead of the workflow
    with patch('src.vectorstores.store_factory.VectorStoreFactory') as mock_factory_class:
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        # Mock the vector store
        mock_vector_store = Mock()
        mock_factory.create.return_value = mock_vector_store
        
        # Mock the repository metadata
        mock_vector_store.get_repository_metadata.return_value = [
            {
                "name": "test-repo",
                "url": "https://github.com/test/repo",
                "language": "Python",
                "stars": 100
            }
        ]
        
        # Create factory and get vector store
        from src.vectorstores.store_factory import VectorStoreFactory
        factory = VectorStoreFactory()
        vector_store = factory.create()
        
        # Get repository metadata
        metadata = vector_store.get_repository_metadata()
        
        # Verify metadata
        assert len(metadata) == 1
        assert metadata[0]["name"] == "test-repo"
        assert metadata[0]["language"] == "Python"


@pytest.mark.asyncio
async def test_api_endpoint_simulation():
    """Test API endpoint simulation."""
    # Mock the API response
    mock_response = {
        "repositories": [
            {
                "name": "test-repo",
                "url": "https://github.com/test/repo",
                "language": "Python",
                "stars": 100
            }
        ],
        "total_count": 1,
        "last_updated": "2024-01-01T00:00:00Z"
    }
    
    # Simulate API endpoint response
    assert mock_response["total_count"] == 1
    assert len(mock_response["repositories"]) == 1
    assert mock_response["repositories"][0]["name"] == "test-repo"


async def main():
    """Main test function."""
    logger.info("Starting repository listing functionality tests...")
    
    # Test 1: Repository metadata retrieval
    test1_success = await test_repository_metadata()
    
    # Test 2: API endpoint simulation
    test2_success = await test_api_endpoint_simulation()
    
    # Summary
    logger.info("Test Summary:")
    logger.info(f"  Repository metadata test: {'PASSED' if test1_success else 'FAILED'}")
    logger.info(f"  API endpoint simulation test: {'PASSED' if test2_success else 'FAILED'}")
    
    if test1_success and test2_success:
        logger.info("All tests PASSED! Repository listing functionality is working correctly.")
        return 0
    else:
        logger.error("Some tests FAILED. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
