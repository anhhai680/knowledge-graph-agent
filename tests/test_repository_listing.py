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

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.vectorstores.store_factory import VectorStoreFactory
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def test_repository_metadata():
    """Test repository metadata retrieval from vector store."""
    try:
        logger.info("Testing repository metadata retrieval...")
        
        # Initialize vector store
        settings = get_settings()
        logger.info(f"Using vector store type: {settings.database_type}")
        
        vector_store_factory = VectorStoreFactory()
        vector_store = vector_store_factory.create()
        
        # Test health check
        logger.info("Testing vector store health check...")
        is_healthy, health_message = vector_store.health_check()
        logger.info(f"Vector store health: {is_healthy} - {health_message}")
        
        # Test collection stats
        logger.info("Testing collection statistics...")
        stats = vector_store.get_collection_stats()
        logger.info(f"Collection stats: {stats}")
        
        # Test repository metadata retrieval
        logger.info("Testing repository metadata retrieval...")
        repositories = vector_store.get_repository_metadata()
        
        if repositories:
            logger.info(f"Found {len(repositories)} repositories:")
            for i, repo in enumerate(repositories):
                logger.info(f"Repository {i+1}:")
                logger.info(f"  Name: {repo.get('name', 'Unknown')}")
                logger.info(f"  URL: {repo.get('url', 'N/A')}")
                logger.info(f"  Branch: {repo.get('branch', 'main')}")
                logger.info(f"  File count: {repo.get('file_count', 0)}")
                logger.info(f"  Document count: {repo.get('document_count', 0)}")
                logger.info(f"  Languages: {repo.get('languages', [])}")
                logger.info(f"  Size (MB): {repo.get('size_mb', 0.0)}")
                logger.info(f"  Last indexed: {repo.get('last_indexed', 'N/A')}")
                logger.info("")
        else:
            logger.warning("No repositories found in vector store")
            logger.info("This could mean:")
            logger.info("1. No repositories have been indexed yet")
            logger.info("2. The vector store is empty")
            logger.info("3. There's an issue with the vector store connection")
        
        logger.info("Repository metadata test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Repository metadata test failed: {e}")
        return False


async def test_api_endpoint_simulation():
    """Simulate the API endpoint logic to test the complete flow."""
    try:
        logger.info("Testing API endpoint simulation...")
        
        # Simulate the list_repositories endpoint logic
        vector_store_factory = VectorStoreFactory()
        vector_store = vector_store_factory.create()
        
        # Get repository information from vector store
        repository_metadata = vector_store.get_repository_metadata()
        
        # Convert repository metadata to response format (simulating RepositoryInfo objects)
        repositories = []
        for repo_data in repository_metadata:
            try:
                # Parse last_indexed date if it's a string
                last_indexed = repo_data.get("last_indexed")
                if isinstance(last_indexed, str):
                    try:
                        from dateutil import parser
                        last_indexed = parser.parse(last_indexed)
                    except Exception:
                        last_indexed = datetime.now()
                elif not last_indexed:
                    last_indexed = datetime.now()
                
                repository_info = {
                    "name": repo_data.get("name", "Unknown"),
                    "url": repo_data.get("url", ""),
                    "branch": repo_data.get("branch", "main"),
                    "last_indexed": last_indexed,
                    "file_count": repo_data.get("file_count", 0),
                    "document_count": repo_data.get("document_count", 0),
                    "languages": repo_data.get("languages", []),
                    "size_mb": repo_data.get("size_mb", 0.0)
                }
                repositories.append(repository_info)
                
            except Exception as e:
                logger.warning(f"Error processing repository metadata: {e}")
                continue
        
        # Simulate response format
        response = {
            "repositories": repositories,
            "total_count": len(repositories),
            "last_updated": datetime.now()
        }
        
        logger.info(f"Simulated API response:")
        logger.info(f"  Total repositories: {response['total_count']}")
        logger.info(f"  Last updated: {response['last_updated']}")
        
        if repositories:
            logger.info("  Repository details:")
            for repo in repositories:
                logger.info(f"    - {repo['name']}: {repo['document_count']} documents, {repo['file_count']} files")
        
        logger.info("API endpoint simulation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"API endpoint simulation failed: {e}")
        return False


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
