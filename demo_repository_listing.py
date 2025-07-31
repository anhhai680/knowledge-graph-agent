#!/usr/bin/env python3
"""
Mock demonstration of the enhanced list_repositories endpoint functionality.

This script demonstrates how the repository listing would work with actual data,
simulating the vector store response to show the complete data flow.
"""

import json
from datetime import datetime
from typing import List, Dict, Any

def simulate_vector_store_metadata() -> List[Dict[str, Any]]:
    """Simulate what the vector store would return with real indexed data."""
    return [
        {
            "name": "microsoft/TypeScript",
            "url": "https://github.com/microsoft/TypeScript",
            "branch": "main", 
            "last_indexed": datetime.now().isoformat(),
            "file_count": 2847,
            "document_count": 15234,
            "languages": ["typescript", "javascript", "markdown"],
            "size_mb": 156.7
        },
        {
            "name": "facebook/react",
            "url": "https://github.com/facebook/react", 
            "branch": "main",
            "last_indexed": datetime.now().isoformat(),
            "file_count": 1205,
            "document_count": 8934,
            "languages": ["javascript", "typescript", "css"],
            "size_mb": 89.3
        },
        {
            "name": "dotnet/aspnetcore",
            "url": "https://github.com/dotnet/aspnetcore",
            "branch": "main", 
            "last_indexed": datetime.now().isoformat(),
            "file_count": 3456,
            "document_count": 18756,
            "languages": ["csharp", "javascript", "razor"],
            "size_mb": 234.9
        }
    ]

def simulate_api_response():
    """Simulate the complete API response flow."""
    print("🚀 Simulating Enhanced Repository Listing API")
    print("=" * 60)
    
    # Simulate vector store query
    print("📊 Querying vector store for repository metadata...")
    repository_metadata = simulate_vector_store_metadata()
    print(f"✅ Found {len(repository_metadata)} repositories in vector store")
    
    # Simulate API response processing
    repositories = []
    for repo_data in repository_metadata:
        try:
            # Parse last_indexed date (simulate the API logic)
            last_indexed = repo_data.get("last_indexed")
            if isinstance(last_indexed, str):
                last_indexed = datetime.fromisoformat(last_indexed.replace('Z', '+00:00'))
            elif not last_indexed:
                last_indexed = datetime.now()
            
            # Create RepositoryInfo equivalent (simulate API model)
            repository_info = {
                "name": repo_data.get("name", "Unknown"),
                "url": repo_data.get("url", ""),
                "branch": repo_data.get("branch", "main"), 
                "last_indexed": last_indexed.isoformat(),
                "file_count": repo_data.get("file_count", 0),
                "document_count": repo_data.get("document_count", 0),
                "languages": repo_data.get("languages", []),
                "size_mb": repo_data.get("size_mb", 0.0)
            }
            repositories.append(repository_info)
            
        except Exception as e:
            print(f"⚠️  Warning: Error processing repository metadata: {e}")
            continue
    
    # Simulate complete API response
    api_response = {
        "repositories": repositories,
        "total_count": len(repositories),
        "last_updated": datetime.now().isoformat()
    }
    
    print("\n📋 API Response Preview:")
    print(json.dumps(api_response, indent=2, default=str))
    
    print(f"\n📊 Response Summary:")
    print(f"  Total repositories: {api_response['total_count']}")
    print(f"  Last updated: {api_response['last_updated']}")
    
    print(f"\n📁 Repository Details:")
    for repo in repositories:
        print(f"  🏢 {repo['name']}:")
        print(f"     📄 {repo['document_count']:,} documents from {repo['file_count']:,} files")
        print(f"     💾 {repo['size_mb']:.1f} MB indexed")
        print(f"     🔧 Languages: {', '.join(repo['languages'])}")
        print(f"     🕒 Last indexed: {repo['last_indexed']}")
        print()

def simulate_statistics_endpoint():
    """Simulate the enhanced statistics endpoint."""
    print("📈 Simulating Enhanced Statistics API")
    print("=" * 60)
    
    repository_metadata = simulate_vector_store_metadata()
    
    # Aggregate statistics (simulate the API logic)
    total_repositories = len(repository_metadata)
    total_files = sum(repo.get("file_count", 0) for repo in repository_metadata)
    total_documents = sum(repo.get("document_count", 0) for repo in repository_metadata)
    total_size_mb = sum(repo.get("size_mb", 0.0) for repo in repository_metadata)
    
    # Language distribution
    language_counts = {}
    for repo in repository_metadata:
        for language in repo.get("languages", []):
            if language:
                language_counts[language] = language_counts.get(language, 0) + repo.get("document_count", 0)
    
    stats_response = {
        "total_repositories": total_repositories,
        "total_documents": total_documents,
        "total_files": total_files,
        "index_size_mb": round(total_size_mb, 2),
        "languages": language_counts,
        "recent_queries": 127,  # Would be tracked separately
        "active_workflows": 0,
        "system_health": "healthy"
    }
    
    print("📊 Statistics Response:")
    print(json.dumps(stats_response, indent=2))
    
    print(f"\n📈 Key Metrics:")
    print(f"  🏢 Repositories: {stats_response['total_repositories']}")
    print(f"  📄 Documents: {stats_response['total_documents']:,}")
    print(f"  📁 Files: {stats_response['total_files']:,}")
    print(f"  💾 Index size: {stats_response['index_size_mb']} MB")
    print(f"  🔧 Top languages: {dict(sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:3])}")

def demonstrate_error_handling():
    """Demonstrate error handling scenarios."""
    print("⚠️  Demonstrating Error Handling Scenarios")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "Empty Vector Store",
            "data": [],
            "expected": "Falls back to appSettings.json or returns empty list"
        },
        {
            "name": "Malformed Repository Data", 
            "data": [{"name": None, "invalid": "data"}],
            "expected": "Skips invalid entries, logs warnings"
        },
        {
            "name": "Vector Store Connection Error",
            "data": "CONNECTION_ERROR",
            "expected": "Returns HTTP 500 with error details"
        }
    ]
    
    for scenario in scenarios:
        print(f"🧪 Scenario: {scenario['name']}")
        print(f"   Expected behavior: {scenario['expected']}")
        print()

if __name__ == "__main__":
    print("🎯 Enhanced Repository Listing - Complete Demonstration")
    print("=" * 80)
    print()
    
    # Main simulation
    simulate_api_response()
    print("\n" + "=" * 80 + "\n")
    
    # Statistics simulation  
    simulate_statistics_endpoint()
    print("\n" + "=" * 80 + "\n")
    
    # Error handling demonstration
    demonstrate_error_handling()
    
    print("✅ Demonstration Complete!")
    print("\n🔑 Key Benefits of the Implementation:")
    print("   • Real-time data from vector store instead of static mocks")
    print("   • Comprehensive repository metadata aggregation")
    print("   • Robust error handling and fallback mechanisms")
    print("   • Type-safe data processing and validation")
    print("   • Production-ready scalable architecture")
    print("   • Extensible design for future enhancements")
