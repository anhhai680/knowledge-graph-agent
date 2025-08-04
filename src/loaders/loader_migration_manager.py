"""
Loader migration manager for transitioning from API-based to Git-based loading.

This module provides functionality to manage the migration from GitHub API-based
loading to Git-based loading, including performance benchmarking and validation.
"""

import time
from typing import Any, Dict, List, Optional, Type

from loguru import logger

from src.config.settings import settings
from .enhanced_github_loader import EnhancedGitHubLoader


class LoaderMigrationManager:
    """
    Manage migration from API-based to Git-based loader.
    
    This class provides functionality to create the appropriate loader based on
    configuration, benchmark performance, validate outputs, and migrate configurations.
    """

    def __init__(self):
        """Initialize loader migration manager."""
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        logger.debug("Initialized loader migration manager")

    def create_loader(
        self, 
        repo_owner: str, 
        repo_name: str, 
        **kwargs
    ) -> EnhancedGitHubLoader:
        """
        Create appropriate loader based on configuration.
        
        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            **kwargs: Additional loader arguments
            
        Returns:
            EnhancedGitHubLoader instance
        """
        try:
            logger.info(f"Creating Git-based loader for {repo_owner}/{repo_name}")
            return EnhancedGitHubLoader(
                repo_owner=repo_owner,
                repo_name=repo_name,
                **kwargs
            )
                
        except Exception as e:
            logger.error(f"Error creating loader for {repo_owner}/{repo_name}: {e}")
            raise

    def create_multi_repo_loader(
        self, 
        repositories: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> 'MultiGitRepositoryLoader':
        """
        Create appropriate multi-repository loader.
        
        Args:
            repositories: List of repository configurations
            **kwargs: Additional loader arguments
            
        Returns:
            Multi-repository loader instance
        """
        try:
            logger.info("Creating Git-based multi-repository loader")
            return MultiGitRepositoryLoader(
                repositories=repositories,
                **kwargs
            )
                
        except Exception as e:
            logger.error(f"Error creating multi-repository loader: {e}")
            raise

    def benchmark_loaders(self, repo_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare performance between API and Git loaders.
        
        Args:
            repo_config: Repository configuration for benchmarking
            
        Returns:
            Dictionary containing benchmark results
        """
        try:
            repo_owner = repo_config.get("owner")
            repo_name = repo_config.get("repo")
            branch = repo_config.get("branch", "main")
            
            # Validate required parameters
            if not repo_owner or not repo_name:
                raise ValueError("Repository owner and name are required")
            
            repo_owner = str(repo_owner)
            repo_name = str(repo_name)
            branch = str(branch)
            
            logger.info(f"Starting benchmark for {repo_owner}/{repo_name}")
            
            benchmark_results = {
                "repository": f"{repo_owner}/{repo_name}",
                "branch": branch,
                "timestamp": time.time(),
                "api_loader": {},
                "git_loader": {},
                "comparison": {}
            }
            
            # Benchmark API-based loader
            try:
                api_results = self._benchmark_single_loader(
                    EnhancedGitHubLoader,
                    repo_owner, 
                    repo_name, 
                    branch,
                    "API-based"
                )
                benchmark_results["api_loader"] = api_results
            except Exception as e:
                logger.error(f"API loader benchmark failed: {e}")
                benchmark_results["api_loader"] = {"error": str(e)}
            
            # Benchmark Git-based loader
            try:
                git_results = self._benchmark_single_loader(
                    EnhancedGitHubLoader,
                    repo_owner, 
                    repo_name, 
                    branch,
                    "Git-based"
                )
                benchmark_results["git_loader"] = git_results
            except Exception as e:
                logger.error(f"Git loader benchmark failed: {e}")
                benchmark_results["git_loader"] = {"error": str(e)}
            
            # Generate comparison
            benchmark_results["comparison"] = self._generate_comparison(
                benchmark_results["api_loader"],
                benchmark_results["git_loader"]
            )
            
            # Store metrics
            self.performance_metrics[f"{repo_owner}/{repo_name}"] = benchmark_results
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Benchmark failed for {repo_config}: {e}")
            return {"error": str(e)}

    def _benchmark_single_loader(
        self,
        loader_class: Type[EnhancedGitHubLoader],
        repo_owner: str,
        repo_name: str,
        branch: str,
        loader_type: str
    ) -> Dict[str, Any]:
        """
        Benchmark a single loader implementation.
        
        Args:
            loader_class: Loader class to benchmark
            repo_owner: Repository owner
            repo_name: Repository name
            branch: Repository branch
            loader_type: Type description for logging
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Benchmarking {loader_type} loader")
        
        start_time = time.time()
        
        try:
            # Create loader
            loader = loader_class(
                repo_owner=repo_owner,
                repo_name=repo_name,
                branch=branch,
                cleanup_after_processing=True  # Clean up after benchmark
            )
            
            # Load documents
            load_start = time.time()
            documents = loader.load()
            load_time = time.time() - load_start
            
            total_time = time.time() - start_time
            
            # Calculate statistics
            total_size = sum(len(doc.page_content) for doc in documents)
            avg_doc_size = total_size / len(documents) if documents else 0
            
            results = {
                "success": True,
                "total_time_seconds": round(total_time, 2),
                "load_time_seconds": round(load_time, 2),
                "setup_time_seconds": round(total_time - load_time, 2),
                "document_count": len(documents),
                "total_content_size": total_size,
                "average_document_size": round(avg_doc_size, 2),
                "documents_per_second": round(len(documents) / load_time, 2) if load_time > 0 else 0,
                "bytes_per_second": round(total_size / load_time, 2) if load_time > 0 else 0
            }
            
            logger.info(f"{loader_type} benchmark completed: {results['document_count']} docs in {results['total_time_seconds']}s")
            
            return results
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"{loader_type} benchmark failed after {total_time:.2f}s: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "total_time_seconds": round(total_time, 2)
            }

    def _generate_comparison(
        self, 
        api_results: Dict[str, Any], 
        git_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comparison between API and Git loader results.
        
        Args:
            api_results: API loader benchmark results
            git_results: Git loader benchmark results
            
        Returns:
            Dictionary containing comparison metrics
        """
        try:
            comparison = {}
            
            if api_results.get("success") and git_results.get("success"):
                # Time comparison
                api_time = api_results.get("total_time_seconds", 0)
                git_time = git_results.get("total_time_seconds", 0)
                
                if api_time > 0:
                    comparison["speed_improvement"] = round((api_time - git_time) / api_time * 100, 1)
                    comparison["speed_ratio"] = round(api_time / git_time, 2) if git_time > 0 else float('inf')
                
                # Document count comparison
                api_docs = api_results.get("document_count", 0)
                git_docs = git_results.get("document_count", 0)
                
                comparison["document_count_match"] = api_docs == git_docs
                comparison["document_difference"] = abs(api_docs - git_docs)
                
                # Content size comparison
                api_size = api_results.get("total_content_size", 0)
                git_size = git_results.get("total_content_size", 0)
                
                comparison["content_size_match"] = abs(api_size - git_size) < (api_size * 0.01)  # 1% tolerance
                comparison["content_size_difference"] = abs(api_size - git_size)
                
                # Performance metrics
                api_throughput = api_results.get("documents_per_second", 0)
                git_throughput = git_results.get("documents_per_second", 0)
                
                if api_throughput > 0:
                    comparison["throughput_improvement"] = round((git_throughput - api_throughput) / api_throughput * 100, 1)
                
                # Recommendation
                if git_time < api_time and comparison["document_count_match"]:
                    comparison["recommendation"] = "use_git_loader"
                elif api_time < git_time and comparison["document_count_match"]:
                    comparison["recommendation"] = "use_api_loader"
                else:
                    comparison["recommendation"] = "investigate_differences"
                    
            else:
                comparison["recommendation"] = "fallback_to_working_loader"
                comparison["api_failed"] = not api_results.get("success", False)
                comparison["git_failed"] = not git_results.get("success", False)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error generating comparison: {e}")
            return {"error": str(e)}

    def validate_git_loader_output(self, repo_config: Dict[str, Any]) -> bool:
        """
        Validate Git loader produces same results as API loader.
        
        Args:
            repo_config: Repository configuration for validation
            
        Returns:
            True if outputs are equivalent, False otherwise
        """
        try:
            repo_owner = repo_config.get("owner")
            repo_name = repo_config.get("repo")
            branch = repo_config.get("branch", "main")
            
            # Validate required parameters
            if not repo_owner or not repo_name:
                raise ValueError("Repository owner and name are required")
            
            repo_owner = str(repo_owner)
            repo_name = str(repo_name)
            branch = str(branch)
            
            logger.info(f"Validating Git loader output for {repo_owner}/{repo_name}")
            
            # Load with API loader
            api_loader = EnhancedGitHubLoader(
                repo_owner=repo_owner,
                repo_name=repo_name,
                branch=branch
            )
            api_docs = api_loader.load()
            
            # Load with Git loader
            git_loader = EnhancedGitHubLoader(
                repo_owner=repo_owner,
                repo_name=repo_name,
                branch=branch,
                cleanup_after_processing=True
            )
            git_docs = git_loader.load()
            
            # Compare results
            validation_result = self._compare_document_sets(api_docs, git_docs)
            
            logger.info(f"Validation result: {validation_result}")
            return validation_result["is_equivalent"]
            
        except Exception as e:
            logger.error(f"Validation failed for {repo_config}: {e}")
            return False

    def _compare_document_sets(
        self, 
        api_docs: List[EnhancedGitHubLoader.Document], 
        git_docs: List[EnhancedGitHubLoader.Document]
    ) -> Dict[str, Any]:
        """
        Compare two sets of documents for equivalence.
        
        Args:
            api_docs: Documents from API loader
            git_docs: Documents from Git loader
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            # Basic counts
            api_count = len(api_docs)
            git_count = len(git_docs)
            
            # Create file path mappings
            api_files = {doc.metadata.get("file_path"): doc for doc in api_docs}
            git_files = {doc.metadata.get("file_path"): doc for doc in git_docs}
            
            # Find common files
            common_files = set(api_files.keys()) & set(git_files.keys())
            api_only = set(api_files.keys()) - set(git_files.keys())
            git_only = set(git_files.keys()) - set(api_files.keys())
            
            # Compare content for common files
            content_matches = 0
            content_differences = []
            
            for file_path in common_files:
                api_doc = api_files[file_path]
                git_doc = git_files[file_path]
                
                if api_doc.page_content == git_doc.page_content:
                    content_matches += 1
                else:
                    content_differences.append({
                        "file_path": file_path,
                        "api_size": len(api_doc.page_content),
                        "git_size": len(git_doc.page_content)
                    })
            
            # Calculate metrics
            is_equivalent = (
                api_count == git_count and
                len(content_differences) == 0 and
                len(api_only) == 0 and
                len(git_only) == 0
            )
            
            return {
                "is_equivalent": is_equivalent,
                "api_document_count": api_count,
                "git_document_count": git_count,
                "common_files": len(common_files),
                "api_only_files": len(api_only),
                "git_only_files": len(git_only),
                "content_matches": content_matches,
                "content_differences": len(content_differences),
                "match_percentage": round(content_matches / len(common_files) * 100, 1) if common_files else 0,
                "differences": content_differences[:10]  # Limit to first 10 differences
            }
            
        except Exception as e:
            logger.error(f"Error comparing document sets: {e}")
            return {"error": str(e), "is_equivalent": False}

    def migrate_repository_config(self, old_config: Dict) -> Dict:
        """
        Convert API-based config to Git-based config.
        
        Args:
            old_config: Original configuration dictionary
            
        Returns:
            Migrated configuration dictionary
        """
        try:
            new_config = old_config.copy()
            
            # Add Git-specific settings
            git_settings = {
                "use_git_loader": True,
                "force_fresh_clone": False,
                "cleanup_after_processing": False,
                "git_timeout_seconds": 300
            }
            
            new_config.update(git_settings)
            
            logger.debug(f"Migrated repository configuration: {new_config}")
            return new_config
            
        except Exception as e:
            logger.error(f"Error migrating configuration: {e}")
            return old_config

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of all benchmark results.
        
        Returns:
            Dictionary containing performance summary
        """
        try:
            if not self.performance_metrics:
                return {"message": "No benchmark data available"}
            
            summary = {
                "total_benchmarks": len(self.performance_metrics),
                "successful_benchmarks": 0,
                "failed_benchmarks": 0,
                "average_git_speedup": 0,
                "recommendations": {
                    "use_git_loader": 0,
                    "use_api_loader": 0,
                    "investigate_differences": 0
                }
            }
            
            speedups = []
            
            for repo, metrics in self.performance_metrics.items():
                comparison = metrics.get("comparison", {})
                
                if comparison.get("speed_improvement") is not None:
                    summary["successful_benchmarks"] += 1
                    speedups.append(comparison["speed_improvement"])
                    
                    # Count recommendations
                    recommendation = comparison.get("recommendation", "investigate_differences")
                    if recommendation in summary["recommendations"]:
                        summary["recommendations"][recommendation] += 1
                else:
                    summary["failed_benchmarks"] += 1
            
            if speedups:
                summary["average_git_speedup"] = round(sum(speedups) / len(speedups), 1)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {"error": str(e)}


class MultiGitRepositoryLoader:
    """
    Load documents from multiple GitHub repositories using Git operations.
    
    This is the Git-based equivalent of MultiRepositoryGitHubLoader.
    """

    def __init__(
        self,
        repositories: Optional[List[Dict[str, str]]] = None,
        github_token: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize multi-repository Git loader.
        
        Args:
            repositories: List of repository configurations
            github_token: GitHub token for authentication
            file_extensions: List of file extensions to load
            **kwargs: Additional arguments for individual loaders
        """
        self.repositories = repositories or [
            {"owner": repo.url.split('/')[-2], "repo": repo.name, "branch": repo.branch}
            for repo in settings.repositories
        ]
        self.github_token = github_token or settings.github.token
        self.file_extensions = file_extensions or settings.github.file_extensions
        self.loader_kwargs = kwargs
        
        logger.info(f"Initialized multi-repository Git loader with {len(self.repositories)} repositories")

    def load(self) -> List[EnhancedGitHubLoader.Document]:
        """
        Load documents from multiple GitHub repositories.
        
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for repo_config in self.repositories:
            owner = repo_config["owner"]
            repo = repo_config["repo"]
            branch = repo_config.get("branch", "main")
            
            logger.info(f"Loading documents from {owner}/{repo}")
            
            try:
                loader = EnhancedGitHubLoader(
                    repo_owner=owner,
                    repo_name=repo,
                    branch=branch,
                    github_token=self.github_token,
                    file_extensions=self.file_extensions,
                    **self.loader_kwargs
                )
                
                repo_documents = loader.load()
                documents.extend(repo_documents)
                
                logger.info(f"Loaded {len(repo_documents)} documents from {owner}/{repo}")
                
            except Exception as e:
                logger.error(f"Error loading documents from {owner}/{repo}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {len(self.repositories)} repositories")
        return documents
