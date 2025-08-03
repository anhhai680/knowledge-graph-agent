"""
Query pattern configuration for key term extraction.

This module provides configurable patterns for extracting key terms from user queries
to improve vector search effectiveness. Patterns are organized by category and focus
on semantic meaning rather than specific repository names.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DomainPattern(BaseModel):
    """Configuration for domain-specific term extraction."""
    
    patterns: List[str] = Field(..., description="Patterns to match in query")
    key_terms: List[str] = Field(..., description="Key terms to extract")


class TechnicalPattern(BaseModel):
    """Configuration for technical term extraction."""
    
    patterns: List[str] = Field(..., description="Patterns to match in query")
    key_terms: List[str] = Field(..., description="Key terms to extract")


class QueryPatternsConfig(BaseModel):
    """Configuration model for query term extraction patterns."""
    
    domain_patterns: List[DomainPattern] = Field(
        default_factory=list,
        description="Domain-specific patterns (e.g., business domains, service types)"
    )
    
    technical_patterns: List[TechnicalPattern] = Field(
        default_factory=list,
        description="Technical term patterns"
    )
    
    programming_patterns: List[TechnicalPattern] = Field(
        default_factory=list,
        description="Programming language patterns"
    )
    
    api_patterns: List[TechnicalPattern] = Field(
        default_factory=list,
        description="API-related patterns"
    )
    
    database_patterns: List[TechnicalPattern] = Field(
        default_factory=list,
        description="Database-related patterns"
    )
    
    architecture_patterns: List[TechnicalPattern] = Field(
        default_factory=list,
        description="Architecture and design patterns"
    )
    
    excluded_words: List[str] = Field(
        default_factory=list,
        description="Words to exclude from general term extraction"
    )
    
    max_terms: int = Field(
        default=5,
        description="Maximum number of terms to include in simplified query"
    )
    
    min_word_length: int = Field(
        default=2,
        description="Minimum word length for general term extraction"
    )


def get_default_query_patterns() -> QueryPatternsConfig:
    """
    Get default query pattern configuration.
    
    This function provides semantic patterns that focus on business domains,
    technical concepts, and architectural patterns rather than specific
    repository names. This makes the configuration reusable across projects.
    
    Returns:
        QueryPatternsConfig: Default configuration with semantic patterns
    """
    return QueryPatternsConfig(
        domain_patterns=[
            # Business domain patterns - focus on semantic meaning
            DomainPattern(
                patterns=["listing", "inventory", "catalog"],
                key_terms=["item", "product", "listing", "catalog"]
            ),
            DomainPattern(
                patterns=["notification", "messaging", "alert"],
                key_terms=["message", "notification", "event", "alert"]
            ),
            DomainPattern(
                patterns=["order", "booking", "reservation"],
                key_terms=["order", "transaction", "booking", "purchase"]
            ),
            DomainPattern(
                patterns=["user", "account", "profile", "auth"],
                key_terms=["user", "account", "authentication", "profile"]
            ),
            DomainPattern(
                patterns=["payment", "billing", "finance"],
                key_terms=["payment", "billing", "transaction", "finance"]
            ),
            DomainPattern(
                patterns=["client", "frontend", "ui", "web"],
                key_terms=["client", "frontend", "interface", "web"]
            ),
        ],
        
        technical_patterns=[
            TechnicalPattern(
                patterns=["component", "components"],
                key_terms=["class", "service", "module"]
            ),
            TechnicalPattern(
                patterns=["main"],
                key_terms=["class", "entry", "main"]
            ),
            TechnicalPattern(
                patterns=["structure"],
                key_terms=["class", "architecture"]
            ),
            TechnicalPattern(
                patterns=["project", "application"],
                key_terms=["class", "module", "application"]
            ),
            TechnicalPattern(
                patterns=["service", "microservice"],
                key_terms=["service", "class", "interface"]
            ),
        ],
        
        programming_patterns=[
            TechnicalPattern(
                patterns=["csharp", "c#"],
                key_terms=["class", "namespace", "method"]
            ),
            TechnicalPattern(
                patterns=["dotnet", ".net"],
                key_terms=["class", "assembly", "namespace"]
            ),
            TechnicalPattern(
                patterns=["python"],
                key_terms=["class", "function", "module"]
            ),
            TechnicalPattern(
                patterns=["javascript", "js", "typescript", "ts"],
                key_terms=["function", "class", "interface", "component"]
            ),
            TechnicalPattern(
                patterns=["java"],
                key_terms=["class", "interface", "method", "package"]
            ),
            TechnicalPattern(
                patterns=["go", "golang"],
                key_terms=["func", "struct", "package", "interface"]
            ),
        ],
        
        api_patterns=[
            TechnicalPattern(
                patterns=["api", "rest", "restful"],
                key_terms=["controller", "service", "endpoint", "api"]
            ),
            TechnicalPattern(
                patterns=["endpoint", "route"],
                key_terms=["controller", "handler", "route"]
            ),
            TechnicalPattern(
                patterns=["controller"],
                key_terms=["controller", "action", "handler"]
            ),
            TechnicalPattern(
                patterns=["graphql", "gql"],
                key_terms=["resolver", "schema", "query", "mutation"]
            ),
            TechnicalPattern(
                patterns=["grpc"],
                key_terms=["service", "proto", "rpc", "method"]
            ),
        ],
        
        database_patterns=[
            TechnicalPattern(
                patterns=["database", "db", "storage"],
                key_terms=["model", "entity", "repository", "table"]
            ),
            TechnicalPattern(
                patterns=["model", "entity"],
                key_terms=["class", "table", "schema"]
            ),
            TechnicalPattern(
                patterns=["repository", "dao"],
                key_terms=["class", "interface", "crud", "data"]
            ),
            TechnicalPattern(
                patterns=["migration", "schema"],
                key_terms=["table", "column", "index", "database"]
            ),
            TechnicalPattern(
                patterns=["nosql", "mongodb"],
                key_terms=["collection", "document", "query"]
            ),
            TechnicalPattern(
                patterns=["sql", "postgres", "mysql"],
                key_terms=["table", "column", "index", "query"]
            ),
        ],
        
        architecture_patterns=[
            TechnicalPattern(
                patterns=["architecture", "design"],
                key_terms=["pattern", "architecture", "design", "structure"]
            ),
            TechnicalPattern(
                patterns=["pattern", "patterns"],
                key_terms=["design", "pattern", "architecture"]
            ),
            TechnicalPattern(
                patterns=["workflow", "pipeline"],
                key_terms=["workflow", "process", "pipeline", "flow"]
            ),
            TechnicalPattern(
                patterns=["container", "docker"],
                key_terms=["container", "docker", "deployment"]
            ),
            TechnicalPattern(
                patterns=["kubernetes", "k8s"],
                key_terms=["pod", "service", "deployment", "cluster"]
            ),
        ],
        
        excluded_words=[
            "the", "and", "or", "for", "with", "this", "that", "what", "are",
            "main", "components", "project", "how", "why", "when", "where",
            "can", "will", "should", "would", "could", "does", "is", "was",
            "have", "has", "had", "do", "did", "get", "got", "make", "made"
        ],
        
        max_terms=5,
        min_word_length=2
    )


def load_query_patterns(config_path: Optional[str] = None) -> QueryPatternsConfig:
    """
    Load query patterns from configuration file or return defaults.
    
    Args:
        config_path: Optional path to JSON configuration file
        
    Returns:
        QueryPatternsConfig: Loaded or default configuration
        
    Raises:
        ValueError: If configuration file is invalid
    """
    if config_path is None:
        return get_default_query_patterns()
    
    try:
        import json
        from pathlib import Path
        
        config_file = Path(config_path)
        if not config_file.exists():
            return get_default_query_patterns()
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return QueryPatternsConfig(**config_data)
    
    except Exception as e:
        raise ValueError(f"Failed to load query patterns configuration: {str(e)}")


# Singleton instance for default patterns
default_patterns = get_default_query_patterns()
