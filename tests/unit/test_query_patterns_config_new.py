"""
Tests for query patterns configuration system.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.config.query_patterns import (
    QueryPatternsConfig,
    DomainPattern,
    TechnicalPattern,
    get_default_query_patterns,
    load_query_patterns,
)


class TestQueryPatternsModels:
    """Test the pattern models."""

    def test_domain_pattern_creation(self):
        """Test DomainPattern can be created properly."""
        pattern = DomainPattern(
            patterns=["listing", "inventory", "catalog"],
            key_terms=["car", "listing", "vehicle", "inventory"]
        )
        
        assert "listing" in pattern.patterns
        assert "car" in pattern.key_terms
        assert "listing" in pattern.key_terms
        assert len(pattern.key_terms) == 4

    def test_technical_pattern_creation(self):
        """Test TechnicalPattern can be created properly."""
        pattern = TechnicalPattern(
            patterns=["api", "endpoint", "service"],
            key_terms=["class", "interface"]
        )
        
        assert "api" in pattern.patterns
        assert pattern.key_terms == ["class", "interface"]

    def test_default_configuration(self):
        """Test getting default query patterns configuration."""
        config = get_default_query_patterns()
        
        # Verify structure
        assert isinstance(config, QueryPatternsConfig)
        assert len(config.domain_patterns) > 0
        assert len(config.technical_patterns) > 0
        assert len(config.programming_patterns) > 0
        assert len(config.api_patterns) > 0
        assert len(config.database_patterns) > 0
        assert len(config.excluded_words) > 0
        assert config.max_terms == 5
        assert config.min_word_length == 2

    def test_domain_patterns_default(self):
        """Test default domain patterns."""
        config = get_default_query_patterns()
        
        # Check domain patterns have expected semantic terms
        all_domain_terms = []
        for pattern in config.domain_patterns:
            all_domain_terms.extend(pattern.key_terms)
        
        # Should contain business domain terms
        domain_terms_found = any(term in all_domain_terms for term in ["listing", "inventory", "catalog", "management"])
        assert domain_terms_found

    def test_technical_patterns_default(self):
        """Test that default technical patterns contain expected terms."""
        config = get_default_query_patterns()
        
        # Extract all technical terms
        tech_terms = []
        for pattern in config.technical_patterns:
            tech_terms.extend(pattern.key_terms)
        
        # Should contain technical terms that actually exist in the patterns
        assert any(term in tech_terms for term in ["class", "service", "module", "architecture"])


class TestQueryPatternsConfiguration:
    """Test configuration loading and validation."""

    def test_empty_configuration(self):
        """Test creating configuration with minimal data."""
        config = QueryPatternsConfig(
            domain_patterns=[],
            technical_patterns=[],
            programming_patterns=[],
            api_patterns=[],
            database_patterns=[],
            architecture_patterns=[],
            excluded_words=["the", "and", "or"],
            max_terms=3,
            min_word_length=2
        )
        
        assert len(config.domain_patterns) == 0
        assert len(config.technical_patterns) == 0
        assert config.max_terms == 3
        assert config.min_word_length == 2
        assert "the" in config.excluded_words

    def test_configuration_validation(self):
        """Test configuration validation works."""
        # This should not raise any validation errors
        config = get_default_query_patterns()
        default_config = get_default_query_patterns()
        
        # Verify we got configurations
        assert len(config.domain_patterns) == len(default_config.domain_patterns)
        assert len(config.technical_patterns) == len(default_config.technical_patterns)
        assert len(config.programming_patterns) == len(default_config.programming_patterns)

    def test_configuration_immutable_defaults(self):
        """Test that getting default config doesn't modify shared state."""
        config1 = get_default_query_patterns()
        config2 = get_default_query_patterns()
        
        # Should have same structure
        assert len(config1.domain_patterns) == len(config2.domain_patterns)
        assert len(config1.technical_patterns) == len(config2.technical_patterns)
        assert config1.max_terms == config2.max_terms


class TestQueryPatternsLoading:
    """Test loading configurations from files."""

    def test_load_from_valid_json(self):
        """Test loading configuration from valid JSON file."""
        config_data = {
            "domain_patterns": [
                {
                    "patterns": ["custom-domain"],
                    "key_terms": ["custom", "domain"]
                }
            ],
            "technical_patterns": [],
            "programming_patterns": [],
            "api_patterns": [],
            "database_patterns": [],
            "architecture_patterns": [],
            "excluded_words": ["the", "and"],
            "max_terms": 3,
            "min_word_length": 2
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            
            config = load_query_patterns(str(f.name))
            assert len(config.domain_patterns) == 1
            assert config.domain_patterns[0].patterns == ["custom-domain"]
            assert config.domain_patterns[0].key_terms == ["custom", "domain"]

    def test_load_from_nonexistent_file(self):
        """Test loading from nonexistent file returns default config."""
        config = load_query_patterns("nonexistent.json")
        default_config = get_default_query_patterns()
        
        # Should match default configuration
        assert len(config.domain_patterns) == len(default_config.domain_patterns)
        assert len(config.technical_patterns) == len(default_config.technical_patterns)

    def test_configuration_extensibility(self):
        """Test that configuration can be extended with new patterns."""
        config = get_default_query_patterns()
        
        # Add new domain pattern
        new_domain = DomainPattern(
            patterns=["analytics", "metrics"],
            key_terms=["analysis", "report", "metric"]
        )
        config.domain_patterns.append(new_domain)
        
        # Verify it was added
        assert len(config.domain_patterns) > 4  # Original patterns plus new one
        domain_pattern_lists = [p.patterns for p in config.domain_patterns]
        assert any("analytics" in patterns for patterns in domain_pattern_lists)

    def test_custom_configuration_creation(self):
        """Test creating custom configuration from scratch."""
        config = QueryPatternsConfig(
            domain_patterns=[
                DomainPattern(patterns=["test-domain"], key_terms=["test", "domain"]),
                DomainPattern(patterns=["test-service"], key_terms=["test", "service"])
            ],
            technical_patterns=[],
            programming_patterns=[],
            api_patterns=[],
            database_patterns=[],
            architecture_patterns=[],
            excluded_words=["the", "and"],
            max_terms=5,
            min_word_length=2
        )
        
        assert len(config.domain_patterns) == 2
        assert config.max_terms == 5
        assert "the" in config.excluded_words


class TestQueryPatternsIntegration:
    """Test integration scenarios."""

    def test_pattern_structure_consistency(self):
        """Test that all patterns follow consistent structure."""
        config = get_default_query_patterns()
        
        # All domain patterns should have patterns and key_terms
        for pattern in config.domain_patterns:
            assert isinstance(pattern.patterns, list)
            assert isinstance(pattern.key_terms, list)
            assert len(pattern.patterns) > 0
            assert len(pattern.key_terms) > 0
        
        # All technical patterns should have patterns and key_terms
        for pattern in config.technical_patterns:
            assert isinstance(pattern.patterns, list)
            assert isinstance(pattern.key_terms, list)
            assert len(pattern.patterns) > 0
            assert len(pattern.key_terms) > 0

    def test_configuration_serialization(self):
        """Test that configuration can be serialized and deserialized."""
        config = get_default_query_patterns()
        
        # Convert to dict and back
        config_dict = config.dict()
        new_config = QueryPatternsConfig(**config_dict)
        
        # Should be equivalent
        assert len(new_config.domain_patterns) == len(config.domain_patterns)
        assert len(new_config.technical_patterns) == len(config.technical_patterns)
        assert new_config.max_terms == config.max_terms
        assert new_config.min_word_length == config.min_word_length
