"""
Feature flags management for the Knowledge Graph Agent.

This module provides feature flag management for enabling/disabling
graph features and other experimental functionality.
"""

from typing import Dict, Any, Optional
from functools import wraps
from ..config.settings import settings


class FeatureFlags:
    """Feature flags management class."""
    
    # Graph-related feature flags
    GRAPH_FEATURES = "enable_graph_features"
    HYBRID_SEARCH = "enable_hybrid_search"
    GRAPH_VISUALIZATION = "enable_graph_visualization"
    
    # Feature flag cache
    _flag_cache: Dict[str, bool] = {}
    
    @classmethod
    def is_enabled(cls, flag_name: str) -> bool:
        """
        Check if a feature flag is enabled.
        
        Args:
            flag_name: Name of the feature flag
            
        Returns:
            bool: True if enabled, False otherwise
        """
        # Check cache first
        if flag_name in cls._flag_cache:
            return cls._flag_cache[flag_name]
        
        # Get from settings
        flag_value = getattr(settings, flag_name, False)
        cls._flag_cache[flag_name] = flag_value
        return flag_value
    
    @classmethod
    def set_flag(cls, flag_name: str, value: bool) -> None:
        """
        Set a feature flag value (runtime override).
        
        Args:
            flag_name: Name of the feature flag
            value: Flag value
        """
        cls._flag_cache[flag_name] = value
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the feature flag cache."""
        cls._flag_cache.clear()
    
    @classmethod
    def get_all_flags(cls) -> Dict[str, bool]:
        """
        Get all feature flag values.
        
        Returns:
            Dict[str, bool]: Dictionary of flag names and values
        """
        return {
            cls.GRAPH_FEATURES: cls.is_enabled(cls.GRAPH_FEATURES),
            cls.HYBRID_SEARCH: cls.is_enabled(cls.HYBRID_SEARCH),
            cls.GRAPH_VISUALIZATION: cls.is_enabled(cls.GRAPH_VISUALIZATION),
        }
    
    @classmethod
    def require_graph_features(cls, func):
        """
        Decorator to require graph features to be enabled.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not cls.is_enabled(cls.GRAPH_FEATURES):
                raise RuntimeError("Graph features are not enabled")
            return func(*args, **kwargs)
        return wrapper
    
    @classmethod
    def require_hybrid_search(cls, func):
        """
        Decorator to require hybrid search to be enabled.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not cls.is_enabled(cls.HYBRID_SEARCH):
                raise RuntimeError("Hybrid search is not enabled")
            return func(*args, **kwargs)
        return wrapper
    
    @classmethod
    def require_graph_visualization(cls, func):
        """
        Decorator to require graph visualization to be enabled.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not cls.is_enabled(cls.GRAPH_VISUALIZATION):
                raise RuntimeError("Graph visualization is not enabled")
            return func(*args, **kwargs)
        return wrapper


def require_graph_features(func):
    """
    Decorator to require graph features to be enabled.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    return FeatureFlags.require_graph_features(func)


def require_hybrid_search(func):
    """
    Decorator to require hybrid search to be enabled.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    return FeatureFlags.require_hybrid_search(func)


def require_graph_visualization(func):
    """
    Decorator to require graph visualization to be enabled.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    return FeatureFlags.require_graph_visualization(func)


def is_graph_enabled() -> bool:
    """
    Check if graph features are enabled.
    
    Returns:
        bool: True if graph features are enabled
    """
    return FeatureFlags.is_enabled(FeatureFlags.GRAPH_FEATURES)


def is_hybrid_search_enabled() -> bool:
    """
    Check if hybrid search is enabled.
    
    Returns:
        bool: True if hybrid search is enabled
    """
    return FeatureFlags.is_enabled(FeatureFlags.HYBRID_SEARCH)


def is_graph_visualization_enabled() -> bool:
    """
    Check if graph visualization is enabled.
    
    Returns:
        bool: True if graph visualization is enabled
    """
    return FeatureFlags.is_enabled(FeatureFlags.GRAPH_VISUALIZATION) 