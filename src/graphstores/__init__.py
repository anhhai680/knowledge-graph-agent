"""
Graph stores module for the Knowledge Graph Agent.

This module provides abstract interfaces and implementations for graph database
operations, supporting MemGraph and other graph databases.
"""

from .base_graph_store import BaseGraphStore
from .memgraph_store import MemGraphStore

__all__ = ["BaseGraphStore", "MemGraphStore"] 