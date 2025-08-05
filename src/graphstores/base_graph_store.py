"""
Abstract base interface for graph stores.

This module defines the abstract interface that all graph store implementations
must follow, ensuring consistent behavior across different graph databases.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GraphQueryResult(BaseModel):
    """Result model for graph query execution."""
    
    data: List[Dict[str, Any]] = Field(..., description="Query result data")
    metadata: Dict[str, Any] = Field(..., description="Query metadata")
    execution_time_ms: float = Field(..., description="Query execution time in milliseconds")
    query: str = Field(..., description="Original query executed")
    node_count: Optional[int] = Field(None, description="Number of nodes in result")
    relationship_count: Optional[int] = Field(None, description="Number of relationships in result")


class GraphNode(BaseModel):
    """Represents a node in the graph."""
    
    id: str
    labels: List[str]
    properties: Dict[str, Any]


class GraphRelationship(BaseModel):
    """Represents a relationship in the graph."""
    
    id: str
    type: str
    start_node_id: str
    end_node_id: str
    properties: Dict[str, Any]


class BaseGraphStore(ABC):
    """
    Abstract base class for graph store implementations.
    
    This class defines the interface that all graph store implementations
    must follow, ensuring consistent behavior across different graph databases.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the graph database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the graph database."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to the graph database.
        
        Returns:
            bool: True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> GraphQueryResult:
        """
        Execute a Cypher query against the graph database.
        
        Args:
            query: Cypher query string
            parameters: Optional parameters for the query
            
        Returns:
            GraphQueryResult: Query execution result
            
        Raises:
            Exception: If query execution fails
        """
        pass
    
    @abstractmethod
    def create_node(self, labels: List[str], properties: Dict[str, Any]) -> str:
        """
        Create a new node in the graph.
        
        Args:
            labels: List of labels for the node
            properties: Node properties
            
        Returns:
            str: ID of the created node
        """
        pass
    
    @abstractmethod
    def create_relationship(
        self, 
        start_node_id: str, 
        end_node_id: str, 
        relationship_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a relationship between two nodes.
        
        Args:
            start_node_id: ID of the start node
            end_node_id: ID of the end node
            relationship_type: Type of relationship
            properties: Optional relationship properties
            
        Returns:
            str: ID of the created relationship
        """
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """
        Retrieve a node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Optional[GraphNode]: Node if found, None otherwise
        """
        pass
    
    @abstractmethod
    def find_nodes(self, labels: Optional[List[str]] = None, properties: Optional[Dict[str, Any]] = None) -> List[GraphNode]:
        """
        Find nodes matching criteria.
        
        Args:
            labels: Optional list of labels to match
            properties: Optional properties to match
            
        Returns:
            List[GraphNode]: List of matching nodes
        """
        pass
    
    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node by ID.
        
        Args:
            node_id: ID of the node to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def clear_graph(self) -> bool:
        """
        Clear all data from the graph.
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get information about the graph database.
        
        Returns:
            Dict[str, Any]: Graph database information
        """
        pass 