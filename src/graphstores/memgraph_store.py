"""
MemGraph graph store implementation.

This module provides the MemGraph-specific implementation of the graph store
interface, handling connection management and query execution.
"""

import time
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError, ClientError

from .base_graph_store import BaseGraphStore, GraphQueryResult, GraphNode, GraphRelationship
from ..config.settings import settings


class MemGraphStore(BaseGraphStore):
    """
    MemGraph graph store implementation.
    
    This class provides MemGraph-specific functionality for graph database
    operations, including connection management and query execution.
    """
    
    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize MemGraph store.
        
        Args:
            uri: MemGraph connection URI (defaults to settings)
            username: Username for authentication (defaults to settings)
            password: Password for authentication (defaults to settings)
        """
        self.uri = uri or settings.graph_store.url
        self.username = username or settings.graph_store.username
        self.password = password or settings.graph_store.password
        self.driver: Optional[Driver] = None
        self._connected = False
    
    def connect(self, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
        """
        Establish connection to MemGraph with retry logic.
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retries in seconds
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Create driver with configuration
                if self.username and self.password:
                    self.driver = GraphDatabase.driver(
                        self.uri, 
                        auth=(self.username, self.password),
                        connection_timeout=10.0,  # 10 second timeout
                        max_connection_lifetime=3600,  # 1 hour lifetime
                    )
                else:
                    self.driver = GraphDatabase.driver(
                        self.uri,
                        connection_timeout=10.0,
                        max_connection_lifetime=3600,
                    )
                
                # Test connection with a simple query
                with self.driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    result.single()  # Consume the result
                
                self._connected = True
                print(f"Successfully connected to MemGraph at {self.uri}")
                return True
                
            except (ServiceUnavailable, AuthError) as e:
                last_error = e
                print(f"MemGraph connection attempt {attempt + 1}/{max_retries} failed: {e}")
                
                # Clean up failed driver
                if self.driver:
                    try:
                        self.driver.close()
                    except Exception:
                        pass
                    self.driver = None
                
                # Wait before retrying (except on last attempt)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    
            except Exception as e:
                last_error = e
                print(f"Unexpected error connecting to MemGraph: {e}")
                
                # Clean up failed driver
                if self.driver:
                    try:
                        self.driver.close()
                    except Exception:
                        pass
                    self.driver = None
                break  # Don't retry on unexpected errors
        
        self._connected = False
        print(f"Failed to connect to MemGraph after {max_retries} attempts. Last error: {last_error}")
        return False
    
    def disconnect(self) -> None:
        """Close connection to MemGraph."""
        if self.driver:
            self.driver.close()
            self.driver = None
        self._connected = False
    
    def is_connected(self) -> bool:
        """
        Check if connected to MemGraph.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.driver or not self._connected:
            self._connected = False
            return False
        
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            self._connected = False
            return False
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> GraphQueryResult:
        """
        Execute a Cypher query against MemGraph.
        
        Args:
            query: Cypher query string
            parameters: Optional parameters for the query
            
        Returns:
            GraphQueryResult: Query execution result
            
        Raises:
            Exception: If query execution fails
        """
        if not self.is_connected():
            raise Exception("Not connected to MemGraph")
        
        start_time = time.time()
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                data = [dict(record) for record in result]
                
                execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                return GraphQueryResult(
                    data=data,
                    metadata={
                        "node_count": len([r for r in data if "node" in r]),
                        "relationship_count": len([r for r in data if "relationship" in r]),
                        "result_count": len(data)
                    },
                    execution_time_ms=execution_time,
                    query=query
                )
                
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")
    
    def create_node(self, labels: List[str], properties: Dict[str, Any]) -> str:
        """
        Create a new node in MemGraph.
        
        Args:
            labels: List of labels for the node
            properties: Node properties
            
        Returns:
            str: ID of the created node
        """
        if not self.is_connected():
            raise Exception("Not connected to MemGraph")
        
        label_string = ":".join(labels)
        properties_string = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        
        query = f"CREATE (n:{label_string} {{{properties_string}}}) RETURN id(n) as node_id"
        
        try:
            with self.driver.session() as session:
                result = session.run(query, properties)
                record = result.single()
                return str(record["node_id"])
        except Exception as e:
            raise Exception(f"Failed to create node: {str(e)}")
    
    def create_relationship(
        self, 
        start_node_id: str, 
        end_node_id: str, 
        relationship_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a relationship between two nodes in MemGraph.
        
        Args:
            start_node_id: ID of the start node
            end_node_id: ID of the end node
            relationship_type: Type of relationship
            properties: Optional relationship properties
            
        Returns:
            str: ID of the created relationship
        """
        if not self.is_connected():
            raise Exception("Not connected to MemGraph")
        
        properties = properties or {}
        properties_string = ", ".join([f"{k}: ${k}" for k in properties.keys()]) if properties else ""
        rel_props = f"{{{properties_string}}}" if properties_string else ""
        
        query = f"""
        MATCH (a), (b) 
        WHERE id(a) = $start_id AND id(b) = $end_id 
        CREATE (a)-[r:{relationship_type} {rel_props}]->(b) 
        RETURN id(r) as rel_id
        """
        
        params = {"start_id": int(start_node_id), "end_id": int(end_node_id), **properties}
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                record = result.single()
                return str(record["rel_id"])
        except Exception as e:
            raise Exception(f"Failed to create relationship: {str(e)}")
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """
        Retrieve a node by ID from MemGraph.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Optional[GraphNode]: Node if found, None otherwise
        """
        if not self.is_connected():
            raise Exception("Not connected to MemGraph")
        
        query = "MATCH (n) WHERE id(n) = $node_id RETURN n"
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {"node_id": int(node_id)})
                record = result.single()
                
                if record:
                    node = record["n"]
                    return GraphNode(
                        id=str(node.id),
                        labels=list(node.labels),
                        properties=dict(node)
                    )
                return None
                
        except Exception as e:
            raise Exception(f"Failed to get node: {str(e)}")
    
    def find_nodes(self, labels: Optional[List[str]] = None, properties: Optional[Dict[str, Any]] = None) -> List[GraphNode]:
        """
        Find nodes matching criteria in MemGraph.
        
        Args:
            labels: Optional list of labels to match
            properties: Optional properties to match
            
        Returns:
            List[GraphNode]: List of matching nodes
        """
        if not self.is_connected():
            raise Exception("Not connected to MemGraph")
        
        # Build query based on criteria
        if labels and properties:
            label_string = ":".join(labels)
            properties_string = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])
            query = f"MATCH (n:{label_string}) WHERE {properties_string} RETURN n"
        elif labels:
            label_string = ":".join(labels)
            query = f"MATCH (n:{label_string}) RETURN n"
        elif properties:
            properties_string = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])
            query = f"MATCH (n) WHERE {properties_string} RETURN n"
        else:
            query = "MATCH (n) RETURN n"
        
        try:
            with self.driver.session() as session:
                result = session.run(query, properties or {})
                nodes = []
                
                for record in result:
                    node = record["n"]
                    nodes.append(GraphNode(
                        id=str(node.id),
                        labels=list(node.labels),
                        properties=dict(node)
                    ))
                
                return nodes
                
        except Exception as e:
            raise Exception(f"Failed to find nodes: {str(e)}")
    
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node by ID from MemGraph.
        
        Args:
            node_id: ID of the node to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        if not self.is_connected():
            raise Exception("Not connected to MemGraph")
        
        query = "MATCH (n) WHERE id(n) = $node_id DETACH DELETE n RETURN count(n) as deleted_count"
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {"node_id": int(node_id)})
                record = result.single()
                return record["deleted_count"] > 0
                
        except Exception as e:
            raise Exception(f"Failed to delete node: {str(e)}")
    
    def clear_graph(self) -> bool:
        """
        Clear all data from MemGraph.
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        if not self.is_connected():
            raise Exception("Not connected to MemGraph")
        
        query = "MATCH (n) DETACH DELETE n"
        
        try:
            with self.driver.session() as session:
                session.run(query)
                return True
                
        except Exception as e:
            raise Exception(f"Failed to clear graph: {str(e)}")
    
    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get information about the MemGraph database.
        
        Returns:
            Dict[str, Any]: Graph database information
        """
        if not self.is_connected():
            raise Exception("Not connected to MemGraph")
        
        try:
            # Get node count
            node_query = "MATCH (n) RETURN count(n) as node_count"
            rel_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
            
            with self.driver.session() as session:
                node_result = session.run(node_query)
                rel_result = session.run(rel_query)
                
                node_count = node_result.single()["node_count"]
                rel_count = rel_result.single()["rel_count"]
                
                return {
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "database_type": "MemGraph",
                    "connected": True
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "connected": False
            } 