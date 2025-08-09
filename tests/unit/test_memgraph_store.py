"""
Unit tests for MemGraph store functionality.

This module tests the MemGraph store implementation including
connection management, query execution, and graph operations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.graphstores.memgraph_store import MemGraphStore
from src.graphstores.base_graph_store import GraphNode
from src.graphstores.base_graph_store import GraphQueryResult
from src.config.settings import settings


class TestMemGraphStore:
    """Test cases for MemGraph store functionality."""
    
    @pytest.fixture
    def mock_driver(self):
        """Mock Neo4j driver for testing."""
        mock_driver = Mock()
        mock_session = Mock()
        
        # Create a context manager mock
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=None)
        mock_driver.session.return_value = mock_context
        
        return mock_driver, mock_session
    
    @pytest.fixture
    def memgraph_store(self):
        """Create MemGraph store instance for testing."""
        return MemGraphStore(
            uri="bolt://localhost:7687",
            username="test_user",
            password="test_pass"
        )
    
    def test_init(self, memgraph_store):
        """Test MemGraph store initialization."""
        assert memgraph_store.uri == "bolt://localhost:7687"
        assert memgraph_store.username == "test_user"
        assert memgraph_store.password == "test_pass"
        assert memgraph_store.driver is None
        assert memgraph_store._connected is False
    
    @patch('src.graphstores.memgraph_store.GraphDatabase')
    def test_connect_success(self, mock_graph_database, memgraph_store, mock_driver):
        """Test successful connection to MemGraph."""
        mock_driver_instance, _ = mock_driver
        mock_graph_database.driver.return_value = mock_driver_instance
        
        result = memgraph_store.connect()
        
        assert result is True
        assert memgraph_store._connected is True
        assert memgraph_store.driver == mock_driver_instance
        # Check that driver was called with the expected parameters (including additional ones)
        mock_graph_database.driver.assert_called_once()
        call_args = mock_graph_database.driver.call_args
        assert call_args[0][0] == "bolt://localhost:7687"
        assert call_args[1]["auth"] == ("test_user", "test_pass")
        assert "connection_timeout" in call_args[1]
        assert "max_connection_lifetime" in call_args[1]
    
    @patch('src.graphstores.memgraph_store.GraphDatabase')
    def test_connect_failure(self, mock_graph_database, memgraph_store):
        """Test connection failure to MemGraph."""
        # Create a custom exception class that mimics ServiceUnavailable
        class ServiceUnavailableException(Exception):
            pass
        
        mock_graph_database.driver.side_effect = ServiceUnavailableException("Connection failed")
        
        result = memgraph_store.connect()
        
        assert result is False
        assert memgraph_store._connected is False
        assert memgraph_store.driver is None
    
    def test_disconnect(self, memgraph_store, mock_driver):
        """Test disconnection from MemGraph."""
        mock_driver_instance, _ = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        memgraph_store.disconnect()
        
        assert memgraph_store.driver is None
        assert memgraph_store._connected is False
        mock_driver_instance.close.assert_called_once()
    
    def test_is_connected_true(self, memgraph_store, mock_driver):
        """Test connection check when connected."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        mock_session.run.return_value = Mock()
        
        result = memgraph_store.is_connected()
        
        assert result is True
    
    def test_is_connected_false_no_driver(self, memgraph_store):
        """Test connection check when no driver."""
        memgraph_store.driver = None
        memgraph_store._connected = True
        
        result = memgraph_store.is_connected()
        
        assert result is False
        # The method should set _connected to False when driver is None
        assert memgraph_store._connected is False
    
    def test_is_connected_false_exception(self, memgraph_store, mock_driver):
        """Test connection check when connection fails."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        mock_session.run.side_effect = Exception("Connection failed")
        
        result = memgraph_store.is_connected()
        
        assert result is False
        assert memgraph_store._connected is False
    
    def test_execute_query_success(self, memgraph_store, mock_driver):
        """Test successful query execution with type-based counting."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        # Create mock Neo4j objects for testing type-based counting
        mock_node = Mock()
        mock_node.__class__.__name__ = 'Node'
        mock_node.id = 1
        mock_node.labels = ["Person"]
        mock_node.items.return_value = [("name", "John"), ("age", 30)]
        
        mock_relationship = Mock()
        mock_relationship.__class__.__name__ = 'Relationship'
        mock_relationship.id = 1
        mock_relationship.type = "KNOWS"
        mock_relationship.start_node.id = 1
        mock_relationship.end_node.id = 2
        mock_relationship.items.return_value = [("since", "2020")]
        
        # Mock record that contains actual Neo4j objects
        mock_record = Mock()
        mock_record.values.return_value = [mock_node, mock_relationship]
        mock_record.items.return_value = [("person", mock_node), ("relationship", mock_relationship)]
        
        mock_result = Mock()
        mock_result.__iter__ = lambda self: iter([mock_record])
        mock_session.run.return_value = mock_result
        
        # Mock the isinstance checks for our mock objects
        with patch('src.graphstores.memgraph_store.isinstance') as mock_isinstance:
            def isinstance_side_effect(obj, cls):
                if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Node':
                    return cls.__name__ == 'Node'
                elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'Relationship':
                    return cls.__name__ == 'Relationship'
                return False
            
            mock_isinstance.side_effect = isinstance_side_effect
            
            result = memgraph_store.execute_query("MATCH (n)-[r]->(m) RETURN n, r")
            
            assert isinstance(result, GraphQueryResult)
            assert result.query == "MATCH (n)-[r]->(m) RETURN n, r"
            assert len(result.data) == 1
            assert result.execution_time_ms > 0
            assert result.metadata["node_count"] == 1  # Should find 1 node
            assert result.metadata["relationship_count"] == 1  # Should find 1 relationship
            assert result.node_count == 1
            assert result.relationship_count == 1
    
    def test_execute_query_not_connected(self, memgraph_store):
        """Test query execution when not connected."""
        memgraph_store._connected = False
        
        with pytest.raises(Exception, match="Not connected to MemGraph"):
            memgraph_store.execute_query("MATCH (n) RETURN n")
    
    def test_execute_query_failure(self, memgraph_store, mock_driver):
        """Test query execution failure."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        # First call for connection check, second call for actual query
        mock_session.run.side_effect = [Mock(), Exception("Query failed")]
        
        with pytest.raises(Exception, match="Query execution failed: Query failed"):
            memgraph_store.execute_query("MATCH (n) RETURN n")
    
    def test_execute_query_type_based_counting(self, memgraph_store, mock_driver):
        """Test that node and relationship counting is based on object types, not key names."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        # Create mock objects that represent different scenarios
        mock_node1 = Mock()
        mock_node1.__class__.__name__ = 'Node'
        
        mock_node2 = Mock()
        mock_node2.__class__.__name__ = 'Node'
        
        mock_relationship = Mock()
        mock_relationship.__class__.__name__ = 'Relationship'
        
        mock_path = Mock()
        mock_path.__class__.__name__ = 'Path'
        mock_path.nodes = [mock_node1, mock_node2]  # Path contains 2 nodes
        mock_path.relationships = [mock_relationship]  # Path contains 1 relationship
        
        # Mock records with different alias names (not 'n', 'node', 'r', 'rel')
        mock_record1 = Mock()
        mock_record1.values.return_value = [mock_node1]  # Record with alias like 'person'
        mock_record1.items.return_value = [("person", mock_node1)]
        
        mock_record2 = Mock()
        mock_record2.values.return_value = [mock_relationship]  # Record with alias like 'friendship'
        mock_record2.items.return_value = [("friendship", mock_relationship)]
        
        mock_record3 = Mock()
        mock_record3.values.return_value = [mock_path]  # Record with a path
        mock_record3.items.return_value = [("path", mock_path)]
        
        mock_record4 = Mock()
        mock_record4.values.return_value = ["some_string", 42]  # Record with non-graph objects
        mock_record4.items.return_value = [("data", "some_string"), ("count", 42)]
        
        mock_result = Mock()
        mock_result.__iter__ = lambda self: iter([mock_record1, mock_record2, mock_record3, mock_record4])
        mock_session.run.return_value = mock_result
        
        # Mock both isinstance and _serialize_record to isolate the counting logic
        with patch('src.graphstores.memgraph_store.isinstance') as mock_isinstance, \
             patch('src.graphstores.memgraph_store._serialize_record') as mock_serialize:
            
            def isinstance_side_effect(obj, cls):
                if hasattr(obj, '__class__'):
                    if obj.__class__.__name__ == 'Node' and cls.__name__ == 'Node':
                        return True
                    elif obj.__class__.__name__ == 'Relationship' and cls.__name__ == 'Relationship':
                        return True
                    elif obj.__class__.__name__ == 'Path' and cls.__name__ == 'Path':
                        return True
                return False
            
            mock_isinstance.side_effect = isinstance_side_effect
            mock_serialize.return_value = {"serialized": "data"}
            
            result = memgraph_store.execute_query("MATCH path = (person)-[friendship]->(friend) RETURN person, friendship, path")
            
            # Verify that _serialize_record was called for each record
            assert mock_serialize.call_count == 4
            
            # Verify that counting is based on object types, not key names
            # Expected: 1 direct node + 2 nodes from path = 3 nodes total
            # Expected: 1 direct relationship + 1 relationship from path = 2 relationships total
            assert result.metadata["node_count"] == 3
            assert result.metadata["relationship_count"] == 2
            assert result.node_count == 3
            assert result.relationship_count == 2
            assert len(result.data) == 4  # 4 records total
    
    def test_create_node_success(self, memgraph_store, mock_driver):
        """Test successful node creation."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        mock_result = Mock()
        mock_result.single.return_value = {"node_id": 123}
        mock_session.run.return_value = mock_result
        
        node_id = memgraph_store.create_node(
            labels=["File"],
            properties={"name": "test.py", "path": "/test/test.py"}
        )
        
        assert node_id == "123"
        # Check that run was called twice: once for connection check, once for query
        assert mock_session.run.call_count == 2
        # Check the actual query call
        call_args = mock_session.run.call_args_list[1]  # Second call is the actual query
        assert "CREATE (n:File" in call_args[0][0]
        assert "name: $name" in call_args[0][0]
        assert "path: $path" in call_args[0][0]
    
    def test_create_relationship_success(self, memgraph_store, mock_driver):
        """Test successful relationship creation."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        mock_result = Mock()
        mock_result.single.return_value = {"rel_id": 456}
        mock_session.run.return_value = mock_result
        
        rel_id = memgraph_store.create_relationship(
            start_node_id="123",
            end_node_id="789",
            relationship_type="IMPORTS",
            properties={"line": 10}
        )
        
        assert rel_id == "456"
        # Check that run was called twice: once for connection check, once for query
        assert mock_session.run.call_count == 2
        # Check the actual query call
        call_args = mock_session.run.call_args_list[1]  # Second call is the actual query
        assert "CREATE (a)-[r:IMPORTS" in call_args[0][0]
        assert "line: $line" in call_args[0][0]
    
    def test_get_node_success(self, memgraph_store, mock_driver):
        """Test successful node retrieval."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        # Mock node with proper dict conversion
        mock_node = Mock()
        mock_node.id = 123
        mock_node.labels = ["File"]
        # Mock the dict conversion
        mock_node.__iter__ = lambda self: iter([("name", "test.py"), ("path", "/test/test.py")])
        mock_node.keys = lambda: ["name", "path"]
        mock_node.__getitem__ = lambda self, key: {"name": "test.py", "path": "/test/test.py"}[key]
        
        mock_result = Mock()
        mock_result.single.return_value = {"n": mock_node}
        mock_session.run.return_value = mock_result
        
        node = memgraph_store.get_node("123")
        
        assert isinstance(node, GraphNode)
        assert node.id == "123"
        assert node.labels == ["File"]
        assert node.properties["name"] == "test.py"
        assert node.properties["path"] == "/test/test.py"
    
    def test_get_node_not_found(self, memgraph_store, mock_driver):
        """Test node retrieval when node not found."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        mock_result = Mock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result
        
        node = memgraph_store.get_node("999")
        
        assert node is None
    
    def test_find_nodes_by_labels(self, memgraph_store, mock_driver):
        """Test finding nodes by labels."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        # Mock nodes with proper dict conversion
        mock_node1 = Mock()
        mock_node1.id = 1
        mock_node1.labels = ["File"]
        mock_node1.__iter__ = lambda self: iter([("name", "file1.py")])
        mock_node1.keys = lambda: ["name"]
        mock_node1.__getitem__ = lambda self, key: {"name": "file1.py"}[key]
        
        mock_node2 = Mock()
        mock_node2.id = 2
        mock_node2.labels = ["File"]
        mock_node2.__iter__ = lambda self: iter([("name", "file2.py")])
        mock_node2.keys = lambda: ["name"]
        mock_node2.__getitem__ = lambda self, key: {"name": "file2.py"}[key]
        
        mock_result = Mock()
        mock_result.__iter__ = lambda self: iter([
            {"n": mock_node1},
            {"n": mock_node2}
        ])
        mock_session.run.return_value = mock_result
        
        nodes = memgraph_store.find_nodes(labels=["File"])
        
        assert len(nodes) == 2
        assert nodes[0].id == "1"
        assert nodes[0].labels == ["File"]
        assert nodes[1].id == "2"
        assert nodes[1].labels == ["File"]
    
    def test_delete_node_success(self, memgraph_store, mock_driver):
        """Test successful node deletion."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        mock_result = Mock()
        mock_result.single.return_value = {"deleted_count": 1}
        mock_session.run.return_value = mock_result
        
        result = memgraph_store.delete_node("123")
        
        assert result is True
        # Check that run was called twice: once for connection check, once for query
        assert mock_session.run.call_count == 2
        # Check the actual query call
        call_args = mock_session.run.call_args_list[1]  # Second call is the actual query
        assert "DETACH DELETE n" in call_args[0][0]
    
    def test_delete_node_not_found(self, memgraph_store, mock_driver):
        """Test node deletion when node not found."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        mock_result = Mock()
        mock_result.single.return_value = {"deleted_count": 0}
        mock_session.run.return_value = mock_result
        
        result = memgraph_store.delete_node("999")
        
        assert result is False
    
    def test_clear_graph_success(self, memgraph_store, mock_driver):
        """Test successful graph clearing."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        mock_session.run.return_value = Mock()
        
        result = memgraph_store.clear_graph()
        
        assert result is True
        # Check that run was called twice: once for connection check, once for query
        assert mock_session.run.call_count == 2
        # Check the actual query call
        call_args = mock_session.run.call_args_list[1]  # Second call is the actual query
        assert "MATCH (n) DETACH DELETE n" in call_args[0][0]
    
    def test_get_graph_info_success(self, memgraph_store, mock_driver):
        """Test successful graph info retrieval."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        # Mock node count query
        mock_node_result = Mock()
        mock_node_result.single.return_value = {"node_count": 100}
        
        # Mock relationship count query
        mock_rel_result = Mock()
        mock_rel_result.single.return_value = {"rel_count": 250}
        
        # Set up session to return different results for different queries
        def mock_run(query):
            mock_result = Mock()
            if "count(n)" in query:
                mock_result.single.return_value = {"node_count": 100}
            else:
                mock_result.single.return_value = {"rel_count": 250}
            return mock_result
        
        mock_session.run.side_effect = mock_run
        
        info = memgraph_store.get_graph_info()
        
        assert info["connected"] is True
        assert info["node_count"] == 100
        assert info["relationship_count"] == 250
        assert info["database_type"] == "MemGraph"
    
    def test_get_graph_info_failure(self, memgraph_store, mock_driver):
        """Test graph info retrieval failure."""
        mock_driver_instance, mock_session = mock_driver
        memgraph_store.driver = mock_driver_instance
        memgraph_store._connected = True
        
        # First call for connection check, subsequent calls for info queries
        mock_session.run.side_effect = [Mock(), Exception("Database error")]
        
        info = memgraph_store.get_graph_info()
        
        assert info["connected"] is False
        assert "error" in info 