"""
Graph schema definitions for the Knowledge Graph Agent.

This module defines the Cypher schema and constraint definitions for the
graph database, including node labels, relationship types, and constraints.
"""

from typing import List, Dict, Any


class GraphSchemas:
    """Graph schema definitions and utilities."""
    
    # Node labels
    FILE = "File"
    CLASS = "Class"
    METHOD = "Method"
    FUNCTION = "Function"
    MODULE = "Module"
    PACKAGE = "Package"
    
    # Relationship types
    IMPORTS = "IMPORTS"
    DEPENDS_ON = "DEPENDS_ON"
    CONTAINS = "CONTAINS"
    HAS_METHOD = "HAS_METHOD"
    CALLS = "CALLS"
    EXTENDS = "EXTENDS"
    IMPLEMENTS = "IMPLEMENTS"
    
    @classmethod
    def get_constraints(cls) -> List[str]:
        """
        Get Cypher constraint definitions for the graph schema.
        
        Returns:
            List[str]: List of Cypher constraint statements
        """
        return [
            # File constraints
            "CREATE CONSTRAINT file_id_unique IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT file_path_unique IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE",
            
            # Class constraints
            "CREATE CONSTRAINT class_id_unique IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT class_name_unique IF NOT EXISTS FOR (c:Class) REQUIRE c.name IS UNIQUE",
            
            # Method constraints
            "CREATE CONSTRAINT method_id_unique IF NOT EXISTS FOR (m:Method) REQUIRE m.id IS UNIQUE",
            
            # Function constraints
            "CREATE CONSTRAINT function_id_unique IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE",
            
            # Module constraints
            "CREATE CONSTRAINT module_id_unique IF NOT EXISTS FOR (m:Module) REQUIRE m.id IS UNIQUE",
            
            # Package constraints
            "CREATE CONSTRAINT package_id_unique IF NOT EXISTS FOR (p:Package) REQUIRE p.id IS UNIQUE",
        ]
    
    @classmethod
    def get_indexes(cls) -> List[str]:
        """
        Get Cypher index definitions for the graph schema.
        
        Returns:
            List[str]: List of Cypher index statements
        """
        return [
            # File indexes
            "CREATE INDEX file_path_index IF NOT EXISTS FOR (f:File) ON (f.path)",
            "CREATE INDEX file_extension_index IF NOT EXISTS FOR (f:File) ON (f.extension)",
            
            # Class indexes
            "CREATE INDEX class_name_index IF NOT EXISTS FOR (c:Class) ON (c.name)",
            "CREATE INDEX class_file_index IF NOT EXISTS FOR (c:Class) ON (c.file_path)",
            
            # Method indexes
            "CREATE INDEX method_name_index IF NOT EXISTS FOR (m:Method) ON (m.name)",
            "CREATE INDEX method_class_index IF NOT EXISTS FOR (m:Method) ON (m.class_name)",
            
            # Function indexes
            "CREATE INDEX function_name_index IF NOT EXISTS FOR (f:Function) ON (f.name)",
            "CREATE INDEX function_file_index IF NOT EXISTS FOR (f:Function) ON (f.file_path)",
        ]
    
    @classmethod
    def get_schema_setup_queries(cls) -> List[str]:
        """
        Get all schema setup queries (constraints and indexes).
        
        Returns:
            List[str]: List of all schema setup queries
        """
        return cls.get_constraints() + cls.get_indexes()
    
    @classmethod
    def get_file_node_properties(cls, file_path: str, content: str = "", metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get properties for a File node.
        
        Args:
            file_path: Path to the file
            content: File content (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Dict[str, Any]: File node properties
        """
        import os
        
        metadata = metadata or {}
        
        return {
            "id": file_path,
            "path": file_path,
            "name": os.path.basename(file_path),
            "extension": os.path.splitext(file_path)[1],
            "directory": os.path.dirname(file_path),
            "content": content,
            "size": len(content),
            "created_at": metadata.get("created_at"),
            "modified_at": metadata.get("modified_at"),
            "language": metadata.get("language"),
            **metadata
        }
    
    @classmethod
    def get_class_node_properties(cls, class_name: str, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get properties for a Class node.
        
        Args:
            class_name: Name of the class
            file_path: Path to the file containing the class
            metadata: Additional metadata (optional)
            
        Returns:
            Dict[str, Any]: Class node properties
        """
        metadata = metadata or {}
        
        return {
            "id": f"{file_path}:{class_name}",
            "name": class_name,
            "file_path": file_path,
            "full_name": f"{file_path}:{class_name}",
            "visibility": metadata.get("visibility", "public"),
            "is_abstract": metadata.get("is_abstract", False),
            "is_final": metadata.get("is_final", False),
            "line_number": metadata.get("line_number"),
            **metadata
        }
    
    @classmethod
    def get_method_node_properties(cls, method_name: str, class_name: str, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get properties for a Method node.
        
        Args:
            method_name: Name of the method
            class_name: Name of the class containing the method
            file_path: Path to the file containing the method
            metadata: Additional metadata (optional)
            
        Returns:
            Dict[str, Any]: Method node properties
        """
        metadata = metadata or {}
        
        return {
            "id": f"{file_path}:{class_name}:{method_name}",
            "name": method_name,
            "class_name": class_name,
            "file_path": file_path,
            "full_name": f"{file_path}:{class_name}:{method_name}",
            "visibility": metadata.get("visibility", "public"),
            "is_static": metadata.get("is_static", False),
            "is_abstract": metadata.get("is_abstract", False),
            "return_type": metadata.get("return_type"),
            "parameters": metadata.get("parameters", []),
            "line_number": metadata.get("line_number"),
            **metadata
        }
    
    @classmethod
    def get_function_node_properties(cls, function_name: str, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get properties for a Function node.
        
        Args:
            function_name: Name of the function
            file_path: Path to the file containing the function
            metadata: Additional metadata (optional)
            
        Returns:
            Dict[str, Any]: Function node properties
        """
        metadata = metadata or {}
        
        return {
            "id": f"{file_path}:{function_name}",
            "name": function_name,
            "file_path": file_path,
            "full_name": f"{file_path}:{function_name}",
            "visibility": metadata.get("visibility", "public"),
            "return_type": metadata.get("return_type"),
            "parameters": metadata.get("parameters", []),
            "line_number": metadata.get("line_number"),
            **metadata
        }
    
    @classmethod
    def get_import_relationship_properties(cls, import_statement: str = "", line_number: int = None) -> Dict[str, Any]:
        """
        Get properties for an IMPORTS relationship.
        
        Args:
            import_statement: The import statement text
            line_number: Line number of the import
            
        Returns:
            Dict[str, Any]: Import relationship properties
        """
        return {
            "import_statement": import_statement,
            "line_number": line_number,
            "type": "import"
        }
    
    @classmethod
    def get_dependency_relationship_properties(cls, dependency_type: str = "unknown", strength: str = "medium") -> Dict[str, Any]:
        """
        Get properties for a DEPENDS_ON relationship.
        
        Args:
            dependency_type: Type of dependency
            strength: Strength of the dependency
            
        Returns:
            Dict[str, Any]: Dependency relationship properties
        """
        return {
            "type": dependency_type,
            "strength": strength
        }
    
    @classmethod
    def get_contains_relationship_properties(cls, line_number: int = None) -> Dict[str, Any]:
        """
        Get properties for a CONTAINS relationship.
        
        Args:
            line_number: Line number where the containment occurs
            
        Returns:
            Dict[str, Any]: Contains relationship properties
        """
        return {
            "line_number": line_number,
            "type": "contains"
        } 