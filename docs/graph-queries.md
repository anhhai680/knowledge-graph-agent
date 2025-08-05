# Graph Query Documentation

This document provides information about the graph database features in the Knowledge Graph Agent.

## Overview

The Knowledge Graph Agent now supports graph database functionality using MemGraph. This allows for complex graph queries and relationship analysis of code repositories.

## Features

- **Graph Database**: MemGraph 2.11.0 with Bolt protocol
- **Feature Flags**: Enable/disable graph features at runtime
- **Cypher Queries**: Execute Cypher queries against the knowledge graph
- **API Integration**: REST API endpoints for graph operations
- **Backward Compatibility**: Existing vector search functionality preserved

## Configuration

### Environment Variables

Add the following to your `.env` file:

```bash
# Graph Database Configuration
ENABLE_GRAPH_FEATURES=false
GRAPH_STORE_TYPE=memgraph
GRAPH_STORE_URL=bolt://localhost:7687
GRAPH_STORE_USER=
GRAPH_STORE_PASSWORD=
GRAPH_STORE_DATABASE=memgraph
ENABLE_HYBRID_SEARCH=false
ENABLE_GRAPH_VISUALIZATION=false
```

### Docker Setup

The MemGraph service is included in `docker-compose.yml`:

```yaml
memgraph:
  image: memgraph/memgraph:2.11.0
  ports:
    - "7687:7687"  # Bolt protocol
    - "7444:7444"  # HTTP for management
  environment:
    - MEMGRAPH_LOG_LEVEL=WARNING
  volumes:
    - memgraph_data:/var/lib/memgraph
```

## API Endpoints

### Graph Query Endpoint

**POST** `/api/v1/graph/query`

Execute Cypher queries against the knowledge graph.

**Request Body:**
```json
{
  "query": "MATCH (f:File) RETURN f LIMIT 10",
  "parameters": {
    "file_path": "src/main.py"
  },
  "timeout_seconds": 30
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "data": [
      {"f": {"id": 1, "name": "main.py", "path": "src/main.py"}}
    ],
    "metadata": {
      "node_count": 1,
      "relationship_count": 0,
      "result_count": 1
    },
    "execution_time_ms": 15.2,
    "query": "MATCH (f:File) RETURN f LIMIT 10"
  },
  "error": null,
  "processing_time": 0.016,
  "query": "MATCH (f:File) RETURN f LIMIT 10"
}
```

### Graph Info Endpoint

**GET** `/api/v1/graph/info`

Get information about the graph database.

**Response:**
```json
{
  "connected": true,
  "node_count": 150,
  "relationship_count": 300,
  "database_type": "MemGraph",
  "schema_info": null,
  "performance_metrics": null
}
```

## Example Queries

### Basic File Queries

```cypher
-- Get all files
MATCH (f:File) RETURN f LIMIT 10

-- Get files by extension
MATCH (f:File) WHERE f.extension = '.py' RETURN f

-- Get files in specific directory
MATCH (f:File) WHERE f.directory = 'src' RETURN f
```

### Relationship Queries

```cypher
-- Find files that import other files
MATCH (f1:File)-[:IMPORTS]->(f2:File) RETURN f1, f2

-- Find dependencies between files
MATCH (f1:File)-[:DEPENDS_ON]->(f2:File) RETURN f1, f2

-- Find classes in files
MATCH (f:File)-[:CONTAINS]->(c:Class) RETURN f, c
```

### Complex Queries

```cypher
-- Find files with most dependencies
MATCH (f:File)-[:DEPENDS_ON]->(other:File)
RETURN f.path, count(other) as dependency_count
ORDER BY dependency_count DESC
LIMIT 10

-- Find circular dependencies
MATCH path = (f:File)-[:DEPENDS_ON*]->(f)
RETURN f.path, length(path) as cycle_length
ORDER BY cycle_length
```

## Feature Flags

Graph features can be controlled using feature flags:

- `ENABLE_GRAPH_FEATURES`: Enable/disable all graph functionality
- `ENABLE_HYBRID_SEARCH`: Enable hybrid search (vector + graph)
- `ENABLE_GRAPH_VISUALIZATION`: Enable graph visualization features

## Error Handling

When graph features are disabled, API endpoints return:

```json
{
  "detail": "Graph features are not enabled"
}
```

## Performance Considerations

- Graph queries should complete within 3 seconds
- Use appropriate LIMIT clauses for large result sets
- Consider indexing frequently queried properties
- Monitor query execution times using the `execution_time_ms` field

## Security

- Graph queries are validated to ensure they start with allowed keywords
- Query parameters are properly sanitized
- Access control can be implemented through feature flags

## Troubleshooting

### Connection Issues

1. Ensure MemGraph container is running:
   ```bash
   docker-compose ps memgraph
   ```

2. Check MemGraph logs:
   ```bash
   docker-compose logs memgraph
   ```

3. Verify connection settings in `.env` file

### Query Issues

1. Ensure queries start with allowed keywords: `MATCH`, `CREATE`, `MERGE`, `RETURN`, `WITH`
2. Check query syntax using MemGraph's built-in validation
3. Use appropriate timeout settings for complex queries

## Future Enhancements

- Graph visualization interface
- Hybrid search combining vector and graph results
- Advanced relationship analysis
- Graph-based code recommendations
- Performance optimization and caching 