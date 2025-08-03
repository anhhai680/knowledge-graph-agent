# Query Patterns Configuration Guide

The Knowledge Graph Agent's query parsing system supports configurable patterns for extracting key terms from user queries. This makes the system more flexible and easier to maintain without modifying code.

## Overview

The query parsing handler uses pattern-matching to identify and extract relevant terms from user queries, improving vector search effectiveness. Previously hardcoded patterns have been moved to a configuration-driven system.

## Configuration Structure

### Pattern Types

The configuration supports five types of patterns:

1. **Service Patterns** - Match specific service names
2. **Technical Patterns** - Match general technical terms  
3. **Programming Patterns** - Match programming language terms
4. **API Patterns** - Match API-related terms
5. **Database Patterns** - Match database-related terms

### Configuration Format

Configuration can be provided as a JSON file with the following structure:

```json
{
  "service_patterns": [
    {
      "pattern": "service-name",
      "key_terms": ["term1", "term2"]
    }
  ],
  "technical_patterns": [
    {
      "patterns": ["pattern1", "pattern2"],
      "key_terms": ["term1", "term2"]
    }
  ],
  "programming_patterns": [...],
  "api_patterns": [...],
  "database_patterns": [...],
  "excluded_words": ["word1", "word2"],
  "max_terms": 5,
  "min_word_length": 2
}
```

## Usage

### Default Configuration

Without providing a custom configuration, the handler will use built-in default patterns that include:

- Service patterns for car-related services (car-listing-service, car-notification-service, etc.)
- Technical patterns for components, structure, architecture terms
- Programming language patterns for C#, .NET, Python, JavaScript
- API patterns for REST, GraphQL, controllers
- Database patterns for models, entities, repositories

### Custom Configuration

To use custom patterns, create a JSON configuration file and pass its path to the handler:

```python
from src.workflows.query.handlers.query_parsing_handler import QueryParsingHandler

# Use custom configuration
handler = QueryParsingHandler(query_patterns_config="/path/to/config.json")

# Use default configuration  
handler = QueryParsingHandler()
```

### Example Custom Configuration

```json
{
  "service_patterns": [
    {
      "pattern": "user-management-service",
      "key_terms": ["user", "auth", "account"]
    },
    {
      "pattern": "payment-service", 
      "key_terms": ["payment", "billing", "transaction"]
    }
  ],
  "technical_patterns": [
    {
      "patterns": ["microservice", "service"],
      "key_terms": ["service", "api"]
    },
    {
      "patterns": ["container", "docker"],
      "key_terms": ["docker", "container", "deployment"]
    }
  ],
  "programming_patterns": [
    {
      "patterns": ["python", "py"],
      "key_terms": ["function", "class", "module"]
    },
    {
      "patterns": ["go", "golang"],
      "key_terms": ["func", "struct", "package"]
    }
  ],
  "api_patterns": [
    {
      "patterns": ["graphql", "gql"],
      "key_terms": ["resolver", "schema", "query", "mutation"]
    },
    {
      "patterns": ["grpc"],
      "key_terms": ["service", "proto", "rpc"]
    }
  ],
  "database_patterns": [
    {
      "patterns": ["nosql", "mongodb"],
      "key_terms": ["collection", "document", "query"]
    },
    {
      "patterns": ["sql", "postgres", "mysql"],
      "key_terms": ["table", "column", "index", "query"]
    }
  ],
  "excluded_words": [
    "the", "and", "or", "for", "with", "this", "that", "what",
    "how", "why", "when", "where", "can", "will", "should"
  ],
  "max_terms": 5,
  "min_word_length": 2
}
```

## Pattern Matching Logic

### Priority Order

1. **Service Patterns** - First match wins, extracts specific service terms
2. **Technical Patterns** - All matching patterns contribute terms
3. **Programming Patterns** - All matching patterns contribute terms
4. **API Patterns** - All matching patterns contribute terms  
5. **Database Patterns** - All matching patterns contribute terms
6. **General Terms** - Fallback extraction if no patterns match

### Term Processing

1. **Duplicate Removal** - Duplicate terms are removed while preserving order
2. **Max Terms Limit** - Results are limited to `max_terms` (default: 5)
3. **Word Length Filter** - General terms must meet `min_word_length` (default: 2)
4. **Exclusion Filter** - Terms in `excluded_words` are filtered out

## Benefits

### Maintainability
- Add new patterns without code changes
- Easy to update existing patterns
- Clear separation of configuration from logic

### Flexibility  
- Different configurations for different domains
- Environment-specific patterns
- A/B testing of different extraction strategies

### Extensibility
- Easy to add new pattern types
- Support for complex matching rules
- Integration with external configuration systems

## Migration from Hardcoded Patterns

The new system maintains backward compatibility by providing default patterns that match the previous hardcoded behavior. Existing functionality continues to work without changes.

### Before (Hardcoded)
```python
if "car-listing-service" in query_lower:
    key_terms.append("car")
    key_terms.append("listing")
```

### After (Configuration-Driven)
```python
for service_pattern in self.patterns_config.service_patterns:
    if service_pattern.pattern in query_lower:
        key_terms.extend(service_pattern.key_terms)
        break
```

## Testing

The configuration system includes comprehensive test coverage:

- Default configuration validation
- Custom configuration loading
- Pattern matching accuracy
- Error handling for invalid configurations
- Performance with large pattern sets

See `tests/unit/test_query_patterns_config.py` and updated `tests/workflows/query/test_query_parsing_handler.py` for detailed test examples.

## Performance Considerations

- Patterns are loaded once at handler initialization
- Pattern matching uses efficient string operations
- Large pattern sets may impact query processing time
- Consider caching for frequently accessed configurations

## Future Enhancements

Possible future improvements include:

- Regular expression patterns
- Weighted term extraction
- Context-aware pattern selection
- Machine learning-based pattern discovery
- Integration with external knowledge bases
