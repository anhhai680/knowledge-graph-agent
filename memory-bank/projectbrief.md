# Project Brief - Knowledge Graph Agent

**Project Name:** Knowledge Graph Agent  
**Timeline:** 2 Weeks MVP (July 19 - August 2, 2025)  
**Document Version:** 1.0  
**Date Created:** July 30, 2025  

## Project Summary

The Knowledge Graph Agent is an AI-powered system that automatically indexes GitHub repositories and provides intelligent, context-aware responses about codebases through a RAG (Retrieval-Augmented Generation) architecture. The system creates a searchable knowledge base from code, documentation, and configuration files, enabling natural language queries about complex technical information.

## Core Requirements

### Primary Goals
- **Repository Indexing**: Automatically process GitHub repositories from configuration
- **Intelligent Querying**: Enable natural language questions about codebases using RAG
- **Context-Aware Responses**: Provide detailed technical insights with proper context
- **Scalable Architecture**: Support multiple repositories and vector storage backends

### Technical Foundation
- **LangChain Framework**: Document processing, embeddings, and RAG chains
- **LangGraph Workflows**: Stateful orchestration for indexing and query processing
- **Vector Storage**: Switchable backends (Pinecone and Chroma) via configuration
- **OpenAI Integration**: Embeddings and LLM responses
- **REST API**: FastAPI with authentication middleware
- **Configuration-Driven**: Repository settings via appSettings.json

### Success Criteria
1. Successfully index configured GitHub repositories
2. Process and chunk documents with metadata enrichment
3. Store vectors in configurable backend (Pinecone/Chroma)
4. Answer natural language queries about indexed code
5. Provide structured API responses with proper authentication
6. Maintain stateful workflow processing with error recovery

## Project Scope

### MVP Features (In Scope)
- GitHub repository indexing with private access support
- LangGraph stateful workflows for indexing and querying
- Language-aware chunking for .NET (C#) and React (JS/TS)
- Metadata-enriched chunks with code symbols and context
- Dual vector storage support (Pinecone and Chroma)
- RAG query processing with OpenAI integration
- REST API with API key authentication
- Environment-based configuration with validation
- Structured logging and basic health monitoring
- Web UI chatbot interface

### Post-MVP Features (Out of Scope)
- Advanced chunking strategies (semantic, AST-based)
- Multiple LLM provider support beyond OpenAI
- Advanced authentication (OAuth, RBAC)
- Comprehensive monitoring dashboard
- Incremental indexing with Git diff detection
- Performance optimization and horizontal scaling
- GitHub Enterprise specific features

## Key Constraints

### Technical Constraints
- Python 3.11+ requirement
- OpenAI API dependency for embeddings and LLM
- GitHub API rate limits
- Vector storage provider limitations
- 2-week MVP timeline constraint

### Business Constraints
- Focus on developer productivity and code understanding
- Must support private repository access
- API-first design for future integrations
- Extensible architecture for post-MVP enhancements

## Stakeholders

### Primary Users
- Software developers and engineers
- Technical architects and team leads
- DevOps engineers and infrastructure specialists
- New team members during onboarding

### Secondary Users
- Product managers requiring technical insights
- Documentation teams needing code context

## Success Metrics

### Technical Metrics
- Repository indexing success rate > 95%
- Query response time < 10 seconds
- API availability > 99%
- Vector storage utilization efficiency

### User Experience Metrics
- Query relevance and accuracy
- Time reduction in codebase exploration
- User satisfaction with natural language responses
- Successful onboarding time reduction

This project brief serves as the foundation document that shapes all other memory bank files and guides development decisions throughout the MVP implementation.
