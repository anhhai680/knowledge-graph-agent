# GitHub Copilot Instructions for Knowledge Graph Agent

## Repository Overview

The **Knowledge Graph Agent** is an AI-powered system that automatically indexes GitHub repositories and provides intelligent, context-aware responses about codebases through a RAG (Retrieval-Augmented Generation) architecture. This is a Python 3.11+ FastAPI application using LangChain/LangGraph workflows for orchestration.

**Key Technologies:**
- **Framework:** FastAPI with async/await patterns
- **AI/ML:** LangChain, LangGraph workflows, OpenAI GPT models
- **Vector Storage:** Dual support for Chroma DB and Pinecone (configurable)
- **Graph Database:** MemGraph (for relationship modeling)
- **Development:** Python 3.11-3.12, pytest, Docker Compose
- **Code Quality:** Black, isort, flake8, mypy, pre-commit hooks

**Repository Size:** ~87 Python files, ~15K lines of code

## Build and Development Setup

### Environment Requirements
- **Python Version:** 3.11 or 3.12 (use `python --version` to check)
- **Docker:** Required for services (Chroma, MemGraph, web interface)
- **Environment Variables:** Copy `.env.example` to `.env` and configure

### Critical Setup Sequence

**ALWAYS follow this exact order for initial setup:**

```bash
# 1. Install dependencies (required before any other commands)
make install-dev

# 2. Setup pre-commit hooks (prevents commit failures)
make pre-commit

# 3. Copy and configure environment
cp .env.example .env
# Edit .env to add required API keys (OPENAI_API_KEY, GITHUB_TOKEN)
```

### Running Tests

**CRITICAL:** Tests require PYTHONPATH to be set explicitly:

```bash
# Unit tests (fast, no external dependencies)
PYTHONPATH=. pytest tests/unit/ -v

# Integration tests (require services to be running)
make docker-up  # Start services first
PYTHONPATH=. pytest tests/integration/ -v

# All tests with coverage
PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=term-missing
```

**Common Test Failure:** If you see `ModuleNotFoundError: No module named 'src'`, you forgot to set `PYTHONPATH=.`

### Code Quality and Formatting

**ALWAYS run before committing changes:**

```bash
# Format code (auto-fixes most issues)
make format

# Run all quality checks
make lint

# Type checking
make type-check
```

**Note:** The `make format` command will modify many files. This is expected behavior - the codebase uses strict Black formatting with 88-character line length.

### Docker Services

**Development workflow with Docker:**

```bash
# Start all services (API, Chroma, MemGraph, Web UI)
make docker-up

# View logs from all services
make docker-logs

# Stop all services
make docker-down

# Alternative: Use development script for faster iteration
./web/dev.sh start-backend  # Start only backend services
./web/dev.sh start-web      # Start web UI with live reload
```

**Service Ports:**
- Main API: http://localhost:8000
- Chroma DB: http://localhost:8001
- MemGraph: bolt://localhost:7687, HTTP: http://localhost:7444
- Web UI: http://localhost:3000

### Running the Application

```bash
# Method 1: Direct Python execution
python main.py

# Method 2: Using Docker (recommended for development)
make docker-up

# Method 3: Development server with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Project Architecture

### Directory Structure

```
src/
├── api/           # FastAPI routes and request/response models
├── agents/        # RAG agent implementations
├── config/        # Environment settings and validation
├── workflows/     # LangGraph stateful workflows
│   ├── query/     # Query processing handlers
│   └── indexing_workflow.py
├── loaders/       # GitHub repository loading and processing
├── vectorstores/  # Chroma and Pinecone implementations
├── graphstores/   # MemGraph integration
├── llm/           # OpenAI and embedding providers
├── processors/    # Document chunking and processing
└── utils/         # Logging, prompt management, utilities

tests/
├── unit/          # Fast unit tests (no external deps)
├── integration/   # Tests requiring external services
└── workflows/     # Workflow-specific test suites
```

### Key Configuration Files

- `pyproject.toml` - Python package configuration, dependencies
- `requirements.txt` - Production dependencies
- `requirements-dev.txt` - Development dependencies (includes production)
- `Makefile` - All build, test, and quality commands
- `pytest.ini` - Test configuration with coverage settings
- `setup.cfg` - flake8, mypy, and pydocstyle configuration
- `docker-compose.yml` - Multi-service Docker setup
- `appSettings.json` - Repository configuration for indexing

### Environment Variables

**Required for basic functionality:**
- `OPENAI_API_KEY` - OpenAI API access
- `GITHUB_TOKEN` - GitHub repository access

**Important configurations:**
- `DATABASE_TYPE` - Switch between "chroma" and "pinecone"
- `APP_ENV` - "development" or "production"
- `LOG_LEVEL` - "DEBUG", "INFO", "WARNING", "ERROR"
- `WORKFLOW_STATE_BACKEND` - "memory" or "database"

## CI/CD Pipeline

### GitHub Actions Workflows

**CI Pipeline (`.github/workflows/ci.yml`):**
- Runs on: Push to main, Pull Requests
- Python versions: 3.11, 3.12
- Steps: Dependencies → Unit Tests → Coverage Report
- **Critical:** Requires `OPENAI_API_KEY` and `GITHUB_TOKEN` in repository secrets

**Release Pipeline (`.github/workflows/release.yml`):**
- Trigger: Git tags starting with 'v' (e.g., v1.0.0)
- Full test suite → Code quality checks → Build artifacts

### Pre-commit Hooks

**Configured tools:**
- Black (code formatting)
- isort (import sorting)
- flake8 (linting)
- trailing-whitespace removal
- end-of-file-fixer

**Setup required:** Run `make pre-commit` after initial setup

## Common Issues and Workarounds

### Import Errors in Tests
**Problem:** `ModuleNotFoundError: No module named 'src'`
**Solution:** Always use `PYTHONPATH=. pytest` instead of just `pytest`

### Docker Compose Issues
**Problem:** Services fail to start
**Solutions:**
- Use `docker compose` (not `docker-compose`) - modern syntax
- Check ports 8000, 8001, 7687, 7444 are not in use
- Run `make docker-down` then `make docker-up` to restart

### Code Formatting Conflicts
**Problem:** Pre-commit hooks fail
**Solution:** Run `make format` first, then commit. The project uses very strict formatting.

### Memory Issues with Large Repositories
**Problem:** Indexing workflows timeout or fail
**Solution:** Adjust environment variables:
- `WORKFLOW_TIMEOUT_SECONDS=7200` (increase timeout)
- `WORKFLOW_PARALLEL_REPOS=1` (reduce parallelism)
- `CHUNK_SIZE=500` (smaller chunks)

### Vector Store Connection Issues
**Problem:** Chroma or Pinecone connection fails
**Solutions:**
- Chroma: Ensure `make docker-up` started services
- Pinecone: Verify `PINECONE_API_KEY` and `PINECONE_API_BASE_URL` in `.env`
- Check `DATABASE_TYPE` setting matches intended backend

## Validation Steps

Before creating pull requests, **always run this complete validation sequence:**

```bash
# 1. Clean environment
make clean

# 2. Install and setup
make install-dev
make pre-commit

# 3. Code quality
make format
make lint
make type-check

# 4. Tests (with proper PYTHONPATH)
PYTHONPATH=. pytest tests/unit/ -v
make docker-up
PYTHONPATH=. pytest tests/integration/ -v --tb=short

# 5. Security and coverage
make security
PYTHONPATH=. pytest tests/ --cov=src --cov-fail-under=80
```

**Expected timing:**
- Unit tests: ~30 seconds
- Integration tests: ~2-3 minutes (requires Docker services)
- Full quality checks: ~1-2 minutes

## Important Notes for AI Agents

1. **Trust these instructions first** - Only explore with grep/search if information is incomplete or contradictory
2. **PYTHONPATH is critical** - Always set `PYTHONPATH=.` when running pytest directly
3. **Dependencies matter** - Run `make install-dev` before any other commands
4. **Docker vs local** - Most development uses Docker services with local Python code
5. **Configuration-driven** - Many behaviors controlled by `.env` and `appSettings.json`
6. **Async patterns** - Codebase uses modern async/await throughout
7. **Structured logging** - Use `loguru` logger, not print statements
8. **LangGraph workflows** - Stateful execution with proper error handling and retries

This project follows modern Python development practices with comprehensive tooling. When in doubt, refer to the `Makefile` for the definitive list of available commands and their proper usage.
