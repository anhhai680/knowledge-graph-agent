# CI/CD Pipeline Documentation

This document explains the Continuous Integration and Continuous Deployment (CI/CD) setup for the Knowledge Graph Agent project.

## Overview

The CI/CD pipeline ensures code quality, runs comprehensive tests, and maintains security standards for every change to the codebase. The pipeline is triggered on:

- **Push** to `main` or `develop` branches
- **Pull Requests** to `main` or `develop` branches
- **Daily security scans** (scheduled)
- **Release tags** (for deployment)

## Pipeline Components

### 1. Main CI Pipeline (`.github/workflows/ci.yml`)

#### Test Job
- **Multi-Python Support**: Tests run on Python 3.11 and 3.12
- **Unit Tests**: Located in `tests/unit/` with coverage requirements (80% minimum)
- **Integration Tests**: Located in `tests/integration/` for end-to-end testing
- **Coverage Reporting**: Uploads coverage reports to Codecov
- **Dependency Caching**: Speeds up builds by caching pip dependencies

#### Code Quality Job
- **Black**: Code formatting check (88 character line limit)
- **isort**: Import sorting validation
- **Flake8**: PEP 8 compliance and style checking
- **Pydocstyle**: Docstring compliance (PEP 257)
- **MyPy**: Static type checking

#### Security Job
- **Safety**: Scans for known security vulnerabilities in dependencies
- **Bandit**: Static security analysis for common security issues

#### Auto-formatting Job (PR only)
- **Automatic Fixes**: Runs Black and isort to auto-format code
- **Auto-commit**: Commits formatting changes back to the PR branch

### 2. Security Pipeline (`.github/workflows/security.yml`)

- **Daily Scans**: Automated security scanning every day at 2 AM UTC
- **Dependency Monitoring**: Triggered when requirements files change
- **Multiple Tools**: Safety, Bandit, and Semgrep for comprehensive coverage
- **Dependency Review**: GitHub's native dependency review for PRs

### 3. Release Pipeline (`.github/workflows/release.yml`)

- **Triggered**: On version tags (e.g., `v1.0.0`)
- **Full Testing**: Complete test suite before release
- **Package Building**: Creates distributable packages
- **Docker Images**: Multi-architecture Docker image builds
- **GitHub Releases**: Automatic release creation

## Local Development Setup

### Prerequisites

```bash
# Install Python 3.11 or 3.12
python --version

# Install project dependencies
make install-dev
# or
pip install -r requirements-dev.txt
```

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit to catch issues early:

```bash
# Install pre-commit hooks
make pre-commit
# or
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files
```

### Running Tests Locally

#### Using Make Commands
```bash
# Run all tests
make test

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration

# Run tests with coverage
make coverage

# Run all quality checks
make lint

# Format code
make format

# Type checking
make type-check

# Security scanning
make security

# Complete CI simulation
make ci
```

#### Using the Test Runner Script
```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --type unit

# Run with coverage
python run_tests.py --coverage

# Run in parallel
python run_tests.py --parallel 4

# Verbose output with fail-fast
python run_tests.py -v -x
```

#### Direct pytest Commands
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Parallel execution
pytest tests/ -n 4
```

## Code Quality Standards

### Code Formatting
- **Black**: Line length of 88 characters
- **isort**: Import sorting with Black profile
- **Trailing whitespace**: Automatically removed

### Type Checking
- **MyPy**: Strict type checking enabled
- **Type hints**: Required for all function definitions
- **Return types**: Must be specified

### Documentation
- **Docstrings**: Required for all public functions and classes
- **PEP 257**: Google-style docstrings preferred
- **Comments**: Explain complex logic and design decisions

### Testing Requirements
- **Coverage**: Minimum 80% code coverage
- **Test Types**: Both unit and integration tests required
- **Async Support**: pytest-asyncio for async test support
- **Mocking**: pytest-mock for dependency mocking

## Security Standards

### Dependency Scanning
- **Safety**: Checks for known vulnerabilities in Python packages
- **Dependency Review**: GitHub native scanning for PRs
- **Regular Updates**: Daily automated scans

### Static Analysis
- **Bandit**: Identifies common security issues in Python code
- **Semgrep**: Advanced static analysis patterns
- **Secret Scanning**: GitHub secret scanning enabled

### Security Best Practices
- **No hardcoded secrets**: Use environment variables
- **Input validation**: Validate all external inputs
- **Error handling**: Don't expose sensitive information in errors

## Configuration Files

### `.pre-commit-config.yaml`
Defines pre-commit hooks for local development quality checks.

### `pyproject.toml`
Modern Python project configuration including tool settings for Black, isort, MyPy, and pytest.

### `setup.cfg`
Legacy configuration file for tools that don't support pyproject.toml.

### `pytest.ini`
Pytest-specific configuration for test discovery and execution.

### `Makefile`
Convenient commands for common development tasks.

## Troubleshooting

### Common Issues

#### Test Failures
```bash
# Run specific test
pytest tests/unit/test_workflows.py::TestWorkflowMetadata::test_metadata_initialization -v

# Debug with pdb
pytest --pdb tests/unit/test_workflows.py

# Show all output
pytest -s tests/unit/test_workflows.py
```

#### Code Quality Issues
```bash
# Fix formatting automatically
black src/ tests/ main.py
isort src/ tests/ main.py

# Check specific flake8 issues
flake8 src/workflows/base_workflow.py --show-source

# Type checking specific file
mypy src/workflows/base_workflow.py
```

#### Pre-commit Hook Failures
```bash
# Skip hooks for emergency commits (not recommended)
git commit --no-verify -m "Emergency fix"

# Update hooks
pre-commit autoupdate

# Clean cache
pre-commit clean
```

### CI/CD Issues

#### Failed GitHub Actions
1. Check the specific job that failed in the GitHub Actions tab
2. Review the logs for detailed error messages
3. Run the same commands locally to reproduce
4. Fix issues and push again

#### Coverage Below Threshold
1. Run `make coverage` locally to see missing coverage
2. Add tests for uncovered code paths
3. Update coverage thresholds if appropriate

## Best Practices

### Commit Messages
- Use conventional commit format: `feat:`, `fix:`, `docs:`, `test:`
- Keep first line under 50 characters
- Provide detailed description if needed

### Pull Request Workflow
1. Create feature branch from `develop`
2. Make changes with tests
3. Run local quality checks: `make lint`
4. Push branch and create PR
5. Address any CI/CD feedback
6. Merge after approval and passing checks

### Release Process
1. Update version in relevant files
2. Update CHANGELOG.md
3. Create and push version tag: `git tag v1.0.0`
4. GitHub Actions will handle the rest

## Monitoring and Alerts

### GitHub Actions Notifications
- Failed builds notify repository maintainers
- Security scan results are uploaded as artifacts
- Coverage reports are available in PR comments

### Dependency Monitoring
- Daily security scans catch new vulnerabilities
- Dependabot creates PRs for dependency updates
- Manual review required for major version updates

This comprehensive CI/CD setup ensures high code quality, security, and reliability for the Knowledge Graph Agent project.
