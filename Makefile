.PHONY: help test test-unit test-integration coverage lint format type-check security clean install install-dev setup pre-commit

# Default target
help:
	@echo "Available commands:"
	@echo "  setup           - Initial setup: install deps and pre-commit hooks"
	@echo "  install         - Install production dependencies"
	@echo "  install-dev     - Install development dependencies"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  coverage        - Run tests with coverage report"
	@echo "  lint            - Run all linting checks"
	@echo "  format          - Format code with black and isort"
	@echo "  type-check      - Run mypy type checking"
	@echo "  security        - Run security checks"
	@echo "  pre-commit      - Install and run pre-commit hooks"
	@echo "  clean           - Clean up generated files"

# Setup
setup: install-dev pre-commit
	@echo "✅ Setup complete!"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# Testing
test: test-unit test-integration

test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v --tb=short

test-integration:
	@echo "🔗 Running integration tests..."
	pytest tests/integration/ -v --tb=short

coverage:
	@echo "📊 Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Code Quality
lint: format type-check security
	@echo "🔍 Running flake8..."
	flake8 src/ tests/ main.py
	@echo "📚 Checking docstrings..."
	pydocstyle src/
	@echo "✅ All linting checks passed!"

format:
	@echo "🎨 Formatting code..."
	black src/ tests/ main.py
	isort src/ tests/ main.py
	@echo "✅ Code formatted!"

type-check:
	@echo "🔎 Running type checks..."
	mypy src/ --ignore-missing-imports --disallow-untyped-defs --warn-return-any

security:
	@echo "🔒 Running security checks..."
	safety check -r requirements.txt -r requirements-dev.txt
	bandit -r src/ -ll

# Pre-commit
pre-commit:
	@echo "🪝 Installing pre-commit hooks..."
	pre-commit install
	pre-commit run --all-files

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "coverage.xml" -delete
	@echo "✅ Cleanup complete!"

# CI/CD simulation
ci: clean install-dev lint coverage
	@echo "🚀 CI pipeline simulation complete!"
