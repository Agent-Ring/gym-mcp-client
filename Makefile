.PHONY: help install test quickstart clean lint format typecheck check all

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies with uv"
	@echo "  make test       - Run test suite with pytest"
	@echo "  make lint       - Run ruff linter (check only)"
	@echo "  make format     - Format code with ruff"
	@echo "  make typecheck  - Run mypy type checker"
	@echo "  make check      - Run all checks (lint + typecheck + test)"
	@echo "  make all        - Format, then run all checks"
	@echo "  make quickstart - Run quickstart example (with remote server)"
	@echo "  make clean      - Clean build artifacts and cache"

install:
	@echo "Installing dependencies with uv..."
	uv sync
	uv pip install -e .

test:
	@echo "Running tests with pytest..."
	uv run pytest tests/ -v --tb=short

lint:
	@echo "Running ruff linter..."
	uv run ruff check agentring/ tests/

format:
	@echo "Formatting code with ruff..."
	uv run ruff format agentring/ tests/
	@echo "Auto-fixing linting issues..."
	uv run ruff check --fix agentring/ tests/

typecheck:
	@echo "Running mypy type checker..."
	uv run mypy agentring/ --strict --pretty

check: lint typecheck test
	@echo ""
	@echo "✅ All checks passed!"

all: format check
	@echo ""
	@echo "✅ Format and all checks completed!"

quickstart:
	@echo "Running quickstart example..."
	uv run python quickstart.py

clean:
	@echo "Cleaning build artifacts and cache..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	@echo "Clean complete!"

