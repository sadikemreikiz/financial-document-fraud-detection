.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check clean

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Create the environment and install git hooks
	uv sync
	uv run pre-commit install

lint:  ## Check code style and common bugs
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## Run mypy in strict mode
	uv run mypy

test:  ## Run tests with coverage (all tests, including those that need data)
	uv run pytest --cov

check: lint typecheck test  ## Everything CI runs

clean:  ## Remove caches and build outputs (never touches data/ or artifacts/)
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
