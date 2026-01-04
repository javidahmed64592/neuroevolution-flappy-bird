# GitHub Workflows

This document details the CI/CD workflows to build and release the Flappy Bird neuroevolution simulation.
They run automated code quality checks to ensure code remains robust, maintainable, and testable.

## CI Workflow

The CI workflow runs on pushes and pull requests to the `main` branch.
It consists of the following jobs:

### validate-pyproject
- Checkout code
- Setup Python environment with dev dependencies (via custom action)
- Validate `pyproject.toml` structure using `validate-pyproject`

### ruff
- Checkout code
- Setup Python environment with dev dependencies (via custom action)
- Run Ruff linter with `uv run -m ruff check`

### mypy
- Checkout code
- Setup Python environment with dev dependencies (via custom action)
- Run mypy type checking with `uv run -m mypy .`

### pytest
- Checkout code
- Setup Python environment with dev dependencies (via custom action)
- Run pytest with coverage (HTML and terminal reports) using `uv run -m pytest --cov-report html --cov-report term`
- Fails if coverage drops below 80% (configured in `pyproject.toml`)
- Upload HTML coverage report as artifact

### bandit
- Checkout code
- Setup Python environment with dev dependencies (via custom action)
- Run security scanning with bandit on `example/` directory
- Generate JSON report for artifacts
- Fail if security vulnerabilities are found

### pip-audit
- Checkout code
- Setup Python environment with dev dependencies (via custom action)
- Audit dependencies for known CVEs using `pip-audit --desc`

### version-check
- Checkout code
- Setup Python environment with dev dependencies (via custom action)
- Check version consistency across `pyproject.toml` and `uv.lock`

## Build Workflow

The Build workflow runs on pushes and pull requests to the `main` branch.
It consists of the following jobs:

### build-wheel
- Checkout code
- Setup Python environment with dev dependencies (via custom action)
- Build wheel with `uv build`
- Inspect wheel contents for verification
- Upload wheel artifact (`template_python_wheel`)

### verify-structure
- Depends on `build-wheel` job
- Checkout code
- Setup Python environment (via custom action)
- Download wheel artifact
- Install wheel using `uv pip install`
- Verify installed package structure in site-packages
