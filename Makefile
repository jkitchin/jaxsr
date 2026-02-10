VENV := .venv
UV := uv
PYTHON := $(VENV)/bin/python
NOTEBOOKS := $(shell find . -name '*.ipynb' -not -path './.ipynb_checkpoints/*' -not -path './.venv/*')

.PHONY: venv install install-all test lint format check notebooks docs docs-clean clean

# --- Environment ---

venv:
	$(UV) venv $(VENV)

install: venv
	$(UV) pip install -e ".[dev]" --python $(PYTHON)

install-all: venv
	$(UV) pip install -e ".[all]" --python $(PYTHON)

# --- Quality ---

test:
	$(PYTHON) -m pytest tests/ -v --tb=short --timeout=60

lint:
	$(PYTHON) -m black --check src/ tests/
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m black src/ tests/ examples/
	$(PYTHON) -m ruff check --fix src/ tests/ examples/

check: lint test

# --- Notebooks & Docs ---

notebooks:
	@for nb in $(NOTEBOOKS); do \
		echo "Running $$nb"; \
		JAX_PLATFORMS=cpu $(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done

docs:
	JAX_PLATFORMS=cpu $(PYTHON) -m jupyter_book build docs/

docs-clean:
	$(PYTHON) -m jupyter_book clean docs/

# --- Cleanup ---

clean:
	rm -rf $(VENV) htmlcov .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
