VENV := .venv
UV := uv
PYTHON := $(VENV)/bin/python
NOTEBOOKS := $(shell find ./docs/examples -name '*.ipynb' -not -name 'gpu_benchmarks.ipynb' -not -path './.ipynb_checkpoints/*' -not -path './docs/_build/*')
GPU_NOTEBOOK := docs/examples/gpu_benchmarks.ipynb
NB_STAMP_DIR := .nb_stamps
NB_STAMPS := $(patsubst %.ipynb,$(NB_STAMP_DIR)/%.stamp,$(NOTEBOOKS))

.PHONY: venv install install-all test lint format check notebooks notebooks-all notebooks-gpu docs docs-clean clean

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
	$(PYTHON) -m black src/ tests/ docs/examples/
	$(PYTHON) -m ruff check --fix src/ tests/ docs/examples/

check: lint test

# --- Notebooks & Docs ---

notebooks: $(NB_STAMPS)

$(NB_STAMP_DIR)/%.stamp: %.ipynb
	@mkdir -p $(dir $@)
	JAX_PLATFORMS=cpu $(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace "$<"
	@touch "$@"

notebooks-all:
	@rm -rf $(NB_STAMP_DIR)
	$(MAKE) notebooks

notebooks-gpu:
	@echo "Running $(GPU_NOTEBOOK)"
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace "$(GPU_NOTEBOOK)"

docs:
	JAX_PLATFORMS=cpu $(PYTHON) -m jupyter_book build docs/

docs-clean:
	$(PYTHON) -m jupyter_book clean docs/

# --- Cleanup ---

clean:
	rm -rf $(VENV) htmlcov .pytest_cache .mypy_cache .ruff_cache $(NB_STAMP_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
