NOTEBOOKS := $(shell find . -name '*.ipynb' -not -path './.ipynb_checkpoints/*')

.PHONY: notebooks docs docs-clean

notebooks:
	@for nb in $(NOTEBOOKS); do \
		echo "Running $$nb"; \
		JAX_PLATFORMS=cpu jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done

docs:
	JAX_PLATFORMS=cpu jupyter-book build docs/

docs-clean:
	jupyter-book clean docs/
