# neuropt Integration Guide

[neuropt](https://github.com/loevlie/neuropt) is an LLM-guided hyperparameter
optimizer. Instead of random search or Bayesian methods, it uses an LLM to read
training results and reason about what to try next.

JAXSR works with neuropt out of the box because it implements the sklearn
estimator protocol (`get_params` / `set_params`).

## Installation

```bash
pip install jaxsr[neuropt]
# or separately:
pip install "neuropt[llm]"
```

## Quick Start

```python
import numpy as np
from jaxsr import BasisLibrary, SymbolicRegressor
from neuropt import ArchSearch

library = (BasisLibrary(n_features=2)
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions()
)
model = SymbolicRegressor(basis_library=library, max_terms=5)

X = np.loadtxt("data.csv", delimiter=",", usecols=[0, 1])
y = np.loadtxt("data.csv", delimiter=",", usecols=[2])

def train_fn(config):
    m = config["model"]  # neuropt injects a cloned model with the config applied
    m.fit(X, y)
    y_pred = np.asarray(m.predict(X))
    mse = float(np.mean((y - y_pred) ** 2))
    return {"score": mse}

search = ArchSearch.from_model(model, train_fn, backend="claude")
search.run(max_evals=15)
```

neuropt auto-discovers tunable parameters from `get_params()` and asks the
LLM for reasonable search ranges. Each evaluation takes a fraction of a
second for symbolic regression, so 15 evals is nearly instant.

## Custom Search Space

Override the auto-detected space with specific ranges:

```python
search = ArchSearch(
    train_fn=train_fn,
    search_space={
        "max_terms": (2, 8),
        "strategy": ["greedy_forward", "greedy_backward", "lasso_path"],
        "information_criterion": ["aic", "aicc", "bic"],
    },
    backend="claude",
)
search.run(max_evals=15)
```

## When to Use neuropt vs GridSearchCV

| | GridSearchCV | neuropt |
|--|-------------|---------|
| Best for | Small, discrete grids | Larger mixed spaces |
| Search method | Exhaustive | LLM-guided |
| Reads results | No | Yes — adapts based on scores |
| Cost | Free | ~$0.01–0.05 per run (Claude Haiku) |
