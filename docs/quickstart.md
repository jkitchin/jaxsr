# Quickstart Guide

This guide will get you started with JAXSR in a few minutes.

## Installation

```bash
pip install jaxsr
```

## Basic Usage

### 1. Import and Prepare Data

```python
import jax.numpy as jnp
import numpy as np
from jaxsr import BasisLibrary, SymbolicRegressor

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = 2.5 * X[:, 0] + 1.2 * X[:, 0] * X[:, 1] - 0.8 * X[:, 1]**2

X = jnp.array(X)
y = jnp.array(y)
```

### 2. Build a Basis Library

The basis library defines the candidate functions to consider:

```python
library = (BasisLibrary(n_features=2, feature_names=["x", "y"])
    .add_constant()           # 1
    .add_linear()             # x, y
    .add_polynomials(max_degree=3)  # x^2, x^3, y^2, y^3
    .add_interactions()       # x*y
)

print(f"Library has {len(library)} basis functions")
```

### 3. Fit the Model

```python
model = SymbolicRegressor(
    basis_library=library,
    max_terms=5,
    strategy="greedy_forward",
    information_criterion="bic",
)

model.fit(X, y)
```

### 4. Examine Results

```python
# Expression
print(f"Discovered: {model.expression_}")

# Metrics
print(f"R² = {model.metrics_['r2']:.4f}")
print(f"MSE = {model.metrics_['mse']:.6f}")

# Predict
y_pred = model.predict(X)
```

## Using the Convenience Function

For quick exploration, use `fit_symbolic`:

```python
from jaxsr import fit_symbolic

model = fit_symbolic(
    X, y,
    feature_names=["x", "y"],
    max_terms=5,
    max_poly_degree=3,
)

print(model.expression_)
```

## Adding Constraints

Incorporate domain knowledge:

```python
from jaxsr import Constraints

constraints = (Constraints()
    .add_bounds("y", lower=0)  # Non-negative output
    .add_sign_constraint("x", sign="positive")  # Positive coefficient
)

model = SymbolicRegressor(
    basis_library=library,
    constraints=constraints,
)
model.fit(X, y)
```

## Exploring the Pareto Front

View the trade-off between complexity and accuracy:

```python
for result in model.pareto_front_:
    print(f"Complexity {result.complexity}: MSE={result.mse:.4f}")
    print(f"  {result.expression()}")
```

## Exporting Models

```python
# SymPy expression
sympy_expr = model.to_sympy()

# LaTeX
latex = model.to_latex()

# Pure NumPy callable (no JAX dependency)
predict_fn = model.to_callable()
y_pred = predict_fn(np.array(X))

# Save/load
model.save("model.json")
loaded = SymbolicRegressor.load("model.json")
```

## Selection Strategies

Choose the appropriate strategy for your problem:

| Strategy | Best For | Speed |
|----------|----------|-------|
| `greedy_forward` | Default, most problems | Fast |
| `greedy_backward` | Starting with many terms | Fast |
| `exhaustive` | Small libraries (<20) | Slow |
| `lasso_path` | Large libraries, screening | Medium |

```python
model = SymbolicRegressor(
    basis_library=library,
    strategy="lasso_path",  # or "exhaustive", "greedy_backward"
)
```

## Next Steps

- See the [Examples](examples/) for application-specific tutorials
- Read the [API Reference](api.md) for detailed documentation
- Check the [Literature Review](literature_review.md) for background
