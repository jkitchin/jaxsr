# JAXSR: JAX-based Symbolic Regression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

JAXSR is a fully open-source symbolic regression library built on JAX that discovers interpretable algebraic expressions from data. Inspired by ALAMO (Automated Learning of Algebraic Models for Optimization), it uses sparse optimization techniques with JAX for automatic differentiation, JIT compilation, and GPU acceleration.

## Features

- **Flexible Basis Library**: Easily define candidate basis functions including polynomials, interactions, transcendentals, ratios, and custom functions
- **Multiple Selection Strategies**: Greedy forward/backward selection, exhaustive search, LASSO path screening
- **Physical Constraints**: Enforce monotonicity, bounds, convexity, and linear constraints
- **Adaptive Sampling**: Intelligently suggest new data points to improve model quality
- **JAX-Accelerated**: JIT compilation and GPU support for fast computation
- **Scikit-learn Compatible**: Familiar `fit`/`predict` interface
- **Symbolic Export**: Export to SymPy, LaTeX, or pure Python/NumPy functions

## Installation

```bash
pip install jaxsr
```

Or install from source:

```bash
git clone https://github.com/jkitchin/jaxsr.git
cd jaxsr
pip install -e ".[dev]"
```

## Quick Start

```python
import jax.numpy as jnp
import numpy as np
from jaxsr import BasisLibrary, SymbolicRegressor

# Generate synthetic data: y = 2.5*x0 + 1.2*x0*x1 - 0.8*x1^2 + noise
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 2) * 2
y = 2.5 * X[:, 0] + 1.2 * X[:, 0] * X[:, 1] - 0.8 * X[:, 1]**2 + np.random.randn(n_samples) * 0.1

X_jax = jnp.array(X)
y_jax = jnp.array(y)

# Build basis library
library = (BasisLibrary(n_features=2, feature_names=["x0", "x1"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions(max_order=2)
)

# Fit model
model = SymbolicRegressor(
    basis_library=library,
    max_terms=5,
    strategy="greedy_forward",
    information_criterion="bic",
)
model.fit(X_jax, y_jax)

# Results
print(f"Discovered: {model.expression_}")
print(f"R² = {model.metrics_['r2']:.4f}")

# Predict
y_pred = model.predict(X_jax)
```

## Basis Functions

JAXSR provides a flexible system for defining candidate basis functions:

```python
from jaxsr import BasisLibrary

library = (BasisLibrary(n_features=3, feature_names=["T", "P", "C"])
    .add_constant()                                    # Intercept term
    .add_linear()                                      # T, P, C
    .add_polynomials(max_degree=3)                     # T^2, T^3, P^2, ...
    .add_interactions(max_order=2)                     # T*P, T*C, P*C
    .add_transcendental(["log", "exp", "sqrt", "inv"]) # log(T), exp(T), ...
    .add_ratios()                                      # T/P, T/C, P/T, ...
    .add_custom(                                       # Custom functions
        name="Arrhenius",
        func=lambda X: jnp.exp(-X[:, 0] / X[:, 1]),
        complexity=3
    )
)

# Evaluate to get design matrix
Phi = library.evaluate(X)  # Shape: (n_samples, n_basis)
```

## Selection Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `greedy_forward` | Forward stepwise selection | Default, fast for large libraries |
| `greedy_backward` | Backward elimination | When starting with many terms |
| `exhaustive` | All combinations | Small libraries (<20 terms) |
| `lasso_path` | LASSO regularization path | Fast screening |

```python
model = SymbolicRegressor(
    basis_library=library,
    max_terms=5,
    strategy="greedy_forward",      # Selection strategy
    information_criterion="bic",    # or "aic", "aicc", "cv"
)
```

## Physical Constraints

Incorporate domain knowledge through constraints:

```python
from jaxsr import Constraints

constraints = (Constraints()
    .add_bounds("y", lower=0)                        # Non-negative output
    .add_monotonic("T", direction="increasing")      # y increases with T
    .add_convex("P")                                 # Convex in P
    .add_sign_constraint("T", sign="positive")       # Positive coefficient
    .add_linear_constraint(A, b)                     # A @ coeffs <= b
)

model = SymbolicRegressor(
    basis_library=library,
    constraints=constraints,
)
```

## Adaptive Sampling

Request new data points to improve model quality:

```python
from jaxsr import AdaptiveSampler

sampler = AdaptiveSampler(
    model=model,
    bounds=[(300, 500), (1, 10), (0.1, 1.0)],
    strategy="uncertainty",  # or "error", "leverage", "gradient"
)

# Get suggested points
X_new = sampler.suggest(n_points=5)

# After obtaining y_new from experiments:
model.update(X_new, y_new)
```

## Export Options

```python
# Human-readable expression
print(model.expression_)  # "y = 2.5*T + 1.2*T*P - 0.8*P^2"

# SymPy expression for symbolic manipulation
sympy_expr = model.to_sympy()

# LaTeX for papers
latex_str = model.to_latex()

# Pure Python/NumPy function (no JAX dependency)
predict_func = model.to_callable()
y_pred = predict_func(X_numpy)

# Save/load models
model.save("model.json")
loaded_model = SymbolicRegressor.load("model.json")
```

## Visualization

```python
from jaxsr.plotting import (
    plot_pareto_front,
    plot_parity,
    plot_residuals,
    plot_coefficient_path,
)

# Pareto front: complexity vs accuracy
plot_pareto_front(model.pareto_front_, highlight_best=True)

# Parity plot
plot_parity(y_true, y_pred)

# Residual diagnostics
plot_residuals(model, X, y)
```

## Examples

See the `examples/` directory for complete worked examples:

- `basic_usage.py`: Simple polynomial fitting
- `chemical_kinetics.py`: Discovering rate laws from kinetic data
- `heat_transfer.py`: Heat transfer correlations
- `benchmark_comparison.py`: Comparison with other methods

## API Reference

See the [documentation](docs/) for full API details.

## Citation

If you use JAXSR in your research, please cite:

```bibtex
@software{jaxsr2024,
  title = {JAXSR: JAX-based Symbolic Regression},
  author = {Kitchin, John},
  year = {2024},
  url = {https://github.com/jkitchin/jaxsr}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see our contributing guidelines and open an issue or pull request.
