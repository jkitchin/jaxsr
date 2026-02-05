# JAXSR Documentation

**JAX-based Symbolic Regression**

JAXSR is a Python library for discovering interpretable algebraic expressions from data using sparse optimization techniques.

## Overview

JAXSR provides tools for symbolic regression - the task of finding mathematical expressions that describe relationships in data. Unlike black-box machine learning methods, symbolic regression produces human-readable equations that can provide scientific insight.

Key features:

- **Flexible Basis Functions**: Build custom libraries of candidate functions
- **Multiple Selection Strategies**: Choose from greedy, exhaustive, or LASSO-based methods
- **Physical Constraints**: Incorporate domain knowledge through constraints
- **JAX-Powered**: GPU acceleration, JIT compilation, automatic differentiation
- **Scikit-learn Compatible**: Familiar fit/predict interface

## Installation

```bash
pip install jaxsr
```

For development:

```bash
git clone https://github.com/jkitchin/jaxsr.git
cd jaxsr
pip install -e ".[dev]"
```

## Quick Start

```python
from jaxsr import BasisLibrary, SymbolicRegressor
import jax.numpy as jnp

# Create basis library
library = (BasisLibrary(n_features=2, feature_names=["x", "y"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions()
)

# Fit model
model = SymbolicRegressor(basis_library=library, max_terms=5)
model.fit(X, y)

# Results
print(model.expression_)
print(f"R² = {model.metrics_['r2']:.4f}")
```

## Documentation Contents

- [Quickstart Guide](quickstart.md) - Get started quickly
- [API Reference](api.md) - Complete API documentation
- [Literature Review](literature_review.md) - Background on symbolic regression
- [Examples](examples/) - Worked examples for various applications

## How It Works

JAXSR follows the ALAMO (Automated Learning of Algebraic Models for Optimization) methodology:

1. **Basis Library Construction**: Define a library of candidate basis functions (polynomials, transcendentals, interactions, etc.)

2. **Design Matrix Evaluation**: Evaluate all basis functions on training data to create a design matrix Φ

3. **Sparse Selection**: Use information criteria (BIC, AIC) to select a sparse subset of basis functions

4. **Coefficient Fitting**: Fit coefficients via least squares, optionally with constraints

5. **Model Analysis**: Examine Pareto front, export to various formats

## When to Use JAXSR

JAXSR is ideal when you:

- Want **interpretable** models rather than black boxes
- Have **domain knowledge** to constrain the solution space
- Need to discover **physical laws** or empirical correlations
- Require **reproducible** results (deterministic algorithms)
- Want to explore the **accuracy-complexity trade-off**

## Comparison with Other Tools

| Feature | JAXSR | ALAMO | PySR | GP |
|---------|-------|-------|------|-----|
| Open Source | ✓ | ✗ | ✓ | ✓ |
| Deterministic | ✓ | ✓ | ✗ | ✗ |
| Constraints | ✓ | ✓ | Limited | Limited |
| GPU Support | ✓ | ✗ | ✓ | Varies |

## License

JAXSR is released under the MIT License.
