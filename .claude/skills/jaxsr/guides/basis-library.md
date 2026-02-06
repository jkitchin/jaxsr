# Basis Library Selection Guide

## What is a Basis Library?

A basis library is a collection of candidate functions that JAXSR considers when building
your model. The final model will be a linear combination of selected basis functions:

    y = c_0 + c_1*f_1(X) + c_2*f_2(X) + ... + c_k*f_k(X)

The key insight: you provide many candidates, and the selection algorithm finds the
smallest subset that best explains your data.

## Building a Library: Method Chaining API

```python
from jaxsr import BasisLibrary

library = (BasisLibrary(n_features=3, feature_names=["T", "P", "v"])
    .add_constant()           # intercept term (1)
    .add_linear()             # T, P, v
    .add_polynomials(max_degree=3)  # T^2, T^3, P^2, P^3, v^2, v^3
    .add_interactions(max_order=2)  # T*P, T*v, P*v
    .add_transcendental()     # log(T), exp(T), sqrt(T), sin(T), ...
)
```

Every `.add_*()` method returns `self`, so you chain as many as needed.

## Available Basis Function Types

### `add_constant()`

Adds the intercept term (value = 1). Almost always include this.

**When to omit:** Only if you know the model must pass through the origin (y=0 when all x=0).

### `add_linear()`

Adds one term per feature: `x_0, x_1, ..., x_{n-1}`.

**Always include this.** Even nonlinear models usually have linear components.

### `add_polynomials(max_degree=2)`

Adds pure powers: `x_i^2, x_i^3, ..., x_i^max_degree` for each feature.

| max_degree | Terms per feature | Use when |
|------------|------------------|----------|
| 2 | 1 (quadratic) | Default. Captures curvature. |
| 3 | 2 (cubic) | Inflection points expected |
| 4+ | 3+ | Rarely needed. Risk of overfitting. |

**Guidance:**
- With n_features > 5, keep max_degree <= 2 or the library explodes.
- Polynomial terms are n_features * (max_degree - 1) additional terms.

### `add_interactions(max_order=2)`

Adds cross-product terms between features.

| max_order | What it adds | Count |
|-----------|-------------|-------|
| 2 | `x_i * x_j` for all pairs | n*(n-1)/2 |
| 3 | Also `x_i * x_j * x_k` for all triples | + n*(n-1)*(n-2)/6 |

**Guidance:**
- Order 2 interactions are usually sufficient.
- Order 3 should only be used with <= 4 features (otherwise too many terms).
- Interactions capture how features modify each other's effects.

### `add_transcendental(funcs=None, safe=True)`

Adds nonlinear transformations of each feature.

**Available functions:** `log`, `exp`, `sqrt`, `inv` (1/x), `sin`, `cos`, `tan`,
`sinh`, `cosh`, `tanh`, `abs`, `square`

**Default (funcs=None):** Adds all of: `log`, `exp`, `sqrt`, `inv`, `sin`, `cos`, `abs`

```python
# All defaults
.add_transcendental()

# Only specific functions
.add_transcendental(funcs=["log", "exp", "sqrt"])

# Must always use safe=True (default) to guard against NaN
.add_transcendental(safe=True)
```

**Which functions for which domain:**

| Domain | Useful functions |
|--------|-----------------|
| Engineering correlations | `log`, `exp`, `sqrt`, `inv` |
| Chemical kinetics | `exp`, `inv`, `log` |
| Oscillatory data | `sin`, `cos` |
| Heat transfer (Nusselt) | `log`, `sqrt`, `inv` |
| General exploration | All defaults |

**Always use `safe=True`** — it wraps functions to handle edge cases:
- `log(x)` → `log(|x| + epsilon)`
- `1/x` → `1/(x + epsilon*sign(x))`
- `sqrt(x)` → `sqrt(|x|)`

### `add_ratios(safe=True)`

Adds `x_i / x_j` for all feature pairs (both orderings).

**Count:** n*(n-1) terms (not symmetric: x/y != y/x).

**When to use:**
- Dimensionless numbers in engineering (Reynolds, Prandtl, etc.)
- Concentration ratios in chemistry
- Rate ratios in kinetics

**When to avoid:**
- If features are not meaningfully divisible
- If library is already large (this doubles the size)

### `add_custom(name, func, complexity=3, feature_indices=None)`

Add any arbitrary function as a basis function.

```python
import jax.numpy as jnp

# Gaussian-like term
library.add_custom(
    name="exp(-T^2)",
    func=lambda X: jnp.exp(-X[:, 0]**2),
    complexity=3,
    feature_indices=[0]  # depends on feature 0 (T)
)

# Product of specific features with transformation
library.add_custom(
    name="T*log(P)",
    func=lambda X: X[:, 0] * jnp.log(jnp.abs(X[:, 1]) + 1e-10),
    complexity=4,
    feature_indices=[0, 1]
)
```

### `add_parametric(name, func, param_bounds, complexity=3, log_scale=False)`

Add basis functions with optimizable nonlinear parameters.

```python
import jax.numpy as jnp

# Arrhenius-type term: exp(-Ea/(R*T))
library.add_parametric(
    name="Arrhenius",
    func=lambda X, Ea: jnp.exp(-Ea / (8.314 * X[:, 0])),
    param_bounds={"Ea": (1000.0, 100000.0)},
    complexity=4,
    log_scale=True  # optimize Ea in log-space for better convergence
)

# Stretched exponential
library.add_parametric(
    name="stretched_exp",
    func=lambda X, beta: jnp.exp(-X[:, 0]**beta),
    param_bounds={"beta": (0.1, 3.0)},
    complexity=3
)
```

**Guidance:**
- `log_scale=True` helps when parameters span orders of magnitude.
- Parametric fits are slower (requires nonlinear optimization per candidate).
- Only add parametric terms when you have physical motivation.

### `add_categorical_indicators(features=None)`

For categorical features, adds binary indicator (dummy) variables.

```python
library = BasisLibrary(
    n_features=3,
    feature_names=["T", "P", "catalyst"],
    feature_types=["continuous", "continuous", "categorical"],
    categories={2: ["A", "B", "C"]}
)
library.add_categorical_indicators()  # adds I(catalyst=B), I(catalyst=C)
```

Reference category (first level) is dropped to avoid multicollinearity.

### `add_categorical_interactions(cat_features=None, cont_features=None)`

Adds interaction terms between categorical indicators and continuous features.

```python
library.add_categorical_interactions()  # I(catalyst=B)*T, I(catalyst=B)*P, ...
```

This allows different slopes for each category.

## Library Size Guidelines

| Library Size | Fits In | Notes |
|-------------|---------|-------|
| < 20 terms | Seconds | Can use `exhaustive` search |
| 20-100 terms | Seconds | Use `greedy_forward` (default) |
| 100-500 terms | Seconds-minutes | Use `greedy_forward` or `lasso_path` |
| 500+ terms | Minutes | Use `lasso_path` for screening, then refine |

Check your library size:
```python
print(f"Library has {len(library)} basis functions")
```

## Recipes by Domain

### Polynomial Response Surface (DOE)
```python
library = (BasisLibrary(n_features=k, feature_names=names)
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=2)
    .add_interactions(max_order=2)
)
```

### Engineering Correlation (Nusselt, friction factor)
```python
library = (BasisLibrary(n_features=3, feature_names=["Re", "Pr", "L_D"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=2)
    .add_transcendental(funcs=["log", "exp", "sqrt", "inv"])
    .add_interactions(max_order=2)
    .add_ratios()
)
```

### Chemical Kinetics
```python
library = (BasisLibrary(n_features=2, feature_names=["T", "C"])
    .add_constant()
    .add_linear()
    .add_transcendental(funcs=["exp", "inv", "log"])
    .add_ratios()
    .add_parametric("Arrhenius", arrhenius_fn,
                    param_bounds={"Ea": (5000, 80000)}, log_scale=True)
)
```

### Minimal Screening (many features)
```python
library = (BasisLibrary(n_features=20, feature_names=names)
    .add_constant()
    .add_linear()
    .add_interactions(max_order=2)
)
# Use lasso_path for fast screening
model = SymbolicRegressor(basis_library=library, strategy="lasso_path")
```

## Saving and Loading Libraries

```python
# Save for reuse
library.save("my_library.json")

# Load later
from jaxsr import BasisLibrary
library = BasisLibrary.load("my_library.json")
```
