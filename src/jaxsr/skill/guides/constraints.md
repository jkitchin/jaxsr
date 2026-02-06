# Physical Constraints Guide

## Why Use Constraints?

Symbolic regression can discover equations that fit data well but violate known physics.
Constraints inject domain knowledge to ensure physically meaningful models:

- A reaction rate cannot be negative
- Heat transfer increases with temperature difference
- Concentration is bounded between 0 and 1
- Pressure drop is monotonically increasing with flow rate

## The Constraints API

```python
from jaxsr import Constraints

constraints = (Constraints()
    .add_bounds("y", lower=0)
    .add_monotonic("T", direction="increasing")
    .add_concave("P")
)
```

Method chaining returns `self`, so you can compose multiple constraints fluently.

## Constraint Types

### Output Bounds: `add_bounds()`

Constrain the model output (predicted y) to a range.

```python
.add_bounds("y", lower=0)                    # y >= 0
.add_bounds("y", upper=100)                  # y <= 100
.add_bounds("y", lower=0, upper=1)           # 0 <= y <= 1
.add_bounds("y", lower=0, weight=10.0)       # Stronger penalty
.add_bounds("y", lower=0, hard=True)         # Strict enforcement
```

**Use when:** Output has known physical limits (concentrations, probabilities, positive quantities).

### Monotonicity: `add_monotonic()`

Constrain the model to be monotonically increasing or decreasing in a feature.

```python
.add_monotonic("T", direction="increasing")   # dy/dT >= 0
.add_monotonic("P", direction="decreasing")   # dy/dP <= 0
```

**Use when:** You know the directional effect of a variable:
- Reaction rate increases with temperature
- Viscosity decreases with temperature
- Yield increases with catalyst loading (up to a point)

### Convexity: `add_convex()`

Constrain the model to be convex in a feature (curves upward, d²y/dx² >= 0).

```python
.add_convex("concentration")
```

**Use when:** Diminishing costs, accelerating growth, U-shaped relationships.

### Concavity: `add_concave()`

Constrain the model to be concave in a feature (curves downward, d²y/dx² <= 0).

```python
.add_concave("catalyst_loading")
```

**Use when:** Diminishing returns, saturation effects, inverted-U relationships.

### Sign Constraints: `add_sign_constraint()`

Force a specific basis function's coefficient to be positive or negative.

```python
.add_sign_constraint("T", sign="positive")       # coefficient of T > 0
.add_sign_constraint("P^2", sign="negative")      # coefficient of P^2 < 0
```

**Note:** The `basis_name` must exactly match a name in the basis library.
Check available names with `[bf.name for bf in library.basis_functions]`.

### Known Coefficients: `add_known_coefficient()`

Fix a coefficient to a known value (e.g., from theory).

```python
.add_known_coefficient("1", 0.0)       # No intercept (force through origin)
.add_known_coefficient("T", 8.314)     # Known theoretical slope
```

### Linear Constraints: `add_linear_constraint()`

Impose arbitrary linear constraints on coefficients: `A @ coefficients <= b`.

```python
import numpy as np

# Sum of first two coefficients <= 1
A = np.array([[0, 1, 1, 0, 0]])  # shape (n_constraints, n_basis_functions)
b = np.array([1.0])
.add_linear_constraint(A, b)
```

**Use when:** You have complex relationships between coefficients from theory.

## Soft vs. Hard Constraints

Every constraint has a `hard` parameter:

| Mode | Behavior | When to Use |
|------|----------|-------------|
| `hard=False` (default) | Penalty added to objective | Data is noisy; constraint is approximate |
| `hard=True` | Strict enforcement via projection | Constraint must hold exactly |

```python
# Soft: violations are penalized but allowed
.add_bounds("y", lower=0, hard=False, weight=1.0)

# Hard: strictly enforced
.add_bounds("y", lower=0, hard=True)
```

**Guidance:**
- Start with soft constraints. If the model still violates them, increase the weight.
- Use hard constraints only when violations are physically impossible.
- Hard constraints can make optimization harder — the feasible region may be small.

## Weight Parameter

The `weight` parameter controls how strongly soft constraints are enforced.

```python
.add_bounds("y", lower=0, weight=1.0)    # Normal penalty
.add_bounds("y", lower=0, weight=10.0)   # 10x stronger penalty
.add_bounds("y", lower=0, weight=0.1)    # Mild suggestion
```

**Guidance:**
- Default weight of 1.0 works in most cases.
- Increase weight if the constraint is frequently violated.
- Very high weights (>100) can dominate the fit and degrade accuracy.

## Complete Example

```python
import numpy as np
from jaxsr import BasisLibrary, SymbolicRegressor, Constraints

# Build library
library = (BasisLibrary(n_features=2, feature_names=["T", "P"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=2)
    .add_interactions(max_order=2)
)

# Define constraints
constraints = (Constraints()
    .add_bounds("y", lower=0)                          # Positive output
    .add_monotonic("T", direction="increasing")        # Rate increases with T
    .add_monotonic("P", direction="increasing")        # Rate increases with P
    .add_concave("T")                                  # Diminishing returns in T
    .add_sign_constraint("T", sign="positive")         # Positive T coefficient
)

# Prepare data (replace with your own)
X = np.random.randn(100, 2)
y = 3.0 * X[:, 0] + 1.5 * X[:, 1] - 0.5 * X[:, 0]**2

# Fit with constraints
model = SymbolicRegressor(
    basis_library=library,
    max_terms=5,
    constraints=constraints,
    information_criterion="bic"
)
model.fit(X, y)

print(model.expression_)
```

## Tips

1. **Don't over-constrain.** Too many constraints can prevent the optimizer from finding
   a good fit. Start with the most important constraint and add more as needed.

2. **Check constraint satisfaction.** After fitting, verify constraints hold on your data:
   ```python
   y_pred = model.predict(X)
   print(f"Min prediction: {y_pred.min()}")  # Should be >= 0 if lower bound is 0
   ```

3. **Monotonicity is checked at data points.** The constraint is enforced at training
   points, not globally. For interpolation this is fine; for extrapolation, verify
   behavior in the region of interest.

4. **Constraints affect the Pareto front.** Constrained models may have slightly higher
   MSE than unconstrained ones. This is expected — you're trading accuracy for physical
   consistency.
