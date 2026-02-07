"""
Constrained Symbolic Regression — Inject physical knowledge.

This template shows how to:
1. Define physical constraints (monotonicity, bounds, signs)
2. Fit a model that respects those constraints
3. Verify constraint satisfaction
"""

import numpy as np

from jaxsr import BasisLibrary, Constraints, SymbolicRegressor

# =============================================================================
# 1. Prepare data
# =============================================================================
np.random.seed(42)
n = 60

# Simulated reaction rate: increases with T, positive, concave in T
T = np.random.uniform(300, 500, n)
P = np.random.uniform(1, 10, n)
rate = 0.01 * T + 0.5 * P - 0.00002 * T**2 + 0.3 * np.random.randn(n)

X = np.column_stack([T, P])
y = rate

# =============================================================================
# 2. Build basis library
# =============================================================================
library = (
    BasisLibrary(n_features=2, feature_names=["T", "P"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=2)
    .add_interactions(max_order=2)
)

# =============================================================================
# 3. Define constraints
# =============================================================================
constraints = (
    Constraints()
    .add_bounds("y", lower=0)  # Rate is non-negative
    .add_monotonic("T", direction="increasing")  # Rate increases with temperature
    .add_monotonic("P", direction="increasing")  # Rate increases with pressure
    .add_concave("T")  # Diminishing returns with temperature
    .add_sign_constraint("T", sign="positive")  # T coefficient is positive
)

# =============================================================================
# 4. Fit constrained model
# =============================================================================
model = SymbolicRegressor(
    basis_library=library,
    max_terms=5,
    constraints=constraints,
    information_criterion="bic",
)
model.fit(X, y)

print(model.summary())

# =============================================================================
# 5. Verify constraint satisfaction
# =============================================================================
y_pred = model.predict(X)
print(f"\nMin prediction: {float(y_pred.min()):.4f} (should be >= 0)")
print(f"Max prediction: {float(y_pred.max()):.4f}")

# Check monotonicity by evaluating on a grid
T_grid = np.linspace(300, 500, 100)
P_fixed = np.full(100, 5.0)
X_grid = np.column_stack([T_grid, P_fixed])
y_grid = model.predict(X_grid)

diffs = np.diff(np.asarray(y_grid))
print(f"Monotonicity in T: {'satisfied' if np.all(diffs >= -1e-6) else 'VIOLATED'}")

# =============================================================================
# 6. Compare with unconstrained model
# =============================================================================
model_free = SymbolicRegressor(
    basis_library=library,
    max_terms=5,
    information_criterion="bic",
)
model_free.fit(X, y)

print(f"\nConstrained model:   {model.expression_}")
print(f"  R² = {model.metrics_['r2']:.4f}, MSE = {model.metrics_['mse']:.6g}")

print(f"Unconstrained model: {model_free.expression_}")
print(f"  R² = {model_free.metrics_['r2']:.4f}, MSE = {model_free.metrics_['mse']:.6g}")
