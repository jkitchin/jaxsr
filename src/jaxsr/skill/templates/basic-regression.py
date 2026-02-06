"""
Basic Symbolic Regression — Discover an equation from data.

This template shows the minimal workflow:
1. Prepare data (X, y)
2. Build a basis library
3. Fit a model
4. Inspect and export results
"""

import numpy as np

from jaxsr import BasisLibrary, SymbolicRegressor

# =============================================================================
# 1. Prepare your data
# =============================================================================
# Replace this with your own data loading
# X shape: (n_samples, n_features)
# y shape: (n_samples,)

np.random.seed(42)
n = 50
x1 = np.random.uniform(0, 5, n)
x2 = np.random.uniform(0, 5, n)
y = 2.5 * x1 + 1.2 * x1 * x2 - 0.8 * x2**2 + 0.5 * np.random.randn(n)

X = np.column_stack([x1, x2])
feature_names = ["x1", "x2"]

# =============================================================================
# 2. Build basis library
# =============================================================================
library = (
    BasisLibrary(n_features=2, feature_names=feature_names)
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions(max_order=2)
)

print(f"Basis library has {len(library)} candidate functions")

# =============================================================================
# 3. Fit model
# =============================================================================
model = SymbolicRegressor(
    basis_library=library,
    max_terms=5,
    strategy="greedy_forward",
    information_criterion="bic",
)
model.fit(X, y)

# =============================================================================
# 4. Inspect results
# =============================================================================
print("\n" + model.summary())

# Key attributes
print(f"Expression: {model.expression_}")
print(f"R²: {model.metrics_['r2']:.4f}")
print(f"MSE: {model.metrics_['mse']:.6g}")
print(f"BIC: {model.metrics_['bic']:.2f}")
print(f"Complexity: {model.complexity_}")

# Coefficient significance
print("\nCoefficient intervals (95%):")
intervals = model.coefficient_intervals(alpha=0.05)
for name, (lo, hi) in intervals.items():
    sig = " *" if lo * hi > 0 else ""
    print(f"  {name}: [{lo:.4f}, {hi:.4f}]{sig}")

# Pareto front
print("\nPareto front:")
for result in model.pareto_front_:
    print(f"  {result.n_terms} terms, MSE={result.mse:.6g}")

# =============================================================================
# 5. Export
# =============================================================================
# LaTeX equation
print(f"\nLaTeX: {model.to_latex()}")

# Pure NumPy callable (no JAX dependency needed)
predict_fn = model.to_callable()
y_check = predict_fn(X)
print(f"Callable check — max error: {np.max(np.abs(y_check - model.predict(X))):.2e}")

# Save model
model.save("model.json")
print("Model saved to model.json")
