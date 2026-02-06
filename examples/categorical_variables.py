"""
Categorical Variables in JAXSR
==============================

This example demonstrates how to use JAXSR with mixed continuous and
categorical input features.  A common scenario in science and engineering
is modelling a response that depends on continuous conditions (temperature,
pressure) *and* a discrete factor such as catalyst type or material grade.

JAXSR uses indicator (dummy) encoding to represent categorical variables
as binary basis functions, and can discover different intercepts and slopes
per category level automatically.
"""

import numpy as np

from jaxsr import BasisLibrary, SymbolicRegressor
from jaxsr.sampling import grid_sample, latin_hypercube_sample

# -----------------------------------------------------------------------
# 1. Generate synthetic data
#
# Ground truth: y = 2*T + 5*I(catalyst=1) + 10*I(catalyst=2) + noise
#
# Three catalysts are encoded as integers: 0 (reference), 1, 2.
# -----------------------------------------------------------------------

rng = np.random.RandomState(0)
n = 80

T = rng.uniform(300, 500, n)
catalyst = rng.choice([0, 1, 2], n)

y_true = 2.0 * T + 5.0 * (catalyst == 1) + 10.0 * (catalyst == 2)
y = y_true + rng.normal(0, 1.0, n)

X = np.column_stack([T, catalyst])

print("=== Data summary ===")
print(f"  {n} observations, 1 continuous + 1 categorical feature")
print(f"  Catalyst levels: {sorted(set(catalyst))}")

# -----------------------------------------------------------------------
# 2. Build a basis library with categorical support
#
# Key points:
#   - feature_types marks which columns are "continuous" or "categorical"
#   - categories maps the column index to its possible integer values
#   - add_categorical_indicators() creates K-1 dummy variables
#   - add_categorical_interactions() creates indicator * continuous terms
#   - Continuous-only methods (polynomials, transcendental) automatically
#     skip categorical features, so you don't have to filter manually.
# -----------------------------------------------------------------------

library = (
    BasisLibrary(
        n_features=2,
        feature_names=["T", "catalyst"],
        feature_types=["continuous", "categorical"],
        categories={1: [0, 1, 2]},
    )
    .add_constant()
    .add_linear()  # only adds T (skips categorical)
    .add_polynomials(max_degree=2)  # only T^2
    .add_categorical_indicators()  # I(catalyst=1), I(catalyst=2)
    .add_categorical_interactions()  # I(catalyst=1)*T, I(catalyst=2)*T
)

print(f"\n=== Basis library: {len(library)} functions ===")
for bf in library.basis_functions:
    print(f"  {bf.name}  (complexity={bf.complexity})")

# -----------------------------------------------------------------------
# 3. Fit and inspect
# -----------------------------------------------------------------------

model = SymbolicRegressor(basis_library=library, max_terms=5, strategy="greedy_forward")
model.fit(X, y)

print("\n=== Fitted model ===")
print(f"  Expression: {model.expression_}")
print(f"  R²: {model.score(X, y):.6f}")
print(f"  Selected features: {model.selected_features_}")

# -----------------------------------------------------------------------
# 4. Export to pure NumPy callable (no JAX needed at prediction time)
# -----------------------------------------------------------------------

predict_fn = model.to_callable()
y_pred = predict_fn(np.array(X))
residual = np.abs(y - y_pred).mean()
print("\n=== NumPy callable ===")
print(f"  Mean absolute residual: {residual:.4f}")

# -----------------------------------------------------------------------
# 5. Sampling with discrete dimensions
#
# When designing experiments, categorical dimensions should only take
# their valid levels.  Pass discrete_dims to the sampling functions.
# -----------------------------------------------------------------------

bounds = [(300, 500), (0, 2)]
discrete = {1: [0, 1, 2]}

X_lhs = latin_hypercube_sample(20, bounds, random_state=42, discrete_dims=discrete)
print("\n=== LHS with discrete dims ===")
print(f"  Shape: {X_lhs.shape}")
print(f"  Catalyst values: {sorted(set(np.array(X_lhs[:, 1]).tolist()))}")

X_grid = grid_sample(5, bounds, discrete_dims=discrete)
print("\n=== Grid with discrete dims ===")
print(f"  Shape: {X_grid.shape}  (5 continuous * 3 discrete = 15)")

# -----------------------------------------------------------------------
# 6. build_default() handles categorical features automatically
#
# If any features are categorical, build_default() adds indicators and
# categorical interactions alongside the standard continuous basis set.
# -----------------------------------------------------------------------

auto_library = BasisLibrary(
    n_features=2,
    feature_names=["T", "catalyst"],
    feature_types=["continuous", "categorical"],
    categories={1: [0, 1, 2]},
).build_default()

auto_model = SymbolicRegressor(basis_library=auto_library, max_terms=5)
auto_model.fit(X, y)
print("\n=== build_default() with categorical ===")
print(f"  Library size: {len(auto_library)}")
print(f"  Expression: {auto_model.expression_}")
print(f"  R²: {auto_model.score(X, y):.6f}")
