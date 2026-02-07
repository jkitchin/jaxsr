"""
Active Learning Loop — Iteratively improve a model with targeted experiments.

This template shows how to:
1. Start with a small initial dataset
2. Fit a model
3. Use the model to suggest informative next experiments
4. Collect data, refit, and repeat
"""

import numpy as np

from jaxsr import AdaptiveSampler, BasisLibrary, SymbolicRegressor

# =============================================================================
# Configuration
# =============================================================================
N_INITIAL = 10  # Initial number of experiments
N_PER_ROUND = 5  # New experiments per round
N_ROUNDS = 5  # Number of adaptive rounds
BOUNDS = [(0, 5), (0, 5)]  # Feature bounds
FEATURE_NAMES = ["x1", "x2"]


# =============================================================================
# True function (replace with your actual experiments)
# =============================================================================
def true_function(X):
    """Simulates running experiments. Replace with actual data collection."""
    x1, x2 = X[:, 0], X[:, 1]
    return 2.0 * x1 + 0.5 * x1 * x2 - 0.3 * x2**2 + 0.3 * np.random.randn(len(x1))


# =============================================================================
# 1. Build basis library (same for all rounds)
# =============================================================================
library = (
    BasisLibrary(n_features=2, feature_names=FEATURE_NAMES)
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions(max_order=2)
)

# =============================================================================
# 2. Initial data
# =============================================================================
np.random.seed(42)
X_all = np.random.uniform(
    [b[0] for b in BOUNDS], [b[1] for b in BOUNDS], size=(N_INITIAL, len(BOUNDS))
)
y_all = true_function(X_all)

print(f"Initial data: {len(y_all)} points")
print("-" * 60)

# =============================================================================
# 3. Active learning loop
# =============================================================================
for round_num in range(N_ROUNDS):
    # Fit model
    model = SymbolicRegressor(basis_library=library, max_terms=5, information_criterion="bic")
    model.fit(X_all, y_all)

    print(f"\nRound {round_num + 1}: {len(y_all)} total points")
    print(f"  Model: {model.expression_}")
    print(f"  R²: {model.metrics_['r2']:.4f}")
    print(f"  MSE: {model.metrics_['mse']:.6g}")

    # Suggest next experiments
    sampler = AdaptiveSampler(
        model=model,
        bounds=BOUNDS,
        strategy="uncertainty",  # Try "error", "leverage", "gradient" too
        batch_size=N_PER_ROUND,
        n_candidates=1000,
        random_state=round_num,
    )
    result = sampler.suggest(n_points=N_PER_ROUND)
    X_next = result.points

    print(f"  Suggested {len(X_next)} new experiments")

    # "Run" experiments
    y_next = true_function(X_next)

    # Accumulate data
    X_all = np.vstack([X_all, X_next])
    y_all = np.concatenate([y_all, y_next])

# =============================================================================
# 4. Final model
# =============================================================================
model_final = SymbolicRegressor(basis_library=library, max_terms=5, information_criterion="bic")
model_final.fit(X_all, y_all)

print("\n" + "=" * 60)
print("FINAL MODEL")
print("=" * 60)
print(f"Total data points: {len(y_all)}")
print(f"Expression: {model_final.expression_}")
print(f"R²: {model_final.metrics_['r2']:.4f}")
print(f"MSE: {model_final.metrics_['mse']:.6g}")
print(f"LaTeX: {model_final.to_latex()}")

# Prediction intervals at a test point
X_test = np.array([[2.5, 2.5]])
y_pred, lower, upper = model_final.predict_interval(X_test, alpha=0.05)
print(
    f"\nPrediction at x1=2.5, x2=2.5: "
    f"{float(y_pred[0]):.4f} [{float(lower[0]):.4f}, {float(upper[0]):.4f}]"
)
