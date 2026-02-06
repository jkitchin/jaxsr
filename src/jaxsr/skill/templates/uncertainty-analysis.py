"""
Uncertainty Quantification — Compare all UQ methods.

This template shows how to:
1. Fit a model
2. Compute OLS prediction intervals
3. Bayesian Model Averaging
4. Conformal prediction (distribution-free)
5. Bootstrap analysis
6. ANOVA (variable importance)
"""

import numpy as np

from jaxsr import (
    BasisLibrary,
    BayesianModelAverage,
    SymbolicRegressor,
    anova,
    bootstrap_coefficients,
    bootstrap_predict,
)

# =============================================================================
# 1. Prepare data
# =============================================================================
np.random.seed(42)
n = 80
x1 = np.random.uniform(0, 5, n)
x2 = np.random.uniform(0, 5, n)
y = 3.0 * x1 - 1.5 * x2**2 + 0.8 * x1 * x2 + 0.5 * np.random.randn(n)

X = np.column_stack([x1, x2])
feature_names = ["x1", "x2"]

# Test points
X_test = np.column_stack([np.random.uniform(0, 5, 20), np.random.uniform(0, 5, 20)])

# =============================================================================
# 2. Fit model
# =============================================================================
library = (
    BasisLibrary(n_features=2, feature_names=feature_names)
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions(max_order=2)
)

model = SymbolicRegressor(basis_library=library, max_terms=5, information_criterion="bic")
model.fit(X, y)
print(f"Model: {model.expression_}")
print(f"R²: {model.metrics_['r2']:.4f}\n")

# =============================================================================
# 3. OLS Prediction Intervals (classical)
# =============================================================================
print("=" * 50)
print("OLS Prediction Intervals")
print("=" * 50)

# Prediction interval (for individual new observations)
y_pred, pi_lo, pi_hi = model.predict_interval(X_test, alpha=0.05)

# Confidence band (for the mean response)
_, cb_lo, cb_hi = model.confidence_band(X_test, alpha=0.05)

print(f"Average prediction interval width: {np.mean(np.asarray(pi_hi) - np.asarray(pi_lo)):.4f}")
print(f"Average confidence band width:     {np.mean(np.asarray(cb_hi) - np.asarray(cb_lo)):.4f}")

# Coefficient intervals
print("\nCoefficient significance (95% CI):")
intervals = model.coefficient_intervals(alpha=0.05)
for name, (lo, hi) in intervals.items():
    sig = " *" if lo * hi > 0 else ""
    print(f"  {name}: [{lo:.4f}, {hi:.4f}]{sig}")

# =============================================================================
# 4. Bayesian Model Averaging
# =============================================================================
print("\n" + "=" * 50)
print("Bayesian Model Averaging")
print("=" * 50)

y_bma, bma_lo, bma_hi = model.predict_bma(X_test, criterion="bic", alpha=0.05)
print(f"Average BMA interval width: {np.mean(np.asarray(bma_hi) - np.asarray(bma_lo)):.4f}")

# Model weights
bma = BayesianModelAverage(model, criterion="bic")
print(f"Number of models averaged: {len(bma.weights_)}")
for i, w in enumerate(bma.weights_):
    print(f"  Model {i + 1}: weight = {w:.4f}")

# =============================================================================
# 5. Conformal Prediction (distribution-free)
# =============================================================================
print("\n" + "=" * 50)
print("Conformal Prediction (jackknife+)")
print("=" * 50)

y_conf, conf_lo, conf_hi = model.predict_conformal(X_test, alpha=0.05, method="jackknife+")
print(
    f"Average conformal interval width: "
    f"{np.mean(np.asarray(conf_hi) - np.asarray(conf_lo)):.4f}"
)

# =============================================================================
# 6. Bootstrap Analysis
# =============================================================================
print("\n" + "=" * 50)
print("Bootstrap Analysis")
print("=" * 50)

# Bootstrap prediction intervals
boot_pred = bootstrap_predict(model, X_test, n_bootstrap=500, alpha=0.05)
print(
    f"Average bootstrap interval width: "
    f"{np.mean(np.asarray(boot_pred.upper) - np.asarray(boot_pred.lower)):.4f}"
)

# Bootstrap coefficient intervals
boot_coef = bootstrap_coefficients(model, n_bootstrap=500, alpha=0.05)
print("\nBootstrap coefficient intervals:")
for name, (lo, hi) in boot_coef.intervals.items():
    print(f"  {name}: [{lo:.4f}, {hi:.4f}]")

# =============================================================================
# 7. ANOVA — Variable Importance
# =============================================================================
print("\n" + "=" * 50)
print("ANOVA Decomposition")
print("=" * 50)

result = anova(model, X)
for row in result.rows:
    print(f"  {row.source}: SS={row.sum_of_squares:.4f}, %={row.percent_contribution:.1f}%")

# =============================================================================
# 8. Compare All Methods
# =============================================================================
print("\n" + "=" * 50)
print("Method Comparison (average interval width)")
print("=" * 50)

widths = {
    "OLS prediction": np.mean(np.asarray(pi_hi) - np.asarray(pi_lo)),
    "OLS confidence": np.mean(np.asarray(cb_hi) - np.asarray(cb_lo)),
    "BMA": np.mean(np.asarray(bma_hi) - np.asarray(bma_lo)),
    "Conformal": np.mean(np.asarray(conf_hi) - np.asarray(conf_lo)),
    "Bootstrap": np.mean(np.asarray(boot_pred.upper) - np.asarray(boot_pred.lower)),
}

for method, width in sorted(widths.items(), key=lambda x: x[1]):
    print(f"  {method:20s}: {width:.4f}")
