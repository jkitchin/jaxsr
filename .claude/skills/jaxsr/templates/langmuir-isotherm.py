"""
Known-Model Fitting: Langmuir Isotherm Parameter Estimation.

This template shows how to:
1. Design experiments for a known model (Langmuir isotherm)
2. Build a parametric basis library encoding the model form
3. Fit and extract parameters with uncertainty
4. Use ANOVA to verify model adequacy
5. Use active learning to suggest the most informative next experiments

The Langmuir isotherm: q = q_max * K * P / (1 + K * P)
  - q_max: maximum monolayer capacity (linear parameter, estimated by OLS)
  - K: equilibrium constant (nonlinear parameter, optimized by profile likelihood)
  - P: pressure (independent variable)
  - q: amount adsorbed (response)
"""

import numpy as np

from jaxsr import (
    AdaptiveSampler,
    BasisLibrary,
    DOEStudy,
    SymbolicRegressor,
    anova,
    bootstrap_coefficients,
    bootstrap_predict,
)

# =============================================================================
# 1. Design the experiment
# =============================================================================
study = DOEStudy(
    name="langmuir_adsorption",
    factor_names=["pressure"],
    bounds=[(0.01, 10.0)],  # Pressure in bar
    description="Langmuir isotherm parameter estimation",
)
X_design = study.create_design(method="latin_hypercube", n_points=15, random_state=42)

print(f"Designed {len(X_design)} experiments")
print(f"Pressure range: [{X_design.min():.3f}, {X_design.max():.3f}] bar")

# =============================================================================
# 2. Collect experimental data (replace with your measurements)
# =============================================================================
# True parameters: q_max=5.0, K=2.0
P = X_design.flatten()
q_true = 5.0 * 2.0 * P / (1 + 2.0 * P)
q_measured = q_true + 0.1 * np.random.randn(len(P))

study.add_observations(X_design, q_measured, notes="Initial measurements")

# =============================================================================
# 3. Build Langmuir basis library
# =============================================================================
library = (
    BasisLibrary(n_features=1, feature_names=["P"])
    .add_constant()  # intercept (should be ~0 for pure Langmuir)
    .add_parametric(
        name="K*P/(1+K*P)",
        func=lambda X, K: K * X[:, 0] / (1 + K * X[:, 0]),
        param_bounds={"K": (0.01, 100.0)},
        complexity=3,
        feature_indices=(0,),
        log_scale=True,  # K spans orders of magnitude
    )
)

print(f"\nBasis library: {len(library)} functions")

# =============================================================================
# 4. Fit the model
# =============================================================================
model = SymbolicRegressor(
    basis_library=library,
    max_terms=2,
    strategy="greedy_forward",
    information_criterion="aicc",  # AICc for small samples
)
model.fit(X_design, q_measured)

print("\n" + model.summary())
print(f"Expression: {model.expression_}")

# =============================================================================
# 5. Extract parameters
# =============================================================================
print("\n--- Parameter Estimates ---")
for i, name in enumerate(model.selected_features_):
    print(f"  {name}: coefficient = {float(model.coefficients_[i]):.4f}")

# The coefficient of the Langmuir term is q_max
# K is embedded in the parametric basis function

# =============================================================================
# 6. Coefficient confidence intervals (OLS)
# =============================================================================
print("\n--- 95% Confidence Intervals (OLS) ---")
intervals = model.coefficient_intervals(alpha=0.05)
for name, (lo, hi) in intervals.items():
    sig = " *" if lo * hi > 0 else ""
    print(f"  {name}: [{lo:.4f}, {hi:.4f}]{sig}")

# =============================================================================
# 7. Bootstrap intervals (captures nonlinear parameter uncertainty)
# =============================================================================
print("\n--- Bootstrap Intervals (500 resamples) ---")
boot_coef = bootstrap_coefficients(model, n_bootstrap=500, alpha=0.05)
for name, (lo, hi) in boot_coef.intervals.items():
    print(f"  {name}: [{lo:.4f}, {hi:.4f}]")

# Prediction intervals via bootstrap
P_pred = np.linspace(0.01, 10.0, 50).reshape(-1, 1)
boot_pred = bootstrap_predict(model, P_pred, n_bootstrap=500, alpha=0.05)
print(
    f"\nAverage bootstrap 95% PI width: "
    f"{np.mean(np.asarray(boot_pred.upper) - np.asarray(boot_pred.lower)):.4f}"
)

# =============================================================================
# 8. ANOVA — term importance
# =============================================================================
print("\n--- ANOVA Decomposition ---")
result = anova(model)
total_ss = sum(row.sum_sq for row in result.rows)
for row in result.rows:
    pct = 100 * row.sum_sq / total_ss if total_ss > 0 else 0.0
    print(f"  {row.source:30s}  SS = {row.sum_sq:10.4f}  ({pct:5.1f}%)")

# =============================================================================
# 9. Suggest next experiments (active learning)
# =============================================================================
sampler = AdaptiveSampler(
    model=model,
    bounds=[(0.01, 10.0)],
    strategy="uncertainty",
    batch_size=5,
    random_state=123,
)
suggestions = sampler.suggest(n_points=5)

print("\n--- Suggested Next Experiments ---")
for i, pt in enumerate(suggestions.points):
    print(f"  Experiment {i + 1}: P = {pt[0]:.4f} bar")

# =============================================================================
# 10. Save and export
# =============================================================================
study.save("langmuir.jaxsr")
model.save("langmuir_model.json")

print(f"\nLaTeX: ${model.to_latex()}$")
print(f"R²: {model.metrics_['r2']:.6f}")
print(f"MSE: {model.metrics_['mse']:.6g}")
print(f"AICc: {model.metrics_['aicc']:.2f}")
