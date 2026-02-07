"""
DOE Study Workflow — Complete design-to-report pipeline.

This template shows the full lifecycle:
1. Create a study with factor definitions
2. Generate an experimental design
3. Simulate running experiments (replace with real data)
4. Import observations and fit a model
5. Suggest next experiments (adaptive)
6. Generate a report
"""

import numpy as np

from jaxsr import DOEStudy

# =============================================================================
# 1. Create the study
# =============================================================================
study = DOEStudy(
    name="catalyst_optimization",
    factor_names=["temperature", "pressure", "flow_rate"],
    bounds=[(300, 500), (1, 10), (0.1, 2.0)],
    description="Optimize yield as a function of temperature, pressure, and flow rate",
)

print(f"Created study: {study.name}")
print(f"Factors: {study.factor_names}")
print(f"Bounds: {study.bounds}")

# =============================================================================
# 2. Generate experimental design
# =============================================================================
X_design = study.create_design(method="latin_hypercube", n_points=20, random_state=42)
print(f"\nGenerated {len(X_design)} design points")
print(f"First 5 runs:\n{X_design[:5]}")

# Save study (design is stored)
study.save("catalyst.jaxsr")

# =============================================================================
# 3. "Run experiments" — replace this with your real experiment results
# =============================================================================


def simulate_experiment(X):
    """Replace this with actual experimental measurements."""
    T, P, flow = X[:, 0], X[:, 1], X[:, 2]
    yield_val = 0.01 * T + 0.5 * P + 2.0 * flow - 0.00002 * T**2 - 0.05 * P**2 + 0.3 * T * P / 500
    noise = 0.5 * np.random.randn(len(T))
    return yield_val + noise


y_measured = simulate_experiment(X_design)
print(f"\nCollected {len(y_measured)} measurements")
print(f"Response range: [{y_measured.min():.2f}, {y_measured.max():.2f}]")

# =============================================================================
# 4. Import observations and fit model
# =============================================================================
study = DOEStudy.load("catalyst.jaxsr")
study.add_observations(X_design, y_measured, notes="Round 1: initial screening")
print(f"\nTotal observations: {study.n_observations}")

model = study.fit(max_terms=5, strategy="greedy_forward", information_criterion="bic")
print(f"\nModel: {model.expression_}")
print(f"R²: {model.metrics_['r2']:.4f}")
print(f"MSE: {model.metrics_['mse']:.6g}")
study.save("catalyst.jaxsr")

# =============================================================================
# 5. Suggest next experiments
# =============================================================================
study = DOEStudy.load("catalyst.jaxsr")
next_pts = study.suggest_next(n_points=5, strategy="uncertainty")
print(f"\nSuggested {len(next_pts)} next experiments:")
for i, pt in enumerate(next_pts):
    print(f"  Run {i + 1}: T={pt[0]:.1f}, P={pt[1]:.2f}, flow={pt[2]:.3f}")

# Run those experiments too
y_next = simulate_experiment(next_pts)
study.add_observations(next_pts, y_next, notes="Round 2: uncertainty-guided")

# Refit with combined data
model = study.fit(max_terms=5)
print(f"\nUpdated model: {model.expression_}")
print(f"R²: {model.metrics_['r2']:.4f} (with {study.n_observations} total points)")
study.save("catalyst.jaxsr")

# =============================================================================
# 6. Final results
# =============================================================================
print("\n" + "=" * 60)
print("FINAL STUDY SUMMARY")
print("=" * 60)
print(study.summary())

# Prediction at optimal conditions (example)
X_test = np.array([[400, 5, 1.0]])
y_pred, lower, upper = model.predict_interval(X_test, alpha=0.05)
print("\nPrediction at T=400, P=5, flow=1.0:")
print(f"  Yield = {float(y_pred[0]):.2f} [{float(lower[0]):.2f}, {float(upper[0]):.2f}]")
