"""
Asynchronous DOE Workflow with DOEStudy
=======================================

Demonstrates how to use DOEStudy to persist a Design of Experiments
workflow across multiple sessions. In practice, each "session" below
would be a separate script run (days or weeks apart) with lab work
in between.

The .jaxsr file is a ZIP archive containing:
- meta.json — version, timestamps
- study.json — factor config, design spec, model config, iteration history
- X_design.npy, X_observed.npy, y_observed.npy — NumPy binary arrays
"""

import numpy as np

from jaxsr import DOEStudy

# ============================================================================
# Session 1: Set up the study and create the initial design
# ============================================================================
print("=" * 60)
print("SESSION 1: Study setup and initial design")
print("=" * 60)

study = DOEStudy(
    name="polymer_strength",
    description="Optimize tensile strength of a polymer blend",
    factor_names=["temperature", "pressure", "additive_pct"],
    bounds=[(150, 250), (5, 20), (0, 10)],
)

# Generate a space-filling design
X_design = study.create_design(method="latin_hypercube", n_points=15, random_state=42)
print(f"Created {len(X_design)} design points:")
for i, row in enumerate(X_design):
    print(f"  Run {i + 1}: T={row[0]:.1f}°C, P={row[1]:.1f} bar, additive={row[2]:.1f}%")

# Save and share with the lab
study.save("/tmp/polymer_study.jaxsr")
print(f"\nStudy saved. {len(study.pending_points)} experiments pending.")
print(study.summary())


# ============================================================================
# Session 2: Add first batch of lab results and fit initial model
# ============================================================================
print("\n" + "=" * 60)
print("SESSION 2: First batch of results")
print("=" * 60)

# Reload from disk (simulating a new Python session)
study = DOEStudy.load("/tmp/polymer_study.jaxsr")
print(f"Loaded study: {study.n_observations} observations, {len(study.pending_points)} pending")

# Simulate lab results for the first 8 experiments
# (In practice, these come from real measurements)
X_batch1 = study.design_points[:8]


def true_response(X):
    """Simulated ground truth for the polymer system."""
    T, P, A = X[:, 0], X[:, 1], X[:, 2]
    return 50 + 0.3 * T + 1.5 * P + 2.0 * A - 0.001 * T**2 + np.random.randn(len(T)) * 2


y_batch1 = true_response(X_batch1)

study.add_observations(X_batch1, y_batch1, notes="First 8 experiments from Lab A")
print(f"Added {len(y_batch1)} observations. Total: {study.n_observations}")

# Fit an initial model
model = study.fit(max_terms=5)
print(f"\nInitial model: {model.expression_}")
print(f"  MSE = {model._result.mse:.4f}")

# Save progress
study.save("/tmp/polymer_study.jaxsr")
print(f"\nStudy saved. {len(study.pending_points)} experiments still pending.")


# ============================================================================
# Session 3: Add remaining results and refine
# ============================================================================
print("\n" + "=" * 60)
print("SESSION 3: Complete results and refinement")
print("=" * 60)

study = DOEStudy.load("/tmp/polymer_study.jaxsr")
print(f"Loaded: {study.n_observations} observations, model fitted: {study.is_fitted}")
print(f"Current model: {study.model.expression_}")

# Add the remaining 7 experiments
X_batch2 = study.design_points[8:]
y_batch2 = true_response(X_batch2)
study.add_observations(X_batch2, y_batch2, notes="Remaining 7 experiments")

# Refit with all data
model = study.fit(max_terms=5)
print(f"\nRefined model: {model.expression_}")
print(f"  MSE = {model._result.mse:.4f}")

# Suggest next experiments for a follow-up round
next_points = study.suggest_next(n_points=3, strategy="space_filling")
print("\nSuggested next experiments:")
for i, row in enumerate(next_points):
    print(f"  Run {i + 1}: T={row[0]:.1f}°C, P={row[1]:.1f} bar, additive={row[2]:.1f}%")

study.save("/tmp/polymer_study.jaxsr")


# ============================================================================
# Session 4: Review and share
# ============================================================================
print("\n" + "=" * 60)
print("SESSION 4: Review complete study")
print("=" * 60)

study = DOEStudy.load("/tmp/polymer_study.jaxsr")
print(study.summary())

# The .jaxsr file can be shared with colleagues
# They can load it and continue the analysis
print("\nFile /tmp/polymer_study.jaxsr is ready to share!")
print(f"  Schema version: {study.meta['schema_version']}")
print(f"  Created: {study.meta['created']}")
print(f"  Last modified: {study.meta['modified']}")
