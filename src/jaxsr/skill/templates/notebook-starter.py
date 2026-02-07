"""
Jupyter Notebook Starter — Cell-by-cell structure for interactive analysis.

Copy these sections into separate notebook cells. Each section is marked with
a comment header showing the cell boundary.

To convert this to a notebook:
    pip install jupytext
    jupytext --to notebook notebook-starter.py
"""

# %% [markdown]
# # Symbolic Regression Analysis with JAXSR
#
# This notebook walks through a complete symbolic regression workflow:
# 1. Load and explore data
# 2. Build a basis library
# 3. Fit and evaluate models
# 4. Uncertainty quantification
# 5. Export results

# %% Cell 1: Imports and setup
import matplotlib.pyplot as plt
import numpy as np

from jaxsr import BasisLibrary, SymbolicRegressor, anova

# %% Cell 2: Load your data
# Replace this cell with your data loading code
# Required: X shape (n_samples, n_features), y shape (n_samples,)

np.random.seed(42)
n = 100
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 10, n)
y = 3.0 * x1 - 0.5 * x2**2 + 0.4 * x1 * x2 + np.random.randn(n)

X = np.column_stack([x1, x2])
feature_names = ["x1", "x2"]

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Response range: [{y.min():.2f}, {y.max():.2f}]")

# %% Cell 3: Explore the data
fig, axes = plt.subplots(1, X.shape[1], figsize=(5 * X.shape[1], 4))
if X.shape[1] == 1:
    axes = [axes]
for i, (ax, name) in enumerate(zip(axes, feature_names, strict=False)):
    ax.scatter(X[:, i], y, alpha=0.5, s=20)
    ax.set_xlabel(name)
    ax.set_ylabel("y")
    ax.set_title(f"y vs {name}")
plt.tight_layout()
plt.show()

# %% Cell 4: Build basis library
# Adjust this based on your domain knowledge
library = (
    BasisLibrary(n_features=len(feature_names), feature_names=feature_names)
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions(max_order=2)
    # Uncomment as needed:
    # .add_transcendental(funcs=["log", "exp", "sqrt"])
    # .add_ratios()
)

print(f"Library: {len(library)} candidate basis functions")

# %% Cell 5: Fit model
# Adjust max_terms, strategy, and criterion as needed
model = SymbolicRegressor(
    basis_library=library,
    max_terms=5,
    strategy="greedy_forward",  # or "exhaustive" if library < 20
    information_criterion="bic",  # or "aicc" for small datasets
)
model.fit(X, y)

print(model.summary())

# %% Cell 6: Parity plot (predicted vs actual)
y_pred = model.predict(X)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y, y_pred, alpha=0.5, s=20)
lims = [min(y.min(), float(y_pred.min())), max(y.max(), float(y_pred.max()))]
ax.plot(lims, lims, "r--", alpha=0.5)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title(f"Parity Plot (R² = {model.metrics_['r2']:.4f})")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %% Cell 7: Residual analysis
residuals = np.asarray(y) - np.asarray(y_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Residuals vs predicted
axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
axes[0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs Predicted")

# Residual histogram
axes[1].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.show()

print(f"Residual mean: {residuals.mean():.4f}")
print(f"Residual std:  {residuals.std():.4f}")

# %% Cell 8: Pareto front
pareto = model.pareto_front_

fig, ax = plt.subplots(figsize=(8, 5))
n_terms_list = [r.n_terms for r in pareto]
mse_list = [r.mse for r in pareto]
ax.plot(n_terms_list, mse_list, "bo-")
ax.scatter(
    [model._result.n_terms],
    [model._result.mse],
    color="red",
    s=100,
    zorder=5,
    label="Selected",
)
ax.set_xlabel("Number of Terms")
ax.set_ylabel("MSE")
ax.set_title("Pareto Front: Complexity vs Accuracy")
ax.set_yscale("log")
ax.legend()
plt.tight_layout()
plt.show()

# %% Cell 9: Prediction intervals
y_pred_i, lower, upper = model.predict_interval(X, alpha=0.05)

# Sort by predicted value for visualization
sort_idx = np.argsort(np.asarray(y_pred_i))
y_sorted = np.asarray(y_pred_i)[sort_idx]
lo_sorted = np.asarray(lower)[sort_idx]
hi_sorted = np.asarray(upper)[sort_idx]
y_actual = np.asarray(y)[sort_idx]

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(range(len(y_sorted)), lo_sorted, hi_sorted, alpha=0.3, label="95% PI")
ax.plot(y_sorted, "b-", label="Predicted")
ax.scatter(range(len(y_actual)), y_actual, color="red", s=10, alpha=0.5, label="Actual")
ax.set_xlabel("Observation (sorted by prediction)")
ax.set_ylabel("Response")
ax.set_title("Prediction Intervals")
ax.legend()
plt.tight_layout()
plt.show()

# %% Cell 10: ANOVA — variable importance
result = anova(model)
print("ANOVA Decomposition:")
summary_sources = {"Model", "Residual", "Total"}
term_rows = [r for r in result.rows if r.source not in summary_sources]
model_ss = sum(r.sum_sq for r in term_rows)
for row in term_rows:
    pct = 100 * row.sum_sq / model_ss if model_ss > 0 else 0.0
    print(f"  {row.source}: {pct:.1f}%")

# %% Cell 11: Export results
# LaTeX equation (for papers)
print(f"LaTeX: ${model.to_latex()}$")

# Save model for later use
model.save("model.json")

# Pure NumPy callable (for deployment without JAX)
predict_fn = model.to_callable()
print(f"\nCallable created. Test: predict_fn(X[:1]) = {predict_fn(X[:1])}")
