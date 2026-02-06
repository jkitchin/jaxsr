# Uncertainty Quantification Guide

## Why UQ Matters

A prediction without uncertainty bounds is incomplete. JAXSR provides five UQ methods,
each with different assumptions, strengths, and computational costs.

## Method Overview

| Method | Assumptions | Speed | Best For |
|--------|-------------|-------|----------|
| OLS intervals | Normal errors, correct model | Instant | Default, quick assessment |
| Ensemble | Multiple models exist | Fast | Model structure uncertainty |
| BMA | Models have posterior weights | Fast | Principled model averaging |
| Conformal | Exchangeable data | Medium | Distribution-free guarantees |
| Bootstrap | Resampling | Slower | Robustness, no assumptions |

## Method 1: OLS Prediction Intervals (Default)

Classical linear regression intervals based on the t-distribution.

```python
# Prediction interval (for new individual observations)
y_pred, lower, upper = model.predict_interval(X_new, alpha=0.05)

# Confidence band (for the mean response)
y_pred, conf_lo, conf_hi = model.confidence_band(X_new, alpha=0.05)

# Coefficient intervals
intervals = model.coefficient_intervals(alpha=0.05)
for name, (lo, hi) in intervals.items():
    print(f"  {name}: [{lo:.4f}, {hi:.4f}]")
```

**Assumptions:**
- Errors are normally distributed
- Errors have constant variance (homoscedasticity)
- The selected model is the true model

**When to use:** First-pass analysis. Always available after fitting.

**When NOT to use:** If residuals are clearly non-normal or heteroscedastic.

## Method 2: Ensemble Prediction

Uses all models on the Pareto front to make predictions. Returns statistics
across models.

```python
result = model.predict_ensemble(X_new)
# result is a dict with: mean, std, predictions (array of all model predictions)
```

**When to use:** You want to see how much predictions vary across plausible models.
If ensemble spread is large, model structure is uncertain.

## Method 3: Bayesian Model Averaging (BMA)

Weights Pareto-front models by their information criterion score and averages predictions.

```python
y_pred, lower, upper = model.predict_bma(X_new, criterion="bic", alpha=0.05)
```

**How it works:**
1. Compute weight for each Pareto model: w_i = exp(-0.5 * delta_criterion_i)
2. Normalize weights to sum to 1
3. Weighted average of predictions and uncertainties

```python
# Access the BMA object for more detail
from jaxsr import BayesianModelAverage
bma = BayesianModelAverage(model, criterion="bic")
print(bma.weights_)   # Model weights
print(bma.models_)    # Contributing models
```

**When to use:**
- Multiple models on the Pareto front have similar criterion values
- You want to hedge against model selection uncertainty
- More principled than picking a single "best" model

**Visualize model weights:**
```python
from jaxsr.plotting import plot_bma_weights
plot_bma_weights(model, criterion="bic")
```

## Method 4: Conformal Prediction

Distribution-free prediction intervals with guaranteed finite-sample coverage.

```python
# Jackknife+ (recommended — uses all training data)
y_pred, lower, upper = model.predict_conformal(X_new, alpha=0.05, method="jackknife+")

# Split conformal (faster but wastes calibration data)
from jaxsr import conformal_predict_split
y_pred, lower, upper = conformal_predict_split(
    model, X_cal, y_cal, X_new, alpha=0.05
)
```

**Guarantees:** Under exchangeability (i.i.d. data), conformal intervals have
at least (1-alpha) coverage probability regardless of the error distribution.

**When to use:**
- You need provable coverage guarantees
- Residuals are non-normal (heavy tails, skewed)
- For publication-quality intervals

**When NOT to use:**
- Very small datasets (n < 30) — intervals become very wide
- Sequential/time-series data (exchangeability violated)

## Method 5: Bootstrap

Resamples the training data and refits to assess stability.

```python
from jaxsr import bootstrap_predict, bootstrap_coefficients

# Prediction intervals via bootstrap
result = bootstrap_predict(model, X_new, n_bootstrap=1000, alpha=0.05)
print(result.mean)      # Mean prediction
print(result.lower)     # Lower bound
print(result.upper)     # Upper bound

# Coefficient stability
coeff_result = bootstrap_coefficients(model, n_bootstrap=1000, alpha=0.05)
print(coeff_result.means)
print(coeff_result.intervals)  # Per-coefficient intervals

# Model selection stability
from jaxsr import bootstrap_model_selection
stability = bootstrap_model_selection(model, X, y, n_bootstrap=100)
# Shows how often each term is selected across bootstrap samples
```

**When to use:**
- Assess sensitivity to individual data points
- No distributional assumptions needed
- Check if model structure is stable (same terms selected across resamples)

**Computational cost:** Refits the model n_bootstrap times. Use n_bootstrap=200-1000.

## Decision Flowchart

```
Do you need quick intervals?
├── YES → model.predict_interval() (OLS)
└── NO
    ├── Is model structure uncertain? (multiple good models on Pareto front)
    │   ├── YES → model.predict_bma() (Bayesian Model Averaging)
    │   └── NO
    │       ├── Are residuals non-normal?
    │       │   ├── YES → model.predict_conformal() (distribution-free)
    │       │   └── NO
    │       │       └── Are you worried about data sensitivity?
    │       │           ├── YES → bootstrap_predict() (resampling)
    │       │           └── NO → model.predict_interval() (OLS, fast)
    └── Want all of the above?
        └── Run all methods and compare!
```

## ANOVA: Variable Importance

Decompose the total variance into contributions from each term:

```python
from jaxsr import anova

result = anova(model, X)
for row in result.rows:
    print(f"  {row.source}: SS={row.sum_of_squares:.4f}, "
          f"%={row.percent_contribution:.1f}%")
```

This tells you which basis functions contribute most to the model's explanatory power.

## Visualization

```python
from jaxsr.plotting import (
    plot_prediction_intervals,
    plot_coefficient_intervals,
    plot_bma_weights
)

# Prediction intervals
plot_prediction_intervals(model, X, y, alpha=0.05)

# Coefficient intervals (shows significance)
plot_coefficient_intervals(model, alpha=0.05)

# BMA weights
plot_bma_weights(model, criterion="bic")
```

## Comparing Methods

Run all methods and compare their interval widths:

```python
# OLS
_, ols_lo, ols_hi = model.predict_interval(X_test, alpha=0.05)

# BMA
_, bma_lo, bma_hi = model.predict_bma(X_test, alpha=0.05)

# Conformal
_, conf_lo, conf_hi = model.predict_conformal(X_test, alpha=0.05)

# Bootstrap
boot = bootstrap_predict(model, X_test, n_bootstrap=500, alpha=0.05)

# Compare average interval widths
import numpy as np
print(f"OLS width:      {np.mean(ols_hi - ols_lo):.4f}")
print(f"BMA width:      {np.mean(bma_hi - bma_lo):.4f}")
print(f"Conformal width: {np.mean(conf_hi - conf_lo):.4f}")
print(f"Bootstrap width: {np.mean(boot.upper - boot.lower):.4f}")
```
