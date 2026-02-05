# API Reference

## Core Classes

### BasisLibrary

```python
class BasisLibrary(n_features, feature_names=None, feature_bounds=None)
```

Library of candidate basis functions for symbolic regression.

**Parameters:**
- `n_features` (int): Number of input features
- `feature_names` (list of str, optional): Names for features
- `feature_bounds` (list of tuple, optional): Bounds for each feature

**Methods:**

| Method | Description |
|--------|-------------|
| `add_constant()` | Add constant term (intercept) |
| `add_linear()` | Add linear terms |
| `add_polynomials(max_degree)` | Add polynomial terms |
| `add_interactions(max_order)` | Add interaction terms |
| `add_transcendental(funcs)` | Add transcendental functions |
| `add_ratios()` | Add ratio terms |
| `add_custom(name, func, complexity)` | Add custom function |
| `build_default(max_poly_degree)` | Build default library |
| `evaluate(X)` | Evaluate all basis functions |
| `save(filepath)` | Save library to JSON |
| `load(filepath)` | Load library from JSON |

### SymbolicRegressor

```python
class SymbolicRegressor(
    basis_library=None,
    max_terms=5,
    strategy="greedy_forward",
    information_criterion="bic",
    cv_folds=5,
    regularization=None,
    constraints=None,
    random_state=None,
)
```

Main symbolic regression model with scikit-learn interface.

**Parameters:**
- `basis_library` (BasisLibrary): Candidate basis functions
- `max_terms` (int): Maximum terms in expression
- `strategy` (str): Selection strategy
- `information_criterion` (str): Model selection criterion
- `constraints` (Constraints): Physical constraints

**Attributes (after fitting):**
- `expression_`: Human-readable expression
- `coefficients_`: Fitted coefficients
- `selected_features_`: Selected basis function names
- `complexity_`: Total complexity score
- `metrics_`: Dictionary of evaluation metrics
- `pareto_front_`: Pareto-optimal models
- `sigma_`: Estimated noise std dev `sqrt(SSR/(n-p))`
- `covariance_matrix_`: Coefficient covariance matrix `s^2 * (Phi^T Phi)^{-1}`

**Methods:**

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Fit the model |
| `predict(X)` | Make predictions |
| `score(X, y)` | Compute R² score |
| `update(X_new, y_new)` | Update with new data |
| `predict_interval(X, alpha)` | Prediction intervals for new observations |
| `confidence_band(X, alpha)` | Confidence band on mean response |
| `coefficient_intervals(alpha)` | CIs for all coefficients |
| `predict_ensemble(X)` | Pareto-front ensemble predictions |
| `predict_bma(X, criterion, alpha)` | Bayesian Model Averaging prediction |
| `predict_conformal(X, alpha, method)` | Conformal prediction intervals |
| `to_sympy()` | Export to SymPy |
| `to_latex()` | Export to LaTeX |
| `to_callable()` | Export to NumPy function |
| `save(filepath)` | Save model to JSON |
| `load(filepath)` | Load model from JSON |
| `summary()` | Get model summary |

### Constraints

```python
class Constraints()
```

Builder for physical constraints.

**Methods:**

| Method | Description |
|--------|-------------|
| `add_bounds(target, lower, upper)` | Add output bounds |
| `add_monotonic(feature, direction)` | Add monotonicity |
| `add_convex(feature)` | Add convexity |
| `add_concave(feature)` | Add concavity |
| `add_sign_constraint(basis_name, sign)` | Constrain coefficient sign |
| `add_linear_constraint(A, b)` | Add linear constraint |
| `add_known_coefficient(name, value)` | Fix coefficient value |

### AdaptiveSampler

```python
class AdaptiveSampler(
    model,
    bounds,
    strategy="uncertainty",
    batch_size=5,
    n_candidates=1000,
    random_state=None,
)
```

Adaptive sampling for iterative model improvement.

**Methods:**

| Method | Description |
|--------|-------------|
| `suggest(n_points)` | Suggest new points to query |

## Convenience Functions

### fit_symbolic

```python
fit_symbolic(
    X, y,
    feature_names=None,
    max_terms=5,
    max_poly_degree=3,
    include_transcendental=True,
    include_ratios=False,
    strategy="greedy_forward",
    information_criterion="bic",
)
```

Quick symbolic regression with automatic library construction.

## Metrics Functions

```python
from jaxsr import (
    compute_aic,
    compute_bic,
    compute_aicc,
    compute_mse,
    compute_rmse,
    compute_mae,
    compute_r2,
    compute_all_metrics,
    cross_validate,
)
```

## Selection Functions

```python
from jaxsr import (
    greedy_forward_selection,
    greedy_backward_elimination,
    exhaustive_search,
    lasso_path_selection,
    compute_pareto_front,
)
```

## Sampling Utilities

```python
from jaxsr import (
    latin_hypercube_sample,
    sobol_sample,
    halton_sample,
    grid_sample,
)
```

## Uncertainty Quantification

### Core Functions

```python
from jaxsr import (
    compute_unbiased_variance,    # s^2 = SSR/(n-p)
    compute_coeff_covariance,     # Cov(beta) = s^2 * (Phi^T Phi)^{-1}
    coefficient_intervals,         # t-based CIs for coefficients
    prediction_interval,           # Prediction + confidence intervals
    ensemble_predict,              # Pareto-front ensemble predictions
)
```

### Bayesian Model Averaging

```python
from jaxsr import BayesianModelAverage
```

```python
class BayesianModelAverage(model, criterion="bic", use_pareto=True, top_k=None)
```

IC-weighted model averaging across multiple models.

**Properties:**
- `weights`: dict mapping expression string to weight
- `expressions`: list of model expressions

**Methods:**

| Method | Description |
|--------|-------------|
| `predict(X)` | Returns `(y_mean, y_std)` — BMA mean and total std |
| `predict_interval(X, alpha)` | Returns `(y_pred, lower, upper)` — Gaussian approximation intervals |

### Conformal Prediction

```python
from jaxsr import conformal_predict_split, conformal_predict_jackknife_plus
```

| Function | Description |
|----------|-------------|
| `conformal_predict_split(model, X_cal, y_cal, X_new, alpha)` | Split conformal using held-out calibration set |
| `conformal_predict_jackknife_plus(model, X_new, alpha)` | Jackknife+ using LOO residuals (no calibration set needed) |

Both return a dict with keys `y_pred`, `lower`, `upper`.

### Bootstrap Methods

```python
from jaxsr import bootstrap_coefficients, bootstrap_predict, bootstrap_model_selection
```

| Function | Description |
|----------|-------------|
| `bootstrap_coefficients(model, n_bootstrap, alpha, seed)` | Residual bootstrap for coefficient CIs |
| `bootstrap_predict(model, X_new, n_bootstrap, alpha, seed)` | Bootstrap prediction intervals |
| `bootstrap_model_selection(model, X, y, n_bootstrap, seed)` | Pairs bootstrap for model selection stability |

`bootstrap_coefficients` returns a dict with keys: `coefficients`, `mean`, `std`, `lower`, `upper`, `names`.

`bootstrap_predict` returns a dict with keys: `y_pred`, `y_mean`, `y_std`, `lower`, `upper`.

`bootstrap_model_selection` returns a dict with keys: `feature_frequencies`, `stability_score`, `expressions`.

## Plotting Functions

```python
from jaxsr.plotting import (
    plot_pareto_front,
    plot_parity,
    plot_residuals,
    plot_coefficient_path,
    plot_feature_importance,
    plot_model_selection,
    plot_prediction_surface,
    plot_comparison,
    plot_learning_curve,
    # Uncertainty Quantification plots
    plot_prediction_intervals,
    plot_coefficient_intervals,
    plot_bma_weights,
)
```

### UQ Plots

| Function | Description |
|----------|-------------|
| `plot_prediction_intervals(model, X, y, alpha, sort_by, ax, figsize)` | Fan chart with confidence band (inner) and prediction interval (outer) |
| `plot_coefficient_intervals(model, alpha, ax, figsize)` | Forest plot with horizontal error bars for coefficient CIs |
| `plot_bma_weights(model, criterion, ax, figsize)` | Horizontal bar chart of BMA model weights |
