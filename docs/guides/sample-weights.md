# Sample Weights

Not every measurement deserves the same say in a fit. Use `sample_weight` when
observations differ in precision — different instruments, replicate-derived variances, or
regions of the design space that carry little information about the quantity you care
about.

```python
import numpy as np
from jaxsr import BasisLibrary, SymbolicRegressor

X = np.random.randn(100, 2)
y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2
variances = np.repeat([0.01, 1.0], 50)  # first half measured 100x more precisely

library = (
    BasisLibrary(n_features=2, feature_names=["a", "b"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=2)
)
model = SymbolicRegressor(basis_library=library, max_terms=3)
model.fit(X, y, sample_weight=1.0 / variances)
print(model.expression_)
```

## What a weight means

Observation `i` is modelled as having variance `sigma² / wᵢ`, so the fit minimises

$$\sum_i w_i \left(y_i - f(x_i)\right)^2$$

A point with twice the weight is treated as twice as precise, i.e. as carrying twice the
information.

**Only ratios matter.** Weights are normalised to average 1 internally, so `w`, `2*w` and
`w/1000` give an identical model, MSE and information criteria. The units of your weights
are irrelevant — pass raw inverse variances without rescaling.

## Weights apply everywhere, not just to the coefficients

A weighted fit whose model *selection* is unweighted, or whose reported R² is unweighted,
is worse than no weighting at all — it looks right and isn't. In JAXSR the weights run
through the whole pipeline:

| Stage | Effect of weights |
|-------|-------------------|
| Term selection (`greedy_forward`, `greedy_backward`, `exhaustive`, `lasso_path`) | Every candidate subset is fitted and scored on the weighted objective |
| `model.metrics_["mse"]` | Weighted MSE, $\sum_i w_i r_i^2 / n$ |
| AIC / BIC / AICc | Computed from the weighted MSE |
| `model.metrics_["r2"]`, `model.score()` | Weighted R², about the weighted mean of `y` |
| Negligible-term pruning (`prune_tol`) | Contributions measured on the weighted design matrix |
| Constraint refitting | The least-squares term is weighted; the constraints themselves are not |
| `sigma_`, `covariance_matrix_`, `coefficient_intervals()` | Weighted $s^2 (\Phi^\top W \Phi)^{-1}$ |
| `predict_interval()`, `confidence_band()` | Weighted leverages |
| `bootstrap_coefficients()`, `bootstrap_predict()` | Residuals resampled in whitened space, each replicate refit by weighted least squares |
| `bootstrap_model_selection()` | Each weight follows its row into the resample, groups included |
| `anova()` | Weighted sums of squares, about the weighted mean |
| `cross_validate()` | Each weight follows its row into the folds, under every splitting strategy |
| `predict_conformal(method="jackknife+")` | Weighted leverages, whitened LOO residuals |
| Parametric bases (`add_parametric`) | Nonlinear parameters are optimised against the weighted objective |

## Effective sample size

The `n` used by AIC, BIC and AICc stays the **nominal** number of observations. Weighting
says how much you trust each measurement; it does not create or destroy measurements.

This is why **duplicating rows is not a valid way to emulate weights**. Duplication
inflates `n`, and the `k·log(n)` term in BIC moves with it, so the comparison between
models shifts for a reason unrelated to the data.

When most of the weight sits on a handful of rows, the nominal `n` overstates how much
information a model comparison actually rests on. Check the Kish effective sample size:

```python
model.effective_sample_size_   # (Σw)² / Σw²
```

For 200 points where half carry weight ≈ 0, this returns ≈ 100. Use that number, not `n`,
as your intuition for how strong the evidence is.

## Recipes

### Known measurement variances

```python
model.fit(X, y, sample_weight=1.0 / variances)
```

### Replicate means

If `yᵢ` is the mean of `nᵢ` replicates sharing a noise variance, its variance is
`sigma²/nᵢ`, so the weight is the replicate count:

```python
model.fit(X_means, y_means, sample_weight=replicate_counts)
```

### A smooth taper instead of a hard threshold

Some rows carry little information about the quantity being fitted — for example when a
derived response is a ratio whose denominator passes through zero. Dropping those rows
below a cutoff discards partially-informative data and puts a step where a taper belongs.
Down-weight them continuously instead:

```python
# information about the slope scales with |y_x|; taper smoothly near zero
w = y_x**2 / (y_x**2 + eps**2)
model.fit(X, y_ratio, sample_weight=w)
```

Choose `eps` on the scale of the noise in `y_x`, not as a keep/drop boundary.

### Excluding a row

A weight of exactly `0` removes a row from the fit but keeps it in `n`. If you mean to
drop the observation, drop it from `X` and `y` instead — that also corrects `n` and every
criterion computed from it.

## Interpreting weighted intervals

For a weighted fit, `sigma_` is the noise level of a **unit-weight** observation, and
`predict_interval()` gives the interval for a new observation of unit weight — one as
precise as an average training point. If the new observation has a known precision
`w_new`, scale the half-width by `1/sqrt(w_new)`.

```python
model.fit(X, y, sample_weight=1.0 / variances)
y_pred, lo, hi = model.predict_interval(X_test)   # for a unit-weight observation
```

## Validation

Bad weights are rejected up front rather than silently absorbed. Each of these raises
`ValueError`:

- an array whose length differs from `len(y)`
- negative entries
- `NaN` or `inf`
- all zeros

## Weights are orthogonal to the resampling level

`cross_validate` and `bootstrap_model_selection` both let you choose *what* gets
resampled — rows, or whole `groups`. That choice and the weights are independent: a
weight describes how precise a row is, a group describes which rows are not independent
of each other. Pass both when both are true.

```python
cross_validate(model, X, y, groups=run_id, sample_weight=1 / variances)
bootstrap_model_selection(model, X, y, groups=run_id, sample_weight=1 / variances)
```

A group whose rows all carry zero weight is scored `NaN` rather than `0.0`, so it cannot
be mistaken for a group the model predicted perfectly. A *fold* with no weight on either
side raises instead — there is nothing there to fit or score.

`bootstrap_model_selection(..., resample_fn=...)` rejects `sample_weight`: each replicate
regenerates its own rows, so a stored weight has no row to belong to. Apply the weighting
inside your `resample_fn`.

## What is not weighted

- **`conformal_predict_split()`** — its coverage guarantee comes from exchangeability of
  the user-supplied calibration residuals, not from the fit. With calibration points of
  unequal precision the interval remains valid but is uninformatively wide for the
  precise ones. Use `method="jackknife+"` for a weight-aware interval.
- **`SymbolicClassifier`** — weights are not implemented for classification.
- **`max_error`** in `compute_all_metrics()` — a maximum has no weighted analogue; it is
  reported over the rows with non-zero weight.

## See also

- [Uncertainty quantification example](../examples/uncertainty_quantification.ipynb)
- [API reference](../api/index.rst)
