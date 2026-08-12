# Sample Weights

Use `sample_weight` when your observations are **not equally trustworthy**: different
measurement precisions, replicate-derived variances, or regions of the design space that
carry little information about the quantity you care about.

```python
model = SymbolicRegressor(basis_library=library, max_terms=4)
model.fit(X, y, sample_weight=1.0 / variances)
```

## What weights mean

Observation `i` is modelled as having variance `sigma^2 / w_i`, so the fit minimises

```
sum_i  w_i * (y_i - f(x_i))^2
```

A point with twice the weight is treated as twice as precise — equivalently, as carrying
twice the information.

**Only ratios matter.** Weights are normalised to average 1 internally, so `w`, `2*w` and
`w/1000` give the identical model, MSE and information criteria. You never have to worry
about what units your weights are in.

## Where the weights are used

Weights are not a cosmetic adjustment to the final coefficients — they run through the
whole pipeline:

| Stage | Effect |
|-------|--------|
| Term selection (all four strategies) | Every candidate subset is fitted and scored on the weighted objective |
| `model.metrics_["mse"]` | Weighted MSE, `sum_i w_i r_i^2 / n` |
| AIC / BIC / AICc | Computed from the weighted MSE (see effective sample size below) |
| `model.metrics_["r2"]`, `model.score()` | Weighted R², about the weighted mean of `y` |
| Term pruning (`prune_tol`) | Contributions measured on the weighted design matrix |
| Constraint refitting | Least-squares term weighted; the constraints themselves are not |
| `sigma_`, `covariance_matrix_`, `coefficient_intervals()` | Weighted `s^2 (Phi^T W Phi)^{-1}` |
| `predict_interval()`, `confidence_band()` | Weighted leverages |
| `bootstrap_coefficients()`, `bootstrap_predict()` | Residuals resampled in whitened space, refit by WLS |
| `bootstrap_model_selection()` | Weights follow their rows into each resample |
| `anova()` | Weighted sums of squares, about the weighted mean |
| `cross_validate()` | Weights follow their rows into the folds |
| `predict_conformal(method="jackknife+")` | Weighted leverages and whitened LOO residuals |

## Effective sample size

The `n` used by AIC/BIC/AICc stays the **nominal** number of observations. Weighting
describes how much you trust each measurement; it does not create or destroy
measurements.

This is also why **duplicating rows is not a valid way to emulate weights** — duplication
inflates `n`, and `k*log(n)` in BIC shifts with it, so the comparison between models
changes for a reason that has nothing to do with the data.

When most of the weight sits on a few rows, the nominal `n` overstates how much
information the model comparison rests on. Check:

```python
model.effective_sample_size_   # Kish ESS: (sum w)^2 / sum w^2
```

For 200 points where half carry weight ~0, this returns ~100. Treat that, not `n`, as
your intuition for how strong the evidence is.

## Recipes

### Known measurement variances

```python
model.fit(X, y, sample_weight=1.0 / variances)
```

### Replicate means

If `y_i` is the mean of `n_i` replicates with common noise variance, its variance is
`sigma^2 / n_i`, so the weight is the replicate count:

```python
model.fit(X_means, y_means, sample_weight=replicate_counts)
```

### A smooth taper instead of a hard threshold

Rows where a derived quantity is near zero carry little information about it. Rather than
dropping them below a cutoff — which discards partially-informative rows and puts a step
where a taper belongs — down-weight them continuously:

```python
# information about the slope scales with |y_x|; taper smoothly near zero
w = y_x**2 / (y_x**2 + eps**2)
model.fit(X, y_ratio, sample_weight=w)
```

Pick `eps` on the scale of the noise in `y_x`, not as a hard "keep/drop" boundary.

### Excluding a row entirely

A weight of exactly `0` removes a row from the fit but keeps it in `n`. If you mean to
drop the observation, drop it from `X` and `y` instead — that also corrects `n` and every
criterion computed from it.

## Interpreting weighted intervals

For a weighted fit, `sigma_` is the noise level of a **unit-weight** observation, and
`predict_interval()` returns the interval for a new observation of unit weight — one as
precise as an average training point. If the new observation has a known precision
`w_new`, scale the half-width by `1 / sqrt(w_new)`.

## Validation

Bad weights are rejected up front rather than silently absorbed:

- wrong length → `ValueError`
- negative entries → `ValueError`
- `NaN` / `inf` → `ValueError`
- all zeros → `ValueError`

## Not weighted

- `conformal_predict_split()` — its coverage comes from exchangeability of the
  user-supplied calibration residuals, not from the fit. With unequal-precision
  calibration points the interval stays valid but is uninformatively wide for the precise
  ones. Use `method="jackknife+"` for a weight-aware interval.
- `SymbolicClassifier` — classification weights are not implemented.
- `max_error` in `compute_all_metrics()` — a maximum has no weighted analogue; it is
  reported over the rows with non-zero weight.

## See also

- `guides/uncertainty.md` — interval methods and when to use each
- `guides/model-fitting.md` — selection strategies and criteria
