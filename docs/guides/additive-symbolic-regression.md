# Additive Symbolic Regression

Additive symbolic regression fits a model as a **sum of small symbolic
expressions**:

```
f(x) = c + eta_1 * g_1(x) + eta_2 * g_2(x) + ... + eta_K * g_K(x)
```

where each `g_k(x)` is a small, interpretable symbolic expression discovered by
the existing JAXSR machinery. This is analogous to **gradient boosting**, except
each weak learner is a symbolic expression rather than a decision tree.

The submodule lives in `jaxsr.additive`.

## Three flavours of symbolic regression

| Approach | What it does | Status |
|----------|--------------|--------|
| **Single-expression** (`jaxsr.SymbolicRegressor`) | Fits one sparse expression over a fixed basis library. | Available |
| **Stagewise additive** (`jaxsr.additive.StagewiseSymbolicRegressor`) | Repeatedly fits a small expression to the *residual* and adds it to the ensemble. Old terms are **frozen**. | Available |
| **Backfitting additive** (`jaxsr.additive.BackfittingSymbolicRegressor`) | Maintains a fixed set of terms and **revises** each one in place across sweeps (GAM-style). | Available (squared error); Bayesian variant planned |

The key distinction between the two additive variants:

- **Stagewise**: once a term is discovered it never changes; only its linear
  weight may be re-estimated.
- **Backfitting**: terms are revised repeatedly, each conditioned on the current
  fit of all the others.

## Quick start

```python
import numpy as np
from jaxsr.additive import StagewiseSymbolicRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(200, 2))
y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.normal(size=200)

model = StagewiseSymbolicRegressor(
    n_terms=5,
    learning_rate=0.2,
    max_complexity=6,
    refit_coefficients=True,
)
model.fit(X, y)

print(model)                 # pretty structural summary
print(model.expressions_)    # per-term expression strings
print(model.coefficients_)   # per-term weights
print(model.intercept_)      # additive intercept
y_pred = model.predict(X)
```

The `print(model)` output looks like:

```
StagewiseSymbolicRegressor(
    intercept = 1.07
    terms =
        + 1 * (y = 2*x0 - 1.07 + 0.5*x1^2)
        ...
)
```

## The stagewise algorithm

1. Initialise the intercept to `mean(y)` and the prediction to that constant.
2. Compute the residual `y - prediction`.
3. Fit a small symbolic expression `g_k` to the residual (via
   `jaxsr.fit_symbolic`).
4. Append `g_k` to the ensemble.
5. If `refit_coefficients=True`, rebuild the design matrix `Phi[:, j] = g_j(X)`
   and re-solve `y ~= intercept + Phi @ coefficients` by least squares.
   Otherwise, update `prediction += learning_rate * g_k(X)`.
6. Record train (and optional validation) loss.
7. Repeat until `n_terms` terms are added or early stopping triggers.

## Key parameters

| Parameter | Meaning |
|-----------|---------|
| `n_terms` | Maximum number of boosting stages (terms). |
| `learning_rate` | Shrinkage on each stage when `refit_coefficients=False`. |
| `max_complexity` | Complexity budget per term (max basis terms). Keep small to favour many simple terms. |
| `refit_coefficients` | Re-solve all linear weights by OLS after each stage. |
| `loss` | `"squared_error"` (default), `"absolute_error"`, `"huber"`, `"quantile"`, or a `Loss` instance. See [Losses](#losses-robust-and-quantile-regression). |
| `early_stopping` | Hold out a validation split and stop when it stops improving. |
| `validation_fraction`, `patience`, `min_delta` | Early-stopping controls. |
| `max_poly_degree`, `include_transcendental`, `include_ratios` | Which basis functions each term may use. |
| `information_criterion` | Complexity control within each term (`"aic"`, `"aicc"`, `"bic"`). |

## Coefficient refitting

- `refit_coefficients=False`: the weights are the learning-rate-scaled stagewise
  weights (`coefficients_[k] == learning_rate`).
- `refit_coefficients=True`: after each new term, the intercept and *all* per-term
  weights are re-solved by ordinary least squares over the discovered symbolic
  features. This decouples term discovery (nonlinear, greedy) from term weighting
  (linear, global) and typically improves accuracy.

The refit uses `jnp.linalg.lstsq` (SVD-based, minimum-norm), so the highly
correlated columns produced by later boosting stages do not cause instability.

## Combined expression

`model.to_expression()` returns a single simplified SymPy expression combining
all terms:

```python
expr = model.to_expression()   # requires sympy
```

## Saving and loading

Fitted models serialize to JSON (each term is stored via the underlying
`SymbolicRegressor` state), mirroring the rest of jaxsr. Note that the models
are **not picklable** — the basis-function closures cannot be pickled — so use
`save`/`load` rather than `pickle`:

```python
model.save("additive_model.json")
loaded = StagewiseSymbolicRegressor.load("additive_model.json")
```

## Structural uncertainty (bootstrap)

A single fitted expression can hide the fact that the *structure* itself is
uncertain — several different basis sets may explain the data about equally
well (this is common with collinear features). `bootstrap_additive` refits the
model on bootstrap resamples and reports, for each basis function, how often it
is selected — a cheap approximation to a posterior inclusion probability —
together with a predictive ensemble:

```python
from jaxsr.additive import (
    StagewiseSymbolicRegressor,
    bootstrap_additive,
    bootstrap_predict_additive,
)

est = StagewiseSymbolicRegressor(n_terms=3, max_complexity=2)
res = bootstrap_additive(est, X, y, n_bootstrap=100, random_state=0)

# How stable is the discovered structure?
for name, prob in res["inclusion_probabilities"].items():
    print(f"{name:10s} selected in {prob:.0%} of resamples")

# Prediction intervals that reflect *structural* variability, not just noise
pi = bootstrap_predict_additive(res["models"], X_new, alpha=0.1)
pi["mean"], pi["lower"], pi["upper"]
```

**How to read it.** Inclusion probabilities near 0 or 1 mean the structure is
identifiable and the single fitted expression is trustworthy. **Diffuse** values
(e.g. a basis selected 50–60% of the time) mean the data do not determine one
expression — no single symbolic model should be over-trusted, and the bootstrap
intervals are the honest summary. This also works as a decision gate for heavier
Bayesian modelling: if the probabilities are already crisp, there is little
structural uncertainty left to quantify. It works for both the stagewise and
backfitting regressors.

## Early stopping

With `early_stopping=True`, a validation split (`validation_fraction`) is held
out. After each stage the validation loss is recorded; training stops once it
fails to improve by at least `min_delta` for `patience` consecutive stages, and
the model rolls back to the best iteration.

## Losses: robust and quantile regression

This is where additive symbolic regression goes beyond ordinary least-squares
symbolic regression. Each weak learner fits the negative gradient `-dL/dy_pred`
(gradient boosting), so you can target losses that OLS selection cannot:

| `loss` | Class | Use when |
|--------|-------|----------|
| `"squared_error"` (default) | `SquaredError` | Standard regression |
| `"absolute_error"` | `AbsoluteError` | Outliers present (fits the median) |
| `"huber"` | `HuberLoss(delta=1.35)` | Outliers, but keep efficiency near zero |
| `"quantile"` | `QuantileLoss(quantile=0.5)` | Quantiles / prediction intervals / asymmetric cost |

Pass a name for defaults, or an instance to customise:

```python
from jaxsr.additive import StagewiseSymbolicRegressor, QuantileLoss, HuberLoss

# Robust regression: heavy outliers barely move the fit
robust = StagewiseSymbolicRegressor(loss="huber", learning_rate=0.5).fit(X, y)

# 90th-percentile regression (build intervals by fitting several quantiles)
q90 = StagewiseSymbolicRegressor(loss=QuantileLoss(0.9), learning_rate=0.5).fit(X, y)
```

**How non-squared losses are fit.** Each stage fits a symbolic term to the
negative gradient, then a **line search** picks the step size that minimises the
loss (`learning_rate` shrinks that step). Because the ordinary least-squares
coefficient refit targets squared error, `refit_coefficients=True` is ignored for
non-squared losses (a warning is issued) and gradient boosting is used instead —
so set `refit_coefficients=False` explicitly for robust/quantile models.

The optimal constant initialisation adapts to the loss: mean for squared error,
median for absolute/Huber, and the empirical quantile for quantile loss.

Add further losses (Poisson, logistic, ...) by subclassing `Loss` and
registering them in `jaxsr.additive.losses._LOSSES`.

## Backfitting (GAM-style)

`BackfittingSymbolicRegressor` maintains a **fixed** number of terms and
*revises* each one across sweeps, instead of freezing them. Each sweep removes a
term, re-discovers its expression on the partial residual, and puts it back:

```python
from jaxsr.additive import BackfittingSymbolicRegressor

model = BackfittingSymbolicRegressor(n_terms=4, n_sweeps=6, max_complexity=3)
model.fit(X, y)   # warm-started from a stagewise fit, then refined by sweeps
```

```
for sweep in 1..n_sweeps:
    for term j:
        partial_residual = y - intercept - sum_{i != j} coef_i * g_i(X)
        g_j = fit_symbolic(X, partial_residual, ...)   # re-discover structure
    intercept, coef = OLS refit over all terms
(stop when the training loss stops improving by `tol`)
```

It is warm-started from a stagewise fit and currently supports **squared error
only**. Structure re-discovery makes the sweep a heuristic (no monotonicity
guarantee), so the best-loss iterate is kept.

**When does it actually help?** Backfitting starts from the stagewise+refit fit
and keeps the best-loss iterate, so **it is never worse than
`StagewiseSymbolicRegressor(refit_coefficients=True)` on the training data** —
the only question is whether the sweeps improve on it. That hinges entirely on
whether re-discovery changes the *set* of selected basis functions:

- **Generous per-term budget** (`max_complexity` ≥ 2–3): greedy usually already
  finds a sufficient basis set, so the joint least-squares refit makes the two
  essentially identical. Backfitting adds nothing here — prefer the stagewise
  regressor.
- **Small per-term budget and collinear features** (`max_complexity=1`, the
  GAM-style single-basis regime): greedy forward selection can lock into a
  *suboptimal* basis set that a single forward pass cannot undo. Backfitting's
  coordinate-descent re-discovery escapes it, changing the basis union and
  improving the fit — we have measured up to roughly **+0.04 train / +0.06 test
  R²** in this regime, with no downside in the cases where it does not help.

So reach for backfitting when you want **small, revisable single-basis terms
over correlated features** (or a fixed-size GAM-style decomposition); use the
stagewise regressor for larger per-term expressions. Its other forward-looking
value is as the foundation for a future **Bayesian backfitting** variant
(BART/iBART-style), which would sample a *posterior over symbolic structure* —
genuinely beyond point-estimate SR — using the same partial-residual sweep with
conjugate marginal likelihoods.
