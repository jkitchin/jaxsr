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
| **Backfitting additive** (`jaxsr.additive.BackfittingSymbolicRegressor`) | Maintains a fixed set of terms and **revises** each one in place across sweeps (BART/iBART-style). | Scaffold / planned |

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
| `loss` | Loss function. Currently only `"squared_error"`. |
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

## Early stopping

With `early_stopping=True`, a validation split (`validation_fraction`) is held
out. After each stage the validation loss is recorded; training stops once it
fails to improve by at least `min_delta` for `patience` consecutive stages, and
the model rolls back to the best iteration.

## Losses (extensibility)

The loss abstraction in `jaxsr.additive.losses` is designed for future gradient
boosting: each weak learner fits the negative gradient `-dL/dy_pred`. Only
`SquaredError` (whose pseudo-residual is the ordinary residual `y - y_pred`) is
implemented today. Additional losses (absolute error, Huber, Poisson, logistic,
quantile) can be added by subclassing `Loss` and registering them.

## Backfitting (planned)

`BackfittingSymbolicRegressor` is currently a documented scaffold that raises
`NotImplementedError`. The planned sweep is:

```
for term j in 1..m:
    partial_residual = y - (intercept + sum_{i != j} coef_i * g_i(X))
    re-discover / refit term j on partial_residual
    reinsert term j
(repeat sweeps until convergence)
```

Use `StagewiseSymbolicRegressor` for additive symbolic regression today.
