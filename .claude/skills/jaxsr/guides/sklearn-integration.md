# Scikit-learn Integration Guide

JAXSR estimators implement the full scikit-learn estimator protocol (`get_params`,
`set_params`, `__sklearn_tags__`, `__sklearn_is_fitted__`), enabling seamless use
with sklearn's meta-tools without requiring sklearn as a hard dependency.

## What Works

| sklearn tool | Works? | Notes |
|-------------|--------|-------|
| `sklearn.base.clone()` | Yes | Creates unfitted copy with same config |
| `cross_val_score()` | Yes | Any sklearn scoring metric |
| `cross_validate()` | Yes | Train/test scores, timing |
| `GridSearchCV` | Yes | Tune `max_terms`, `strategy`, `information_criterion` |
| `Pipeline` | Yes | Preprocessing + symbolic regression |
| `check_is_fitted()` | Yes | Uses `_is_fitted` attribute |

## Installation

```bash
pip install jaxsr[sklearn]
# or separately:
pip install scikit-learn
```

sklearn is optional — all JAXSR functionality works without it.

## Estimator Protocol

All three JAXSR estimator classes inherit from `_SklearnCompatMixin`:

- `SymbolicRegressor` — 11 constructor params
- `MultiOutputSymbolicRegressor` — 2 params (`estimator`, `target_names`)
- `SymbolicClassifier` — 9 constructor params

### `get_params(deep=True)`

Returns a dict of all constructor parameters. When `deep=True`, recurses
into sub-estimators (relevant for `MultiOutputSymbolicRegressor.estimator`),
producing nested keys like `estimator__max_terms`.

```python
model = SymbolicRegressor(basis_library=lib, max_terms=5)
model.get_params()
# {'basis_library': <BasisLibrary>, 'max_terms': 5, 'strategy': 'greedy_forward', ...}

mo = MultiOutputSymbolicRegressor(estimator=model)
mo.get_params(deep=True)
# {'estimator': <SymbolicRegressor>, 'target_names': None,
#  'estimator__max_terms': 5, 'estimator__strategy': 'greedy_forward', ...}
```

### `set_params(**params)`

Sets parameters. Supports double-underscore nested syntax.

```python
model.set_params(max_terms=3, strategy="exhaustive")
mo.set_params(estimator__max_terms=8)
```

Raises `ValueError` for invalid parameter names.

## Cross-Validation

### sklearn's `cross_val_score`

```python
from sklearn.model_selection import cross_val_score
from jaxsr import BasisLibrary, SymbolicRegressor

library = BasisLibrary(n_features=2).add_constant().add_linear().add_polynomials(3)
model = SymbolicRegressor(basis_library=library, max_terms=5)

scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"R² = {scores.mean():.3f} ± {scores.std():.3f}")
```

### JAXSR's built-in `cross_validate`

JAXSR provides its own `cross_validate` that works without sklearn:

```python
from jaxsr import cross_validate

results = cross_validate(model, X, y, cv=5, scoring="r2")
print(f"R² = {results['mean_test_score']:.3f} ± {results['std_test_score']:.3f}")
```

Both give equivalent results. Use sklearn's version for ecosystem compatibility.

## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_terms": [3, 5, 7],
    "strategy": ["greedy_forward", "greedy_backward"],
    "information_criterion": ["aic", "bic"],
}

grid = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=1)
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")
print(f"Best expression: {grid.best_estimator_.expression_}")
```

## Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("sr", SymbolicRegressor(basis_library=library, max_terms=5)),
])
pipe.fit(X, y)
y_pred = pipe.predict(X)
```

## Model Comparison

```python
from sklearn.linear_model import Lasso, LinearRegression

models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.1),
    "SymbolicRegressor": SymbolicRegressor(basis_library=library, max_terms=5),
}

for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=5, scoring="r2")
    print(f"{name:25s}  R² = {scores.mean():.3f} ± {scores.std():.3f}")
```

## Caveats and Common Mistakes

| Issue | Guidance |
|-------|----------|
| `n_jobs` | Always use `n_jobs=1`. JAX handles parallelism internally. |
| `BasisLibrary` in Pipeline | Not a transformer. Configure before creating the estimator. |
| JAX arrays | Accepted transparently — sklearn will pass NumPy arrays which JAXSR converts. |
| `check_is_fitted` | Works because JAXSR sets `_is_fitted` attribute. |
| Cloning | `sklearn.base.clone()` and JAXSR's `_clone_estimator()` both use `get_params()`. |
