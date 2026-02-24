# Scikit-learn Integration

JAXSR estimators implement the scikit-learn estimator protocol (`get_params`, `set_params`),
which means they work seamlessly with sklearn's meta-tools: `clone`, `cross_val_score`,
`GridSearchCV`, `Pipeline`, and more.

**scikit-learn is not a required dependency.** Install it separately if you want to use
these tools:

```bash
pip install jaxsr[sklearn]
# or
pip install scikit-learn
```

## What Works

| sklearn tool | Status | Notes |
|-------------|--------|-------|
| `clone()` | Works | Creates unfitted copy with same config |
| `cross_val_score()` | Works | Any sklearn scoring metric |
| `GridSearchCV` | Works | Tune `max_terms`, `strategy`, etc. |
| `Pipeline` | Works | Preprocessing + symbolic regression |
| `cross_validate()` | Works | Train/test scores, timing |

## Cross-Validation

### sklearn's `cross_val_score`

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from jaxsr import BasisLibrary, SymbolicRegressor

X = np.random.randn(100, 2)
y = 2.0 * X[:, 0] + 3.0 * X[:, 1] ** 2

library = BasisLibrary(n_features=2).add_constant().add_linear().add_polynomials(max_degree=3)
model = SymbolicRegressor(basis_library=library, max_terms=5)

scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"R² = {scores.mean():.3f} ± {scores.std():.3f}")
```

### JAXSR's built-in `cross_validate`

JAXSR also provides its own `cross_validate` that works without sklearn:

```python
from jaxsr import cross_validate

results = cross_validate(model, X, y, cv=5, scoring="r2")
print(f"R² = {results['mean_test_score']:.3f} ± {results['std_test_score']:.3f}")
```

Both approaches give equivalent results. Use sklearn's version when you need
compatibility with sklearn's ecosystem (e.g., combining with other sklearn scorers).

## Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

model = SymbolicRegressor(basis_library=library)

param_grid = {
    "max_terms": [3, 5, 7],
    "strategy": ["greedy_forward", "greedy_backward"],
    "information_criterion": ["aic", "bic"],
}

grid = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=1)
grid.fit(X, y)

print(f"Best params: {grid.best_params_}")
print(f"Best R²: {grid.best_score_:.3f}")
best_model = grid.best_estimator_
print(f"Expression: {best_model.expression_}")
```

Note: Set `n_jobs=1` because JAX already parallelizes internally. Using `n_jobs=-1`
may cause issues with JAX's device management.

## Pipelines

Combine preprocessing with symbolic regression:

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

**Caveat:** The `BasisLibrary` must be pre-configured before creating the pipeline.
It is not a sklearn transformer and cannot be included as a pipeline step.

## Model Comparison

Compare symbolic regression against standard sklearn regressors:

```python
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import cross_val_score

models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.1),
    "SymbolicRegressor": SymbolicRegressor(basis_library=library, max_terms=5),
}

for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=5, scoring="r2")
    print(f"{name:25s}  R² = {scores.mean():.3f} ± {scores.std():.3f}")
```

## Using `get_params` and `set_params` Directly

```python
model = SymbolicRegressor(basis_library=library, max_terms=5)

# Inspect all parameters
print(model.get_params())

# Modify parameters
model.set_params(max_terms=3, strategy="exhaustive")
print(model.max_terms)  # 3

# Clone manually
from sklearn.base import clone
model_copy = clone(model)
```

## MultiOutputSymbolicRegressor

Nested parameters work with double-underscore syntax:

```python
from jaxsr import MultiOutputSymbolicRegressor

template = SymbolicRegressor(basis_library=library, max_terms=5)
mo = MultiOutputSymbolicRegressor(estimator=template)

# Get nested params
params = mo.get_params(deep=True)
print(params["estimator__max_terms"])  # 5

# Set nested params
mo.set_params(estimator__max_terms=3)
print(mo.estimator.max_terms)  # 3
```

## Caveats

- **JAX arrays**: JAXSR accepts both NumPy and JAX arrays transparently.
- **`n_jobs`**: Always use `n_jobs=1` with sklearn tools. JAX handles parallelism internally.
- **`BasisLibrary`**: Not a sklearn transformer. Configure it before creating the model.
- **Fitted attributes**: sklearn's `check_is_fitted` works because JAXSR sets `_is_fitted`.
