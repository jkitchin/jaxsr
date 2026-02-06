# Active Learning & Adaptive Sampling Guide

## What is Active Learning?

Instead of running all experiments upfront, active learning iteratively:
1. Fit a model to current data
2. Use the model to suggest the most informative next experiments
3. Run those experiments
4. Repeat until the model is good enough

This can dramatically reduce the number of experiments needed.

## AdaptiveSampler: Simple Interface

```python
from jaxsr import AdaptiveSampler

sampler = AdaptiveSampler(
    model=model,  # a fitted SymbolicRegressor
    bounds=[(300, 500), (1, 10)],  # Feature bounds
    strategy="uncertainty",
    batch_size=5,
    n_candidates=1000,
    random_state=42
)

# Get suggestions
result = sampler.suggest(n_points=5)
X_next = result.points    # shape (5, n_features)
scores = result.scores    # acquisition function values
```

## Sampling Strategies

### `uncertainty`

Suggests points where the model is most uncertain (largest prediction intervals).

**Best for:** General exploration, reducing model error everywhere.

### `error`

Suggests points where the model prediction error is expected to be highest.
Uses ensemble disagreement or residual extrapolation.

**Best for:** Improving accuracy in poorly-modeled regions.

### `leverage`

Suggests points with high statistical leverage — positions that would have the
most influence on the fitted coefficients.

**Best for:** Stabilizing coefficient estimates, optimal experimental design.

### `gradient`

Suggests points where the model gradient is steepest — regions of rapid change.

**Best for:** Resolving sharp transitions, nonlinear behavior.

### `space_filling`

Suggests points that fill gaps in the existing design.
Uses maximin distance criterion.

**Best for:** Initial exploration, uniform coverage, no model required.

### `random`

Uniform random sampling within bounds.

**Best for:** Baseline comparison, sanity check.

## Strategy Selection Guide

```
Do you have a fitted model?
├── NO → Use "space_filling" or "random"
└── YES
    ├── Want to reduce overall uncertainty?
    │   └── Use "uncertainty"
    ├── Want to improve accuracy in worst regions?
    │   └── Use "error"
    ├── Want to stabilize coefficients?
    │   └── Use "leverage"
    └── Want to resolve sharp features?
        └── Use "gradient"
```

## Acquisition Functions: Advanced Interface

An **acquisition function** scores candidate points by how useful they would be as
the next experiment. Higher scores mean more informative. For fine-grained control,
use acquisition functions directly:

```python
from jaxsr.acquisition import (
    ActiveLearner,
    PredictionVariance,
    UCB, LCB,
    ExpectedImprovement,
    EnsembleDisagreement,
    BMAUncertainty,
    AOptimal, DOptimal,
    Composite,
    suggest_points
)
```

### Prediction Variance

```python
acq = PredictionVariance()
learner = ActiveLearner(model, bounds=[(0, 1), (0, 1)], acquisition=acq)
result = learner.suggest(n_points=5)
```

### Upper/Lower Confidence Bound

```python
# Explore high-response regions (maximize)
acq = UCB(kappa=2.0)     # mean + kappa * std

# Explore low-response regions (minimize)
acq = LCB(kappa=2.0)     # mean - kappa * std
```

`kappa` controls exploration vs. exploitation:
- High kappa (> 2) → more exploration (wider search)
- Low kappa (< 1) → more exploitation (stay near known good areas)

### Expected Improvement

```python
acq = ExpectedImprovement()  # For finding the maximum of the response
```

Classic Bayesian optimization acquisition function. Balances exploitation
(predicted value) and exploration (uncertainty).

### Ensemble Disagreement

```python
acq = EnsembleDisagreement()  # Where Pareto-front models disagree most
```

### Optimal Design Criteria

```python
acq = AOptimal()    # Minimize average variance of coefficients
acq = DOptimal()    # Minimize volume of coefficient confidence ellipsoid
```

### Composite Acquisition Functions

Combine multiple acquisition functions with weights:

```python
# 70% uncertainty + 30% space-filling
combined = 0.7 * PredictionVariance() + 0.3 * AOptimal()

learner = ActiveLearner(model, bounds=bounds, acquisition=combined)
result = learner.suggest(n_points=5)
```

You can also use the `Composite` class:
```python
combined = Composite(functions=[(0.7, PredictionVariance()), (0.3, AOptimal())])
```

## Complete Active Learning Loop

```python
import numpy as np
from jaxsr import BasisLibrary, SymbolicRegressor, AdaptiveSampler

# Define your experiment (replace with real measurements)
def true_function(X):
    """Placeholder — replace with your actual experiment or simulation."""
    return 3.0 * X[:, 0] + 0.5 * X[:, 1] - 0.01 * X[:, 0]**2

# Initial data (small)
X_init = np.random.uniform([300, 1], [500, 10], size=(10, 2))
y_init = true_function(X_init)

# Build library
library = (BasisLibrary(n_features=2, feature_names=["T", "P"])
    .add_constant().add_linear()
    .add_polynomials(max_degree=2)
    .add_interactions(max_order=2)
)

# Fit initial model
model = SymbolicRegressor(basis_library=library, max_terms=5)
model.fit(X_init, y_init)

X_all, y_all = X_init.copy(), y_init.copy()
bounds = [(300, 500), (1, 10)]

for iteration in range(5):
    print(f"\n--- Iteration {iteration + 1} ---")
    print(f"Model: {model.expression_}")
    print(f"R²: {model.metrics_['r2']:.4f}")

    # Suggest next experiments
    sampler = AdaptiveSampler(model, bounds, strategy="uncertainty", batch_size=5)
    result = sampler.suggest(n_points=5)
    X_next = result.points

    # Run experiments (replace with actual experiments)
    y_next = true_function(X_next)

    # Update model
    X_all = np.vstack([X_all, X_next])
    y_all = np.concatenate([y_all, y_next])
    model.fit(X_all, y_all)

print(f"\nFinal model ({len(y_all)} points): {model.expression_}")
```

## DOE Integration

The `DOEStudy` class wraps active learning into the study workflow:

```python
from jaxsr import DOEStudy

study = DOEStudy.load("my_study.jaxsr")

# Suggest next experiments using the fitted model
next_pts = study.suggest_next(n_points=5, strategy="uncertainty")

# Or via CLI
# jaxsr suggest my_study.jaxsr -n 5 --strategy uncertainty
```

## Discrete/Integer Variables

If some features take only discrete values:

```python
sampler = AdaptiveSampler(
    model, bounds,
    strategy="uncertainty",
    discrete_dims={1: [1, 2, 3, 4, 5]}  # Feature index 1: valid discrete values
)
```

## Excluding Regions

Avoid suggesting points too close to existing data:

```python
result = sampler.suggest(
    n_points=5,
    exclude_points=X_existing,    # Don't suggest near these
    min_distance=0.01             # Minimum distance threshold
)
```

## Batch vs. Sequential

- **Sequential** (n_points=1): Optimal but requires one experiment at a time.
- **Batch** (n_points=5+): Less optimal but allows parallel experiments.

For batch suggestions, points are selected greedily to avoid clustering:
each new point considers previously selected points in the batch.
