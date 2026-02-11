# Design of Experiments with JAXSR

This guide explains how to use JAXSR for **design of experiments (DOE)** —
systematically choosing which experiments to run so you get the most
information from the fewest measurements.

---

## Why JAXSR for DOE?

Traditional DOE tools (factorial designs, response surface methods) create
a fixed experimental plan *before* collecting any data. JAXSR goes further
by combining **symbolic regression** with **active learning**, enabling:

1. **Adaptive designs** — each batch of experiments is informed by
   everything you've learned so far.
2. **Exact uncertainty** — because JAXSR models are linear-in-parameters
   (`y = Φ(X) β`), prediction variance is available in closed form. No
   Gaussian process approximations or MCMC needed.
3. **Interpretable models** — you get an algebraic equation, not a black
   box. This lets you verify that the model obeys domain knowledge.
4. **Physical constraints** — encode monotonicity, bounds, convexity, or
   sign constraints directly into the regression.

---

## Overview of the DOE Workflow

```
┌────────────────────────┐
│  1. Initial Design     │  Generate a small space-filling design
│     (LHS, Sobol, etc.) │  to bootstrap the model.
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│  2. Run Experiments    │  Collect measurements y at the
│                        │  suggested input points X.
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│  3. Fit Model          │  SymbolicRegressor discovers an
│                        │  interpretable expression y = f(X).
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│  4. Suggest Next       │  Acquisition functions score candidate
│     Experiments        │  points; pick the most informative ones.
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│  5. Converged?         │  Check if the model is good enough.
│     If not, go to 2.   │  If not, repeat.
└────────────────────────┘
```

---

## Step 1: Initial Experimental Design

Before fitting any model you need a small initial dataset. JAXSR provides
several space-filling designs in `jaxsr.sampling`:

```python
from jaxsr.sampling import (
    latin_hypercube_sample,
    sobol_sample,
    halton_sample,
    grid_sample,
    d_optimal_select,
)

# Define the experimental region
bounds = [
    (300, 500),   # Temperature (K)
    (1, 10),      # Pressure (atm)
    (0.01, 0.5),  # Concentration (mol/L)
]

# Latin Hypercube — good general-purpose default
X_init = latin_hypercube_sample(20, bounds, random_state=42)

# Sobol sequence — better uniformity in high dimensions
X_init = sobol_sample(20, bounds, random_state=42)

# Halton sequence — similar to Sobol, different construction
X_init = halton_sample(20, bounds, random_state=42)

# Full factorial grid — use only for ≤3 features with few levels
X_init = grid_sample(5, bounds)
```

### How many initial points?

A rule of thumb: start with **10–20 points per input feature**. For 3
features, 30–60 initial experiments is a reasonable range. The active
learning loop will fill in gaps efficiently.

---

## Step 2: Run Experiments and Fit a Model

After collecting measurements at the initial design points:

```python
import jax.numpy as jnp
from jaxsr import BasisLibrary, SymbolicRegressor

# Define candidate basis functions
library = (
    BasisLibrary(n_features=3, feature_names=["T", "P", "C"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=2)
    .add_interactions(max_order=2)
    .add_transcendental(["log", "exp", "sqrt", "inv"])
)

# Fit
model = SymbolicRegressor(
    basis_library=library,
    max_terms=6,
    strategy="greedy_forward",
)
model.fit(jnp.array(X_init), jnp.array(y_init))

print(model.expression_)
# e.g. "y = 2.1 + 0.05*T - 1.3/P + 4.7*C*T"
```

---

## Step 3: Choose an Acquisition Function

The acquisition function decides **where to sample next**. Your choice
depends on your goal.

### Goal A: Build an Accurate Model (Exploration)

Use these when you want the model to be accurate everywhere:

| Function | When to use |
|----------|-------------|
| `PredictionVariance()` | **Default choice.** Samples where prediction uncertainty is highest. Fast and exact. |
| `DOptimal()` | Maximises information per experiment. Best for building precise models with minimal data. |
| `AOptimal()` | Minimises uncertainty in the model coefficients. Use when you need tight confidence intervals on parameters. |
| `ConfidenceBandWidth(alpha=0.05)` | Like PredictionVariance but reports the actual 95% CI width. |
| `EnsembleDisagreement()` | Samples where Pareto-front models disagree. Good when you are unsure about model complexity. |
| `BMAUncertainty(criterion="bic")` | Most comprehensive — combines noise uncertainty and model-selection uncertainty via Bayesian Model Averaging. |

```python
from jaxsr.acquisition import ActiveLearner, PredictionVariance

learner = ActiveLearner(
    model,
    bounds=bounds,
    acquisition=PredictionVariance(),
    random_state=42,
)
result = learner.suggest(n_points=5)
```

### Goal B: Find an Optimum (Bayesian Optimisation)

Use these when you want to find the input that minimises (or maximises)
the response:

| Function | When to use |
|----------|-------------|
| `ExpectedImprovement(minimize=True)` | **Recommended default.** Naturally balances exploration and exploitation. |
| `LCB(kappa=2.0)` | Lower Confidence Bound for minimisation. `kappa` controls the explore/exploit tradeoff (higher = more exploration). |
| `UCB(kappa=2.0)` | Upper Confidence Bound for maximisation. |
| `ProbabilityOfImprovement(minimize=True)` | When you care about the probability of beating the current best, not the magnitude of improvement. |
| `ThompsonSampling(minimize=True, seed=42)` | Randomised exploration. Naturally produces diverse batches. |
| `ModelMin()` / `ModelMax()` | Pure exploitation — only use when you fully trust the model. |

```python
from jaxsr.acquisition import ActiveLearner, ExpectedImprovement

learner = ActiveLearner(
    model,
    bounds=bounds,
    acquisition=ExpectedImprovement(minimize=True),
    random_state=42,
)
result = learner.suggest(n_points=5, batch_strategy="penalized")
```

### Goal C: Discriminate Between Models

When multiple models on the Pareto front fit equally well but disagree in
unexplored regions:

```python
from jaxsr.acquisition import ModelDiscrimination, suggest_points

result = suggest_points(
    model, bounds, ModelDiscrimination(), n_points=5, random_state=42
)
# Run experiments at result.points to determine which model is correct
```

### Goal D: Multiple Objectives

Combine acquisition functions with `+` and `*`. Components are min-max
normalised to [0, 1] before weighting:

```python
from jaxsr.acquisition import ExpectedImprovement, PredictionVariance, DOptimal

# 70% optimisation, 30% exploration
acq = 0.7 * ExpectedImprovement(minimize=True) + 0.3 * PredictionVariance()

# Or balance three objectives equally
acq = ExpectedImprovement(minimize=True) + PredictionVariance() + DOptimal()
```

---

## Step 4: Batch Selection Strategies

When requesting multiple points at once, use a batch strategy to ensure
diversity:

```python
result = learner.suggest(n_points=10, batch_strategy="penalized")
```

| Strategy | How it works | Best for |
|----------|-------------|----------|
| `"greedy"` | Top-k by raw score | Single-point suggestions or when speed matters |
| `"penalized"` | After picking the best, penalise nearby candidates, repeat | **Good default for batches.** Simple and fast spatial diversity |
| `"kriging_believer"` | After picking each point, temporarily update the model with a fantasy observation, then re-score | Most information-aware batches, but slower |
| `"d_optimal"` | Select the batch maximising `det(Φᵀ Φ)` | Pure model-building with maximum statistical efficiency |

---

## Step 5: The Full Active Learning Loop

Here is a complete example tying everything together:

```python
import numpy as np
import jax.numpy as jnp
from jaxsr import BasisLibrary, SymbolicRegressor
from jaxsr.sampling import latin_hypercube_sample
from jaxsr.acquisition import ActiveLearner, ExpectedImprovement

# --- Problem setup ---
bounds = [(0.0, 5.0), (0.0, 5.0)]

def oracle(X):
    """Stand-in for your real experiment or simulation."""
    x1, x2 = X[:, 0], X[:, 1]
    return x1**2 + x2**2 - 3*x1 - 2*x2 + np.random.randn(len(X)) * 0.2

# --- Initial design ---
X_init = latin_hypercube_sample(30, bounds, random_state=0)
y_init = oracle(np.array(X_init))

# --- Define basis library ---
library = (
    BasisLibrary(n_features=2, feature_names=["x1", "x2"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions(max_order=2)
)

# --- Fit initial model ---
model = SymbolicRegressor(
    basis_library=library, max_terms=6, strategy="greedy_forward"
)
model.fit(jnp.array(X_init), jnp.array(y_init))
print(f"Initial model: {model.expression_}")

# --- Active learning loop ---
learner = ActiveLearner(
    model, bounds,
    acquisition=ExpectedImprovement(minimize=True),
    random_state=42,
)

for iteration in range(10):
    # Suggest next batch
    result = learner.suggest(n_points=5, batch_strategy="penalized")

    # Run experiments
    y_new = oracle(np.array(result.points))

    # Update model
    learner.update(result.points, jnp.array(y_new))

    print(
        f"Iter {iteration + 1}: "
        f"n={learner.n_observations}, "
        f"MSE={model.metrics_['mse']:.4f}, "
        f"best_y={learner.best_y:.3f}, "
        f"model={model.expression_}"
    )

    # Check convergence
    if learner.converged(tol=1e-3):
        print("Converged!")
        break

print(f"\nFinal model: {model.expression_}")
print(f"Best observed y: {learner.best_y:.3f} at X = {learner.best_X}")
```

---

## Candidate Generation Options

The `ActiveLearner` generates candidate points internally. You can
control the method and pool size:

```python
learner = ActiveLearner(
    model, bounds,
    acquisition=PredictionVariance(),
    n_candidates=2000,           # more candidates = finer resolution
    candidate_method="sobol",    # "lhs", "sobol", "halton", "random"
)
```

Or provide your own candidate set (useful when your experimental inputs
are constrained to specific values):

```python
my_candidates = jnp.array([[300, 1.0], [350, 2.5], [400, 5.0], ...])
result = learner.suggest(n_points=5, candidates=my_candidates)
```

---

## One-Shot Suggestion (No Loop)

If you only need a single batch of suggestions without an iterative loop,
use `suggest_points` directly:

```python
from jaxsr.acquisition import suggest_points, UCB

result = suggest_points(
    model,
    bounds=[(0, 10), (1, 5)],
    acquisition=UCB(kappa=2.0),
    n_points=5,
    random_state=42,
)
print(result.points)   # (5, 2) array of suggested experiments
print(result.scores)   # acquisition score for each point
```

---

## Incorporating Physical Constraints

Domain knowledge reduces the search space and improves model quality:

```python
from jaxsr import Constraints

constraints = (
    Constraints()
    .add_bounds(target="y", lower=0)                       # response must be non-negative
    .add_monotonic(feature="T", direction="increasing")    # rate increases with T
    .add_sign_constraint("T", sign="positive")             # positive temperature terms
    .add_sign_constraint("T^2", sign="positive")
)

model = SymbolicRegressor(
    basis_library=library,
    max_terms=6,
    constraints=constraints,
)
model.fit(X, y)
```

Constrained models produce more physically meaningful predictions, which
in turn makes the acquisition function suggestions more reliable.

---

## Uncertainty Quantification During DOE

Track how uncertainty shrinks as you add data:

```python
from jaxsr.uncertainty import prediction_interval, coefficient_intervals

# Prediction intervals at test points
pi = prediction_interval(Phi_train, y_train, model.coefficients_, Phi_test, alpha=0.05)
print(f"Mean PI width: {(pi['pred_upper'] - pi['pred_lower']).mean():.3f}")

# Coefficient confidence intervals
ci = coefficient_intervals(Phi_train, y_train, model.coefficients_, model.selected_features_, alpha=0.05)
for name, (est, lo, hi, se) in ci.items():
    print(f"  {name}: [{lo:.4f}, {hi:.4f}]")
```

You can use the CI width as a stopping criterion: stop adding experiments
when all prediction intervals are narrower than your tolerance.

---

## Decision Cheatsheet

```
WHAT IS YOUR GOAL?

1. BUILD AN ACCURATE MODEL EVERYWHERE
   ├── Simple & fast             → PredictionVariance
   ├── Maximum info per run      → DOptimal
   ├── Tight coefficient CIs     → AOptimal
   ├── Unsure about model form   → EnsembleDisagreement
   └── Comprehensive UQ          → BMAUncertainty

2. FIND THE OPTIMUM
   ├── Balanced (recommended)    → ExpectedImprovement
   ├── Tunable explore/exploit   → LCB(kappa) or UCB(kappa)
   ├── Probability of beating    → ProbabilityOfImprovement
   ├── Randomised                → ThompsonSampling
   └── Trust the model fully     → ModelMin / ModelMax

3. DECIDE WHICH MODEL IS CORRECT
   ├── Pareto models disagree    → ModelDiscrimination
   └── Structural uncertainty    → EnsembleDisagreement

4. MULTIPLE OBJECTIVES
   └── Combine: 0.7 * EI + 0.3 * PredictionVariance

BATCH STRATEGY
   ├── Don't need diversity      → greedy
   ├── Simple diversity          → penalized  (recommended default)
   ├── Information-aware batches → kriging_believer
   └── Maximum design efficiency → d_optimal
```

---

## Further Reading

- [Acquisition function API reference](acquisition.md)
- [Active learning example](../../examples/active_learning.ipynb)
- [Comprehensive tutorial notebook](../../examples/comprehensive_tutorial.ipynb)
- [API reference](../api/index.rst)
