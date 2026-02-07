# Active Learning & Acquisition Functions

JAXSR provides a composable framework for **active learning** and
**design of experiments (DOE)**.  After fitting a symbolic regression
model, you can use acquisition functions to intelligently decide
**where to sample next**, balancing exploration (reducing uncertainty)
and exploitation (optimising the response).

## Why Active Learning with Symbolic Regression?

Because JAXSR models are **linear-in-parameters** (`y = Phi @ beta`),
the posterior uncertainty is available in **closed form**:

- Prediction variance: `sigma^2(x) = sigma_hat^2 * phi(x)^T (Phi^T Phi)^{-1} phi(x)`
- Coefficient covariance: `Cov(beta) = sigma_hat^2 * (Phi^T Phi)^{-1}`
- Leverage (influence): hat matrix diagonal

This means acquisition functions that rely on posterior uncertainty are
**exact** --- no GP approximations needed, no MCMC, no variational inference.
The result is fast, reliable active learning that scales to large candidate pools.

## Quick Start

```python
from jaxsr import BasisLibrary, SymbolicRegressor
from jaxsr.acquisition import ActiveLearner, UCB, PredictionVariance

# 1. Fit a model
model = SymbolicRegressor(basis_library=library, max_terms=5)
model.fit(X, y)

# 2. Create a learner
learner = ActiveLearner(
    model,
    bounds=[(0, 10), (1, 5)],     # feature bounds
    acquisition=UCB(kappa=2.0),    # exploration-exploitation tradeoff
)

# 3. Get suggestions
result = learner.suggest(n_points=5)
print(result.points)   # shape (5, n_features)

# 4. Run experiments and update
y_new = run_experiment(result.points)
learner.update(result.points, y_new)
```

## Choosing an Acquisition Function

### Decision Flowchart

```
WHAT IS YOUR GOAL?

1. IMPROVE MODEL ACCURACY (explore everywhere)
   |-- Simple & fast?            --> PredictionVariance
   |-- Need coverage guarantee?  --> ConfidenceBandWidth(alpha=0.05)
   |-- Unsure about model form?  --> EnsembleDisagreement or BMAUncertainty
   |-- Tighten coefficient CIs?  --> AOptimal
   '-- Max info per experiment?  --> DOptimal

2. FIND THE OPTIMUM (minimise or maximise y)
   |-- Trust the model fully?         --> ModelMin / ModelMax
   |-- Want balanced exploration?     --> ExpectedImprovement  (recommended)
   |-- Probability of beating target? --> ProbabilityOfImprovement
   |-- Want explicit exploration knob --> LCB(kappa) / UCB(kappa)
   '-- Want randomised exploration?   --> ThompsonSampling

3. DECIDE WHICH MODEL IS CORRECT
   |-- Pareto models disagree?            --> ModelDiscrimination
   '-- Quantify structural uncertainty?   --> EnsembleDisagreement

4. MULTIPLE OBJECTIVES
   '-- Combine with weights:  0.7 * EI + 0.3 * PredictionVariance
```

### Detailed Descriptions

#### Exploration (Reduce Uncertainty)

| Function | Score | Best For |
|----------|-------|----------|
| `PredictionVariance()` | `sigma^2(x)` from OLS posterior | Default for "where is the model least certain?" Fast and exact. |
| `ConfidenceBandWidth(alpha)` | Width of the `1-alpha` confidence band on `E[y|x]` | When you have a specific coverage target (e.g. 95% intervals). |
| `EnsembleDisagreement()` | Std across Pareto-front model predictions | When structural uncertainty matters --- which model form is right? |
| `BMAUncertainty(criterion)` | BMA std (within + between model variance) | Most comprehensive uncertainty. Combines noise and model-selection uncertainty. |
| `AOptimal()` | Reduction in `tr(Cov(beta))` from adding a point | Tighten confidence intervals on *all* coefficients. |
| `DOptimal()` | Leverage `h(x) = phi^T (Phi^T Phi)^{-1} phi` | Maximum information per experiment. Classic optimal design. |

#### Exploitation (Optimise the Response)

| Function | Score | Best For |
|----------|-------|----------|
| `ModelMin()` | `-y_hat(x)` | Pure exploitation (minimise y). Trusts the model completely. |
| `ModelMax()` | `y_hat(x)` | Pure exploitation (maximise y). |

#### Exploration-Exploitation Tradeoffs

| Function | Score | Best For |
|----------|-------|----------|
| `UCB(kappa)` | `y_hat + kappa * sigma` | Maximising y with tunable exploration. `kappa=0` is pure exploitation, `kappa~2` is balanced, `kappa>3` is exploratory. |
| `LCB(kappa)` | `-y_hat + kappa * sigma` | Minimising y with tunable exploration. Mirror image of UCB. |
| `ExpectedImprovement(minimize, xi)` | `E[max(0, f_best - y)]` | Gold standard for Bayesian optimisation. Naturally balances explore/exploit without sensitive hyperparameters. |
| `ProbabilityOfImprovement(minimize, xi)` | `P(y < f_best - xi)` | When you care about *probability* of beating a threshold, not magnitude. More exploitative than EI. |
| `ThompsonSampling(minimize, seed)` | Score from a posterior draw of beta | Randomised exploration. Each call draws a different model from the posterior. Produces diverse batches naturally. |

#### Optimal Experimental Design

| Function | Score | Best For |
|----------|-------|----------|
| `AOptimal()` | Trace-reduction in coefficient covariance | Reduce average parameter uncertainty. Good when every coefficient matters. |
| `DOptimal()` | Leverage (information gain) | Maximise determinant of information matrix. Classic choice for building precise models with minimal data. |

## Composing Acquisition Functions

Acquisition functions can be combined with `+` and `*`:

```python
# Weighted combination (components are min-max normalised first)
acq = 0.7 * ExpectedImprovement(minimize=True) + 0.3 * PredictionVariance()

# Equal weighting
acq = UCB(kappa=2) + DOptimal() + AOptimal()

# Single term scaled
acq = 2.0 * PredictionVariance()
```

Each component is **min-max normalised to [0, 1]** before weighting, so
the weights represent the relative importance of each objective regardless
of their natural scales.

### Recommended Recipes

| Goal | Recipe |
|------|--------|
| Optimise while exploring | `0.7 * EI + 0.3 * PredictionVariance` |
| Build accurate model fast | `0.5 * PredictionVariance + 0.5 * DOptimal` |
| Optimise & tighten CIs | `0.6 * LCB(kappa=2) + 0.4 * AOptimal` |
| Multi-objective | `0.4 * ModelMin + 0.3 * PredictionVariance + 0.3 * DOptimal` |

## Batch Selection Strategies

When requesting multiple points at once, you can choose how to ensure
batch diversity:

```python
result = learner.suggest(n_points=10, batch_strategy="penalized")
```

| Strategy | Method | Best For |
|----------|--------|----------|
| `"greedy"` | Top-k by raw score | Fast, when diversity doesn't matter. |
| `"penalized"` | Select best, penalise nearby, repeat | Simple diversity. Good default for batches. |
| `"kriging_believer"` | Select best, fantasise `y_hat`, re-score, repeat | Information-aware batches. Later selections account for what earlier ones teach. |
| `"d_optimal"` | Maximise `det(Phi^T Phi)` for the batch | Maximum design efficiency. Ignores acquisition function, purely geometric. |

### When to use each strategy

- **`greedy`**: Use when `n_points=1` (where batch strategy is irrelevant),
  or when speed matters more than diversity.  Also fine when the
  acquisition function itself provides diversity (e.g. `ThompsonSampling`).

- **`penalized`**: Good default for most situations. Computationally cheap
  and produces well-spread batches.  The bandwidth of the Gaussian penalty
  automatically scales with `n_points`.

- **`kriging_believer`**: Best when information gain matters.  After selecting
  each point, the model is temporarily updated with a fantasy observation
  (`y = y_hat`), so the acquisition scores adjust.  More expensive but
  produces batches where each point adds genuinely new information.

- **`d_optimal`**: Use when your primary goal is building an accurate model
  (not optimisation).  Produces the most statistically efficient design
  regardless of the acquisition function.

## Candidate Generation

The `ActiveLearner` generates internal candidate points using space-filling
designs.  You can control the method:

```python
learner = ActiveLearner(
    model, bounds,
    acquisition=UCB(kappa=2),
    n_candidates=2000,           # size of candidate pool
    candidate_method="sobol",    # "lhs", "sobol", "halton", "random"
)
```

You can also provide your own candidates:

```python
my_candidates = jnp.array([[1.0, 2.0], [3.0, 4.0], ...])
result = learner.suggest(n_points=5, candidates=my_candidates)
```

## The Active Learning Loop

A typical active learning workflow:

```python
# Initial fit
model = SymbolicRegressor(basis_library=library, max_terms=5)
model.fit(X_initial, y_initial)

# Set up learner
learner = ActiveLearner(
    model, bounds,
    acquisition=ExpectedImprovement(minimize=True),
    random_state=42,
)

# Iterate
for iteration in range(budget):
    # 1. Suggest
    result = learner.suggest(
        n_points=5,
        batch_strategy="penalized",
    )

    # 2. Experiment (replace with your actual experiment/simulation)
    y_new = run_experiment(result.points)

    # 3. Update
    learner.update(result.points, y_new)

    # 4. Check convergence
    print(f"Iter {iteration}: n={learner.n_observations}, "
          f"MSE={model.metrics_['mse']:.4f}, "
          f"model={model.expression_}")

    if learner.converged(tol=1e-3):
        print("Converged!")
        break
```

### Tracking Progress

The learner tracks its history:

```python
learner.iteration        # number of suggest-update cycles
learner.n_observations   # total training set size
learner.best_y           # best observed y
learner.best_X           # corresponding input
learner.history_X        # list of X arrays from each iteration
learner.history_y        # list of y arrays from each iteration
```

## One-Shot Convenience

If you don't need the loop, use `suggest_points` directly:

```python
from jaxsr.acquisition import suggest_points, UCB

result = suggest_points(
    model, bounds, UCB(kappa=2), n_points=5, random_state=42
)
```

## Mathematical Details

### OLS Prediction Variance

For a fitted model `y = Phi @ beta` with `beta = (Phi^T Phi)^{-1} Phi^T y`,
the prediction variance at a new point `x` is:

```
Var(y_hat(x)) = sigma_hat^2 * phi(x)^T (Phi^T Phi)^{-1} phi(x)
```

where:
- `sigma_hat^2 = SSR / (n - p)` is the unbiased noise variance estimate
- `phi(x)` is the vector of basis function values at `x`
- `(Phi^T Phi)^{-1}` is computed via SVD for numerical stability

### Expected Improvement

For minimisation with Gaussian posterior:

```
EI(x) = (f_best - mu(x) - xi) * Phi(z) + sigma(x) * phi(z)
z = (f_best - mu(x) - xi) / sigma(x)
```

where `Phi` is the standard normal CDF and `phi` is the PDF.

### A-Optimal Criterion

The reduction in trace of coefficient covariance from adding one point:

```
Delta_tr = phi^T (Phi^T Phi)^{-1} Cov(beta) (Phi^T Phi)^{-1} phi
           / (1 + phi^T (Phi^T Phi)^{-1} phi)
```

This is computed efficiently in vectorised form over all candidates.

## API Reference

### Acquisition Functions

All acquisition functions inherit from `AcquisitionFunction` and implement
the `score(X_candidates, model)` method.  The convention is **higher score
= more desirable to sample**.

```python
class AcquisitionFunction:
    def score(self, X_candidates, model) -> jnp.ndarray:
        """Return scores for each candidate (higher = better)."""
    @property
    def name(self) -> str: ...
```

### ActiveLearner

```python
class ActiveLearner:
    def __init__(self, model, bounds, acquisition, n_candidates=1000,
                 candidate_method="lhs", random_state=None): ...
    def suggest(self, n_points=5, batch_strategy="greedy",
                min_distance=0.01, candidates=None) -> AcquisitionResult: ...
    def update(self, X_new, y_new, refit=True) -> None: ...
    def converged(self, tol=1e-3, window=3, metric="mse") -> bool: ...
```

### AcquisitionResult

```python
@dataclass
class AcquisitionResult:
    points: jnp.ndarray     # (n_points, n_features)
    scores: jnp.ndarray     # (n_points,)
    acquisition: str         # name of the acquisition function
    metadata: dict           # extra info (batch strategy, etc.)
```
