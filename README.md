# JAXSR: JAX-based Symbolic Regression

[![Tests](https://github.com/jkitchin/jaxsr/actions/workflows/tests.yml/badge.svg)](https://github.com/jkitchin/jaxsr/actions/workflows/tests.yml)
[![Docs](https://github.com/jkitchin/jaxsr/actions/workflows/docs.yml/badge.svg)](https://jkitchin.github.io/jaxsr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

![img](jaxsr-image.png)

JAXSR is a fully open-source symbolic regression library built on JAX that discovers interpretable algebraic expressions from data. It uses sparse optimization techniques with JAX for automatic differentiation, JIT compilation, and GPU acceleration.

## Features

- **Flexible Basis Library**: Easily define candidate basis functions including polynomials, interactions, transcendentals, ratios, and custom functions
- **Multiple Selection Strategies**: Greedy forward/backward selection, exhaustive search, LASSO path screening
- **Uncertainty Quantification**: Classical OLS intervals, Bayesian Model Averaging, conformal prediction, bootstrap methods, and Pareto ensemble predictions
- **Physical Constraints**: Enforce monotonicity, bounds, convexity, and linear constraints
- **Adaptive Sampling**: Intelligently suggest new data points to improve model quality
- **JAX-Accelerated**: JIT compilation and GPU support for fast computation
- **Symbolic Classification**: Discover interpretable logistic models for binary and multiclass problems via IRLS + sparse selection
- **Scikit-learn Compatible**: Full estimator protocol (`get_params`/`set_params`/`clone`) — works with `cross_val_score`, `GridSearchCV`, `Pipeline`
- **Symbolic Export**: Export to SymPy, LaTeX, or pure Python/NumPy functions

## Installation

```bash
pip install jaxsr
```

Or install from source:

```bash
git clone https://github.com/jkitchin/jaxsr.git
cd jaxsr
pip install -e ".[dev]"
```

## Quick Start

```python
import jax.numpy as jnp
import numpy as np
from jaxsr import BasisLibrary, SymbolicRegressor

# Generate synthetic data: y = 2.5*x0 + 1.2*x0*x1 - 0.8*x1^2 + noise
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 2) * 2
y = 2.5 * X[:, 0] + 1.2 * X[:, 0] * X[:, 1] - 0.8 * X[:, 1]**2 + np.random.randn(n_samples) * 0.1

X_jax = jnp.array(X)
y_jax = jnp.array(y)

# Build basis library
library = (BasisLibrary(n_features=2, feature_names=["x0", "x1"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions(max_order=2)
)

# Fit model
model = SymbolicRegressor(
    basis_library=library,
    max_terms=5,
    strategy="greedy_forward",
    information_criterion="bic",
)
model.fit(X_jax, y_jax)

# Results
print(f"Discovered: {model.expression_}")
print(f"R² = {model.metrics_['r2']:.4f}")

# Predict
y_pred = model.predict(X_jax)
```

## Basis Functions

JAXSR provides a flexible system for defining candidate basis functions:

```python
import jax.numpy as jnp
from jaxsr import BasisLibrary

library = (BasisLibrary(n_features=3, feature_names=["T", "P", "C"])
    .add_constant()                                    # Intercept term
    .add_linear()                                      # T, P, C
    .add_polynomials(max_degree=3)                     # T^2, T^3, P^2, ...
    .add_interactions(max_order=2)                     # T*P, T*C, P*C
    .add_transcendental(["log", "exp", "sqrt", "inv"]) # log(T), exp(T), ...
    .add_ratios()                                      # T/P, T/C, P/T, ...
    .add_custom(                                       # Custom functions
        name="Arrhenius",
        func=lambda X: jnp.exp(-X[:, 0] / X[:, 1]),
        complexity=3
    )
)
```

## Selection Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `greedy_forward` | Forward stepwise selection | Default, fast for large libraries |
| `greedy_backward` | Backward elimination | When starting with many terms |
| `exhaustive` | All combinations | Small libraries (<20 terms) |
| `lasso_path` | LASSO regularization path | Fast screening |

```python
from jaxsr import SymbolicRegressor

model = SymbolicRegressor(
    basis_library=library,
    max_terms=5,
    strategy="greedy_forward",      # Selection strategy
    information_criterion="bic",    # or "aic", "aicc"
)
```

## Physical Constraints

Incorporate domain knowledge through constraints:

```python
from jaxsr import Constraints

constraints = (Constraints()
    .add_bounds("y", lower=0)                        # Non-negative output
    .add_monotonic("T", direction="increasing")      # y increases with T
    .add_convex("P")                                 # Convex in P
    .add_sign_constraint("T", sign="positive")       # Positive coefficient
)

model = SymbolicRegressor(
    basis_library=library,
    constraints=constraints,
)
```

## Adaptive Sampling

Request new data points to improve model quality:

```python
from jaxsr import AdaptiveSampler

sampler = AdaptiveSampler(
    model=model,
    bounds=[(300, 500), (1, 10), (0.1, 1.0)],
    strategy="uncertainty",  # or "error", "leverage", "gradient"
)

# Get suggested points
result = sampler.suggest(n_points=5)
X_next = result.points    # shape (5, n_features)
scores = result.scores    # acquisition function values
```

## Export Options

```python
# Human-readable expression
print(model.expression_)  # "y = 2.5*T + 1.2*T*P - 0.8*P^2"

# SymPy expression for symbolic manipulation
sympy_expr = model.to_sympy()

# LaTeX for papers
latex_str = model.to_latex()

# Pure Python/NumPy function (no JAX dependency)
predict_func = model.to_callable()
y_pred = predict_func(X_numpy)

# Save/load models
model.save("model.json")
loaded_model = SymbolicRegressor.load("model.json")
```

## Uncertainty Quantification

JAXSR provides comprehensive UQ capabilities for linear-in-parameters models:

```python
# Prediction intervals (classical OLS)
y_pred, lower, upper = model.predict_interval(X_new, alpha=0.05)

# Confidence band on the mean response
y_pred, conf_lo, conf_hi = model.confidence_band(X_new, alpha=0.05)

# Coefficient confidence intervals
intervals = model.coefficient_intervals(alpha=0.05)
for name, (est, lo, hi, se) in intervals.items():
    print(f"  {name}: {est:.4f} [{lo:.4f}, {hi:.4f}]")

# Noise standard deviation and coefficient covariance
print(f"sigma = {model.sigma_:.4f}")
cov = model.covariance_matrix_

# Bayesian Model Averaging across Pareto-front models
y_pred, lower, upper = model.predict_bma(X_new, criterion="bic")

# Distribution-free conformal prediction (jackknife+ or split)
y_pred, lower, upper = model.predict_conformal(X_new, method="jackknife+")

# Pareto front ensemble predictions
result = model.predict_ensemble(X_new)
print(f"Ensemble std: {result['y_std']}")

# Residual bootstrap (no Gaussian assumption needed)
from jaxsr import bootstrap_predict
result = bootstrap_predict(model, X_new, n_bootstrap=1000)
```

## Symbolic Classification

JAXSR also supports interpretable classification — discover sparse logistic models that explain class boundaries:

```python
import jax.numpy as jnp
import numpy as np
from jaxsr import BasisLibrary, SymbolicClassifier

# Generate binary classification data
np.random.seed(42)
X = np.random.randn(200, 2)
y = (X[:, 0] + 0.5 * X[:, 1] ** 2 > 0).astype(float)

# Build basis library and fit classifier
library = (BasisLibrary(n_features=2, feature_names=["x0", "x1"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions(max_order=2)
)

clf = SymbolicClassifier(basis_library=library, max_terms=4, strategy="greedy_forward")
clf.fit(jnp.array(X), jnp.array(y))

# Results
print(f"Expression: {clf.expression_}")
print(f"Accuracy: {clf.score(jnp.array(X), jnp.array(y)):.4f}")

# Probabilities and class predictions
proba = clf.predict_proba(jnp.array(X))
y_pred = clf.predict(jnp.array(X))
```

Multiclass problems are handled automatically via one-vs-rest (OVR), giving each class its own interpretable expression. The classifier also supports coefficient intervals, conformal prediction sets, SymPy/LaTeX export, and save/load.

## Visualization

```python
from jaxsr.plotting import (
    plot_pareto_front,
    plot_parity,
    plot_residuals,
    plot_coefficient_path,
    plot_prediction_intervals,
    plot_coefficient_intervals,
    plot_bma_weights,
)

# Pareto front: complexity vs accuracy
plot_pareto_front(model.pareto_front_, highlight_best=True)

# Parity plot
plot_parity(y_true, y_pred)

# Residual diagnostics
plot_residuals(model, X, y)

# Prediction intervals fan chart
plot_prediction_intervals(model, X, y)

# Coefficient confidence intervals (forest plot)
plot_coefficient_intervals(model)

# BMA model weights
plot_bma_weights(model)
```

## Claude Code Skills

JAXSR ships with [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill files
that let an AI assistant guide you through symbolic regression workflows interactively.
The skill files live in `.claude/skills/jaxsr/` (and are mirrored in `src/jaxsr/skill/`
for packaging).

**What's included:**

| Resource | Description |
|----------|-------------|
| `SKILL.md` | Main skill definition — activation triggers, assistant-mode decision trees, quick-reference API and CLI cheat sheets |
| `guides/basis-library.md` | Choosing and building basis function libraries |
| `guides/model-fitting.md` | Selection strategies and information criteria |
| `guides/uncertainty.md` | UQ methods: OLS intervals, BMA, conformal, bootstrap |
| `guides/constraints.md` | Adding physical constraints (monotonicity, bounds, convexity) |
| `guides/doe-workflow.md` | End-to-end Design of Experiments lifecycle |
| `guides/active-learning.md` | Acquisition functions and adaptive sampling |
| `guides/rsm.md` | Response Surface Methodology designs and analysis |
| `guides/known-model-fitting.md` | Fitting known model forms (Langmuir, Arrhenius, etc.) |
| `guides/sklearn-integration.md` | Using JAXSR with sklearn (cross_val_score, GridSearchCV, Pipeline) |
| `guides/cli.md` | CLI reference for code-free DOE workflows |

**Templates** (ready-to-run starter scripts in `templates/`):

| Template | Use Case |
|----------|----------|
| `basic-regression.py` | Discover an equation from X, y data |
| `constrained-model.py` | Add physical constraints to a model |
| `doe-study.py` | Full DOE workflow from design to report |
| `uncertainty-analysis.py` | Compare all UQ methods |
| `active-learning-loop.py` | Iterative experiment-model loop |
| `langmuir-isotherm.py` | Known-model parameter estimation |
| `notebook-starter.py` | Jupyter notebook cell structure |

When Claude Code is available, it uses these files to provide context-aware help —
recommending basis libraries, selection strategies, UQ methods, and constraint setups
based on your specific problem. See the
[Claude Code Skills guide](https://jkitchin.github.io/jaxsr/guides/claude_code_skills.html)
in the documentation for more details.

## Examples

See the `examples/` directory for complete worked examples:

- `basic_usage.py`: Simple polynomial fitting
- `uncertainty_quantification.py`: Prediction intervals, BMA, conformal, bootstrap
- `chemical_kinetics.py`: Discovering rate laws from kinetic data
- `heat_transfer.py`: Heat transfer correlations

## API Reference

See the [documentation](docs/) for full API details.

## Citation

If you use JAXSR in your research, please cite:

```bibtex
@software{jaxsr2024,
  title = {JAXSR: JAX-based Symbolic Regression},
  author = {Kitchin, John},
  year = {2024},
  url = {https://github.com/jkitchin/jaxsr}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see our contributing guidelines and open an issue or pull request.

