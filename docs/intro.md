# JAXSR Documentation

**JAX-based Symbolic Regression**

JAXSR is a Python library for discovering interpretable algebraic expressions from data using sparse optimization techniques.

## Overview

JAXSR provides tools for symbolic regression - the task of finding mathematical expressions that describe relationships in data. Unlike black-box machine learning methods, symbolic regression produces human-readable equations that can provide scientific insight.

```{admonition} Try it in your browser — nothing to install
:class: tip

**[Open the JAXSR web app](https://kitchingroup.cheme.cmu.edu/jaxsr/app/)**

Upload a spreadsheet, say which columns are features and which is the response, choose the
families of functions to consider, and get a ranked table of candidate equations with
confidence intervals, ANOVA, diagnostic plots, and exports.

The whole library is compiled to WebAssembly and runs client-side, so nothing is uploaded and
unpublished data never leaves your machine. The app offers an example workbook with a known
answer to work through, and can export a Python script that reproduces your fit locally.
```

Key features:

- **Flexible Basis Functions**: Build custom libraries of candidate functions
- **Multiple Selection Strategies**: Choose from greedy, exhaustive, or LASSO-based methods
- **Uncertainty Quantification**: Prediction intervals, Bayesian Model Averaging, conformal prediction, and bootstrap methods
- **Physical Constraints**: Incorporate domain knowledge through constraints
- **Additive Symbolic Regression**: Boosting-style ensembles of small symbolic expressions (`jaxsr.additive`)
- **JAX-Powered**: GPU acceleration, JIT compilation, automatic differentiation
- **Scikit-learn Compatible**: Familiar fit/predict interface
- **Two GUIs**: a hosted browser app that needs no install, and a local Streamlit app for the full design-of-experiments cycle

## Installation

```bash
pip install jaxsr
```

For development:

```bash
git clone https://github.com/jkitchin/jaxsr.git
cd jaxsr
pip install -e ".[dev]"
```

## Quick Start

```python
from jaxsr import BasisLibrary, SymbolicRegressor
import jax.numpy as jnp

# Create basis library
library = (BasisLibrary(n_features=2, feature_names=["x", "y"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions()
)

# Fit model
model = SymbolicRegressor(basis_library=library, max_terms=5)
model.fit(X, y)

# Results
print(model.expression_)
print(f"R² = {model.metrics_['r2']:.4f}")
```

## Interactive apps

Two graphical front ends, for different jobs.

**[Browser app](https://kitchingroup.cheme.cmu.edu/jaxsr/app/)** — nothing to install. Best for fitting a dataset you already have,
comparing candidate models, and sharing a result with someone who does not use Python. Runs on
WebAssembly, so your data stays in the browser.

**Streamlit DOE app** — for the full experimental cycle, where you are choosing what to measure
next rather than analysing a finished dataset:

```bash
pip install "jaxsr[app]"
jaxsr app                      # opens http://localhost:8501
jaxsr app --study my.jaxsr     # resume a saved study
```

Eight pages covering the loop end to end: define factors, generate a design and export an Excel
template for the bench, import the completed results, fit, inspect diagnostics, explore the
response surface, run canonical analysis and get suggested next experiments, then export a Word
or Excel report. State persists in a `.jaxsr` study file, so a campaign can be picked back up
later. See the [Design of Experiments Guide](guides/doe_guide.md).

## Documentation Contents

- [Quickstart Guide](quickstart.md) - Get started quickly
- [Design of Experiments Guide](guides/doe_guide.md) - Adaptive DOE and active learning
- [Acquisition Functions](guides/acquisition.md) - Detailed acquisition function reference
- [CLI Guide](guides/cli_guide.md) - Command-line interface reference
- [Claude Code Skills](guides/claude_code_skills.md) - AI-assisted workflows with Claude Code
- [API Reference](api/index.rst) - Complete API documentation
- [Literature Review](background/literature_review.md) - Background on symbolic regression
- [Examples](examples/basic_usage.ipynb) - Worked examples for various applications

## How It Works

JAXSR follows the ALAMO (Automated Learning of Algebraic Models for Optimization) methodology:

1. **Basis Library Construction**: Define a library of candidate basis functions (polynomials, transcendentals, interactions, etc.)

2. **Design Matrix Evaluation**: Evaluate all basis functions on training data to create a design matrix Φ

3. **Sparse Selection**: Use information criteria (BIC, AIC) to select a sparse subset of basis functions

4. **Coefficient Fitting**: Fit coefficients via least squares, optionally with constraints

5. **Model Analysis**: Examine Pareto front, export to various formats

## When to Use JAXSR

JAXSR is ideal when you:

- Want **interpretable** models rather than black boxes
- Have **domain knowledge** to constrain the solution space
- Need to discover **physical laws** or empirical correlations
- Require **reproducible** results (deterministic algorithms)
- Want to explore the **accuracy-complexity trade-off**

## Comparison with Other Tools

| Feature | JAXSR | ALAMO | PySR | GP |
|---------|-------|-------|------|-----|
| Open Source | ✓ | ✗ | ✓ | ✓ |
| Deterministic | ✓ | ✓ | ✗ | ✗ |
| UQ / Intervals | ✓ | Limited | ✗ | ✗ |
| Constraints | ✓ | ✓ | Limited | Limited |
| GPU Support | ✓ | ✗ | ✓ | Varies |

## License

JAXSR is released under the MIT License.
