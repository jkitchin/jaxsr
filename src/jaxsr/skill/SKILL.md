# JAXSR Skill — Symbolic Regression Assistant

JAXSR is a JAX-based symbolic regression library that discovers interpretable algebraic
expressions from data using sparse optimization. It follows ALAMO-style methodology:
build a rich candidate basis, then select the simplest model that explains the data.

## Skill Activation

Activate this skill when the user wants to:
- Discover algebraic expressions or equations from data
- Set up a Design of Experiments (DOE) study
- Fit, interpret, or export symbolic regression models
- Choose between basis functions, strategies, UQ methods, or design methods
- Generate reports from experimental data
- Use the `jaxsr` CLI tool
- Build notebooks or scripts for symbolic regression workflows

## Assistant Mode

When the user asks for help deciding how to set up, analyze, or report on their problem,
enter **assistant mode**. In this mode, ask diagnostic questions to guide them to the right
configuration. Do not dump all options at once — walk through decisions sequentially.

### Step 1: Characterize the Problem

Ask the user:
1. **What are you modeling?** (physical system, chemical process, ML feature engineering, etc.)
2. **How many input features?** (1-3 is small, 4-8 is medium, 9+ is large)
3. **How many data points?** (< 20 is very small, 20-100 is typical, 100+ is large)
4. **Do you have domain knowledge?** (known physics, monotonicity, bounds, symmetry)
5. **What is the goal?** (interpretable equation, prediction, optimization, screening)

### Step 2: Recommend Basis Library

Based on the answers, recommend a basis library configuration:

| Scenario | Recommended Basis |
|----------|-------------------|
| Unknown relationship, few features | `add_constant + add_linear + add_polynomials(3) + add_interactions(2) + add_transcendental()` |
| Known polynomial behavior | `add_constant + add_linear + add_polynomials(max_degree)` |
| Engineering correlation (Nusselt, friction) | `add_constant + add_linear + add_polynomials(2) + add_transcendental(funcs=["log","exp","sqrt","inv"])` |
| Chemical kinetics (rate laws) | `add_constant + add_linear + add_transcendental(funcs=["exp","inv","log"]) + add_ratios() + add_parametric(Arrhenius)` |
| Large feature space (screening) | `add_constant + add_linear + add_interactions(2)` then use `lasso_path` strategy |
| Response surface (DOE) | `add_constant + add_linear + add_polynomials(2) + add_interactions(2)` — or use `ResponseSurface` directly |
| Categorical factors present | Add `add_categorical_indicators() + add_categorical_interactions()` to any of the above |

**Key guidance:**
- Start simple. You can always add complexity.
- `add_transcendental(safe=True)` guards against log(0), 1/0, sqrt(<0). Always use `safe=True`.
- `add_ratios(safe=True)` adds x_i/x_j terms. Doubles the library size — only use when ratios are physically meaningful.
- `add_parametric()` enables nonlinear parameters (e.g., `exp(-a*x)`). Powerful but slower to fit.
- If n_features > 5, avoid `add_polynomials(degree>2)` — the library becomes enormous.

### Step 3: Recommend Selection Strategy

| Data Size | Library Size | Recommended Strategy |
|-----------|-------------|---------------------|
| Any | < 20 basis functions | `exhaustive` (exact optimal) |
| Any | 20-200 basis functions | `greedy_forward` (default, fast, reliable) |
| Small n | Large library | `lasso_path` (regularized screening) |
| Many terms expected | Any | `greedy_backward` (start full, prune) |

**When to change from defaults:**
- `greedy_forward` is the right choice 80% of the time. It's the default.
- Use `exhaustive` only when the basis library is small enough (< 20 terms). It guarantees the global optimum but scales as O(2^n).
- Use `lasso_path` when you have a very large library and want fast screening. It may miss interaction effects.
- Use `greedy_backward` when you suspect many terms matter and want to start from the full model.

### Step 4: Recommend Information Criterion

| Scenario | Recommended Criterion |
|----------|----------------------|
| Small sample (n < 40) | `aicc` (corrected AIC, penalizes overfitting more) |
| Medium sample (40 < n < 200) | `bic` (stronger complexity penalty, sparser models) |
| Large sample (n > 200) | `aic` or `bic` (both work well) |
| Want simplest model | `bic` (always penalizes complexity more) |
| Want best prediction | `aicc` (balances fit and complexity) |

**Default recommendation:** Use `"bic"` for interpretable models, `"aicc"` for predictive models.
Only `"aic"`, `"aicc"`, and `"bic"` are supported — not `"cv"`.

### Step 5: Recommend Constraints (if applicable)

Ask: "Do you have any physical knowledge about the system?"

| Physical Knowledge | Constraint to Add |
|-------------------|-------------------|
| Output must be positive | `.add_bounds("y", lower=0)` |
| Output in known range | `.add_bounds("y", lower=lo, upper=hi)` |
| Increasing in temperature | `.add_monotonic("T", direction="increasing")` |
| Diminishing returns | `.add_concave(feature)` |
| Accelerating growth | `.add_convex(feature)` |
| Coefficient must be positive | `.add_sign_constraint(basis_name, sign="positive")` |
| Known intercept or slope | `.add_known_coefficient(name, value)` |

Use `hard=True` for strict enforcement; `hard=False` (default) for soft penalty.

### Step 6: Recommend Uncertainty Quantification

| Need | Method | When to Use |
|------|--------|-------------|
| Quick confidence intervals | `model.predict_interval()` | Default. OLS-based. Assumes normality. |
| Coefficient significance | `model.coefficient_intervals()` | Check which terms are statistically significant |
| Robust to model uncertainty | `model.predict_bma()` | Averages over Pareto-front models weighted by criterion |
| No distributional assumptions | `model.predict_conformal()` | Distribution-free. Needs enough data (n > 30). |
| Assess model stability | `bootstrap_predict()` | Resamples data. Shows sensitivity to individual points. |
| Compare model structures | `model.predict_ensemble()` | Returns predictions from all Pareto-front models |
| Variable importance | `anova()` | Decomposes variance by term. Shows which factors matter. |

**Default recommendation:** Start with `predict_interval()` (built-in, fast). Add `predict_bma()` if you have multiple competing models. Use `predict_conformal()` for publication-quality intervals.

### Step 7: Recommend Reporting Format

| Goal | Action |
|------|--------|
| Quick look at results | `model.summary()` or `jaxsr status study.jaxsr` |
| Share with collaborators | `jaxsr report study.jaxsr -o report.xlsx` (Excel) |
| Formal report | `jaxsr report study.jaxsr -o report.docx` (Word) |
| Paper/presentation | `model.to_latex()` for equation, `plot_pareto_front()` for figures |
| Deploy model | `model.to_callable()` (pure NumPy, no JAX dependency) |
| Archive/reproduce | `model.save("model.json")` and `study.save("study.jaxsr")` |

---

## Quick Reference: Installation

```bash
# Core library
pip install jaxsr

# With CLI support
pip install "jaxsr[cli]"

# With Excel reporting
pip install "jaxsr[excel]"

# With Word reports
pip install "jaxsr[reports]"

# Everything for development
pip install -e ".[dev,cli,excel,reports]"
```

## Quick Reference: Python API

### Minimal Example (5 lines)

```python
from jaxsr import fit_symbolic
import numpy as np

X = np.column_stack([x1, x2])  # shape (n_samples, n_features)
model = fit_symbolic(X, y, feature_names=["x1", "x2"], max_terms=5)
print(model.expression_)
```

### Full Control Example

```python
from jaxsr import BasisLibrary, SymbolicRegressor, Constraints

# 1. Build basis library
library = (BasisLibrary(n_features=2, feature_names=["T", "P"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions(max_order=2)
    .add_transcendental(funcs=["log", "exp", "sqrt"])
)

# 2. Define constraints (optional)
constraints = (Constraints()
    .add_monotonic("T", direction="increasing")
    .add_bounds("y", lower=0)
)

# 3. Fit model
model = SymbolicRegressor(
    basis_library=library,
    max_terms=5,
    strategy="greedy_forward",
    information_criterion="bic",
    constraints=constraints,
)
model.fit(X_train, y_train)

# 4. Inspect results
print(model.expression_)
print(model.metrics_)
print(model.summary())

# 5. Predict with uncertainty
y_pred, lower, upper = model.predict_interval(X_test, alpha=0.05)

# 6. Export
model.save("model.json")
latex_eq = model.to_latex()
predict_fn = model.to_callable()  # pure NumPy function
```

### DOE Workflow

```python
from jaxsr import DOEStudy

# Create study
study = DOEStudy("catalyst", ["T", "P", "flow"],
                 bounds=[(300, 500), (1, 10), (0.1, 2.0)])
X_design = study.create_design(method="latin_hypercube", n_points=20)
study.save("catalyst.jaxsr")

# After collecting data
study = DOEStudy.load("catalyst.jaxsr")
study.add_observations(X_measured, y_measured)
model = study.fit(max_terms=5)

# Get next experiments
next_pts = study.suggest_next(n_points=5, strategy="uncertainty")
study.save("catalyst.jaxsr")
```

## Quick Reference: CLI

```bash
# Create study with factors
jaxsr init my_study -f "temp:300:500" -f "pressure:1:10" -f "catalyst:A,B,C"

# Generate experimental design → Excel template
jaxsr design my_study.jaxsr -m latin_hypercube -n 20 --format xlsx -o template.xlsx

# Import completed experiments
jaxsr add my_study.jaxsr completed.xlsx

# Fit model
jaxsr fit my_study.jaxsr --max-terms 5 --strategy greedy_forward --criterion bic

# Suggest next experiments
jaxsr suggest my_study.jaxsr -n 5 --strategy uncertainty

# Generate reports
jaxsr report my_study.jaxsr -o report.xlsx
jaxsr report my_study.jaxsr -o report.docx

# Check study status
jaxsr status my_study.jaxsr
```

## Decision Trees

### "Which basis functions should I use?"

See `guides/basis-library.md` for the complete decision guide.

### "Which selection strategy should I use?"

See `guides/model-fitting.md` for strategy comparison and benchmarks.

### "Which UQ method should I use?"

See `guides/uncertainty.md` for method comparison and selection flowchart.

### "How do I set up a DOE study?"

See `guides/doe-workflow.md` for the complete lifecycle guide.

### "How do I add physical constraints?"

See `guides/constraints.md` for constraint types and examples.

### "How do I use the CLI?"

See `guides/cli.md` for full CLI reference with examples.

### "I already know the model form. How do I estimate parameters?"

See `guides/known-model-fitting.md` for a worked example using the Langmuir isotherm,
including parametric basis functions, experiment design, ANOVA, and uncertainty analysis.
Generalizes to Arrhenius, Michaelis-Menten, power laws, and other known models.

### "How do I use Response Surface Methodology?"

See `guides/rsm.md` for RSM designs, canonical analysis, and optimization.

### "How do I set up active learning?"

See `guides/active-learning.md` for acquisition functions and adaptive sampling.

## Templates

Ready-to-use scripts and notebook starters are in `templates/`:

| Template | Use Case |
|----------|----------|
| `basic-regression.py` | Discover an equation from X, y data |
| `constrained-model.py` | Add physical constraints to model |
| `doe-study.py` | Full DOE workflow from design to report |
| `uncertainty-analysis.py` | Compare all UQ methods |
| `active-learning-loop.py` | Iterative experiment-model loop |
| `langmuir-isotherm.py` | Known-model parameter estimation (Langmuir) |
| `notebook-starter.py` | Jupyter notebook cell structure |

## Common Pitfalls

1. **Library too large for exhaustive search.** If you have > 20 basis functions, use `greedy_forward` instead of `exhaustive`.
2. **Using `information_criterion="cv"`.** Only `"aic"`, `"aicc"`, `"bic"` are supported.
3. **Forgetting `safe=True` for transcendental.** Without it, `log(0)` and `1/0` produce NaN.
4. **Over-specifying the basis.** A library with 500+ terms is slow and prone to overfitting. Start simple.
5. **Not checking collinearity.** Use `from jaxsr.utils import check_collinearity` before fitting if terms are nearly redundant.
6. **Stale metrics after refit.** After applying constraints, metrics are automatically recalculated. Do not copy metrics from a previous result.
7. **Python control flow in JIT.** If writing custom basis functions with `@jit`, use `jnp.where` instead of `if/else`.
8. **Calling `float()` on JAX arrays inside JIT.** Use `.item()` outside JIT or keep values as JAX arrays.
