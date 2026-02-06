# Model Fitting Guide: Strategies and Criteria

## Overview

Model fitting in JAXSR has two key choices:
1. **Selection strategy** — the algorithm that searches through candidate models
2. **Information criterion** — the metric that balances fit quality vs. complexity

## Selection Strategies

### `greedy_forward` (Default, Recommended)

**Algorithm:** Start with empty model. Add one term at a time, choosing the term that
most improves the information criterion. Stop when no addition improves the criterion.

```python
model = SymbolicRegressor(
    basis_library=library,
    max_terms=5,
    strategy="greedy_forward",
    information_criterion="bic"
)
model.fit(X, y)
```

**Pros:**
- Fast: O(k * n_basis) evaluations where k = max_terms
- Works well for most problems
- Naturally produces sparse models
- Builds the Pareto front incrementally

**Cons:**
- Greedy — may miss globally optimal combinations
- Cannot undo a term once added

**Use when:** Default choice. Works for most problems.

### `greedy_backward`

**Algorithm:** Start with all terms (up to max_terms or all basis functions). Remove
one term at a time, choosing the removal that least worsens the criterion. Stop when
any removal worsens the criterion.

```python
model = SymbolicRegressor(strategy="greedy_backward", max_terms=15)
```

**Pros:**
- Can find models that forward selection misses
- Better at retaining important interaction terms

**Cons:**
- Slower than forward (starts with large model)
- Requires max_terms <= n_observations for OLS to work

**Use when:** You expect many terms to be relevant, or forward selection gives poor results.

### `exhaustive`

**Algorithm:** Evaluate every possible subset of basis functions (up to max_terms).
Return the globally optimal model.

```python
model = SymbolicRegressor(strategy="exhaustive", max_terms=5)
```

**Pros:**
- Guaranteed global optimum
- No greedy approximation errors

**Cons:**
- O(2^n_basis) complexity — impractical for n_basis > 20
- Very slow for large libraries

**Use when:** Library has fewer than ~20 basis functions and you want the provably best model.

### `lasso_path`

**Algorithm:** Sweep the LASSO regularization parameter from high to low.
At each regularization level, some coefficients are zero. This traces a path
from empty model to full model. Select the point on the path that optimizes
the information criterion.

```python
model = SymbolicRegressor(strategy="lasso_path")
```

**Pros:**
- Very fast for large libraries (hundreds of terms)
- Good for screening — identifies important features quickly
- Handles correlated features better than forward selection

**Cons:**
- Biased coefficient estimates (LASSO shrinks toward zero)
- May select slightly different terms than OLS-based methods
- Coefficients are re-estimated via OLS after selection

**Use when:** Library has 100+ terms, or you want fast feature screening.

## Strategy Comparison

| Strategy | Speed | Optimality | Best Library Size |
|----------|-------|-----------|-------------------|
| `greedy_forward` | Fast | Local | 20-200 |
| `greedy_backward` | Medium | Local | 20-100 |
| `exhaustive` | Slow | Global | < 20 |
| `lasso_path` | Fast | Approximate | 50-1000+ |

## Information Criteria

All three criteria balance goodness-of-fit (MSE) against model complexity (number of terms k).

### AIC (Akaike Information Criterion)

    AIC = n * ln(MSE) + 2k

- Minimizes expected prediction error
- Less penalty on complexity → tends to select larger models
- Asymptotically equivalent to leave-one-out cross-validation

### AICc (Corrected AIC)

    AICc = AIC + 2k(k+1)/(n-k-1)

- Adds a correction term for small samples
- Converges to AIC as n → infinity
- **Use when n/k < 40** (few data points per parameter)

### BIC (Bayesian Information Criterion)

    BIC = n * ln(MSE) + k * ln(n)

- Stronger complexity penalty than AIC (for n >= 8)
- Tends to select simpler, more interpretable models
- Consistent — selects the true model as n → infinity (if it's in the library)

## Criterion Selection Flowchart

```
Is n/k < 40? (few data points per parameter)
├── YES → Use "aicc"
└── NO
    ├── Want simplest interpretable model? → Use "bic"
    └── Want best prediction accuracy? → Use "aic"
```

**Default recommendation:** `"bic"` for interpretability, `"aicc"` for small datasets.

## Regularization

Optional L2 (ridge) regularization stabilizes coefficient estimates when basis
functions are nearly collinear.

```python
model = SymbolicRegressor(regularization=0.01)  # L2 alpha
```

**When to use:**
- Numerical warnings during fitting
- Very similar basis functions (e.g., x^2 and x^2.1)
- Coefficients are unreasonably large

**Typical values:** 1e-4 to 1e-1. Start small and increase if needed.

## Cross-Validation

Built-in cross-validation for model assessment:

```python
from jaxsr import cross_validate

scores = cross_validate(model, X, y, cv=5)
print(f"CV R²: {scores.mean():.4f} ± {scores.std():.4f}")
```

## Pareto Front

After fitting, JAXSR computes a Pareto front: the set of models where no other model
is both simpler AND more accurate.

```python
model.fit(X, y)

# Access the Pareto front
for result in model.pareto_front_:
    print(f"{result.n_terms} terms, MSE={result.mse:.6g}")

# The best model (by information criterion) is automatically selected
print(model.expression_)
```

Visualize it:
```python
from jaxsr.plotting import plot_pareto_front
plot_pareto_front(model.pareto_front_)
```

## Post-Fit Inspection

```python
# Full summary
print(model.summary())

# Individual attributes
print(model.expression_)        # Human-readable expression
print(model.coefficients_)      # Coefficient values
print(model.selected_features_) # Names of selected basis functions
print(model.selected_indices_)  # Indices into the basis library
print(model.complexity_)        # Total complexity score
print(model.metrics_)           # Dict: mse, aic, bic, aicc, r2
print(model.sigma_)             # Estimated noise standard deviation
```

## Online Updates

When new data arrives, update the model without refitting from scratch:

```python
model.update(X_new, y_new)
print(model.expression_)  # may change if new data shifts the optimal model
```

## Export Options

```python
# SymPy symbolic expression
sympy_expr = model.to_sympy()

# LaTeX string
latex = model.to_latex()

# Pure NumPy callable (no JAX dependency)
predict_fn = model.to_callable()
y_pred = predict_fn(X_numpy)

# Save/load model
model.save("model.json")
loaded = SymbolicRegressor.load("model.json")
```
