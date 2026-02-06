# Known-Model Fitting Guide: Langmuir Isotherm Example

## Scenario

You have a known model form (e.g., Langmuir isotherm, Arrhenius equation, Michaelis-Menten
kinetics) and want to:

1. **Design experiments** to efficiently estimate the parameters
2. **Fit the model** and get parameter estimates with uncertainty
3. **Use ANOVA** to understand which terms contribute most to the response
4. **Report** the results with confidence intervals

This guide uses the Langmuir isotherm as a worked example, but the pattern applies
to any known model.

## The Langmuir Isotherm

The Langmuir adsorption isotherm models surface coverage as a function of pressure:

    q = q_max * K * P / (1 + K * P)

Parameters:
- `q_max` — maximum adsorption capacity (monolayer coverage)
- `K` — equilibrium adsorption constant
- `P` — partial pressure (the independent variable)
- `q` — amount adsorbed (the response)

This is nonlinear in `K`, so we use **parametric basis functions**.

## Step 1: Design the Experiment

For a single-factor model like Langmuir, design points across the pressure range.
Choose points that span the transition from low coverage to saturation.

```python
import numpy as np
from jaxsr import DOEStudy

# Create study with one factor: pressure
study = DOEStudy(
    name="langmuir_adsorption",
    factor_names=["pressure"],
    bounds=[(0.01, 10.0)],  # Pressure range in bar
    description="Langmuir isotherm parameter estimation"
)

# Use log-spaced design for better coverage of the nonlinear region
# Latin hypercube in log space gives good spread
X_design = study.create_design(method="latin_hypercube", n_points=15, random_state=42)
study.save("langmuir.jaxsr")

print(f"Designed {len(X_design)} experiments")
print(f"Pressure range: [{X_design.min():.3f}, {X_design.max():.3f}] bar")
```

**Design advice for Langmuir:**
- Include points at low P (linear regime: q ≈ q_max * K * P)
- Include points at high P (saturation regime: q ≈ q_max)
- Include points in the transition region (where K*P ≈ 1)
- 12-20 points is usually sufficient for 2 parameters

For better coverage of the nonlinear region, consider custom spacing:

```python
# Manually designed points: log-spaced for Langmuir
P_values = np.logspace(np.log10(0.01), np.log10(10.0), 15)
X_custom = P_values.reshape(-1, 1)
```

## Step 2: Collect Data and Import

```python
# After running experiments (replace with real data)
study = DOEStudy.load("langmuir.jaxsr")

# Example: simulated Langmuir data
# True parameters: q_max=5.0, K=2.0
P = X_design.flatten()
q_true = 5.0 * 2.0 * P / (1 + 2.0 * P)
q_measured = q_true + 0.1 * np.random.randn(len(P))  # 0.1 noise

study.add_observations(X_design, q_measured, notes="Initial measurements")
study.save("langmuir.jaxsr")
```

## Step 3: Build the Basis Library with Parametric Functions

The key insight: the Langmuir model `q = q_max * K*P / (1 + K*P)` is **linear in
q_max** but **nonlinear in K**. We encode this using `add_parametric`:

```python
import jax.numpy as jnp
from jaxsr import BasisLibrary

library = (BasisLibrary(n_features=1, feature_names=["P"])
    .add_constant()  # Intercept (should be ~0 for Langmuir)
    .add_parametric(
        name="K*P/(1+K*P)",
        func=lambda X, K: K * X[:, 0] / (1 + K * X[:, 0]),
        param_bounds={"K": (0.01, 100.0)},
        complexity=3,
        feature_indices=(0,),
        log_scale=True  # K spans orders of magnitude
    )
)
```

**How this works:**
- JAXSR optimizes `K` (the nonlinear parameter) by profile likelihood
- For each candidate `K`, it solves for the linear coefficient (which equals `q_max`)
- The combination gives you both `q_max` and `K`

### Alternative: Linearized Langmuir

If you prefer to avoid parametric fitting, the Langmuir model can be linearized.
The reciprocal form is:

    1/q = 1/q_max + 1/(q_max * K) * 1/P

```python
# Linearized approach (no parametric fitting needed)
library_linear = (BasisLibrary(n_features=1, feature_names=["P"])
    .add_constant()
    .add_custom(
        name="1/P",
        func=lambda X: 1.0 / (X[:, 0] + 1e-10),
        complexity=2,
        feature_indices=[0]
    )
)

# Fit to 1/q instead of q
from jaxsr import SymbolicRegressor

model_lin = SymbolicRegressor(basis_library=library_linear, max_terms=2)
model_lin.fit(X_design, 1.0 / q_measured)

# Extract parameters from coefficients
# 1/q = a + b/P  →  a = 1/q_max, b = 1/(q_max*K)
a = model_lin.coefficients_[0]  # intercept
b = model_lin.coefficients_[1]  # slope of 1/P
q_max_est = 1.0 / a
K_est = a / b
print(f"Linearized: q_max = {q_max_est:.3f}, K = {K_est:.3f}")
```

**Trade-offs:**
- Linearized: simpler, standard OLS, but transforms error structure
- Parametric: fits original data directly, proper error model, more accurate UQ

## Step 4: Fit the Model

```python
from jaxsr import SymbolicRegressor

model = SymbolicRegressor(
    basis_library=library,
    max_terms=2,  # Constant + Langmuir term
    strategy="greedy_forward",
    information_criterion="aicc",  # Use AICc for small samples
)
model.fit(X_design, q_measured)

print(model.summary())
print(f"\nExpression: {model.expression_}")
print(f"R²: {model.metrics_['r2']:.6f}")
print(f"MSE: {model.metrics_['mse']:.6g}")
```

**Interpreting the result:**
- The coefficient of the Langmuir term is `q_max`
- The optimized `K` is embedded in the basis function
- The intercept should be near zero (Langmuir says q=0 when P=0)

## Step 5: Extract Parameters and Uncertainty

### Coefficient Intervals (q_max uncertainty)

```python
# 95% confidence intervals for the linear coefficients
intervals = model.coefficient_intervals(alpha=0.05)
for name, (est, lo, hi, se) in intervals.items():
    print(f"  {name}: {est:.4f}  [{lo:.4f}, {hi:.4f}]  SE={se:.4f}")

# The coefficient of the Langmuir term IS q_max
print(f"\nq_max estimate: {float(model.coefficients_[1]):.4f}")
est, lo, hi, se = intervals[model.selected_features_[1]]
print(f"q_max 95% CI: [{lo:.4f}, {hi:.4f}]")
```

### Prediction Intervals

```python
# Predict across the full pressure range
P_pred = np.linspace(0.01, 10.0, 100).reshape(-1, 1)
y_pred, lower, upper = model.predict_interval(P_pred, alpha=0.05)

print(f"Average 95% PI width: {np.mean(np.asarray(upper) - np.asarray(lower)):.4f}")
```

### Bootstrap for K Uncertainty

Since `K` is a nonlinear parameter, OLS intervals don't directly give its uncertainty.
Use bootstrap:

```python
from jaxsr import bootstrap_coefficients, bootstrap_predict

# Bootstrap prediction intervals
boot = bootstrap_predict(model, P_pred, n_bootstrap=500, alpha=0.05)
print(f"Bootstrap 95% PI width: "
      f"{np.mean(np.asarray(boot['upper']) - np.asarray(boot['lower'])):.4f}")

# Bootstrap coefficient intervals
boot_coef = bootstrap_coefficients(model, n_bootstrap=500, alpha=0.05)
print("\nBootstrap coefficient intervals:")
for name, lo, hi in zip(boot_coef["names"], boot_coef["lower"], boot_coef["upper"], strict=False):
    print(f"  {name}: [{float(lo):.4f}, {float(hi):.4f}]")
```

### Conformal Prediction (distribution-free)

```python
# Distribution-free intervals with guaranteed coverage
y_conf, conf_lo, conf_hi = model.predict_conformal(
    P_pred, alpha=0.05, method="jackknife+"
)
print(f"Conformal 95% PI width: "
      f"{np.mean(np.asarray(conf_hi) - np.asarray(conf_lo)):.4f}")
```

## Step 6: ANOVA — Term Importance

ANOVA decomposes the model's explanatory power by term:

```python
from jaxsr import anova

result = anova(model)

print("ANOVA Decomposition:")
print("-" * 50)
summary_sources = {"Model", "Residual", "Total"}
term_rows = [r for r in result.rows if r.source not in summary_sources]
model_ss = sum(r.sum_sq for r in term_rows)
for row in term_rows:
    pct = 100 * row.sum_sq / model_ss if model_ss > 0 else 0.0
    print(f"  {row.source:30s}  SS = {row.sum_sq:10.4f}  ({pct:5.1f}%)")
```

**For Langmuir, expect:**
- The Langmuir term (K*P/(1+K*P)) should account for > 95% of variance
- The constant term should be negligible if the model is correct
- If the constant contributes significantly, there may be an offset in the data

### What ANOVA Tells You

| Finding | Interpretation | Action |
|---------|---------------|--------|
| Langmuir term > 95% | Model fits well | Good — report parameters |
| Constant term > 5% | Offset present | Check calibration, consider baseline correction |
| Low total R² | Model inadequate | Try dual-site Langmuir or Freundlich |

## Step 7: Model Comparison

If you're unsure whether Langmuir is the right model, fit alternatives:

```python
# Langmuir (already fitted above)
# ...

# Freundlich: q = K_f * P^n
library_freundlich = (BasisLibrary(n_features=1, feature_names=["P"])
    .add_constant()
    .add_parametric(
        name="P^n",
        func=lambda X, n: jnp.power(jnp.abs(X[:, 0]) + 1e-10, n),
        param_bounds={"n": (0.1, 2.0)},
        complexity=3,
        feature_indices=(0,),
    )
)

model_freundlich = SymbolicRegressor(
    basis_library=library_freundlich, max_terms=2, information_criterion="aicc"
)
model_freundlich.fit(X_design, q_measured)

# BET: q = q_m * C * P / ((Ps - P) * (1 + (C-1)*P/Ps))
# (more complex — would need add_parametric with two parameters)

# Compare by information criterion
print(f"Langmuir  AICc: {model.metrics_['aicc']:.2f}")
print(f"Freundlich AICc: {model_freundlich.metrics_['aicc']:.2f}")
print(f"Better model: {'Langmuir' if model.metrics_['aicc'] < model_freundlich.metrics_['aicc'] else 'Freundlich'}")
```

## Step 8: Adaptive Design (Optional)

If initial data is insufficient, use active learning to suggest targeted experiments:

```python
from jaxsr import AdaptiveSampler

sampler = AdaptiveSampler(
    model=model,
    bounds=[(0.01, 10.0)],
    strategy="uncertainty",  # Focus on uncertain regions
    batch_size=5,
)
result = sampler.suggest(n_points=5)

print("Suggested next pressures:")
for pt in result.points:
    print(f"  P = {pt[0]:.3f} bar")
```

For Langmuir, uncertainty-guided sampling will typically suggest:
- Points near the transition region (K*P ≈ 1)
- Points at the extremes to nail down q_max and the low-P slope

## Step 9: Report

```python
# LaTeX equation for publication
print(f"LaTeX: ${model.to_latex()}$")

# If using DOEStudy, generate full report
from jaxsr.excel import add_report_sheets
study = DOEStudy.load("langmuir.jaxsr")
add_report_sheets(study, "langmuir_report.xlsx")
```

## Complete Script

```python
"""Langmuir Isotherm: Experiment Design → Parameter Estimation → ANOVA → Report"""
import numpy as np
import jax.numpy as jnp
from jaxsr import (
    BasisLibrary, SymbolicRegressor, DOEStudy, AdaptiveSampler,
    anova, bootstrap_predict, bootstrap_coefficients
)

# --- Design ---
study = DOEStudy("langmuir", ["pressure"], bounds=[(0.01, 10.0)])
X = study.create_design(method="latin_hypercube", n_points=15, random_state=42)

# --- Collect data (replace with real measurements) ---
P = X.flatten()
q = 5.0 * 2.0 * P / (1 + 2.0 * P) + 0.1 * np.random.randn(len(P))
study.add_observations(X, q, notes="Initial batch")

# --- Build Langmuir basis ---
library = (BasisLibrary(n_features=1, feature_names=["P"])
    .add_constant()
    .add_parametric(
        name="K*P/(1+K*P)",
        func=lambda X, K: K * X[:, 0] / (1 + K * X[:, 0]),
        param_bounds={"K": (0.01, 100.0)},
        complexity=3, feature_indices=(0,), log_scale=True
    )
)

# --- Fit ---
model = SymbolicRegressor(basis_library=library, max_terms=2, information_criterion="aicc")
model.fit(X, q)
print(model.summary())

# --- Uncertainty ---
intervals = model.coefficient_intervals(alpha=0.05)
for name, (est, lo, hi, se) in intervals.items():
    print(f"  {name}: {est:.4f}  [{lo:.4f}, {hi:.4f}]  SE={se:.4f}")

boot = bootstrap_coefficients(model, n_bootstrap=500, alpha=0.05)
print("\nBootstrap intervals:")
for name, lo, hi in zip(boot["names"], boot["lower"], boot["upper"], strict=False):
    print(f"  {name}: [{float(lo):.4f}, {float(hi):.4f}]")

# --- ANOVA ---
result = anova(model)
print("\nANOVA:")
summary_sources = {"Model", "Residual", "Total"}
term_rows = [r for r in result.rows if r.source not in summary_sources]
model_ss = sum(r.sum_sq for r in term_rows)
for row in term_rows:
    pct = 100 * row.sum_sq / model_ss if model_ss > 0 else 0.0
    print(f"  {row.source}: {pct:.1f}%")

# --- Adaptive suggestions ---
sampler = AdaptiveSampler(model, bounds=[(0.01, 10.0)], strategy="uncertainty")
next_pts = sampler.suggest(n_points=5)
print(f"\nNext experiments: {next_pts.points.flatten()}")

# --- Save ---
study.save("langmuir.jaxsr")
model.save("langmuir_model.json")
print(f"\nLaTeX: ${model.to_latex()}$")
```

## Generalizing to Other Known Models

The same pattern works for any model that is linear in some parameters and
nonlinear in others:

| Model | Linear Parameter(s) | Nonlinear Parameter(s) | Parametric Basis |
|-------|---------------------|----------------------|-----------------|
| Langmuir: q = q_m·K·P/(1+K·P) | q_m | K | `K*P/(1+K*P)` |
| Arrhenius: k = A·exp(-Ea/RT) | A | Ea | `exp(-Ea/(R*T))` |
| Michaelis-Menten: v = V_max·S/(K_m+S) | V_max | K_m | `S/(K_m+S)` |
| Power law: y = a·x^n | a | n | `x^n` |
| Exponential decay: y = A·exp(-k·t) + B | A, B | k | `1` and `exp(-k*t)` |
| Logistic: y = L/(1+exp(-k·(x-x0))) | L | k, x0 | `1/(1+exp(-k*(x-x0)))` |

For each: encode the nonlinear parameters in `add_parametric`, and let the
linear coefficient(s) be estimated by OLS.
