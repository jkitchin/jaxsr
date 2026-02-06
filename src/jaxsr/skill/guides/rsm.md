# Response Surface Methodology (RSM) Guide

## What is RSM?

Response Surface Methodology is a statistical technique for modeling and optimizing
responses that are influenced by several variables. JAXSR provides RSM-specific
designs, coding/decoding, and canonical analysis.

RSM is particularly useful when:
- You have 2-5 continuous factors
- You want to find optimal operating conditions
- The response can be approximated by a low-order polynomial

## RSM Designs

### Central Composite Design (CCD)

The most popular RSM design. Combines a factorial design with center points and
axial (star) points.

```python
from jaxsr import central_composite_design, decode

# Generate design in coded units (-1 to +1)
X_coded = central_composite_design(n_factors=3)

# Decode to natural units
bounds = [(300, 500), (1, 10), (0.1, 2.0)]
X_natural = decode(X_coded, bounds)

print(f"Design has {len(X_coded)} runs")
```

**Properties:**
- Rotatable (equal prediction variance at equal distance from center)
- Supports estimation of all quadratic terms
- Includes center points for lack-of-fit testing
- For k factors: 2^k factorial + 2k axial + center points

### Box-Behnken Design

An alternative that avoids extreme factor combinations (no corner points).

```python
from jaxsr import box_behnken_design

X_coded = box_behnken_design(n_factors=3)
X_natural = decode(X_coded, bounds)
```

**When to prefer over CCD:**
- Factor extremes are dangerous or infeasible (e.g., very high temperature + pressure)
- Fewer runs needed than CCD for 3-4 factors
- All points are within the factor bounds (no axial points outside)

### Factorial Designs

```python
from jaxsr import factorial_design, fractional_factorial_design

# Full factorial (2^k runs)
X_coded = factorial_design(levels=2, n_factors=3)  # 8 runs

# Fractional factorial (reduced runs via aliasing)
X_coded = fractional_factorial_design(n_factors=5, resolution=3)  # Resolution III design
```

### Design Comparison

| Design | Factors | Runs (k=3) | Runs (k=4) | Runs (k=5) |
|--------|---------|------------|------------|------------|
| Full Factorial | 2+ | 8 | 16 | 32 |
| CCD | 2+ | 20 | 30 | 52 |
| Box-Behnken | 3+ | 15 | 27 | 46 |
| Fractional Factorial | 4+ | — | 8 (half) | 16 (quarter) |

## Coding and Decoding

RSM works in coded units where each factor ranges from -1 to +1.

```python
from jaxsr import encode, decode
import numpy as np

bounds = [(300, 500), (1, 10)]

# Natural to coded
X_natural = np.array([[350, 3], [450, 8]])
X_coded = encode(X_natural, bounds)
# X_coded ≈ [[-0.5, -0.56], [0.5, 0.56]]

# Coded to natural
X_back = decode(X_coded, bounds)
# X_back ≈ [[350, 3], [450, 8]]
```

**Why code?** Coded variables ensure all factors have equal influence in the model,
regardless of their natural units.

## ResponseSurface: Convenience Wrapper

```python
from jaxsr import ResponseSurface

rs = ResponseSurface(
    n_factors=3,
    factor_names=["T", "P", "flow"],
    bounds=[(300, 500), (1, 10), (0.1, 2.0)]
)

# Generate design (use dedicated methods)
X_design = rs.ccd(alpha="rotatable", center_points=3)  # or rs.box_behnken()

# After collecting data
rs.fit(X_data, y_data)

# Results
print(rs.model.expression_)
print(rs.model.summary())

# Find optimal conditions via canonical analysis
ca = rs.canonical()
print(f"Stationary point: {ca.stationary_point}")
print(f"Predicted response: {ca.stationary_response:.4f}")
```

## Canonical Analysis

After fitting a quadratic model, canonical analysis reveals the nature of the
stationary point (maximum, minimum, or saddle point).

```python
from jaxsr import canonical_analysis

analysis = canonical_analysis(model, bounds=[(300, 500), (1, 10), (0.1, 2.0)])

print(f"Stationary point: {analysis.stationary_point}")
print(f"Predicted response at stationary point: {analysis.stationary_response}")
print(f"Nature: {analysis.nature}")  # "maximum", "minimum", or "saddle"
print(f"Eigenvalues: {analysis.eigenvalues}")
```

**Interpreting eigenvalues:**
- All negative → maximum (response decreases in all directions)
- All positive → minimum (response increases in all directions)
- Mixed signs → saddle point (maximum in some directions, minimum in others)

**What to do with a saddle point:**
A saddle point means the optimal conditions lie along a ridge. You can:
1. Explore along the ridge to find where the response is best
2. Fix one variable and optimize the others
3. Constrain the problem to avoid the saddle

## Complete RSM Workflow

```python
from jaxsr import (
    ResponseSurface, BasisLibrary, SymbolicRegressor,
    central_composite_design, decode, encode, canonical_analysis
)
import numpy as np

# 1. Define factors and generate design
bounds = [(300, 500), (1, 10), (0.1, 2.0)]
names = ["T", "P", "flow"]

X_coded = central_composite_design(n_factors=3)
X_natural = decode(X_coded, bounds)

# 2. Run experiments (replace with actual experiments)
y = your_experiment(X_natural)

# 3. Fit quadratic model
library = (BasisLibrary(n_features=3, feature_names=names)
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=2)
    .add_interactions(max_order=2)
)

model = SymbolicRegressor(
    basis_library=library,
    max_terms=10,
    strategy="greedy_forward",
    information_criterion="aicc"
)
model.fit(X_natural, y)
print(model.summary())

# 4. Canonical analysis
analysis = canonical_analysis(model, bounds=bounds)
print(f"\nStationary point: {analysis.stationary_point}")
print(f"Nature: {analysis.nature}")
print(f"Predicted response: {analysis.stationary_response:.4f}")

# 5. Validate at the optimum
X_opt = np.array([analysis.stationary_point])
y_pred, lower, upper = model.predict_interval(X_opt, alpha=0.05)
print(f"Prediction at optimum: {y_pred[0]:.4f} [{lower[0]:.4f}, {upper[0]:.4f}]")
```

## Tips

1. **Always include center point replicates.** They provide an estimate of pure error
   and allow lack-of-fit testing. Most RSM designs include 3-5 center points.

2. **Check for lack of fit.** If a quadratic model doesn't fit well, consider:
   - Adding cubic or transcendental terms
   - Narrowing the factor ranges
   - Transforming the response (log, sqrt)

3. **Coded vs. natural units.** Fit models in natural units for interpretability.
   Use coded units only for design generation and canonical analysis.

4. **Visualization.** Use contour plots or surface plots to visualize the response surface:
   ```python
   from jaxsr.plotting import plot_parity, plot_residuals
   plot_parity(y, model.predict(X))
   plot_residuals(model, X, y)
   ```
