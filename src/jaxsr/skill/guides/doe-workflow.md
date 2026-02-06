# Design of Experiments (DOE) Workflow Guide

## Overview

A DOE study in JAXSR follows this lifecycle:

```
Create Study → Generate Design → Run Experiments → Import Data → Fit Model
                                                                    ↓
                 ← Suggest Next ← Evaluate Model ← Review Results ←
```

The `DOEStudy` class manages this entire lifecycle with persistent storage.

## Step 1: Create the Study

### Python API

```python
from jaxsr import DOEStudy

study = DOEStudy(
    name="catalyst_optimization",
    factor_names=["temperature", "pressure", "flow_rate"],
    bounds=[(300, 500), (1, 10), (0.1, 2.0)],
    description="Optimize catalyst yield as a function of T, P, and flow"
)
study.save("catalyst.jaxsr")
```

### With Categorical Factors

```python
study = DOEStudy(
    name="catalyst_optimization",
    factor_names=["temperature", "pressure", "catalyst"],
    bounds=[(300, 500), (1, 10), (0, 2)],
    feature_types=["continuous", "continuous", "categorical"],
    categories={2: ["Pt", "Pd", "Rh"]},
    description="Compare catalyst types"
)
```

### CLI

```bash
jaxsr init catalyst_optimization \
    -f "temperature:300:500" \
    -f "pressure:1:10" \
    -f "flow_rate:0.1:2.0" \
    -d "Optimize catalyst yield"

# With categorical factor
jaxsr init catalyst_optimization \
    -f "temperature:300:500" \
    -f "pressure:1:10" \
    -f "catalyst:Pt,Pd,Rh"
```

## Step 2: Generate Experimental Design

### Design Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `latin_hypercube` | Space-filling, stratified | Default, general purpose |
| `sobol` | Quasi-random, low-discrepancy | Uniform coverage, many factors |
| `halton` | Quasi-random sequence | Similar to Sobol, simpler |
| `grid` | Full factorial grid | When you need all combinations |

### How Many Points?

| Factors | Minimum (screening) | Recommended | Comprehensive |
|---------|---------------------|-------------|---------------|
| 2 | 6-8 | 15-20 | 30+ |
| 3 | 10-12 | 20-30 | 50+ |
| 4 | 15-20 | 30-50 | 80+ |
| 5+ | 2*k+2 | 5*k to 10*k | 15*k+ |

Rule of thumb: At least 3-5 points per basis function you expect in the final model.

### Python API

```python
study = DOEStudy.load("catalyst.jaxsr")
X_design = study.create_design(method="latin_hypercube", n_points=20, random_state=42)
study.save("catalyst.jaxsr")

# View the design
print(X_design)  # shape (20, 3)
```

### CLI

```bash
# Print as table
jaxsr design catalyst.jaxsr -m latin_hypercube -n 20

# Export to Excel template
jaxsr design catalyst.jaxsr -m latin_hypercube -n 20 --format xlsx -o experiments.xlsx

# Export to CSV
jaxsr design catalyst.jaxsr -n 20 --format csv -o experiments.csv
```

### RSM-Specific Designs

For Response Surface Methodology, use specialized designs:

```python
from jaxsr import central_composite_design, box_behnken_design

# Central Composite Design (CCD) — good for quadratic models
X_coded = central_composite_design(n_factors=3)

# Box-Behnken Design — no extreme corners
X_coded = box_behnken_design(n_factors=3)

# Decode from coded (-1, +1) to natural units
from jaxsr import decode
X_natural = decode(X_coded, bounds=[(300, 500), (1, 10), (0.1, 2.0)])
```

## Step 3: Run Experiments

This is the part you do in the lab or simulation. JAXSR generates the design;
you collect the response values.

**Excel workflow:**
1. `jaxsr design study.jaxsr --format xlsx -o template.xlsx`
2. Open `template.xlsx`, fill in the "Response" column
3. `jaxsr add study.jaxsr template.xlsx`

## Step 4: Import Results

### Python API

```python
import numpy as np

study = DOEStudy.load("catalyst.jaxsr")

# Add observations
X_measured = np.array([[350, 5, 1.0], [400, 3, 0.5], ...])
y_measured = np.array([85.2, 91.3, ...])
study.add_observations(X_measured, y_measured, notes="Batch 1, 2024-01-15")
study.save("catalyst.jaxsr")

print(f"Total observations: {study.n_observations}")
```

### CLI

```bash
# From Excel template (validates columns match)
jaxsr add catalyst.jaxsr completed_experiments.xlsx

# From CSV (last column is response)
jaxsr add catalyst.jaxsr results.csv

# Skip validation
jaxsr add catalyst.jaxsr results.xlsx --skip-validation

# With notes
jaxsr add catalyst.jaxsr results.xlsx --notes "Batch 2, repeat runs"
```

## Step 5: Fit Model

### Python API

```python
study = DOEStudy.load("catalyst.jaxsr")

model = study.fit(
    max_terms=5,
    strategy="greedy_forward",
    information_criterion="aicc"
)

print(model.expression_)
print(model.summary())
study.save("catalyst.jaxsr")
```

### CLI

```bash
jaxsr fit catalyst.jaxsr --max-terms 5 --strategy greedy_forward --criterion aicc
```

### Custom Basis Library

By default, `study.fit()` creates a standard library (constant + linear + quadratic
+ interactions). For custom basis functions:

```python
from jaxsr import BasisLibrary

library = (BasisLibrary(n_features=study.n_factors, feature_names=study.factor_names)
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions(max_order=2)
    .add_transcendental(funcs=["log", "exp"])
)

# Use a custom library by fitting a SymbolicRegressor directly
from jaxsr import SymbolicRegressor
model = SymbolicRegressor(basis_library=library, max_terms=5, information_criterion="aicc")
model.fit(study.X_train, study.y_train)
```

## Step 6: Evaluate & Iterate

### Check Model Quality

```python
print(model.summary())
print(f"R²: {model.metrics_['r2']:.4f}")
print(f"MSE: {model.metrics_['mse']:.6g}")
print(f"BIC: {model.metrics_['bic']:.2f}")

# Coefficient significance
intervals = model.coefficient_intervals(alpha=0.05)
for name, (est, lo, hi, se) in intervals.items():
    sig = "*" if lo * hi > 0 else ""  # Significant if interval excludes 0
    print(f"  {name}: {est:.4f}  [{lo:.4f}, {hi:.4f}] {sig}")
```

### Suggest Next Experiments

```python
next_pts = study.suggest_next(n_points=5, strategy="uncertainty")
print("Suggested next experiments:")
print(next_pts)
study.save("catalyst.jaxsr")
```

```bash
jaxsr suggest catalyst.jaxsr -n 5 --strategy uncertainty
```

### When to Stop

- R² > 0.95 and model is physically sensible
- Coefficient intervals are narrow (all terms are significant)
- Prediction intervals are acceptable for your application
- Adding more data doesn't change the selected model
- Bootstrap model selection shows stable term selection

## Step 7: Report

### Python API

```python
from jaxsr.excel import add_report_sheets

add_report_sheets(study, "report.xlsx")

# Or Word report
from jaxsr.reporting import generate_word_report
generate_word_report(study, "report.docx")
```

### CLI

```bash
jaxsr report catalyst.jaxsr -o report.xlsx
jaxsr report catalyst.jaxsr -o report.docx
```

### Quick Status

```bash
jaxsr status catalyst.jaxsr
```

## Study File Format

The `.jaxsr` file is a ZIP archive containing:
- Study metadata (JSON)
- Design points
- Observations with timestamps and notes
- Fitted model (if available)
- Basis library configuration

You can share `.jaxsr` files with collaborators to reproduce the analysis.

## Multi-Round Example

```python
from jaxsr import DOEStudy

# Round 1: Initial design
study = DOEStudy("reactor", ["T", "P", "residence_time"],
                 bounds=[(350, 550), (1, 20), (1, 60)])
X1 = study.create_design(method="latin_hypercube", n_points=15)
# ... run experiments ...
study.add_observations(X1, y1, notes="Round 1: screening")
model = study.fit(max_terms=5)
study.save("reactor.jaxsr")

# Round 2: Targeted experiments
study = DOEStudy.load("reactor.jaxsr")
X2 = study.suggest_next(n_points=5, strategy="uncertainty")
# ... run experiments ...
study.add_observations(X2, y2, notes="Round 2: uncertainty reduction")
model = study.fit(max_terms=5)
study.save("reactor.jaxsr")

# Round 3: Final refinement
study = DOEStudy.load("reactor.jaxsr")
X3 = study.suggest_next(n_points=3, strategy="error")
# ... run experiments ...
study.add_observations(X3, y3, notes="Round 3: error reduction")
model = study.fit(max_terms=5)
study.save("reactor.jaxsr")

# Final report
from jaxsr.excel import add_report_sheets
add_report_sheets(study, "reactor_report.xlsx")
```
