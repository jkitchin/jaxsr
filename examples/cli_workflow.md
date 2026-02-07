# CLI Workflow: Catalyst Screening Study

This walkthrough demonstrates the complete `jaxsr` command-line workflow —
from study creation through adaptive experiments to final reporting —
without writing any Python code.

## Scenario

You're screening three factors for a heterogeneous catalysis reaction:

| Factor | Range | Type |
|--------|-------|------|
| Temperature | 300–500 K | Continuous |
| Pressure | 1–10 bar | Continuous |
| Catalyst | Pt, Pd, Rh | Categorical |

The response is conversion (%).

---

## Step 1: Create the Study

```bash
jaxsr init catalyst_screening \
    -f "temperature:300:500" \
    -f "pressure:1:10" \
    -f "catalyst:Pt,Pd,Rh" \
    -d "Screen catalyst type, temperature, and pressure for max conversion"
```

Output:
```
Created study 'catalyst_screening' with 3 factors.
Saved to: catalyst_screening.jaxsr
```

The `.jaxsr` file is a portable ZIP archive containing the study metadata.

---

## Step 2: Generate an Experimental Design

Create a 20-point Latin Hypercube design and export to an Excel template
for lab use:

```bash
jaxsr design catalyst_screening.jaxsr \
    -m latin_hypercube \
    -n 20 \
    -s 42 \
    --format xlsx \
    -o lab_template.xlsx
```

Output:
```
Generated 20 design points using latin_hypercube.
Excel template written to: lab_template.xlsx
```

You can also preview the design as a table:

```bash
jaxsr design catalyst_screening.jaxsr -n 20 -s 42
```

Output:
```
Generated 20 design points using latin_hypercube.
  Run   temperature      pressure     catalyst
----------------------------------------------
    1       347.500         3.250           Pd
    2       421.250         7.750           Pt
    3       ...             ...            ...
```

Or export to CSV for scripting:

```bash
jaxsr design catalyst_screening.jaxsr -n 20 --format csv -o design.csv
```

---

## Step 3: Run Experiments in the Lab

1. Open `lab_template.xlsx`
2. For each row, run the experiment at the specified conditions
3. Fill in the **Response** column with the measured conversion (%)
4. Save the file

The Excel template has the factor columns pre-filled. You only need to add
the response values.

---

## Step 4: Import Results

```bash
jaxsr add catalyst_screening.jaxsr lab_template.xlsx \
    --notes "Batch 1: initial screening, 2024-01-15"
```

Output:
```
Added 20 observations. Total: 20
All design points completed!
```

If you have a CSV instead:

```bash
jaxsr add catalyst_screening.jaxsr results.csv --notes "From CSV"
```

**CSV format:** columns must match factor names, with the last column as
the response. Example:

```csv
temperature,pressure,catalyst,Response
347.5,3.25,Pd,62.1
421.25,7.75,Pt,78.3
...
```

---

## Step 5: Fit a Model

```bash
jaxsr fit catalyst_screening.jaxsr \
    --max-terms 5 \
    --strategy greedy_forward \
    --criterion bic
```

Output:
```
Model: y = 0.15*temperature + 2.3*pressure + 12.5*I(catalyst=Pt) - 0.0001*temperature^2
  MSE:   4.231
  AIC:   87.42
  BIC:   91.15
  Terms: 4
```

### Choosing `--criterion`:

| Data Size | Recommendation |
|-----------|----------------|
| < 40 observations | `--criterion aicc` (corrected for small samples) |
| 40–200 observations | `--criterion bic` (sparser models) |
| > 200 observations | `--criterion aic` or `--criterion bic` |

### Choosing `--strategy`:

| Library Size | Recommendation |
|-------------|----------------|
| < 20 basis functions | `--strategy exhaustive` (globally optimal) |
| 20–200 | `--strategy greedy_forward` (default, fast) |
| 200+ | `--strategy lasso_path` (regularized screening) |

---

## Step 6: Check Study Status

```bash
jaxsr status catalyst_screening.jaxsr
```

Output:
```
Study: catalyst_screening
Description: Screen catalyst type, temperature, and pressure for max conversion
Factors:
  temperature: [300, 500] (continuous)
  pressure: [1, 10] (continuous)
  catalyst: Pt, Pd, Rh (categorical)
Observations: 20
Model: y = 0.15*temperature + 2.3*pressure + 12.5*I(catalyst=Pt) - 0.0001*temperature^2
  R² = 0.9234
  MSE = 4.231
```

---

## Step 7: Suggest Next Experiments

The model identifies where to measure next for maximum information gain:

```bash
jaxsr suggest catalyst_screening.jaxsr \
    -n 5 \
    --strategy uncertainty
```

Output:
```
Suggested 5 next experiments:
  Run   temperature      pressure     catalyst
----------------------------------------------
    1       312.000         8.500           Rh
    2       488.000         2.100           Pt
    3       405.000         9.800           Pd
    4       500.000         5.500           Rh
    5       300.000         1.200           Pd
```

### Suggestion strategies:

| Strategy | When to use |
|----------|-------------|
| `space_filling` | No model yet, or want uniform coverage |
| `uncertainty` | Reduce prediction uncertainty everywhere |
| `error` | Fix regions where the model fits poorly |
| `leverage` | Stabilize coefficient estimates |

Export as CSV for automation:

```bash
jaxsr suggest catalyst_screening.jaxsr -n 5 --format csv > next_batch.csv
```

---

## Step 8: Add More Data and Refit

After running the suggested experiments:

```bash
jaxsr add catalyst_screening.jaxsr batch2.xlsx --notes "Batch 2: uncertainty-guided"
jaxsr fit catalyst_screening.jaxsr --max-terms 5 --criterion bic
jaxsr status catalyst_screening.jaxsr
```

Repeat Steps 7–8 until the model is satisfactory.

### When to stop:

- R² > 0.95 and model is physically sensible
- Adding data doesn't change the model expression
- Prediction intervals are narrow enough for your application
- Budget is exhausted

---

## Step 9: Generate Reports

### Excel Report

```bash
jaxsr report catalyst_screening.jaxsr -o report.xlsx
```

The Excel workbook includes:
- Study summary sheet
- Design matrix with responses
- Model coefficients and metrics
- Pareto front (complexity vs. accuracy)

### Word Report

```bash
jaxsr report catalyst_screening.jaxsr -o report.docx
```

The Word document includes:
- Formatted model equation
- Coefficient table with standard errors
- Diagnostic discussion
- Embedded figures

---

## Complete Session

Here's the entire workflow as a single script:

```bash
#!/bin/bash
# Catalyst screening study — complete CLI workflow

# 1. Setup
jaxsr init catalyst_screening \
    -f "temperature:300:500" \
    -f "pressure:1:10" \
    -f "catalyst:Pt,Pd,Rh" \
    -d "Catalyst screening"

# 2. Design → Excel template
jaxsr design catalyst_screening.jaxsr -n 20 -s 42 --format xlsx -o template.xlsx
echo ">>> Fill in template.xlsx with experimental results, then continue"

# 3. Import results (after lab work)
jaxsr add catalyst_screening.jaxsr template.xlsx --notes "Initial batch"

# 4. Fit
jaxsr fit catalyst_screening.jaxsr --max-terms 5 --criterion bic
jaxsr status catalyst_screening.jaxsr

# 5. Adaptive round
jaxsr suggest catalyst_screening.jaxsr -n 5 --strategy uncertainty
echo ">>> Run suggested experiments, save to batch2.xlsx, then continue"

# 6. Import and refit
jaxsr add catalyst_screening.jaxsr batch2.xlsx --notes "Adaptive batch"
jaxsr fit catalyst_screening.jaxsr --max-terms 5 --criterion bic

# 7. Report
jaxsr report catalyst_screening.jaxsr -o final_report.xlsx
jaxsr report catalyst_screening.jaxsr -o final_report.docx
```

---

## Tips

1. **Always use `--notes`** when adding data. It creates an audit trail
   inside the `.jaxsr` file.

2. **The `.jaxsr` file is self-contained.** Share it with collaborators —
   they can run `jaxsr status`, `jaxsr fit`, or `jaxsr report` on their
   own machine.

3. **Seed the design** with `-s 42` (or any integer) for reproducibility.

4. **Start with fewer points.** 15–20 points is enough for a first pass.
   Active learning (Step 7) tells you exactly where to measure next.

5. **Don't over-specify `--max-terms`.** Start with 5 and increase only
   if the model R² is poor. More terms = harder to interpret.
