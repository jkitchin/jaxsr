# JAXSR Command-Line Interface Guide

The `jaxsr` CLI provides a complete workflow for Design of Experiments (DOE)
without writing Python code. It operates on `.jaxsr` study files that store
your experimental design, data, and fitted models.

## Installation

```bash
# Full install (CLI + Excel + Word reports)
pip install jaxsr[cli,excel,reports]

# Minimal CLI only
pip install jaxsr[cli]

# Add Excel support later
pip install jaxsr[excel]
```

## Quick Start

```bash
# 1. Create a study
jaxsr init catalyst_opt \
  -f "temperature:300:500" \
  -f "pressure:1:10" \
  -f "catalyst:A,B,C"

# 2. Generate experiment template
jaxsr design catalyst_opt.jaxsr -n 20 --format xlsx -o experiments.xlsx

# 3. Fill in results in Excel, then import
jaxsr add catalyst_opt.jaxsr experiments.xlsx

# 4. Fit a model
jaxsr fit catalyst_opt.jaxsr --max-terms 5

# 5. Generate report
jaxsr report catalyst_opt.jaxsr -o report.xlsx
```

---

## Commands Reference

### `jaxsr init`

Create a new DOE study.

```
jaxsr init <name> -f <factor_spec> [-f <factor_spec> ...] [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `name` | Study name (used as default filename) |

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--factors` | `-f` | Factor specification (required, repeatable) |
| `--description` | `-d` | Study description |
| `--output` | `-o` | Output file path (default: `<name>.jaxsr`) |

**Factor specification format:**

- **Continuous:** `"name:low:high"` — defines a numeric factor with bounds
- **Categorical:** `"name:level1,level2,..."` — defines a categorical factor

Auto-detection: if the value after the first colon contains a comma, it's treated as categorical.

**Examples:**

```bash
# Simple 2-factor continuous study
jaxsr init yield_study -f "temperature:300:500" -f "pressure:1:10"

# Mixed continuous and categorical
jaxsr init catalyst_opt \
  -f "temperature:300:500" \
  -f "pressure:1:10" \
  -f "catalyst:A,B,C" \
  -d "Optimize reaction yield"

# Custom output path
jaxsr init my_study -f "x:0:1" -f "y:0:1" -o /data/studies/my_study.jaxsr
```

---

### `jaxsr design`

Generate an experimental design and optionally export it.

```
jaxsr design <study_file> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `study_file` | Path to the `.jaxsr` study file |

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--method` | `-m` | `latin_hypercube` | Design method |
| `--n-points` | `-n` | `20` | Number of design points |
| `--seed` | `-s` | (random) | Random seed for reproducibility |
| `--format` | | `table` | Output format: `table`, `csv`, `xlsx` |
| `--output` | `-o` | (auto) | Output file (for csv/xlsx) |

**Available design methods:**

| Method | Description | Use when |
|--------|-------------|----------|
| `latin_hypercube` | Space-filling LHS design | General-purpose, default choice |
| `sobol` | Sobol quasi-random sequence | Uniform coverage, best with power-of-2 points |
| `halton` | Halton quasi-random sequence | Similar to Sobol, fewer restrictions on n |
| `grid` | Full factorial grid | Small n_factors, want all combinations |
| `factorial` | Full factorial at 2 or more levels | Screening designs |
| `ccd` | Central Composite Design | Response surface methodology |
| `box_behnken` | Box-Behnken design | RSM with 3+ factors, avoids extreme corners |
| `fractional_factorial` | Fractional factorial | Screening many factors efficiently |

**Examples:**

```bash
# Print design as table
jaxsr design study.jaxsr -m latin_hypercube -n 20 -s 42

# Export to CSV
jaxsr design study.jaxsr -n 15 --format csv -o design.csv

# Generate Excel template for lab
jaxsr design study.jaxsr -n 20 --format xlsx -o experiments.xlsx

# Central Composite Design (n_points is ignored, CCD determines its own size)
jaxsr design study.jaxsr -m ccd

# Box-Behnken design
jaxsr design study.jaxsr -m box_behnken
```

**Excel template output:**

When `--format xlsx` is used, the output is a formatted Excel workbook with:

- **Instructions** sheet: how to fill in the template
- **Design** sheet: pre-filled factor columns (locked, blue), empty Response column (unlocked, green), and Notes column
- **_Metadata** sheet (hidden): study fingerprint for validation on re-import

Categorical factors get dropdown validation in Excel. The Response column has
numeric validation. Factor columns are protected to prevent accidental edits.

---

### `jaxsr add`

Import experiment results from an Excel template or CSV file.

```
jaxsr add <study_file> <data_file> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `study_file` | Path to the `.jaxsr` study file |
| `data_file` | Path to the completed `.xlsx` template or `.csv` file |

**Options:**

| Option | Description |
|--------|-------------|
| `--notes` | Notes for this batch of observations |
| `--skip-validation` | Skip template fingerprint validation |

**Validation (Excel templates):**

When importing an `.xlsx` file, the CLI validates that:

1. The file was generated for this specific study (fingerprint match)
2. Factor columns have not been modified
3. Response values are numeric

If validation fails, you'll see a clear error message explaining what's wrong.
Use `--skip-validation` to bypass fingerprint checking (useful for manually
created spreadsheets).

**CSV format:**

For CSV import, the file should have:
- One header row (column names, ignored)
- Factor columns first (in the same order as the study)
- Response column last

```csv
temperature,pressure,response
350.2,3.5,85.1
425.1,7.8,92.3
```

**Examples:**

```bash
# Import from Excel template
jaxsr add study.jaxsr completed_experiments.xlsx

# Import with notes
jaxsr add study.jaxsr batch2.xlsx --notes "Second round from Lab B"

# Import from CSV
jaxsr add study.jaxsr results.csv

# Skip validation for manually created files
jaxsr add study.jaxsr manual_data.xlsx --skip-validation
```

---

### `jaxsr fit`

Fit a symbolic regression model to the study's observations.

```
jaxsr fit <study_file> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `study_file` | Path to the `.jaxsr` study file |

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--max-terms` | `-t` | `10` | Maximum number of terms in the model |
| `--strategy` | | `greedy_forward` | Selection strategy |
| `--criterion` | | `aicc` | Information criterion |

**Selection strategies:**

| Strategy | Description |
|----------|-------------|
| `greedy_forward` | Start empty, add best term iteratively (default, fast) |
| `greedy_backward` | Start full, remove worst term iteratively |
| `exhaustive` | Try all subsets (slow, guaranteed optimal for small problems) |
| `lasso_path` | LASSO regularization path screening |

**Information criteria:**

| Criterion | Description |
|-----------|-------------|
| `aicc` | Corrected AIC — best default, accounts for small samples |
| `aic` | Akaike Information Criterion |
| `bic` | Bayesian Information Criterion — stronger penalty for complexity |

**Examples:**

```bash
# Basic fit
jaxsr fit study.jaxsr

# Fit with at most 5 terms
jaxsr fit study.jaxsr --max-terms 5

# Use backward elimination with BIC
jaxsr fit study.jaxsr --strategy greedy_backward --criterion bic

# Output:
# Model: y = 2.01*x1 + 3.02*x2 + 0.15*x1^2
#   MSE:   0.00234
#   AIC:   -125.4321
#   BIC:   -118.9876
#   Terms: 3
```

---

### `jaxsr suggest`

Suggest next experiments using adaptive sampling.

```
jaxsr suggest <study_file> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `study_file` | Path to the `.jaxsr` study file (must have a fitted model) |

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--n-points` | `-n` | `5` | Number of points to suggest |
| `--strategy` | | `space_filling` | Suggestion strategy |
| `--format` | | `table` | Output format: `table` or `csv` |

**Suggestion strategies:**

| Strategy | Description |
|----------|-------------|
| `space_filling` | Fill gaps in the design space (default) |
| `uncertainty` | Target regions of high model uncertainty |
| `error` | Target regions of high prediction error |
| `leverage` | Target high-leverage points |
| `gradient` | Target regions of steep response gradients |
| `random` | Random sampling within bounds |

**Examples:**

```bash
# Suggest 5 space-filling points
jaxsr suggest study.jaxsr -n 5

# Target uncertain regions
jaxsr suggest study.jaxsr -n 3 --strategy uncertainty

# Output as CSV for programmatic use
jaxsr suggest study.jaxsr -n 10 --format csv
```

---

### `jaxsr report`

Generate a report (Excel or Word).

```
jaxsr report <study_file> -o <output_file>
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `study_file` | Path to the `.jaxsr` study file (must have a fitted model) |

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output file path (required). Extension determines format. |

**Excel report (`.xlsx`):**

Contains four sheets:

| Sheet | Contents |
|-------|----------|
| **Summary** | Study name, factors, observations, model expression, all metrics |
| **Coefficients** | Basis function names, coefficient values, bar chart of magnitudes |
| **Predictions** | Actual vs predicted values, residuals, % error, scatter chart |
| **Residuals** | Residuals vs predicted scatter chart |

If the output file already exists, report sheets are appended (or replaced
if they already exist). This means you can add report sheets to an existing
experiment template workbook.

**Word report (`.docx`):**

A formatted document with:

- Title page with study name, description, timestamps
- Factor summary table
- Model expression and metrics table (R², MSE, RMSE, AIC, BIC, AICc)
- Coefficient table
- Embedded matplotlib plots:
  - Predicted vs actual scatter (with R² in title)
  - Residuals vs predicted scatter
  - Coefficient magnitude horizontal bar chart
- Full prediction table
- Iteration history (if multiple rounds were run)

**Examples:**

```bash
# Excel report
jaxsr report study.jaxsr -o report.xlsx

# Word report
jaxsr report study.jaxsr -o analysis_report.docx

# Add report sheets to the experiment template
jaxsr report study.jaxsr -o experiments.xlsx
```

---

### `jaxsr status`

Display a summary of the study's current state.

```
jaxsr status <study_file>
```

**Examples:**

```bash
jaxsr status study.jaxsr

# Output:
# ============================================================
# DOE Study: catalyst_opt
# ============================================================
# Description: Optimize reaction yield
# Factors: temperature, pressure, catalyst
# Bounds: [(300, 500), (1, 10), (0, 2)]
# Feature types: ['continuous', 'continuous', 'categorical']
# Categories: {2: ['A', 'B', 'C']}
#
# Design: 20 points (15 completed, 5 pending)
# Design method: latin_hypercube
# Observations: 15
#
# Model: y = 0.31*temperature + 1.52*pressure + 4.21*I(catalyst=B)
#   MSE: 2.340000
#   AIC: 42.1234
#   Terms: 3
#
# Iterations: 2
#   Round 1: +10 points → 0.28*temperature + 1.48*pressure (First batch)
#   Round 2: +5 points → 0.31*temperature + 1.52*pressure + 4.21*I(catalyst=B)
#
# Created: 2026-02-06T12:00:00+00:00
# Modified: 2026-02-06T14:30:00+00:00
# ============================================================
```

---

## Complete Workflow Example

This walks through a realistic multi-session DOE workflow for optimizing a
chemical process.

### Session 1: Setup

```bash
# Create the study
jaxsr init reactor_optimization \
  -f "temperature:250:450" \
  -f "flow_rate:10:50" \
  -f "catalyst_loading:1:5" \
  -d "Optimize conversion in the packed bed reactor"

# Generate initial design
jaxsr design reactor_optimization.jaxsr \
  -m latin_hypercube -n 15 -s 42 \
  --format xlsx -o run_sheet.xlsx

# Check status
jaxsr status reactor_optimization.jaxsr
```

Give `run_sheet.xlsx` to the lab team. They fill in the Response column.

### Session 2: First analysis

```bash
# Import completed experiments
jaxsr add reactor_optimization.jaxsr run_sheet.xlsx \
  --notes "First 15 experiments from Lab A"

# Fit initial model
jaxsr fit reactor_optimization.jaxsr --max-terms 5

# Check results
jaxsr status reactor_optimization.jaxsr

# Generate report for the team
jaxsr report reactor_optimization.jaxsr -o first_report.docx
```

### Session 3: Follow-up experiments

```bash
# Suggest targeted follow-up points
jaxsr suggest reactor_optimization.jaxsr -n 5 --strategy uncertainty

# Export as CSV for the lab
jaxsr suggest reactor_optimization.jaxsr -n 5 --format csv > next_runs.csv

# After running experiments, import new data
jaxsr add reactor_optimization.jaxsr followup_results.csv \
  --notes "Follow-up experiments targeting uncertain regions"

# Refit with all data
jaxsr fit reactor_optimization.jaxsr --max-terms 5

# Final report
jaxsr report reactor_optimization.jaxsr -o final_report.xlsx
```

### Sharing with colleagues

The `.jaxsr` file is a self-contained ZIP archive. Share it with anyone
who has jaxsr installed:

```bash
# Colleague can inspect the study
jaxsr status shared_study.jaxsr

# Re-generate report
jaxsr report shared_study.jaxsr -o my_report.docx
```

---

## Python API Integration

The CLI is a thin wrapper around the Python API. Everything the CLI does
can also be done programmatically:

```python
from jaxsr import DOEStudy
from jaxsr.excel import generate_template, read_completed_template, add_report_sheets
from jaxsr.reporting import generate_word_report

# Create study (equivalent to `jaxsr init`)
study = DOEStudy(
    name="reactor_optimization",
    factor_names=["temperature", "flow_rate", "catalyst_loading"],
    bounds=[(250, 450), (10, 50), (1, 5)],
)

# Generate design + Excel template
study.create_design(method="latin_hypercube", n_points=15, random_state=42)
generate_template(study, "run_sheet.xlsx")
study.save("reactor_optimization.jaxsr")

# Later: load, import data, fit
study = DOEStudy.load("reactor_optimization.jaxsr")
X, y = read_completed_template(study, "run_sheet.xlsx")
study.add_observations(X, y, notes="First batch")
study.fit(max_terms=5)
study.save("reactor_optimization.jaxsr")

# Generate reports
add_report_sheets(study, "report.xlsx")
generate_word_report(study, "report.docx")
```

---

## Troubleshooting

### "click is required for the jaxsr CLI"

Install the CLI dependencies:

```bash
pip install jaxsr[cli]
```

### "xlsxwriter is required" / "openpyxl is required"

Install the Excel dependencies:

```bash
pip install jaxsr[excel]
```

### "python-docx is required"

Install the reports dependencies:

```bash
pip install jaxsr[reports]
```

### "Template fingerprint mismatch"

The Excel template was generated for a different study. Make sure you are
loading the template into the same study that generated it. If the file was
manually created, use `--skip-validation`.

### "No observations available"

You need to add experiment results before fitting. Run `jaxsr add` first.

### "No fitted model"

You need to fit a model before generating reports or suggesting next points.
Run `jaxsr fit` first.
