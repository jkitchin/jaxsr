# JAXSR Review Skill — Red-Team, Engineering & Pedagogical Review

Systematically review JAXSR code, documentation, guides, and notebooks for
API correctness, engineering quality, and pedagogical clarity.

## Skill Activation

Activate this skill when the user invokes `/jaxsr-review` or asks to review, audit, or
check correctness of JAXSR-related files (notebooks, guides, templates, source code).

## Invocation Syntax

```
/jaxsr-review <TARGET>                          # all scopes on one file/dir
/jaxsr-review --scope api <TARGET>              # API correctness only
/jaxsr-review --scope engineering <TARGET>      # safety/robustness only
/jaxsr-review --scope pedagogy <TARGET>         # clarity/explanation only
```

`TARGET` can be a file path, directory, or glob pattern.
If no target is given, review the whole project (`examples/`, `docs/`, `src/jaxsr/`).

## Scope Mapping

| Scope | What it checks | CLAUDE.md checklists |
|-------|---------------|----------------------|
| `api` | Signatures, return types, imports, ANOVA filtering, copy-paste safety | #1 Red-Team, #5 Cross-Ref, #6 Copy-Paste |
| `engineering` | Numerical hazards, destructive ops, packaging, dead code, docstrings | #2 Software Engineering |
| `pedagogy` | Progression, term definitions, "why" explanations, coverage gaps | #3 Pedagogical, #4 Coverage Gap |

When no `--scope` is specified, run **all three** scopes.

---

## Phase 1: Parse Arguments

1. Extract `--scope` value (default: all three scopes).
2. Extract `TARGET` path(s). Resolve globs. Verify files exist.
3. Classify each file by type: `.ipynb` (notebook), `.md` (guide/doc), `.py` (source/template).
4. Read each target file. For notebooks, examine every code cell.

---

## Phase 2: API Correctness Review (scope = `api`)

This is the highest-priority review. Incorrect examples teach users wrong patterns.

### API Truth Table

For every JAXSR API call found in the target files, verify against this authoritative
reference. Each entry lists: **function/class**, **correct usage**, and **common mistake**.

#### Return Types

| API | Returns | Common Mistake |
|-----|---------|---------------|
| `model.coefficient_intervals()` | `dict[str, (est, lo, hi, se)]` — 4-tuple per term | Unpacking as `(lo, hi)` — it's a 4-tuple |
| `bootstrap_coefficients()` | `dict` with keys `"coefficients"`, `"mean"`, `"std"`, `"lower"`, `"upper"`, `"names"` | Using `result.intervals` or `result.names` — it's a plain dict, use `result["names"]` |
| `bootstrap_predict()` | `dict` with keys `"y_pred"`, `"y_mean"`, `"y_std"`, `"lower"`, `"upper"` | Using `result.upper`, `result.lower` — it's a plain dict, use `result["upper"]` |
| `conformal_predict_split()` | `dict` with keys `"y_pred"`, `"lower"`, `"upper"`, `"quantile"` | Tuple unpacking `y_pred, lo, hi = conformal_predict_split(...)` — returns dict |
| `model.predict_conformal()` | `tuple` `(y_pred, lower, upper)` | Treating like dict — the *method* returns a tuple, the *standalone function* returns dict |
| `cross_validate()` | `dict` with `"mean_test_score"`, `"std_test_score"`, etc. | Treating return as array |
| `canonical_analysis()` | `CanonicalAnalysis` dataclass | — |

#### Class Attributes

| Class | Correct Attribute | Common Mistake |
|-------|-------------------|---------------|
| `BayesianModelAverage` | `.weights` (property, returns `dict[str, float]`) | `.weights_` (no trailing underscore) |
| `BayesianModelAverage` | `.expressions` (property, returns `list[str]`) | `.models_` (wrong name entirely) |
| `CanonicalAnalysis` | `.stationary_response` | `.predicted_response` (wrong name) |
| `CanonicalAnalysis` | `.stationary_point`, `.eigenvalues`, `.eigenvectors`, `.nature`, `.b_vector`, `.B_matrix`, `.warnings` | — |
| `AnovaResult` | `.rows` (list of `AnovaRow`) | — |
| `AnovaRow` | `.source`, `.df`, `.sum_sq`, `.mean_sq`, `.f_value`, `.p_value` | `.p_value` can be `None` for summary rows |

#### Constructor Signatures

| Class/Function | Correct Signature | Common Mistake |
|----------------|-------------------|---------------|
| `ActiveLearner(model, bounds, acquisition)` | `bounds` is 2nd positional arg | `ActiveLearner(model, acq, bounds)` — wrong arg order |
| `Composite(functions=[(w, acq), ...])` | `functions` is list of `(weight, AcquisitionFunction)` tuples | `Composite([...], weights=[...])` — no separate weights param |
| `ResponseSurface(n_factors=..., bounds=..., ...)` | `n_factors` is required first arg | Missing `n_factors` |
| `factorial_design(levels, n_factors, ...)` | `levels` is required first arg | `factorial_design(n_factors=3)` — missing `levels` |
| `fractional_factorial_design(n_factors, resolution=3)` | Parameter is `resolution` | `fraction=2` — wrong param name |
| `canonical_analysis(model, bounds=...)` | Parameter is `bounds` | `factor_names=...` — wrong param name |

#### Sklearn Compatibility Protocol

| API | Correct Usage | Common Mistake |
|-----|---------------|---------------|
| `model.get_params()` | Returns `dict` of constructor params | Accessing non-constructor attrs |
| `model.get_params(deep=True)` | Includes nested `estimator__param` keys for `MultiOutputSymbolicRegressor` | Expecting only flat keys |
| `model.set_params(key=val)` | Sets constructor params, returns `self` | Passing non-constructor param names |
| `mo.set_params(estimator__max_terms=8)` | Double-underscore syntax for nested params | `mo.set_params(max_terms=8)` — wrong level |
| `_clone_estimator(est)` | Uses `type(est)(**est.get_params(deep=False))` | Manual parameter listing |
| sklearn `n_jobs` | Always use `n_jobs=1` | `n_jobs=-1` — conflicts with JAX parallelism |
| `BasisLibrary` in Pipeline | Not a transformer; configure before creating estimator | Including as Pipeline step |

#### Method Names

| Class | Correct Method | Common Mistake |
|-------|---------------|---------------|
| `ResponseSurface` | `.ccd()`, `.box_behnken()`, `.factorial()` — separate methods | `.create_design()` — wrong method name |

#### Parameter Values

| Parameter | Valid Values | Common Mistake |
|-----------|-------------|---------------|
| `SymbolicRegressor(information_criterion=)` | `"aic"`, `"aicc"`, `"bic"` | `"cv"` — not supported as IC |
| `compute_information_criterion(criterion=)` | `"aic"`, `"aicc"`, `"bic"`, `"hqc"`, `"mdl"` | Note: `"hqc"` and `"mdl"` are valid HERE but NOT in `SymbolicRegressor` |
| `add_transcendental(funcs=)` | Defaults to `["log", "exp", "sqrt", "inv"]` (4 funcs) | Listing 7 default funcs (sin, cos, tan are NOT default) |
| `discrete_dims` | `dict[int, list]` — maps dimension index to allowed values | `list[int]` — wrong type |

#### ANOVA Filtering

When iterating over `anova_result.rows` to compute percentages or display tables:
- **MUST filter out** rows with `source` in `{"Model", "Residual", "Total"}` — these are summary rows.
- **MUST null-guard** `p_value` — it is `None` for `"Residual"` and `"Total"` rows.
- Correct pattern:
  ```python
  for row in result.rows:
      if row.source in {"Model", "Residual", "Total"}:
          continue  # skip summary rows
      pct = 100 * row.sum_sq / total_ss
  ```

### Anti-False-Positive Rules

Do NOT flag these as errors:
- `compute_information_criterion(..., criterion="hqc")` — valid for standalone function
- `model.predict_interval()` returning a tuple — the *method* returns `(y_pred, lower, upper)`
- `BayesianModelAverage.weights` without underscore — this IS correct (no trailing `_`)
- Imports from `jaxsr` that exist in `__all__` — even if not commonly used
- Using `.item()` on JAX arrays outside JIT — this is fine outside JIT

### Procedure

For each target file:

1. **Extract all JAXSR API calls** — imports, function calls, class instantiations, attribute access.
2. **Check each call against the truth table above.** Flag mismatches as CRITICAL.
3. **Check imports** — verify every imported symbol exists in `jaxsr.__init__.__all__`.
4. **Check return type usage** — if a function returns a dict and code unpacks as tuple, flag CRITICAL.
5. **Check ANOVA loops** — verify summary rows are filtered before computing percentages.
6. **Copy-paste safety** — verify each code block is self-contained OR explicitly references prerequisites.
7. **Cross-references** — verify any `"See guides/..."` or `"See templates/..."` references point to files that exist.

---

## Phase 3: Software Engineering Review (scope = `engineering`)

### Numerical Hazards

Check for:
- `jnp.linalg.inv()` on potentially singular matrices → suggest `lstsq`, `pinv`, or SVD
- `jnp.log(x)` without guarding `x > 0` → suggest `jnp.log(jnp.clip(x, 1e-30))`
- Division by zero without guards
- Empty array operations (`mean` of empty, indexing empty)
- `float()` or `int()` on JAX tracers inside `@jit`-decorated functions
- Python `if/else` on runtime values inside `@jit` → suggest `jax.lax.cond` or `jnp.where`

### Destructive Operations

Check for:
- `shutil.rmtree()` without path validation against deny-list (`/`, `/home`, `/usr`, `/etc`, `/var`, `/tmp`, `$HOME`)
- `os.remove()` or `os.unlink()` without existence check
- File overwrites without backup or confirmation
- `subprocess` calls with unsanitized inputs

### Packaging & Sync

Check for:
- Symbols in `__all__` that are not imported in `__init__.py`
- Imports in `__init__.py` that are not in `__all__`
- Duplicate file inclusions in `pyproject.toml` (`packages` vs `force-include`)
- `.claude/skills/jaxsr/` and `src/jaxsr/skill/` out of sync

### Dead Code

Check for:
- Unused imports (F401)
- Unused variables (F841)
- Bare expressions with no side effects (`len(x)`, `list(x)` without assignment)
- Functions not exported and not called

### Docstring Quality (for .py files)

Check that public functions have:
- Numpy-style docstring with **Parameters**, **Returns**, **Raises** sections
- Type annotations on all parameters and return type
- Docstring examples using actual function signatures (not stale names)

### Notebook-Specific Checks

For `.ipynb` files:
- Guard against negative values in physical simulations
- Use `scipy.special.erfinv` not `np.math.erfinv`
- Add null-guards for ANOVA `p_value`
- Check for cross-cell variable dependencies that break when cells run out of order
- Check for temp files that should be cleaned up

---

## Phase 4: Pedagogical Review (scope = `pedagogy`)

Read the target as a first-time user and evaluate:

### Structure & Flow

- Is there a logical progression from simple to advanced?
- Are concepts introduced before they're used?
- Does the material start with a motivating example or question?

### Terminology

- Are domain-specific terms explained on first use?
  - Examples: "profile likelihood", "Pareto front", "AICc correction", "conformal prediction",
    "canonical analysis", "stationary point"
- Is terminology consistent within the document?
  - Check: "information criterion" vs "IC" vs "selection criterion"
  - Check: "basis functions" vs "candidate terms" vs "features"
  - Check: "selection strategy" vs "search strategy" vs "algorithm"

### Explanation Quality

- Are the "why" questions answered, not just the "how"?
  - Example: Why AICc over BIC? When would you choose one over the other?
- Are decision points clearly marked with guidance?
- Are warnings and caveats placed before the code they apply to, not after?

### Code Block Self-Sufficiency

Every code block must either:
1. Be self-contained (includes all imports and data setup), or
2. Clearly state "Continuing from above..." with a reference to the prerequisite section

Common failures to flag:
- Missing `import numpy as np` or `from jaxsr import ...`
- Using variables defined in earlier code blocks without re-defining them
- Using `model` variable before the fitting code block

### Coverage Gap Analysis

Check which JAXSR features are NOT covered by any guide, template, or notebook:

**Currently covered:**
- Basis library building, Model fitting & selection, Uncertainty quantification
- Constraints, DOE workflow, Active learning, RSM, Known-model fitting, CLI
- Scikit-learn integration (cross-validation, GridSearchCV, Pipeline, model comparison)

**Known gaps (flag as INFO if relevant to the target):**
- Metrics comparison guide (R^2 vs AIC vs cross-validation)
- Export & reporting guide (JSON, LaTeX, callable, Excel, Word)
- Categorical variables (indicators, interactions, encoding/decoding)
- SISSO / power-law / rational-form basis builders
- Model serialization round-trip
- BayesianModelAverage standalone workflow
- Conformal prediction standalone usage
- Multi-response / multi-objective workflows

---

## Phase 5: Generate Report

Produce a structured markdown report with this exact format:

```markdown
# Review Report

**Target:** <file or directory path>
**Scope:** <api | engineering | pedagogy | all>
**Files reviewed:** <count>

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | N     |
| WARNING  | N     |
| INFO     | N     |

## CRITICAL

### [C1] <short title>
- **File:** <path>, <location (line N or cell N, line N)>
- **Code:** `<offending code snippet>`
- **Problem:** <what is wrong>
- **Fix:** `<corrected code>`
- **Source:** <source file:line where the correct API is defined>

### [C2] ...

## WARNING

### [W1] <short title>
- **File:** <path>, <location>
- **Code:** `<relevant code>`
- **Problem:** <what could go wrong>
- **Suggestion:** <recommended change>

### [W2] ...

## INFO

### [I1] <short title>
- **File:** <path>, <location>
- **Note:** <observation or suggestion>

---
*Review generated by `/jaxsr-review` skill*
```

### Severity Definitions

| Severity | Definition |
|----------|-----------|
| **CRITICAL** | Will cause runtime errors, wrong results, or teaches incorrect API usage. Must fix before merging. |
| **WARNING** | Won't crash but is fragile, unclear, or inconsistent. Should fix. |
| **INFO** | Stylistic, coverage gap, or minor improvement opportunity. Nice to have. |

### Classification Rules

- Wrong return type usage (dict vs tuple, wrong attribute) → **CRITICAL**
- Wrong parameter name or argument order → **CRITICAL**
- Wrong parameter value (e.g., `information_criterion="cv"`) → **CRITICAL**
- Missing ANOVA summary row filter → **CRITICAL**
- Import of non-existent symbol → **CRITICAL**
- Cross-reference to non-existent file → **WARNING**
- Missing null-guard for ANOVA `p_value` → **WARNING**
- `jnp.linalg.inv()` without singularity guard → **WARNING**
- Unexplained domain term → **WARNING**
- Missing imports in code block → **WARNING**
- Coverage gap relevant to the target → **INFO**
- Inconsistent terminology → **INFO**
- Style suggestion → **INFO**

---

## Execution Notes

- Read each target file completely before analyzing. Do not guess at contents.
- For directories, recursively find all `.py`, `.md`, and `.ipynb` files.
- For notebooks, parse all code cells and markdown cells separately.
- When checking imports, read `src/jaxsr/__init__.py` to get the current `__all__` list.
- When verifying cross-references, use Glob to check file existence.
- If no issues are found for a scope, state "No issues found" in that section.
- Always include the Summary table, even if all counts are 0.
