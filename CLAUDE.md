# CLAUDE.md — Development Guidelines for JAXSR

## Project Overview

JAXSR is a JAX-based symbolic regression library. It discovers interpretable algebraic
expressions from data using sparse optimization. The codebase lives in `src/jaxsr/` with
tests in `tests/`.

## Build & Test Commands

```bash
pip install -e ".[dev]"          # Install with dev dependencies
pytest tests/ -v --tb=short --timeout=60   # Run tests
black --check src/ tests/        # Check formatting
ruff check src/ tests/           # Check linting
black src/ tests/                # Auto-format
ruff check --fix src/ tests/     # Auto-fix lint issues

# Coverage
pytest tests/ --cov=jaxsr --cov-report=term-missing   # Coverage to terminal
pytest tests/ --cov=jaxsr --cov-report=html            # Coverage HTML report → htmlcov/
```

## CI Requirements

All PRs must pass before merge:
- **pytest** across Python 3.10, 3.11, 3.12
- **black --check** (line-length 100, target py310)
- **ruff check** (rules: E, F, W, I, UP, B, C4; E501 ignored)

Always run `black` and `ruff check` locally before committing.

## Code Quality Rules

### Every new public function MUST have:
- A numpy-style docstring with **Parameters**, **Returns**, and **Raises** sections
- Type annotations on all parameters and the return type
- Input validation on public API boundaries (check shapes, types, value ranges)
- At least one test in the corresponding `tests/test_<module>.py` file

### Modules that still need dedicated test files:
- `metrics.py` → needs `tests/test_metrics.py`
- `simplify.py` → needs `tests/test_simplify.py`
- `sampling.py` → needs `tests/test_sampling.py`
- `plotting.py` → needs `tests/test_plotting.py`
- `utils.py` → needs `tests/test_utils.py`

When modifying any of these modules, add tests for the code you touch.

### Documentation must match code
- Docstring examples and `docs/*.md` code blocks must use **actual function signatures**
  (parameter names, argument order, return types). This has been a recurring source of bugs.
- When renaming a parameter or changing a signature, grep docs/ and README.md for the old name.
- `information_criterion` only supports `"aic"`, `"aicc"`, `"bic"` — not `"cv"`.

### Numerical robustness
- Never call `jnp.linalg.inv()` on a matrix that could be singular. Use `jnp.linalg.lstsq`,
  `jnp.linalg.pinv`, or SVD-based approaches instead.
- Guard against log(0), division by zero, and empty arrays in metrics/statistics code.
- When adding `@jit`-decorated functions, do not use Python control flow (`if`/`else`) on
  values that depend on runtime data. Use `jax.lax.cond` or `jnp.where` instead.
  Do not call `float()` or `int()` on JAX tracers inside JIT.

### Avoid dead code
- Do not leave bare expressions with no side effects (e.g. `len(x)`, `list(x)` without assignment).
- Do not commit unused imports, variables, or functions. Ruff catches most of these (F401, F841).
- If a function is not exported in `__init__.py` and not called anywhere, remove it or export it.

### Keep `__init__.py` exports in sync
- Every public symbol imported in `__init__.py` must appear in `__all__`.
- When adding a new public class or function, add it to both the import block and `__all__`.
- Keep import blocks sorted (ruff rule I001 enforces this).

### Basis function name parsing
- `regressor.py:to_callable()` and `simplify.py:_parse_term_to_sympy()` parse basis function
  names as strings. When adding new basis function types to `BasisLibrary`, verify that both
  parsers handle the new name format. Test with `model.to_callable()` and `model.to_sympy()`.
- Interaction terms with powers (e.g. `"x^2*y"`) must be handled — split on `*` then
  check each part for `^`.
- Use `float()` not `int()` for exponents to support fractional powers.

### Constraints and information criteria
- After any refit (constrained or otherwise), always recalculate AIC/BIC/AICc from the
  new MSE and coefficient count. Never copy stale values from a previous SelectionResult.
- Use `compute_information_criterion(n, k, mse, criterion)` from `metrics.py`.

### Composition basis functions
- `add_compositions` iterates over feature pairs. The duplicate-pair skip for symmetric
  operations (like `product` where `x*y == y*x`) must only skip that specific inner form,
  not the entire `(i, j)` iteration. Non-symmetric forms like `ratio` need both orderings.

## Review Checklists

When creating or modifying documentation, examples, guides, templates, or notebooks,
run these reviews before merging. Each review targets a different failure mode.

### 1. Red-Team Review (API Correctness)

Verify every code snippet against the actual source code. This is the highest-priority
review — incorrect examples teach users wrong patterns.

**Check every occurrence of:**

| API | Common mistake | Correct usage |
|-----|---------------|---------------|
| `coefficient_intervals()` | Unpacking as `(lo, hi)` | Returns `dict[str, (est, lo, hi, se)]` — 4-tuple |
| `bootstrap_coefficients()` | `result.intervals`, `result.names` | Returns plain `dict` — use `result["names"]`, `result["lower"]`, etc. |
| `bootstrap_predict()` | `result.upper`, `result.lower` | Returns plain `dict` — use `result["upper"]`, `result["lower"]`, etc. |
| `BayesianModelAverage` | `.weights_`, `.models_` | `.weights`, `.expressions` (no trailing underscore) |
| `CanonicalAnalysis` | `.predicted_response` | `.stationary_response` |
| `cross_validate()` | Treating return as array | Returns `dict` with `"mean_test_score"`, `"std_test_score"` |
| `conformal_predict_split()` | Tuple unpacking | Standalone returns `dict`; `model.predict_conformal()` returns tuple |
| `add_transcendental()` | Listing 7 default funcs | Defaults: `["log", "exp", "sqrt", "inv"]` (4 funcs) |
| `ActiveLearner()` | `ActiveLearner(model, acq, bounds)` | `ActiveLearner(model, bounds, acquisition)` — bounds is 2nd arg |
| `Composite()` | `Composite([...], weights=[...])` | `Composite(functions=[(weight, acq), ...])` — no separate weights param |
| `ResponseSurface()` | Missing `n_factors` | `ResponseSurface(n_factors=3, ...)` — required first arg |
| `ResponseSurface` methods | `.create_design()` | `.ccd()`, `.box_behnken()` — separate methods |
| `factorial_design()` | `factorial_design(n_factors=3)` | `factorial_design(levels=2, n_factors=3)` — levels required |
| `fractional_factorial_design()` | `fraction=2` | `resolution=3` — parameter is resolution |
| `canonical_analysis()` | `factor_names=...` | `bounds=...` — not factor_names |
| `discrete_dims` | `list[int]` | `dict[int, list]` — maps dimension index to allowed values |
| ANOVA `result.rows` | Using all rows for % calc | Filter `{"Model", "Residual", "Total"}` — they are summary rows |
| `information_criterion` | `"cv"` | Only `"aic"`, `"aicc"`, `"bic"` are supported |
| `model.get_params()` | Returns `dict` of all constructor params | Accessing non-constructor attributes |
| `model.set_params(**kw)` | Sets constructor params, returns `self` | Passing non-constructor param names (raises `ValueError`) |
| `mo.set_params(estimator__max_terms=8)` | Double-underscore for nested params | `mo.set_params(max_terms=8)` — wrong level |
| sklearn `n_jobs` | Always `n_jobs=1` | `n_jobs=-1` conflicts with JAX parallelism |

**Process:**
1. For each file, extract every JAXSR API call
2. Check parameter names, argument order, and return types against source code
3. Check that every import exists in `__init__.py` `__all__`
4. Verify ANOVA loops filter summary rows before computing percentages

### 2. Software Engineering Review

Review architecture, safety, robustness, and packaging.

**Safety & destructive operations:**
- Any use of `shutil.rmtree()` must validate the target path against a deny-list
  of system directories (`/`, `/home`, `/usr`, `/etc`, `/var`, `/tmp`, `$HOME`)
- File operations that delete or overwrite must check `is_dir()` / `is_file()` first
- CLI commands that modify the filesystem must handle `PermissionError` and `OSError`

**Packaging:**
- `pyproject.toml`: Ensure no duplicate file inclusions between `packages` and `force-include`
- `src/jaxsr/skill/` and `.claude/skills/jaxsr/` must stay in sync — after editing any
  skill file, run: `rm -rf src/jaxsr/skill && cp -r .claude/skills/jaxsr src/jaxsr/skill`
- Verify `__init__.py` exports: every symbol in `__all__` must be imported; `__all__` should
  be sorted alphabetically within each section

**Notebooks & examples:**
- Guard against negative values in physical simulations (adsorption, concentrations, etc.)
- Use `scipy.special.erfinv` not `np.math.erfinv` (deprecated)
- Add null-guards for ANOVA `p_value` (can be `None` for summary rows)
- Avoid cross-cell variable dependencies that break when cells run out of order
- Clean up temp files at the end of notebooks (`os.remove`)

### 3. Pedagogical Review

Read guides and notebooks as a first-time user.

**Check for:**
- Is there a logical progression from simple → advanced?
- Are domain-specific terms explained on first use? (e.g., "profile likelihood",
  "Pareto front", "AICc correction")
- Can a user copy-paste any code block and have it work? Or does it silently depend
  on earlier cells / imports not shown?
- Are the "why" questions answered, not just the "how"? (e.g., why AICc over BIC?)
- Do decision tables cover the user's likely scenario?

### 4. Coverage Gap Review

Map which JAXSR features have guide/template/notebook coverage.

**Currently covered:**
- Basis library building (`guides/basis-library.md`)
- Model fitting & selection (`guides/model-fitting.md`)
- Uncertainty quantification (`guides/uncertainty.md`)
- Constraints (`guides/constraints.md`)
- DOE workflow (`guides/doe-workflow.md`)
- Active learning (`guides/active-learning.md`)
- RSM (`guides/rsm.md`)
- Known-model fitting (`guides/known-model-fitting.md`)
- Scikit-learn integration (`guides/sklearn-integration.md`)
- CLI reference (`guides/cli.md`)

**Gaps to fill:**
- Metrics comparison guide (when to use R² vs AIC vs cross-validation)
- Export & reporting guide (JSON, LaTeX, callable, Excel, Word)
- Categorical variables (indicators, interactions, encoding/decoding)
- SISSO / power-law / rational-form basis builders
- Model serialization round-trip (save/load/share)
- BayesianModelAverage standalone workflow
- Conformal prediction standalone usage
- Multi-response / multi-objective workflows

### 5. Cross-Reference & Terminology Review

- Verify guide cross-references point to files that exist
  (e.g., `"See guides/uncertainty.md"` → file must exist)
- Check for inconsistent terminology across docs:
  - "information criterion" vs "IC" vs "selection criterion"
  - "basis functions" vs "candidate terms" vs "features"
  - "selection strategy" vs "search strategy" vs "algorithm"
- Ensure SKILL.md decision trees route to content that exists

### 6. Copy-Paste Safety Review

Every code block in guides must either:
1. Be self-contained (includes all imports and data setup), or
2. Clearly state "Continuing from above..." with a reference to the prerequisite section

**Common failures:**
- Missing `import numpy as np` or `from jaxsr import ...`
- Using variables defined in earlier code blocks without re-defining them
- Using `model` variable before the fitting code block

## Linting & Formatting (MUST pass before every commit)

**Always run these two commands before committing:**

```bash
black src/ tests/ docs/examples/
ruff check --fix src/ tests/ docs/examples/
```

Then verify they pass in check mode (what CI runs):

```bash
black --check src/ tests/
ruff check src/ tests/
```

### Common ruff errors and how to avoid them

| Rule | What it catches | Prevention |
|------|----------------|------------|
| **I001** | Unsorted imports | Let `ruff check --fix` auto-sort, or use isort-compatible ordering |
| **F401** | Unused imports | Only import what you use; re-exports in `__init__.py` must be in `__all__` |
| **F841** | Unused local variables | Delete variables you don't read; use `_` for intentionally unused values |
| **B905** | `zip()` without `strict=` | Add `strict=False` (or `strict=True` if lengths must match) |
| **B007** | Unused loop variable | Use `_` for loop variables you don't reference |
| **E402** | Module-level import not at top | Move imports above non-import code |
| **UP** rules | Outdated Python syntax | Use `X | None` instead of `Optional[X]`, `list[str]` instead of `List[str]` |
| **C4** rules | Unnecessary list/dict comprehension | Use direct constructors when possible |
| **F541** | f-string without placeholders | Remove the `f` prefix if there are no `{...}` expressions |

### Key principles

- **Fix lint before committing** — CI will reject PRs with lint failures. Never rely on CI to catch
  what you can catch locally.
- **Use auto-fixers first** — `black` and `ruff check --fix` handle most issues automatically.
  Only manually fix what auto-fix cannot.
- **Don't suppress warnings without justification** — Avoid `# noqa` comments unless the rule is
  genuinely a false positive. If you must suppress, add a comment explaining why.
- **Keep imports sorted** — ruff I001 enforces isort-compatible import ordering. Three groups:
  stdlib, third-party, local. Alphabetical within each group.
- **Format docstrings and strings consistently** — Use `"double quotes"` (black default).

## Style

- **Formatter:** black (line-length 100)
- **Linter:** ruff (see pyproject.toml for rule selection: E, F, W, I, UP, B, C4; E501 ignored)
- **Docstrings:** numpy-style
- **Naming:** scikit-learn convention for fitted attributes (trailing underscore: `coefficients_`,
  `expression_`); leading underscore for private attributes (`_X_train`, `_is_fitted`)
- Avoid broad `except Exception` — catch specific exceptions (`ValueError`, `LinAlgError`, etc.)
- Prefer `strict=False` on `zip()` calls to satisfy ruff B905, unless lengths are guaranteed equal

## Test Coverage

Coverage is configured in `pyproject.toml` under `[tool.coverage.*]` sections.
The `fail_under` threshold is **60%**. Coverage reports exclude `pragma: no cover`,
`TYPE_CHECKING` blocks, and `NotImplementedError` stubs.

### Current coverage baseline (294 tests passing)

| Module | Coverage | Notes |
|--------|----------|-------|
| `acquisition.py` | 95% | Well tested |
| `rsm.py` | 94% | Well tested |
| `__init__.py` | 88% | Mostly re-exports |
| `uncertainty.py` | 87% | Well tested |
| `regressor.py` | 80% | Core module, good coverage |
| `sampling.py` | 79% | Needs dedicated test file |
| `selection.py` | 72% | Some advanced strategies untested |
| `constraints.py` | 71% | Some constraint types untested |
| `basis.py` | 58% | Many builder methods untested (SISSO, power laws, rational forms) |
| `utils.py` | 40% | Needs dedicated test file |
| `metrics.py` | 28% | Needs dedicated test file — most metric functions untested |
| `simplify.py` | 11% | Needs dedicated test file — nearly all code untested |
| `plotting.py` | 0% | Needs dedicated test file — entirely untested |
| **TOTAL** | **64%** | |

### Priority modules for new tests

1. **plotting.py** (0%) — At minimum, test that functions don't error with valid inputs
2. **simplify.py** (11%) — Test sympy conversion, expression simplification
3. **metrics.py** (28%) — Test all metric functions with known inputs/outputs
4. **utils.py** (40%) — Test utility functions
5. **basis.py** (58%) — Test SISSO, power laws, rational forms builders
