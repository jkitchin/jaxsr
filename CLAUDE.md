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

## Linting & Formatting (MUST pass before every commit)

**Always run these two commands before committing:**

```bash
black src/ tests/ examples/
ruff check --fix src/ tests/ examples/
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
