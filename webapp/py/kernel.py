"""
Browser-side Python API for the JAXSR web app.

This module is executed inside Pyodide by ``webapp/js/worker.js`` after
``jax_shim.install()`` and ``import jaxsr``.  Every public function takes and
returns a JSON string, so nothing but strings crosses the Python/JavaScript
boundary and there are no ``PyProxy`` lifetimes to manage on the JS side.

Every function returns ``{"ok": true, "data": ...}`` or
``{"ok": false, "error": "...", "kind": "..."}``.
"""

from __future__ import annotations

import json
import math
import traceback
import warnings
from collections.abc import Callable
from math import comb
from typing import Any

import numpy as np

import jaxsr
from jaxsr import BasisLibrary, SymbolicRegressor
from jaxsr.metrics import cross_validate as _cross_validate
from jaxsr.uncertainty import anova as _anova
from jaxsr.uncertainty import coefficient_intervals as _coefficient_intervals

# JAX produces inf/nan silently where NumPy warns; jaxsr already filters
# non-finite basis columns, so match the quieter JAX behaviour.
np.seterr(all="ignore")

# selection.exhaustive_search refuses to run above this many subsets.
MAX_EXHAUSTIVE_COMBINATIONS = 100_000

# ANOVA rows that summarise the table rather than describe a term.
_ANOVA_SUMMARY_SOURCES = {"Model", "Residual", "Total"}

_STATE: dict[str, Any] = {
    "X": None,
    "y": None,
    "feature_names": [],
    "target_name": None,
    "library": None,
    "library_config": None,
    "Phi": None,
    "model": None,
    "fit_config": None,
}


# =============================================================================
# Plumbing
# =============================================================================


def _api(fn: Callable) -> Callable:
    """Wrap a handler so it takes and returns JSON, reporting errors as data."""

    def wrapper(payload: str = "{}") -> str:
        try:
            args = json.loads(payload) if payload else {}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                data = fn(args)
                messages = sorted({str(w.message) for w in caught})
            if messages:
                data = {**data, "warnings": [*data.get("warnings", []), *messages]}
            return json.dumps({"ok": True, "data": data}, allow_nan=False, default=_jsonable)
        except Exception as exc:  # noqa: BLE001 - the boundary reports everything
            return json.dumps(
                {
                    "ok": False,
                    "kind": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _jsonable(obj: Any) -> Any:
    """Fallback encoder for NumPy scalars and arrays."""
    if isinstance(obj, np.ndarray):
        return _clean(obj).tolist()
    if isinstance(obj, np.generic):
        return _num(obj.item())
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")


def _num(value: Any) -> float | None:
    """Convert to a JSON-safe float, mapping non-finite values to ``None``."""
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _clean(arr: Any) -> np.ndarray:
    """Convert to a float array with non-finite values replaced by NaN."""
    return np.asarray(arr, dtype=float)


def _list(arr: Any) -> list:
    """Convert an array to a JSON-safe list of floats/None."""
    return [_num(v) for v in np.asarray(arr, dtype=float).ravel()]


def _require(key: str, message: str) -> Any:
    """Fetch a piece of state, raising a clear error when it is missing."""
    value = _STATE.get(key)
    if value is None:
        raise RuntimeError(message)
    return value


# =============================================================================
# Environment
# =============================================================================


@_api
def get_version(_args: dict) -> dict:
    """Report the jaxsr version and the numeric backend in use."""
    import sys

    return {
        "jaxsr": jaxsr.__version__,
        "backend": getattr(sys.modules.get("jax"), "__version__", "unknown"),
        "numpy": np.__version__,
    }


# =============================================================================
# Data
# =============================================================================


@_api
def set_data(args: dict) -> dict:
    """
    Install the uploaded table as training data.

    Expects ``columns`` (names), ``rows`` (list of row lists) and ``roles``
    (one of ``feature``/``target``/``ignore`` per column).
    """
    columns: list[str] = args["columns"]
    rows: list[list] = args["rows"]
    roles: list[str] = args["roles"]

    if len(roles) != len(columns):
        raise ValueError(f"Got {len(roles)} roles for {len(columns)} columns.")

    feature_idx = [i for i, r in enumerate(roles) if r == "feature"]
    target_idx = [i for i, r in enumerate(roles) if r == "target"]

    if len(target_idx) != 1:
        raise ValueError(f"Select exactly one target column (got {len(target_idx)}).")
    if not feature_idx:
        raise ValueError("Select at least one feature column.")
    if not rows:
        raise ValueError("The selected sheet has no data rows.")

    keep = [*feature_idx, target_idx[0]]
    table = np.array(
        [[_to_float(row[i] if i < len(row) else None) for i in keep] for row in rows],
        dtype=float,
    )

    finite_rows = np.all(np.isfinite(table), axis=1)
    dropped = int((~finite_rows).sum())
    table = table[finite_rows]

    if table.shape[0] == 0:
        raise ValueError(
            "No rows had numeric values in every selected column. "
            "Check that the feature and target columns contain numbers."
        )

    X = table[:, :-1]
    y = table[:, -1]

    notes = []
    if dropped:
        notes.append(f"Dropped {dropped} row(s) with missing or non-numeric values.")
    n, p = X.shape
    if n <= p + 1:
        notes.append(
            f"Only {n} usable rows for {p} feature(s) -- too few to fit anything "
            "beyond a very small model."
        )
    for j, name in enumerate([columns[i] for i in feature_idx]):
        if np.ptp(X[:, j]) == 0:
            notes.append(f"Feature '{name}' is constant and cannot explain anything.")

    _STATE.update(
        X=X,
        y=y,
        feature_names=[columns[i] for i in feature_idx],
        target_name=columns[target_idx[0]],
        library=None,
        library_config=None,
        Phi=None,
        model=None,
        fit_config=None,
    )

    return {
        "n_samples": int(n),
        "n_features": int(p),
        "feature_names": _STATE["feature_names"],
        "target_name": _STATE["target_name"],
        "dropped_rows": dropped,
        "target_summary": {
            "min": _num(y.min()),
            "max": _num(y.max()),
            "mean": _num(y.mean()),
            "std": _num(y.std()),
        },
        "warnings": notes,
    }


def _to_float(value: Any) -> float:
    """Coerce a spreadsheet cell to a float, using NaN for anything unusable."""
    if value is None or value == "":
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# =============================================================================
# Basis library
# =============================================================================


@_api
def build_library(args: dict) -> dict:
    """
    Build the candidate basis library from the checkbox configuration.

    Cheap enough to call on every UI change, which is what drives the live
    candidate-term count.
    """
    X = _require("X", "Load a data file first.")
    config = args.get("config", {})
    library = _library_from_config(config, _STATE["feature_names"])

    _STATE["library"] = library
    _STATE["library_config"] = config
    _STATE["Phi"] = None

    n_basis = len(library)
    max_terms = int(args.get("max_terms", 5))
    return {
        "n_terms": n_basis,
        "names": list(library.names),
        "n_samples": int(X.shape[0]),
        "exhaustive": _exhaustive_feasibility(n_basis, max_terms),
    }


def _library_from_config(config: dict, feature_names: list[str]) -> BasisLibrary:
    """Translate the UI's basis configuration into a ``BasisLibrary``."""
    library = BasisLibrary(n_features=len(feature_names), feature_names=feature_names)

    if config.get("constant", True):
        library.add_constant()
    if config.get("linear", True):
        library.add_linear()

    poly = config.get("polynomials") or {}
    if poly.get("enabled"):
        library.add_polynomials(max_degree=int(poly.get("max_degree", 2)))

    inter = config.get("interactions") or {}
    if inter.get("enabled"):
        library.add_interactions(max_order=int(inter.get("max_order", 2)))

    trans = config.get("transcendental") or {}
    if trans.get("enabled"):
        funcs = trans.get("funcs") or ["log", "exp", "sqrt", "inv"]
        library.add_transcendental(funcs=list(funcs))

    if (config.get("ratios") or {}).get("enabled"):
        library.add_ratios()

    comp = config.get("compositions") or {}
    if comp.get("enabled"):
        library.add_compositions(
            outer_funcs=list(comp.get("outer") or ["log", "exp", "sqrt"]),
            inner_forms=list(comp.get("inner") or ["product", "ratio"]),
        )

    powers = config.get("power_laws") or {}
    if powers.get("enabled"):
        exponents = powers.get("exponents")
        library.add_power_laws(exponents=[float(e) for e in exponents] if exponents else None)

    if len(library) == 0:
        raise ValueError("The basis library is empty -- enable at least one function family.")
    return library


def _exhaustive_feasibility(n_basis: int, max_terms: int) -> dict:
    """Report whether exhaustive search is within its hard combination cap."""
    k_max = min(max_terms, n_basis)
    total = sum(comb(n_basis, k) for k in range(1, k_max + 1))
    return {
        "combinations": total,
        "limit": MAX_EXHAUSTIVE_COMBINATIONS,
        "feasible": total <= MAX_EXHAUSTIVE_COMBINATIONS,
    }


# =============================================================================
# Fitting
# =============================================================================


@_api
def fit(args: dict) -> dict:
    """Fit a :class:`SymbolicRegressor` and return every candidate it scored."""
    X = _require("X", "Load a data file first.")
    y = _STATE["y"]
    library = _require("library", "Configure the basis library first.")

    max_terms = int(args.get("max_terms", 5))
    strategy = args.get("strategy", "greedy_forward")
    criterion = args.get("information_criterion", "aicc")
    regularization = args.get("regularization")

    if strategy == "exhaustive":
        feasibility = _exhaustive_feasibility(len(library), max_terms)
        if not feasibility["feasible"]:
            raise ValueError(
                f"Exhaustive search would evaluate {feasibility['combinations']:,} subsets, "
                f"above the {MAX_EXHAUSTIVE_COMBINATIONS:,} limit. Reduce the number of "
                "candidate terms or max terms, or use greedy_forward."
            )

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=max_terms,
        strategy=strategy,
        information_criterion=criterion,
        regularization=float(regularization) if regularization else None,
    )
    model.fit(X, y)

    _STATE["model"] = model
    _STATE["Phi"] = np.asarray(library.evaluate(X), dtype=float)
    _STATE["fit_config"] = {
        "max_terms": max_terms,
        "strategy": strategy,
        "information_criterion": criterion,
        "regularization": regularization,
    }

    return {
        "candidates": _rank_candidates(model, criterion),
        "best": _model_summary(model),
        "strategy": strategy,
        "information_criterion": criterion,
        "path_semantics": (
            "every subset evaluated"
            if strategy == "exhaustive"
            else "the best model found at each step of the search"
        ),
    }


def _rank_candidates(model: SymbolicRegressor, criterion: str, limit: int = 50) -> dict:
    """
    Rank every evaluated candidate by the chosen information criterion.

    The selection path records the search *before* ``fit`` drops non-finite
    terms and prunes negligible ones, so the winning candidate can carry terms
    the returned model does not.  The row at ``best_index`` is therefore
    replaced with the model jaxsr actually returns, which is also what the
    exported reproduction script produces; every other row is the candidate as
    the search scored it.
    """
    path = model.selection_path_
    y = _STATE["y"]
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    n = len(y)

    pareto_keys = {tuple(r.selected_names) for r in model.pareto_front_}
    fitted_names = list(model.selected_features_)
    fitted_metrics = model.metrics_

    rows = []
    for position, result in enumerate(path.results):
        is_best = position == path.best_index
        if is_best:
            names = fitted_names
            coefficients = np.asarray(model.coefficients_, dtype=float)
            indices = [int(i) for i in np.asarray(model.selected_indices_).ravel()]
            expression = model.expression_
            mse = float(fitted_metrics["mse"])
            r2 = _num(fitted_metrics["r2"])
            aic, aicc, bic = fitted_metrics["aic"], fitted_metrics["aicc"], fitted_metrics["bic"]
            complexity = int(model.complexity_)
        else:
            names = list(result.selected_names)
            coefficients = np.asarray(result.coefficients, dtype=float)
            indices = [int(i) for i in np.asarray(result.selected_indices).ravel()]
            expression = result.expression()
            mse = float(result.mse)
            # An intercept-only model has SSres == SStot exactly, so R² is 0;
            # what survives the subtraction is float noise like -9e-14.
            raw_r2 = 1.0 - (mse * n) / ss_tot if ss_tot > 0 else None
            r2 = _num(0.0 if raw_r2 is not None and abs(raw_r2) < 1e-10 else raw_r2)
            aic, aicc, bic = result.aic, result.aicc, result.bic
            complexity = int(result.complexity)

        rows.append(
            {
                "terms": names,
                "coefficients": _list(coefficients),
                "indices": indices,
                "expression": expression,
                "n_terms": len(names),
                "complexity": complexity,
                "mse": _num(mse),
                "rmse": _num(math.sqrt(max(mse, 0.0))),
                "r2": r2,
                "aic": _num(aic),
                "aicc": _num(aicc),
                "bic": _num(bic),
                "is_pareto": tuple(names) in pareto_keys,
                "is_best": is_best,
                "pruned": is_best and len(names) < int(result.n_terms),
            }
        )

    def sort_key(row: dict) -> tuple:
        score = row.get(criterion)
        return (score is None, score if score is not None else 0.0, row["n_terms"])

    rows.sort(key=sort_key)
    truncated = max(0, len(rows) - limit)
    for rank, row in enumerate(rows[:limit], start=1):
        row["rank"] = rank
    return {"rows": rows[:limit], "total": len(rows), "truncated": truncated}


def _model_summary(model: SymbolicRegressor) -> dict:
    """Summarise the selected model."""
    metrics = model.metrics_
    return {
        "expression": model.expression_,
        "terms": list(model.selected_features_),
        "coefficients": _list(model.coefficients_),
        "complexity": int(model.complexity_),
        "metrics": {k: _num(v) for k, v in metrics.items()},
        "rmse": _num(math.sqrt(max(float(metrics["mse"]), 0.0))),
        "summary": model.summary(),
    }


# =============================================================================
# Diagnostics
# =============================================================================


@_api
def diagnostics(args: dict) -> dict:
    """
    Diagnostics for one candidate model.

    ``indices`` selects a candidate from the ranked table; omitting it uses the
    model the information criterion chose.
    """
    model = _require("model", "Fit a model first.")
    Phi = _STATE["Phi"]
    y = _STATE["y"]
    alpha = float(args.get("alpha", 0.05))

    fitted_indices = _best_indices(model)
    requested = args.get("indices")
    indices = fitted_indices if requested is None else [int(i) for i in requested]

    if indices == fitted_indices:
        names = list(model.selected_features_)
        coeffs = np.asarray(model.coefficients_, dtype=float)
    else:
        candidate = _candidate_by_indices(model, indices)
        names = list(candidate.selected_names)
        coeffs = np.asarray(candidate.coefficients, dtype=float)

    Phi_sub = Phi[:, indices]
    y_pred = Phi_sub @ coeffs
    residuals = y - y_pred

    return {
        "terms": names,
        "parity": {"observed": _list(y), "predicted": _list(y_pred)},
        "residuals": {"predicted": _list(y_pred), "residual": _list(residuals)},
        "qq": _qq_points(residuals),
        "intervals": _intervals(Phi_sub, y, coeffs, names, alpha),
        # ANOVA is defined against the fitted regressor, so it is only
        # meaningful for the model jaxsr actually returned.
        "anova": _anova_table(model) if indices == fitted_indices else None,
        "residual_summary": {
            "mean": _num(residuals.mean()),
            "std": _num(residuals.std(ddof=1)) if len(residuals) > 1 else None,
            "max_abs": _num(np.abs(residuals).max()),
        },
        "alpha": alpha,
    }


def _best_indices(model: SymbolicRegressor) -> list[int]:
    """
    Basis indices of the model ``fit`` actually returned.

    Taken from the regressor rather than from ``selection_path_.best``: the path
    predates pruning, so the winning candidate can carry terms the final model
    does not.
    """
    return [int(i) for i in np.asarray(model.selected_indices_).ravel()]


def _candidate_by_indices(model: SymbolicRegressor, indices: list[int]):
    """Find a candidate in the selection path by its basis indices."""
    target = list(indices)
    for result in model.selection_path_.results:
        if [int(i) for i in np.asarray(result.selected_indices).ravel()] == target:
            return result
    raise ValueError("That candidate is not in the selection path.")


def _intervals(
    Phi_sub: np.ndarray,
    y: np.ndarray,
    coeffs: np.ndarray,
    names: list[str],
    alpha: float,
) -> list[dict]:
    """Coefficient confidence intervals, or estimates alone if they are undefined."""
    try:
        raw = _coefficient_intervals(Phi_sub, y, coeffs, names, alpha)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return [
            {"name": n, "estimate": _num(c), "lower": None, "upper": None, "se": None}
            for n, c in zip(names, coeffs, strict=False)
        ]

    rows = []
    for name in names:
        # coefficient_intervals returns {name: (estimate, lower, upper, se)}
        estimate, lower, upper, se = raw[name]
        rows.append(
            {
                "name": name,
                "estimate": _num(estimate),
                "lower": _num(lower),
                "upper": _num(upper),
                "se": _num(se),
                # A term whose interval excludes zero is distinguishable from noise.
                "significant": bool(
                    np.isfinite(lower) and np.isfinite(upper) and lower * upper > 0
                ),
            }
        )
    return rows


def _anova_table(model: SymbolicRegressor) -> dict | None:
    """Sequential ANOVA with per-term percentage contributions."""
    try:
        result = _anova(model)
    except (ValueError, RuntimeError, ZeroDivisionError):
        return None

    summary_rows = [r for r in result.rows if r.source in _ANOVA_SUMMARY_SOURCES]
    term_rows = [r for r in result.rows if r.source not in _ANOVA_SUMMARY_SOURCES]

    total = next((r for r in summary_rows if r.source == "Total"), None)
    residual = next((r for r in summary_rows if r.source == "Residual"), None)
    total_ss = float(total.sum_sq) if total and total.sum_sq > 0 else 0.0

    def row_to_dict(row: Any, with_pct: bool) -> dict:
        pct = None
        if with_pct and total_ss > 0:
            pct = _num(100.0 * float(row.sum_sq) / total_ss)
        return {
            "source": row.source,
            "df": int(row.df),
            "sum_sq": _num(row.sum_sq),
            "mean_sq": _num(row.mean_sq),
            "f_value": _num(row.f_value) if row.f_value is not None else None,
            "p_value": _num(row.p_value) if row.p_value is not None else None,
            "pct_contribution": pct,
        }

    note = None
    if residual is not None and float(residual.mean_sq) < 1e-10:
        note = (
            "Residual variance is at machine precision, so F-tests and p-values are "
            "unreliable here. Judge terms by their percentage contribution instead."
        )

    return {
        "terms": [row_to_dict(r, True) for r in term_rows],
        "summary": [row_to_dict(r, False) for r in summary_rows],
        "note": note,
    }


def _qq_points(residuals: np.ndarray) -> dict:
    """Normal quantile-quantile points for the residuals."""
    from scipy.stats import norm

    n = len(residuals)
    if n < 2:
        return {"theoretical": [], "sample": [], "line": None}

    ordered = np.sort(np.asarray(residuals, dtype=float))
    std = ordered.std(ddof=1)
    standardized = ordered / std if std > 0 else ordered
    # Blom plotting positions.
    probs = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theoretical = norm.ppf(probs)
    return {
        "theoretical": _list(theoretical),
        "sample": _list(standardized),
        "line": {"slope": 1.0, "intercept": 0.0},
    }


@_api
def run_cross_validation(args: dict) -> dict:
    """K-fold cross-validation of the fitted model."""
    model = _require("model", "Fit a model first.")
    folds = int(args.get("folds", 5))
    n = len(_STATE["y"])
    if folds < 2 or folds > n:
        raise ValueError(f"Folds must be between 2 and {n}.")

    result = _cross_validate(model, _STATE["X"], _STATE["y"], cv=folds)
    return {
        "folds": folds,
        "mean_test_score": _num(result["mean_test_score"]),
        "std_test_score": _num(result["std_test_score"]),
        "mean_train_score": _num(result.get("mean_train_score")),
    }


@_api
def prediction_band(args: dict) -> dict:
    """Prediction intervals for the training points, ordered by fitted value."""
    model = _require("model", "Fit a model first.")
    alpha = float(args.get("alpha", 0.05))
    y_pred, lower, upper = model.predict_interval(_STATE["X"], alpha=alpha)
    order = np.argsort(np.asarray(y_pred, dtype=float))
    return {
        "predicted": _list(np.asarray(y_pred)[order]),
        "lower": _list(np.asarray(lower)[order]),
        "upper": _list(np.asarray(upper)[order]),
        "observed": _list(np.asarray(_STATE["y"])[order]),
        "alpha": alpha,
    }


# =============================================================================
# Export
# =============================================================================


@_api
def export_model(args: dict) -> dict:
    """Render the fitted model as LaTeX, JSON, a Python script, or clean CSV."""
    model = _require("model", "Fit a model first.")
    kind = args.get("kind", "latex")

    if kind == "latex":
        return {"kind": kind, "filename": "model.tex", "content": model.to_latex()}

    if kind == "json":
        return {
            "kind": kind,
            "filename": "model.json",
            "content": json.dumps(model._state_dict(), indent=2),
        }

    if kind == "python":
        return {
            "kind": kind,
            "filename": "reproduce_fit.py",
            "content": _reproduction_script(),
        }

    if kind == "csv":
        return {"kind": kind, "filename": "data.csv", "content": _data_csv()}

    raise ValueError(f"Unknown export kind: {kind!r}")


def _data_csv() -> str:
    """The cleaned training data as CSV, matching what the fit actually used."""
    header = [*_STATE["feature_names"], _STATE["target_name"]]
    table = np.column_stack([_STATE["X"], _STATE["y"]])
    lines = [",".join(header)]
    lines.extend(",".join(repr(float(v)) for v in row) for row in table)
    return "\n".join(lines) + "\n"


def _reproduction_script() -> str:
    """
    Generate a standalone script that reproduces this fit with ``pip install jaxsr``.

    The web app is a front door to the library, so the exported script uses the
    ordinary public API rather than anything app-specific.
    """
    config = _STATE["library_config"] or {}
    fit_config = _STATE["fit_config"] or {}
    names = _STATE["feature_names"]

    chain = [f"BasisLibrary(n_features={len(names)}, feature_names={names!r})"]
    if config.get("constant", True):
        chain.append("    .add_constant()")
    if config.get("linear", True):
        chain.append("    .add_linear()")
    poly = config.get("polynomials") or {}
    if poly.get("enabled"):
        chain.append(f"    .add_polynomials(max_degree={int(poly.get('max_degree', 2))})")
    inter = config.get("interactions") or {}
    if inter.get("enabled"):
        chain.append(f"    .add_interactions(max_order={int(inter.get('max_order', 2))})")
    trans = config.get("transcendental") or {}
    if trans.get("enabled"):
        chain.append(f"    .add_transcendental(funcs={list(trans.get('funcs') or [])!r})")
    if (config.get("ratios") or {}).get("enabled"):
        chain.append("    .add_ratios()")
    comp = config.get("compositions") or {}
    if comp.get("enabled"):
        chain.append(
            "    .add_compositions(outer_funcs={!r}, inner_forms={!r})".format(
                list(comp.get("outer") or []), list(comp.get("inner") or [])
            )
        )
    powers = config.get("power_laws") or {}
    if powers.get("enabled"):
        exps = powers.get("exponents")
        chain.append(
            f"    .add_power_laws(exponents={[float(e) for e in exps] if exps else None!r})"
        )

    reg = fit_config.get("regularization")
    return f'''"""
Reproduce a fit made with the JAXSR web app.

    pip install jaxsr
    python reproduce_fit.py

Expects `data.csv` (downloadable from the app) next to this file. That file
holds the cleaned data the app actually fitted, after dropping rows with
missing or non-numeric values.
"""

import numpy as np

from jaxsr import BasisLibrary, SymbolicRegressor

FEATURES = {names!r}
TARGET = {_STATE["target_name"]!r}

data = np.genfromtxt("data.csv", delimiter=",", names=True)
X = np.column_stack([data[name] for name in data.dtype.names[:-1]])
y = data[data.dtype.names[-1]]

library = (
    {chr(10).join(chain)}
)
print(f"{{len(library)}} candidate basis functions")

model = SymbolicRegressor(
    basis_library=library,
    max_terms={int(fit_config.get("max_terms", 5))},
    strategy={fit_config.get("strategy", "greedy_forward")!r},
    information_criterion={fit_config.get("information_criterion", "aicc")!r},
    regularization={float(reg) if reg else None!r},
)
model.fit(X, y)

print(model.summary())

# Every candidate the search scored, best first.
path = model.selection_path_
for i, result in enumerate(path.results):
    mark = "*" if i == path.best_index else " "
    print(f"{{mark}} {{result.n_terms}} terms  BIC={{result.bic:10.2f}}  {{result.expression()}}")
'''
