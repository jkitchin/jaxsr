"""
Model Selection Algorithms for JAXSR.

Implements multiple strategies for selecting sparse subsets of basis functions:
- Greedy forward selection
- Greedy backward elimination
- Exhaustive search
- LASSO path screening
- Coordinate descent LASSO
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap

from .metrics import compute_information_criterion


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SelectionResult:
    """
    Result of model selection for a single model configuration.

    Parameters
    ----------
    coefficients : jnp.ndarray
        Fitted coefficients for selected features.
    selected_indices : jnp.ndarray
        Indices of selected basis functions.
    selected_names : list of str
        Names of selected basis functions.
    mse : float
        Mean squared error.
    complexity : int
        Total complexity score.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    aicc : float
        Corrected AIC.
    n_samples : int
        Number of training samples.
    """
    coefficients: jnp.ndarray
    selected_indices: jnp.ndarray
    selected_names: List[str]
    mse: float
    complexity: int
    aic: float
    bic: float
    aicc: float
    n_samples: int
    parametric_params: Optional[Dict[int, Dict[str, float]]] = None

    def expression(self) -> str:
        """Return human-readable expression."""
        from .utils import build_expression_string
        return build_expression_string(self.coefficients, self.selected_names)

    @property
    def n_terms(self) -> int:
        """Number of terms in the model."""
        return len(self.selected_indices)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = {
            "coefficients": np.array(self.coefficients).tolist(),
            "selected_indices": np.array(self.selected_indices).tolist(),
            "selected_names": self.selected_names,
            "mse": self.mse,
            "complexity": self.complexity,
            "aic": self.aic,
            "bic": self.bic,
            "aicc": self.aicc,
            "n_samples": self.n_samples,
        }
        if self.parametric_params is not None:
            d["parametric_params"] = {
                str(k): v for k, v in self.parametric_params.items()
            }
        return d


@dataclass
class SelectionPath:
    """
    Full path of model selection (for visualization).

    Parameters
    ----------
    results : list of SelectionResult
        Results at each step.
    strategy : str
        Selection strategy used.
    best_index : int
        Index of best model according to information criterion.
    """
    results: List[SelectionResult]
    strategy: str
    best_index: int

    @property
    def best(self) -> SelectionResult:
        """Return the best model."""
        return self.results[self.best_index]


# =============================================================================
# Core Fitting Functions
# =============================================================================


def fit_ols(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
) -> Tuple[jnp.ndarray, float]:
    """
    Fit ordinary least squares.

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        Target vector.

    Returns
    -------
    coefficients : jnp.ndarray
        Fitted coefficients.
    mse : float
        Mean squared error.
    """
    coeffs, residuals, rank, s = jnp.linalg.lstsq(Phi, y, rcond=None)
    y_pred = Phi @ coeffs
    mse = float(jnp.mean((y - y_pred) ** 2))
    return coeffs, mse


def fit_ridge(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    alpha: float,
) -> Tuple[jnp.ndarray, float]:
    """
    Fit ridge regression (L2 regularized OLS).

    Solves: minimize ||y - Phi @ w||^2 + alpha * ||w||^2

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        Target vector.
    alpha : float
        L2 regularization strength.

    Returns
    -------
    coefficients : jnp.ndarray
        Fitted coefficients.
    mse : float
        Mean squared error (without regularization term).
    """
    n_samples, n_features = Phi.shape

    # Ridge solution: (Phi^T Phi + alpha*I)^{-1} Phi^T y
    PhiTPhi = Phi.T @ Phi
    regularized = PhiTPhi + alpha * jnp.eye(n_features)
    coeffs = jnp.linalg.solve(regularized, Phi.T @ y)

    y_pred = Phi @ coeffs
    mse = float(jnp.mean((y - y_pred) ** 2))
    return coeffs, mse


def _fit_subset_parametric(
    X: jnp.ndarray,
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    indices_list: List[int],
    basis_names: List[str],
    complexities: jnp.ndarray,
    regularization: Optional[float],
    basis_library,
    parametric_in_subset: list,
    param_optimizer: str = "scipy",
    param_optimization_budget: int = 50,
    _param_cache: Optional[dict] = None,
) -> "SelectionResult":
    """Profile-likelihood fit: optimise nonlinear params, solve OLS inside."""
    import re
    from scipy.optimize import minimize as sp_minimize
    from scipy.optimize import minimize_scalar

    param_index_set = {p.basis_index for p in parametric_in_subset}

    # ---- cache check ----
    cache_key = tuple(sorted(indices_list))
    if _param_cache is not None and cache_key in _param_cache:
        best_params = _param_cache[cache_key]
    else:
        # ---- helpers ----
        def _build_Phi(all_pv):
            cols = []
            for idx in indices_list:
                if idx in param_index_set:
                    pi = next(p for p in parametric_in_subset if p.basis_index == idx)
                    cols.append(pi.func(X, **all_pv[idx]))
                else:
                    cols.append(Phi[:, idx])
            return jnp.column_stack(cols)

        def _profile_mse(all_pv):
            Phi_sub = _build_Phi(all_pv)
            if regularization is not None and regularization > 0:
                _, mse = fit_ridge(Phi_sub, y, regularization)
            else:
                _, mse = fit_ols(Phi_sub, y)
            return float(mse)

        # Gather every scalar parameter that needs optimising
        all_params = []  # (p_info, pname, bounds, log_scale)
        for pi in parametric_in_subset:
            for pname, bnds in pi.param_bounds.items():
                all_params.append((pi, pname, bnds, pi.log_scale))

        # ---- single scalar parameter (fast path) ----
        if len(all_params) == 1:
            pi, pname, bnds, log_scale = all_params[0]
            if log_scale and bnds[0] > 0 and bnds[1] > 0:
                import math
                log_bnds = (math.log10(bnds[0]), math.log10(bnds[1]))

                def _obj(log_a):
                    return _profile_mse({pi.basis_index: {pname: 10 ** log_a}})

                res = minimize_scalar(_obj, bounds=log_bnds, method="bounded")
                best_val = 10 ** res.x
            else:
                def _obj(a):
                    return _profile_mse({pi.basis_index: {pname: a}})

                res = minimize_scalar(_obj, bounds=bnds, method="bounded")
                best_val = float(res.x)
            best_params = {pi.basis_index: {pname: float(best_val)}}

        # ---- multi-parameter ----
        else:
            _use_optuna = False
            if param_optimizer == "optuna":
                try:
                    import optuna  # noqa: F811
                    _use_optuna = True
                except ImportError:
                    pass

            if _use_optuna:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def _optuna_obj(trial):
                    pv: Dict[int, Dict[str, float]] = {}
                    for pi, pname, bnds, log_scale in all_params:
                        pv.setdefault(pi.basis_index, {})
                        pv[pi.basis_index][pname] = trial.suggest_float(
                            f"{pi.basis_index}_{pname}",
                            bnds[0], bnds[1], log=log_scale,
                        )
                    return _profile_mse(pv)

                study = optuna.create_study(direction="minimize")
                study.optimize(
                    _optuna_obj,
                    n_trials=param_optimization_budget,
                    show_progress_bar=False,
                )
                best_params: Dict[int, Dict[str, float]] = {}
                for pi, pname, bnds, log_scale in all_params:
                    best_params.setdefault(pi.basis_index, {})
                    best_params[pi.basis_index][pname] = study.best_params[
                        f"{pi.basis_index}_{pname}"
                    ]
            else:
                import math
                x0, sp_bounds = [], []
                for pi, pname, bnds, log_scale in all_params:
                    if log_scale and bnds[0] > 0 and bnds[1] > 0:
                        x0.append(math.log10(pi.initial_params[pname]))
                        sp_bounds.append(
                            (math.log10(bnds[0]), math.log10(bnds[1]))
                        )
                    else:
                        x0.append(pi.initial_params[pname])
                        sp_bounds.append(bnds)

                def _scipy_obj(x):
                    pv: Dict[int, Dict[str, float]] = {}
                    for i, (pi, pname, bnds, log_scale) in enumerate(all_params):
                        pv.setdefault(pi.basis_index, {})
                        val = 10 ** x[i] if (log_scale and bnds[0] > 0) else x[i]
                        pv[pi.basis_index][pname] = float(val)
                    return _profile_mse(pv)

                res = sp_minimize(
                    _scipy_obj, x0, bounds=sp_bounds, method="L-BFGS-B",
                )
                best_params = {}
                for i, (pi, pname, bnds, log_scale) in enumerate(all_params):
                    best_params.setdefault(pi.basis_index, {})
                    val = 10 ** res.x[i] if (log_scale and bnds[0] > 0) else res.x[i]
                    best_params[pi.basis_index][pname] = float(val)

        # store in cache
        if _param_cache is not None:
            _param_cache[cache_key] = best_params

    # ---- final OLS at optimised parameters ----
    cols = []
    for idx in indices_list:
        if idx in param_index_set:
            pi = next(p for p in parametric_in_subset if p.basis_index == idx)
            cols.append(pi.func(X, **best_params[idx]))
        else:
            cols.append(Phi[:, idx])
    Phi_subset = jnp.column_stack(cols)

    if regularization is not None and regularization > 0:
        coeffs, mse = fit_ridge(Phi_subset, y, regularization)
    else:
        coeffs, mse = fit_ols(Phi_subset, y)

    # Resolved names
    resolved_names = []
    for idx in indices_list:
        if idx in param_index_set:
            pi = next(p for p in parametric_in_subset if p.basis_index == idx)
            rn = pi.name
            for pname, val in best_params[idx].items():
                rn = re.sub(r"\b" + re.escape(pname) + r"\b", f"{val:.4g}", rn)
            resolved_names.append(rn)
        else:
            resolved_names.append(basis_names[idx])

    n = len(y)
    k = len(indices_list)
    complexity = int(sum(int(complexities[i]) for i in indices_list))
    mse_f = float(mse)

    return SelectionResult(
        coefficients=coeffs,
        selected_indices=jnp.array(indices_list),
        selected_names=resolved_names,
        mse=mse_f,
        complexity=complexity,
        aic=compute_information_criterion(n, k, mse_f, "aic"),
        bic=compute_information_criterion(n, k, mse_f, "bic"),
        aicc=compute_information_criterion(n, k, mse_f, "aicc"),
        n_samples=n,
        parametric_params=best_params,
    )


def fit_subset(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    indices: Union[List[int], jnp.ndarray],
    basis_names: List[str],
    complexities: jnp.ndarray,
    regularization: Optional[float] = None,
    X: Optional[jnp.ndarray] = None,
    basis_library=None,
    param_optimizer: str = "scipy",
    param_optimization_budget: int = 50,
    _param_cache: Optional[dict] = None,
) -> SelectionResult:
    """
    Fit model using only selected basis functions.

    Parameters
    ----------
    Phi : jnp.ndarray
        Full design matrix.
    y : jnp.ndarray
        Target vector.
    indices : array-like
        Indices of selected basis functions.
    basis_names : list of str
        Names of all basis functions.
    complexities : jnp.ndarray
        Complexity scores for all basis functions.
    regularization : float, optional
        L2 regularization strength (ridge penalty).
    X : jnp.ndarray, optional
        Raw input data – required when parametric basis functions are present.
    basis_library : BasisLibrary, optional
        The basis library – required when parametric basis functions are present.
    param_optimizer : str
        ``"scipy"`` (default) or ``"optuna"``.
    param_optimization_budget : int
        Number of optuna trials (ignored for scipy).
    _param_cache : dict, optional
        Cache for optimised parametric parameter values.

    Returns
    -------
    result : SelectionResult
        Fitting result.
    """
    indices = jnp.array(indices)

    # Delegate to parametric path when needed
    if X is not None and basis_library is not None and basis_library.has_parametric:
        indices_set = set(int(i) for i in indices)
        parametric_in_subset = [
            p for p in basis_library._parametric_info
            if p.basis_index in indices_set
        ]
        if parametric_in_subset:
            return _fit_subset_parametric(
                X, Phi, y, [int(i) for i in indices],
                basis_names, complexities, regularization,
                basis_library, parametric_in_subset,
                param_optimizer, param_optimization_budget,
                _param_cache,
            )

    Phi_subset = Phi[:, indices]

    if regularization is not None and regularization > 0:
        coeffs, mse = fit_ridge(Phi_subset, y, regularization)
    else:
        coeffs, mse = fit_ols(Phi_subset, y)

    n = len(y)
    k = len(indices)
    complexity = int(jnp.sum(complexities[indices]))

    return SelectionResult(
        coefficients=coeffs,
        selected_indices=indices,
        selected_names=[basis_names[int(i)] for i in indices],
        mse=mse,
        complexity=complexity,
        aic=compute_information_criterion(n, k, mse, "aic"),
        bic=compute_information_criterion(n, k, mse, "bic"),
        aicc=compute_information_criterion(n, k, mse, "aicc"),
        n_samples=n,
    )


# =============================================================================
# Greedy Forward Selection
# =============================================================================


def greedy_forward_selection(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    basis_names: List[str],
    complexities: jnp.ndarray,
    max_terms: int = 5,
    information_criterion: str = "bic",
    early_stop: bool = True,
    candidate_indices: Optional[List[int]] = None,
    regularization: Optional[float] = None,
    X: Optional[jnp.ndarray] = None,
    basis_library=None,
    param_optimizer: str = "scipy",
    param_optimization_budget: int = 50,
) -> SelectionPath:
    """
    Greedy forward stepwise selection.

    Starting from empty model, iteratively add the basis function that most
    improves the information criterion until no improvement or max_terms reached.

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix.
    y : jnp.ndarray
        Target vector.
    basis_names : list of str
        Names of basis functions.
    complexities : jnp.ndarray
        Complexity scores.
    max_terms : int
        Maximum number of terms.
    information_criterion : str
        Criterion for selection ("aic", "bic", "aicc").
    early_stop : bool
        If True, stop when IC stops improving.
    candidate_indices : list of int, optional
        Indices of candidate basis functions to consider.
    regularization : float, optional
        L2 regularization strength (ridge penalty).

    Returns
    -------
    path : SelectionPath
        Selection path with all intermediate results.
    """
    n_basis = Phi.shape[1]

    if candidate_indices is None:
        available = set(range(n_basis))
    else:
        available = set(candidate_indices)

    selected: List[int] = []
    results: List[SelectionResult] = []

    current_ic = float("inf")
    best_ic = float("inf")
    best_index = -1

    _param_cache = (
        {} if (basis_library is not None and basis_library.has_parametric) else None
    )
    _fs_kw = dict(
        X=X, basis_library=basis_library,
        param_optimizer=param_optimizer,
        param_optimization_budget=param_optimization_budget,
        _param_cache=_param_cache,
    )

    for step in range(min(max_terms, len(available))):
        best_step_ic = float("inf")
        best_idx = None
        best_result = None

        for idx in available:
            candidate = selected + [idx]
            result = fit_subset(
                Phi, y, candidate, basis_names, complexities,
                regularization, **_fs_kw,
            )

            ic_value = getattr(result, information_criterion)
            if ic_value < best_step_ic:
                best_step_ic = ic_value
                best_idx = idx
                best_result = result

        if best_result is None:
            break

        # Check for improvement
        if early_stop and best_step_ic >= current_ic:
            break

        selected.append(best_idx)
        available.remove(best_idx)
        current_ic = best_step_ic
        results.append(best_result)

        if best_step_ic < best_ic:
            best_ic = best_step_ic
            best_index = len(results) - 1

    if not results:
        # Return single-term model with constant if available
        const_idx = [i for i, name in enumerate(basis_names) if name == "1"]
        if const_idx:
            result = fit_subset(
                Phi, y, const_idx, basis_names, complexities,
                regularization, **_fs_kw,
            )
            results = [result]
            best_index = 0
        else:
            # Fit with first available feature
            idx = list(available)[0] if available else 0
            result = fit_subset(
                Phi, y, [idx], basis_names, complexities,
                regularization, **_fs_kw,
            )
            results = [result]
            best_index = 0

    return SelectionPath(
        results=results,
        strategy="greedy_forward",
        best_index=max(0, best_index),
    )


# =============================================================================
# Greedy Backward Elimination
# =============================================================================


def greedy_backward_elimination(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    basis_names: List[str],
    complexities: jnp.ndarray,
    min_terms: int = 1,
    information_criterion: str = "bic",
    start_indices: Optional[List[int]] = None,
    max_terms: int = 5,  # For API compatibility
    regularization: Optional[float] = None,
    X: Optional[jnp.ndarray] = None,
    basis_library=None,
    param_optimizer: str = "scipy",
    param_optimization_budget: int = 50,
) -> SelectionPath:
    """
    Greedy backward elimination.

    Starting from full model (or specified subset), iteratively remove the
    term whose removal most improves the information criterion.

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix.
    y : jnp.ndarray
        Target vector.
    basis_names : list of str
        Names of basis functions.
    complexities : jnp.ndarray
        Complexity scores.
    min_terms : int
        Minimum number of terms to keep.
    information_criterion : str
        Criterion for selection.
    start_indices : list of int, optional
        Starting indices. If None, uses all basis functions.

    Returns
    -------
    path : SelectionPath
        Selection path.
    """
    n_basis = Phi.shape[1]

    if start_indices is None:
        selected = list(range(n_basis))
    else:
        selected = list(start_indices)

    results: List[SelectionResult] = []
    best_ic = float("inf")
    best_index = -1

    _param_cache = (
        {} if (basis_library is not None and basis_library.has_parametric) else None
    )
    _fs_kw = dict(
        X=X, basis_library=basis_library,
        param_optimizer=param_optimizer,
        param_optimization_budget=param_optimization_budget,
        _param_cache=_param_cache,
    )

    # Initial full model
    result = fit_subset(Phi, y, selected, basis_names, complexities, regularization, **_fs_kw)
    results.append(result)
    current_ic = getattr(result, information_criterion)
    if current_ic < best_ic:
        best_ic = current_ic
        best_index = 0

    while len(selected) > min_terms:
        best_step_ic = float("inf")
        worst_idx = None
        best_result = None

        for i, idx in enumerate(selected):
            # Try removing each term
            candidate = selected[:i] + selected[i + 1 :]
            if not candidate:
                continue

            result = fit_subset(Phi, y, candidate, basis_names, complexities, regularization, **_fs_kw)
            ic_value = getattr(result, information_criterion)

            if ic_value < best_step_ic:
                best_step_ic = ic_value
                worst_idx = i
                best_result = result

        # If removing any term doesn't improve, stop
        if best_step_ic >= current_ic:
            break

        selected.pop(worst_idx)
        current_ic = best_step_ic
        results.append(best_result)

        if best_step_ic < best_ic:
            best_ic = best_step_ic
            best_index = len(results) - 1

    return SelectionPath(
        results=results,
        strategy="greedy_backward",
        best_index=max(0, best_index),
    )


# =============================================================================
# Exhaustive Search
# =============================================================================


def exhaustive_search(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    basis_names: List[str],
    complexities: jnp.ndarray,
    max_terms: int = 5,
    information_criterion: str = "bic",
    candidate_indices: Optional[List[int]] = None,
    max_combinations: int = 100000,
    regularization: Optional[float] = None,
    X: Optional[jnp.ndarray] = None,
    basis_library=None,
    param_optimizer: str = "scipy",
    param_optimization_budget: int = 50,
) -> SelectionPath:
    """
    Exhaustive search over all combinations.

    Enumerate all combinations up to max_terms and select the best.

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix.
    y : jnp.ndarray
        Target vector.
    basis_names : list of str
        Names of basis functions.
    complexities : jnp.ndarray
        Complexity scores.
    max_terms : int
        Maximum number of terms.
    information_criterion : str
        Criterion for selection.
    candidate_indices : list of int, optional
        Indices to consider. If None, uses all.
    max_combinations : int
        Maximum combinations to evaluate (safety limit).
    regularization : float, optional
        L2 regularization strength (ridge penalty).

    Returns
    -------
    path : SelectionPath
        Selection path (all Pareto-optimal models).

    Raises
    ------
    ValueError
        If too many combinations would be evaluated.
    """
    if candidate_indices is None:
        candidate_indices = list(range(Phi.shape[1]))

    n_candidates = len(candidate_indices)

    # Count total combinations
    total = sum(
        len(list(itertools.combinations(candidate_indices, k)))
        for k in range(1, min(max_terms, n_candidates) + 1)
    )

    if total > max_combinations:
        raise ValueError(
            f"Exhaustive search would evaluate {total} combinations, "
            f"exceeding limit of {max_combinations}. Use greedy or LASSO instead."
        )

    _param_cache = (
        {} if (basis_library is not None and basis_library.has_parametric) else None
    )
    _fs_kw = dict(
        X=X, basis_library=basis_library,
        param_optimizer=param_optimizer,
        param_optimization_budget=param_optimization_budget,
        _param_cache=_param_cache,
    )

    results: List[SelectionResult] = []
    best_ic = float("inf")
    best_index = -1

    for k in range(1, min(max_terms, n_candidates) + 1):
        for combo in itertools.combinations(candidate_indices, k):
            result = fit_subset(
                Phi, y, list(combo), basis_names, complexities,
                regularization, **_fs_kw,
            )
            results.append(result)

            ic_value = getattr(result, information_criterion)
            if ic_value < best_ic:
                best_ic = ic_value
                best_index = len(results) - 1

    return SelectionPath(
        results=results,
        strategy="exhaustive",
        best_index=best_index,
    )


# =============================================================================
# LASSO Coordinate Descent
# =============================================================================


def coordinate_descent_lasso(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    alpha: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    warm_start: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Solve LASSO using coordinate descent.

    minimize (1/2n) ||y - Phi @ w||^2 + alpha * ||w||_1

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix (should be standardized).
    y : jnp.ndarray
        Target vector (should be centered).
    alpha : float
        Regularization parameter.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    warm_start : jnp.ndarray, optional
        Initial coefficients.

    Returns
    -------
    coefficients : jnp.ndarray
        LASSO coefficients.
    """
    n_samples, n_features = Phi.shape

    if warm_start is not None:
        w = warm_start.copy()
    else:
        w = jnp.zeros(n_features)

    # Precompute for efficiency
    Phi_sq_sum = jnp.sum(Phi ** 2, axis=0)

    for iteration in range(max_iter):
        w_old = w.copy()

        for j in range(n_features):
            # Compute partial residual
            residual = y - Phi @ w + Phi[:, j] * w[j]

            # Coordinate update with soft thresholding
            rho = jnp.dot(Phi[:, j], residual)
            z = Phi_sq_sum[j]

            if z < 1e-10:
                w = w.at[j].set(0.0)
            else:
                # Soft thresholding
                w_j = jnp.sign(rho) * jnp.maximum(jnp.abs(rho) - n_samples * alpha, 0) / z
                w = w.at[j].set(w_j)

        # Check convergence
        if jnp.max(jnp.abs(w - w_old)) < tol:
            break

    return w


def lasso_path_selection(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    basis_names: List[str],
    complexities: jnp.ndarray,
    max_terms: int = 5,
    information_criterion: str = "bic",
    n_alphas: int = 100,
    alpha_min_ratio: float = 1e-4,
    regularization: Optional[float] = None,
    X: Optional[jnp.ndarray] = None,
    basis_library=None,
    param_optimizer: str = "scipy",
    param_optimization_budget: int = 50,
) -> SelectionPath:
    """
    LASSO path screening for variable selection.

    Use LASSO regularization path to identify promising subsets, then
    refit with OLS for final selection.

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix.
    y : jnp.ndarray
        Target vector.
    basis_names : list of str
        Names of basis functions.
    complexities : jnp.ndarray
        Complexity scores.
    max_terms : int
        Maximum number of terms.
    information_criterion : str
        Criterion for selection.
    n_alphas : int
        Number of regularization values.
    alpha_min_ratio : float
        Ratio of min to max alpha.

    Returns
    -------
    path : SelectionPath
        Selection path.
    """
    n_samples, n_features = Phi.shape

    # Standardize features
    Phi_mean = jnp.mean(Phi, axis=0)
    Phi_std = jnp.std(Phi, axis=0)
    Phi_std = jnp.where(Phi_std < 1e-10, 1.0, Phi_std)
    Phi_standardized = (Phi - Phi_mean) / Phi_std

    y_mean = jnp.mean(y)
    y_centered = y - y_mean

    # Compute alpha_max (minimum alpha that gives all-zero solution)
    alpha_max = jnp.max(jnp.abs(Phi_standardized.T @ y_centered)) / n_samples
    alpha_min = alpha_max * alpha_min_ratio

    # Generate alpha path
    alphas = jnp.logspace(
        jnp.log10(alpha_max),
        jnp.log10(alpha_min),
        n_alphas,
    )

    # Track unique subsets
    seen_subsets: set = set()
    results: List[SelectionResult] = []
    best_ic = float("inf")
    best_index = -1

    _param_cache = (
        {} if (basis_library is not None and basis_library.has_parametric) else None
    )
    _fs_kw = dict(
        X=X, basis_library=basis_library,
        param_optimizer=param_optimizer,
        param_optimization_budget=param_optimization_budget,
        _param_cache=_param_cache,
    )

    w = jnp.zeros(n_features)

    for alpha in alphas:
        # Warm start from previous solution
        w = coordinate_descent_lasso(
            Phi_standardized, y_centered, float(alpha), warm_start=w
        )

        # Get active set (non-zero coefficients)
        active = jnp.where(jnp.abs(w) > 1e-8)[0]

        if len(active) == 0 or len(active) > max_terms:
            continue

        # Check if we've seen this subset
        subset_key = tuple(sorted(int(i) for i in active))
        if subset_key in seen_subsets:
            continue
        seen_subsets.add(subset_key)

        # Refit with OLS on original (unstandardized) data
        result = fit_subset(
            Phi, y, list(active), basis_names, complexities,
            regularization, **_fs_kw,
        )
        results.append(result)

        ic_value = getattr(result, information_criterion)
        if ic_value < best_ic:
            best_ic = ic_value
            best_index = len(results) - 1

    if not results:
        # Fallback to greedy forward if LASSO finds nothing
        return greedy_forward_selection(
            Phi, y, basis_names, complexities, max_terms, information_criterion,
            regularization=regularization,
            X=X, basis_library=basis_library,
            param_optimizer=param_optimizer,
            param_optimization_budget=param_optimization_budget,
        )

    return SelectionPath(
        results=results,
        strategy="lasso_path",
        best_index=best_index,
    )


# =============================================================================
# Elastic Net
# =============================================================================


def coordinate_descent_elastic_net(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    alpha: float,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-6,
    warm_start: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Solve Elastic Net using coordinate descent.

    minimize (1/2n) ||y - Phi @ w||^2 + alpha * (l1_ratio * ||w||_1 + (1-l1_ratio)/2 * ||w||_2^2)

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix.
    y : jnp.ndarray
        Target vector.
    alpha : float
        Overall regularization parameter.
    l1_ratio : float
        Mix between L1 (1) and L2 (0).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    warm_start : jnp.ndarray, optional
        Initial coefficients.

    Returns
    -------
    coefficients : jnp.ndarray
        Elastic net coefficients.
    """
    n_samples, n_features = Phi.shape

    if warm_start is not None:
        w = warm_start.copy()
    else:
        w = jnp.zeros(n_features)

    l1_penalty = alpha * l1_ratio
    l2_penalty = alpha * (1 - l1_ratio)

    Phi_sq_sum = jnp.sum(Phi ** 2, axis=0)

    for iteration in range(max_iter):
        w_old = w.copy()

        for j in range(n_features):
            residual = y - Phi @ w + Phi[:, j] * w[j]
            rho = jnp.dot(Phi[:, j], residual)
            z = Phi_sq_sum[j] + n_samples * l2_penalty

            if z < 1e-10:
                w = w.at[j].set(0.0)
            else:
                # Soft thresholding with L2 modification
                w_j = jnp.sign(rho) * jnp.maximum(jnp.abs(rho) - n_samples * l1_penalty, 0) / z
                w = w.at[j].set(w_j)

        if jnp.max(jnp.abs(w - w_old)) < tol:
            break

    return w


# =============================================================================
# Pareto Front Computation
# =============================================================================


def compute_pareto_front(results: List[SelectionResult]) -> List[SelectionResult]:
    """
    Extract Pareto-optimal models (complexity vs MSE).

    Parameters
    ----------
    results : list of SelectionResult
        All candidate models.

    Returns
    -------
    pareto : list of SelectionResult
        Pareto-optimal models (sorted by complexity).
    """
    if not results:
        return []

    # Sort by complexity, then by MSE
    sorted_results = sorted(results, key=lambda r: (r.complexity, r.mse))

    pareto = []
    best_mse = float("inf")

    for r in sorted_results:
        if r.mse < best_mse:
            pareto.append(r)
            best_mse = r.mse

    return pareto


def compute_pareto_front_multi(
    results: List[SelectionResult],
    objectives: List[str] = ["complexity", "mse"],
) -> List[SelectionResult]:
    """
    Compute Pareto front for multiple objectives.

    Parameters
    ----------
    results : list of SelectionResult
        All candidate models.
    objectives : list of str
        Objective names (attributes of SelectionResult to minimize).

    Returns
    -------
    pareto : list of SelectionResult
        Pareto-optimal models.
    """
    if not results:
        return []

    n = len(results)
    is_dominated = [False] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Check if j dominates i (j is better in all objectives)
            dominates = True
            strictly_better = False

            for obj in objectives:
                val_i = getattr(results[i], obj)
                val_j = getattr(results[j], obj)

                if val_j > val_i:
                    dominates = False
                    break
                if val_j < val_i:
                    strictly_better = True

            if dominates and strictly_better:
                is_dominated[i] = True
                break

    pareto = [results[i] for i in range(n) if not is_dominated[i]]
    return sorted(pareto, key=lambda r: r.complexity)


# =============================================================================
# Selection Strategy Dispatcher
# =============================================================================


def select_features(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    basis_names: List[str],
    complexities: jnp.ndarray,
    strategy: str = "greedy_forward",
    max_terms: int = 5,
    information_criterion: str = "bic",
    **kwargs,
) -> SelectionPath:
    """
    Run feature selection with the specified strategy.

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix.
    y : jnp.ndarray
        Target vector.
    basis_names : list of str
        Names of basis functions.
    complexities : jnp.ndarray
        Complexity scores.
    strategy : str
        One of "greedy_forward", "greedy_backward", "exhaustive", "lasso_path".
    max_terms : int
        Maximum number of terms.
    information_criterion : str
        Information criterion for model selection.
    **kwargs
        Additional arguments for specific strategies.

    Returns
    -------
    path : SelectionPath
        Selection results.
    """
    strategies = {
        "greedy_forward": greedy_forward_selection,
        "greedy_backward": greedy_backward_elimination,
        "exhaustive": exhaustive_search,
        "lasso_path": lasso_path_selection,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}"
        )

    return strategies[strategy](
        Phi=Phi,
        y=y,
        basis_names=basis_names,
        complexities=complexities,
        max_terms=max_terms,
        information_criterion=information_criterion,
        **kwargs,
    )
