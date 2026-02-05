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
        return {
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


def fit_subset(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    indices: Union[List[int], jnp.ndarray],
    basis_names: List[str],
    complexities: jnp.ndarray,
    regularization: Optional[float] = None,
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

    Returns
    -------
    result : SelectionResult
        Fitting result.
    """
    indices = jnp.array(indices)
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

    for step in range(min(max_terms, len(available))):
        best_step_ic = float("inf")
        best_idx = None
        best_result = None

        for idx in available:
            candidate = selected + [idx]
            result = fit_subset(Phi, y, candidate, basis_names, complexities, regularization)

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
            result = fit_subset(Phi, y, const_idx, basis_names, complexities, regularization)
            results = [result]
            best_index = 0
        else:
            # Fit with first available feature
            idx = list(available)[0] if available else 0
            result = fit_subset(Phi, y, [idx], basis_names, complexities, regularization)
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

    # Initial full model
    result = fit_subset(Phi, y, selected, basis_names, complexities, regularization)
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

            result = fit_subset(Phi, y, candidate, basis_names, complexities, regularization)
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

    results: List[SelectionResult] = []
    best_ic = float("inf")
    best_index = -1

    for k in range(1, min(max_terms, n_candidates) + 1):
        for combo in itertools.combinations(candidate_indices, k):
            result = fit_subset(Phi, y, list(combo), basis_names, complexities, regularization)
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
        result = fit_subset(Phi, y, list(active), basis_names, complexities, regularization)
        results.append(result)

        ic_value = getattr(result, information_criterion)
        if ic_value < best_ic:
            best_ic = ic_value
            best_index = len(results) - 1

    if not results:
        # Fallback to greedy forward if LASSO finds nothing
        return greedy_forward_selection(
            Phi, y, basis_names, complexities, max_terms, information_criterion,
            regularization=regularization
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
