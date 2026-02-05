"""
Numerical Utilities for JAXSR.

Provides utility functions for numerical stability, array operations,
and helper functions used throughout the library.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap


# =============================================================================
# Numerical Stability
# =============================================================================


@jit
def safe_solve_lstsq(
    A: jnp.ndarray,
    b: jnp.ndarray,
    rcond: Optional[float] = None,
) -> Tuple[jnp.ndarray, float, int]:
    """
    Solve least squares with numerical stability checks.

    Uses SVD-based approach with appropriate conditioning.

    Parameters
    ----------
    A : jnp.ndarray
        Design matrix of shape (n_samples, n_features).
    b : jnp.ndarray
        Target vector of shape (n_samples,).
    rcond : float, optional
        Cutoff ratio for small singular values. If None, uses machine precision.

    Returns
    -------
    x : jnp.ndarray
        Solution vector.
    residual : float
        Sum of squared residuals.
    rank : int
        Effective rank of A.
    """
    if rcond is None:
        rcond = jnp.finfo(A.dtype).eps * max(A.shape)

    # Use lstsq which handles rank-deficient cases
    x, residuals, rank, s = jnp.linalg.lstsq(A, b, rcond=rcond)

    # Compute residual if not returned (happens when rank < n_features)
    if residuals.size == 0:
        residual = jnp.sum((b - A @ x) ** 2)
    else:
        residual = residuals[0] if residuals.ndim > 0 else residuals

    return x, float(residual), int(rank)


@jit
def solve_lstsq_svd(
    A: jnp.ndarray,
    b: jnp.ndarray,
    rcond: float = 1e-10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve least squares using explicit SVD decomposition.

    More numerically stable for ill-conditioned matrices.

    Parameters
    ----------
    A : jnp.ndarray
        Design matrix.
    b : jnp.ndarray
        Target vector.
    rcond : float
        Cutoff for singular values (relative to largest).

    Returns
    -------
    x : jnp.ndarray
        Solution vector.
    s : jnp.ndarray
        Singular values.
    """
    U, s, Vt = jnp.linalg.svd(A, full_matrices=False)

    # Threshold small singular values
    cutoff = rcond * jnp.max(s)
    s_inv = jnp.where(s > cutoff, 1.0 / s, 0.0)

    # Solution: x = V @ S^-1 @ U.T @ b
    x = Vt.T @ (s_inv * (U.T @ b))

    return x, s


@jit
def condition_number(A: jnp.ndarray) -> float:
    """
    Compute the condition number of a matrix.

    Parameters
    ----------
    A : jnp.ndarray
        Input matrix.

    Returns
    -------
    cond : float
        Condition number (ratio of largest to smallest singular value).
    """
    s = jnp.linalg.svd(A, compute_uv=False)
    return float(s[0] / (s[-1] + 1e-10))


def check_collinearity(
    Phi: jnp.ndarray,
    threshold: float = 1e-6,
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Check for collinear columns in design matrix.

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix.
    threshold : float
        Correlation threshold for collinearity.

    Returns
    -------
    has_collinearity : bool
        True if collinear columns found.
    pairs : list of tuple
        Pairs of collinear column indices.
    """
    # Standardize columns
    Phi_centered = Phi - jnp.mean(Phi, axis=0)
    norms = jnp.linalg.norm(Phi_centered, axis=0)
    Phi_normalized = Phi_centered / (norms + 1e-10)

    # Correlation matrix
    corr = Phi_normalized.T @ Phi_normalized / Phi.shape[0]

    # Find highly correlated pairs (excluding diagonal)
    pairs = []
    n_cols = Phi.shape[1]
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if jnp.abs(corr[i, j]) > 1 - threshold:
                pairs.append((i, j))

    return len(pairs) > 0, pairs


# =============================================================================
# Array Utilities
# =============================================================================


def standardize(
    X: jnp.ndarray,
    mean: Optional[jnp.ndarray] = None,
    std: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Standardize array to zero mean and unit variance.

    Parameters
    ----------
    X : jnp.ndarray
        Input array.
    mean : jnp.ndarray, optional
        Pre-computed mean (for transform).
    std : jnp.ndarray, optional
        Pre-computed std (for transform).

    Returns
    -------
    X_std : jnp.ndarray
        Standardized array.
    mean : jnp.ndarray
        Mean values.
    std : jnp.ndarray
        Standard deviation values.
    """
    if mean is None:
        mean = jnp.mean(X, axis=0)
    if std is None:
        std = jnp.std(X, axis=0)
        std = jnp.where(std < 1e-10, 1.0, std)  # Avoid division by zero

    X_std = (X - mean) / std
    return X_std, mean, std


def unstandardize(
    X_std: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
) -> jnp.ndarray:
    """
    Reverse standardization.

    Parameters
    ----------
    X_std : jnp.ndarray
        Standardized array.
    mean : jnp.ndarray
        Mean values.
    std : jnp.ndarray
        Standard deviation values.

    Returns
    -------
    X : jnp.ndarray
        Original scale array.
    """
    return X_std * std + mean


def normalize(
    X: jnp.ndarray,
    min_val: Optional[jnp.ndarray] = None,
    max_val: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Normalize array to [0, 1] range.

    Parameters
    ----------
    X : jnp.ndarray
        Input array.
    min_val : jnp.ndarray, optional
        Pre-computed min (for transform).
    max_val : jnp.ndarray, optional
        Pre-computed max (for transform).

    Returns
    -------
    X_norm : jnp.ndarray
        Normalized array.
    min_val : jnp.ndarray
        Min values.
    max_val : jnp.ndarray
        Max values.
    """
    if min_val is None:
        min_val = jnp.min(X, axis=0)
    if max_val is None:
        max_val = jnp.max(X, axis=0)

    range_val = max_val - min_val
    range_val = jnp.where(range_val < 1e-10, 1.0, range_val)

    X_norm = (X - min_val) / range_val
    return X_norm, min_val, max_val


# =============================================================================
# Matrix Operations
# =============================================================================


@jit
def compute_hat_matrix(Phi: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the hat (projection) matrix.

    H = Phi @ (Phi.T @ Phi)^-1 @ Phi.T

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix.

    Returns
    -------
    H : jnp.ndarray
        Hat matrix.
    """
    Phi_pinv = jnp.linalg.pinv(Phi)
    return Phi @ Phi_pinv


@jit
def compute_leverage(Phi: jnp.ndarray) -> jnp.ndarray:
    """
    Compute leverage values (diagonal of hat matrix).

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix.

    Returns
    -------
    leverage : jnp.ndarray
        Leverage values for each sample.
    """
    H = compute_hat_matrix(Phi)
    return jnp.diag(H)


@jit
def compute_residuals(
    y: jnp.ndarray,
    y_pred: jnp.ndarray,
    Phi: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    Compute various types of residuals.

    Parameters
    ----------
    y : jnp.ndarray
        True values.
    y_pred : jnp.ndarray
        Predicted values.
    Phi : jnp.ndarray
        Design matrix.

    Returns
    -------
    residuals : dict
        Dictionary containing:
        - "raw": y - y_pred
        - "standardized": raw / std(raw)
        - "studentized": raw / (sigma * sqrt(1 - h_ii))
    """
    raw = y - y_pred
    leverage = compute_leverage(Phi)

    n, p = Phi.shape
    sigma = jnp.sqrt(jnp.sum(raw ** 2) / (n - p))

    standardized = raw / (jnp.std(raw) + 1e-10)
    studentized = raw / (sigma * jnp.sqrt(1 - leverage + 1e-10))

    return {
        "raw": raw,
        "standardized": standardized,
        "studentized": studentized,
    }


# =============================================================================
# Validation Utilities
# =============================================================================


def validate_array(
    X: Any,
    name: str = "X",
    ndim: Optional[int] = None,
    dtype: Optional[jnp.dtype] = None,
) -> jnp.ndarray:
    """
    Validate and convert input to JAX array.

    Parameters
    ----------
    X : array-like
        Input data.
    name : str
        Name for error messages.
    ndim : int, optional
        Expected number of dimensions.
    dtype : dtype, optional
        Expected dtype.

    Returns
    -------
    X : jnp.ndarray
        Validated JAX array.

    Raises
    ------
    ValueError
        If validation fails.
    """
    X = jnp.asarray(X)

    if dtype is not None:
        X = X.astype(dtype)

    if ndim is not None and X.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {X.ndim}D")

    if not jnp.all(jnp.isfinite(X)):
        n_invalid = int(jnp.sum(~jnp.isfinite(X)))
        raise ValueError(f"{name} contains {n_invalid} non-finite values")

    return X


def check_consistent_length(*arrays) -> int:
    """
    Check that all arrays have consistent first dimension.

    Parameters
    ----------
    *arrays : array-like
        Arrays to check.

    Returns
    -------
    length : int
        Common length.

    Raises
    ------
    ValueError
        If lengths don't match.
    """
    lengths = [len(a) for a in arrays if a is not None]
    unique_lengths = set(lengths)

    if len(unique_lengths) > 1:
        raise ValueError(f"Inconsistent array lengths: {lengths}")

    return lengths[0] if lengths else 0


# =============================================================================
# Random Utilities
# =============================================================================


def get_random_key(seed: Optional[int] = None) -> jax.random.PRNGKey:
    """
    Get a JAX random key.

    Parameters
    ----------
    seed : int, optional
        Random seed. If None, uses system time.

    Returns
    -------
    key : PRNGKey
        JAX random key.
    """
    if seed is None:
        seed = np.random.randint(0, 2**31)
    return jax.random.PRNGKey(seed)


def split_key(key: jax.random.PRNGKey, n: int = 2) -> List[jax.random.PRNGKey]:
    """
    Split a random key into multiple keys.

    Parameters
    ----------
    key : PRNGKey
        Input key.
    n : int
        Number of keys to generate.

    Returns
    -------
    keys : list of PRNGKey
        Split keys.
    """
    return list(jax.random.split(key, n))


# =============================================================================
# Expression Utilities
# =============================================================================


def format_coefficient(value: float, precision: int = 4) -> str:
    """
    Format a coefficient for display.

    Parameters
    ----------
    value : float
        Coefficient value.
    precision : int
        Number of significant figures.

    Returns
    -------
    formatted : str
        Formatted string.
    """
    if abs(value) < 1e-10:
        return "0"

    # Use scientific notation for very large/small values
    if abs(value) > 1e4 or abs(value) < 1e-4:
        return f"{value:.{precision}e}"

    return f"{value:.{precision}g}"


def build_expression_string(
    coefficients: jnp.ndarray,
    names: List[str],
    precision: int = 4,
) -> str:
    """
    Build a human-readable expression string.

    Parameters
    ----------
    coefficients : jnp.ndarray
        Coefficient values.
    names : list of str
        Basis function names.
    precision : int
        Coefficient precision.

    Returns
    -------
    expression : str
        Expression like "y = 2.5*x + 1.2*x^2 - 0.8"
    """
    terms = []

    for coef, name in zip(coefficients, names):
        coef = float(coef)
        if abs(coef) < 1e-10:
            continue

        coef_str = format_coefficient(abs(coef), precision)

        if name == "1":
            term = coef_str
        elif coef_str == "1":
            term = name
        else:
            term = f"{coef_str}*{name}"

        if coef < 0:
            terms.append(f"- {term}")
        elif len(terms) > 0:
            terms.append(f"+ {term}")
        else:
            terms.append(term)

    if not terms:
        return "y = 0"

    return "y = " + " ".join(terms)


# =============================================================================
# Serialization Utilities
# =============================================================================


def array_to_list(arr: jnp.ndarray) -> List:
    """Convert JAX array to nested Python list for JSON serialization."""
    return np.array(arr).tolist()


def list_to_array(lst: List) -> jnp.ndarray:
    """Convert nested Python list back to JAX array."""
    return jnp.array(lst)
