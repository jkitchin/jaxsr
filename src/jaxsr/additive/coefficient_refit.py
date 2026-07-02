"""
Coefficient refitting for additive symbolic regression.

After a new symbolic term is discovered, the stagewise model can optionally
re-solve the linear coefficients over *all* discovered symbolic features at
once, treating each term's prediction as a single feature column::

    Phi[:, j] = g_j(X)
    y ~= intercept + Phi @ coefficients

This decouples term *discovery* (nonlinear, greedy) from term *weighting*
(linear, global), which typically improves accuracy relative to fixed
learning-rate-scaled stagewise weights.

Only ordinary least squares is implemented for the first milestone.  Ridge,
lasso, and sparse variants can be added later behind the same interface.
"""

from __future__ import annotations

import jax.numpy as jnp


def refit_ols(Phi: jnp.ndarray, y: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    """
    Refit an intercept and per-term coefficients by ordinary least squares.

    Solves ``y ~= intercept + Phi @ coefficients`` using a least-squares
    solver that is robust to rank-deficient / collinear design matrices
    (later boosting stages fit residuals of earlier ones, so the term columns
    can be highly correlated).

    Parameters
    ----------
    Phi : jnp.ndarray of shape (n_samples, n_terms)
        Design matrix whose column ``j`` is ``g_j(X)`` for term ``j``.  May
        have zero columns (no terms discovered yet).
    y : jnp.ndarray of shape (n_samples,)
        Target values.

    Returns
    -------
    intercept : float
        Fitted intercept.
    coefficients : jnp.ndarray of shape (n_terms,)
        Fitted per-term coefficients.

    Raises
    ------
    ValueError
        If ``Phi`` is not 2-D or its number of rows does not match ``len(y)``.
    """
    Phi = jnp.asarray(Phi)
    y = jnp.asarray(y).ravel()

    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2-D, got shape {Phi.shape}.")
    if Phi.shape[0] != y.shape[0]:
        raise ValueError(f"Phi has {Phi.shape[0]} rows but y has {y.shape[0]} samples.")

    n_samples, n_terms = Phi.shape

    # No terms yet: the best constant model is the mean.
    if n_terms == 0:
        return float(jnp.mean(y)), jnp.zeros((0,))

    # Fit the intercept by centering rather than augmenting Phi with a
    # column of ones.  Augmentation mixes an O(1) ones-column with the term
    # columns, whose scale is arbitrary (it tracks the scale of y); when the
    # terms are tiny (e.g. y ~ 1e-6) that disparity makes the system severely
    # ill-conditioned in float32 and lstsq returns garbage.  Centering keeps
    # the design matrix on a single scale and is the standard, stable way to
    # estimate an intercept.  Still SVD-based via lstsq -- never inv().
    Phi_mean = jnp.mean(Phi, axis=0)
    y_mean = jnp.mean(y)
    coefficients, _, _, _ = jnp.linalg.lstsq(Phi - Phi_mean, y - y_mean, rcond=None)

    intercept = float(y_mean - Phi_mean @ coefficients)
    return intercept, coefficients
