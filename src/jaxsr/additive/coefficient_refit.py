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

    # Augment with an intercept column and solve via lstsq (SVD-based,
    # minimum-norm solution for rank-deficient systems -- never inv()).
    A = jnp.concatenate([jnp.ones((n_samples, 1)), Phi], axis=1)
    solution, _, _, _ = jnp.linalg.lstsq(A, y, rcond=None)

    intercept = float(solution[0])
    coefficients = solution[1:]
    return intercept, coefficients
