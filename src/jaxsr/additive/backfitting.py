"""
Backfitting (BART/iBART-style) additive symbolic regression -- scaffolding.

This module is a documented placeholder for a future backfitting regressor.
Unlike :class:`~jaxsr.additive.stagewise.StagewiseSymbolicRegressor`, where
each discovered term is *frozen*, a backfitting regressor maintains a fixed
number of terms and repeatedly *revises* them.

Planned algorithm (one "sweep" over the ``m`` terms)::

    for term j in 1..m:
        partial_residual = y - (intercept + sum_{i != j} coef_i * g_i(X))
        refit / re-discover term j on partial_residual
        put term j back into the ensemble
    (repeat sweeps until convergence or a sweep budget is exhausted)

This mirrors the additive backfitting idea behind BART / iBART, where
individual components are updated conditional on all the others rather than
added once and fixed.

The class below intentionally raises :class:`NotImplementedError` from
:meth:`fit`; it exists to fix the public API surface and give the stagewise
implementation a sibling to extend.  See ``TODO`` markers for the concrete
next steps.
"""

from __future__ import annotations

import jax.numpy as jnp

from .._compat import _SklearnCompatMixin


class BackfittingSymbolicRegressor(_SklearnCompatMixin):
    """
    Backfitting additive symbolic regression (BART/iBART-style) -- scaffold.

    Maintains ``n_terms`` symbolic components and revises each one in place
    across repeated sweeps, conditioning on the current fit of all other
    terms.  Contrast with the stagewise regressor, which freezes terms once
    discovered.

    .. warning::
        This class is not yet implemented.  :meth:`fit` raises
        :class:`NotImplementedError`.  The constructor and attribute names are
        provided so downstream code and documentation can target a stable API.

    Parameters
    ----------
    n_terms : int
        Number of symbolic components to maintain and revise.
    n_sweeps : int
        Maximum number of backfitting sweeps over the terms.
    max_complexity : int
        Complexity budget (max basis terms) for each component.
    loss : str
        Loss function name.  Currently only ``"squared_error"`` is planned.
    tol : float
        Convergence tolerance on the change in training loss between sweeps.
    max_poly_degree : int
        Maximum polynomial degree available to each component.
    include_transcendental : bool
        Whether to allow transcendental terms in each component.
    include_ratios : bool
        Whether to allow ratio terms in each component.
    strategy : str
        Selection strategy for each component.
    information_criterion : str
        Information criterion for complexity control.
    feature_names : list of str, optional
        Names for the input features.
    random_state : int, optional
        Seed for any stochastic sweep ordering.
    """

    def __init__(
        self,
        n_terms: int = 5,
        n_sweeps: int = 10,
        max_complexity: int = 4,
        loss: str = "squared_error",
        tol: float = 1e-6,
        max_poly_degree: int = 3,
        include_transcendental: bool = False,
        include_ratios: bool = False,
        strategy: str = "greedy_forward",
        information_criterion: str = "bic",
        feature_names: list[str] | None = None,
        random_state: int | None = None,
    ):
        self.n_terms = n_terms
        self.n_sweeps = n_sweeps
        self.max_complexity = max_complexity
        self.loss = loss
        self.tol = tol
        self.max_poly_degree = max_poly_degree
        self.include_transcendental = include_transcendental
        self.include_ratios = include_ratios
        self.strategy = strategy
        self.information_criterion = information_criterion
        self.feature_names = feature_names
        self.random_state = random_state

        self.model_ = None
        self._is_fitted = False

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> BackfittingSymbolicRegressor:
        """
        Fit the backfitting model (not yet implemented).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : BackfittingSymbolicRegressor
            The fitted estimator.

        Raises
        ------
        NotImplementedError
            Always -- this class is a scaffold for future work.
        """
        # TODO: initialise m terms (e.g. from a stagewise warm start), then
        #       loop sweeps: for each term, remove it, re-discover/refit it on
        #       the partial residual, and reinsert it; track loss for the
        #       convergence check against `tol` and cap iterations at n_sweeps.
        raise NotImplementedError(
            "BackfittingSymbolicRegressor is not implemented yet. "
            "Use StagewiseSymbolicRegressor for additive symbolic regression."
        )
