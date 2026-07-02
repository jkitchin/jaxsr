"""
Backfitting additive symbolic regression (GAM-style).

Unlike :class:`~jaxsr.additive.stagewise.StagewiseSymbolicRegressor`, where each
discovered term is *frozen*, the backfitting regressor maintains a fixed number
of terms and repeatedly *revises* each one.  A "sweep" visits every term in
turn, removes it from the ensemble, and re-discovers its symbolic expression on
the *partial residual* (the target minus every other term's contribution)::

    for sweep in 1..n_sweeps:
        for term j:
            partial_residual = y - intercept - sum_{i != j} coef_i * g_i(X)
            g_j = fit_symbolic(X, partial_residual, ...)   # re-discover structure
        intercept, coef = OLS refit over all terms

This lets early terms -- originally fit against a residual still polluted by
effects that had not yet been discovered -- clean themselves up once the other
terms are in place.  It is the classic backfitting algorithm behind generalized
additive models, with symbolic expressions as the smoothers.

The regressor is warm-started from a stagewise fit, so a single ``fit`` call
first runs stagewise boosting and then refines it by backfitting.

Scope
-----
This is the deterministic, squared-error version.  A future Bayesian variant
(BART/iBART-style, sampling a posterior over symbolic structures) can build on
the same partial-residual sweep; see the project notes.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..regressor import fit_symbolic
from .base import _BaseAdditiveRegressor
from .coefficient_refit import refit_ols
from .ensemble import AdditiveSymbolicModel
from .losses import SquaredError, get_loss
from .stagewise import StagewiseSymbolicRegressor


class BackfittingSymbolicRegressor(_BaseAdditiveRegressor):
    """
    Backfitting additive symbolic regression (GAM-style).

    Maintains ``n_terms`` symbolic components and revises each one in place
    across repeated sweeps, conditioning on the current fit of all other terms.
    Contrast with :class:`~jaxsr.additive.stagewise.StagewiseSymbolicRegressor`,
    which freezes terms once discovered.

    The model is warm-started with a stagewise fit and then refined by
    backfitting.  Only squared-error loss is supported.

    Parameters
    ----------
    n_terms : int
        Number of symbolic components to maintain and revise.
    n_sweeps : int
        Maximum number of backfitting sweeps over the terms.
    max_complexity : int
        Complexity budget (max basis terms) for each component.
    loss : str
        Loss function.  Only ``"squared_error"`` is supported.
    tol : float
        Convergence tolerance: stop when the training MSE improves by less than
        ``tol`` between consecutive sweeps.
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
        Seed forwarded to the stagewise warm start.

    Attributes
    ----------
    model_ : AdditiveSymbolicModel
        The fitted additive model.
    intercept_ : float
        Fitted intercept.
    coefficients_ : list of float
        Fitted per-term coefficients.
    expressions_ : list of str
        Human-readable expression for each term.
    terms_ : list of SymbolicRegressor
        The fitted symbolic terms.
    training_history_ : list of dict
        Per-sweep diagnostics (``sweep`` index and ``train_loss``).
    n_terms_ : int
        Number of terms in the fitted model.

    Examples
    --------
    >>> from jaxsr.additive import BackfittingSymbolicRegressor
    >>> model = BackfittingSymbolicRegressor(n_terms=3, n_sweeps=5)
    >>> model.fit(X, y)  # doctest: +SKIP
    >>> print(model)  # doctest: +SKIP
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

        self.model_: AdditiveSymbolicModel | None = None
        self._is_fitted = False

    def _validate_params(self) -> None:
        """Validate constructor parameters."""
        if self.n_terms < 1:
            raise ValueError(f"n_terms must be >= 1, got {self.n_terms}.")
        if self.n_sweeps < 1:
            raise ValueError(f"n_sweeps must be >= 1, got {self.n_sweeps}.")
        if self.max_complexity < 1:
            raise ValueError(f"max_complexity must be >= 1, got {self.max_complexity}.")

    def _fit_term(
        self,
        X: jnp.ndarray,
        target: jnp.ndarray,
        feature_names: list[str],
    ):
        """Discover a single symbolic term for ``target`` (reuses fit_symbolic)."""
        return fit_symbolic(
            X,
            target,
            feature_names=feature_names,
            max_terms=self.max_complexity,
            max_poly_degree=self.max_poly_degree,
            include_transcendental=self.include_transcendental,
            include_ratios=self.include_ratios,
            strategy=self.strategy,
            information_criterion=self.information_criterion,
        )

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> BackfittingSymbolicRegressor:
        """
        Fit the backfitting additive model.

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
        ValueError
            If parameters are invalid, ``X``/``y`` are mismatched or non-finite.
        NotImplementedError
            If ``loss`` is not ``"squared_error"``.
        """
        self._validate_params()
        loss_fn = get_loss(self.loss)
        if not isinstance(loss_fn, SquaredError):
            raise NotImplementedError(
                "BackfittingSymbolicRegressor currently supports only "
                f"loss='squared_error', got {loss_fn.name!r}. Use "
                "StagewiseSymbolicRegressor for other losses."
            )

        X = jnp.atleast_2d(jnp.asarray(X))
        y = jnp.asarray(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}."
            )
        if not bool(jnp.all(jnp.isfinite(X))):
            raise ValueError("X contains non-finite values (NaN or inf).")
        if not bool(jnp.all(jnp.isfinite(y))):
            raise ValueError("y contains non-finite values (NaN or inf).")

        n_features = X.shape[1]
        feature_names = self.feature_names or [f"x{i}" for i in range(n_features)]
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but X has "
                f"{n_features} features."
            )

        # ------------------------------------------------------------------
        # Warm start: a stagewise fit with the same per-term budget provides
        # the initial set of terms that backfitting then refines.
        # ------------------------------------------------------------------
        warm = StagewiseSymbolicRegressor(
            n_terms=self.n_terms,
            max_complexity=self.max_complexity,
            refit_coefficients=True,
            loss="squared_error",
            max_poly_degree=self.max_poly_degree,
            include_transcendental=self.include_transcendental,
            include_ratios=self.include_ratios,
            strategy=self.strategy,
            information_criterion=self.information_criterion,
            feature_names=feature_names,
            random_state=self.random_state,
        ).fit(X, y)

        terms = list(warm.terms_)
        intercept = float(warm.intercept_)
        coefficients = list(warm.coefficients_)

        def full_prediction() -> jnp.ndarray:
            pred = jnp.full((X.shape[0],), intercept)
            for c, t in zip(coefficients, terms, strict=False):
                pred = pred + float(c) * t.predict(X)
            return pred

        history: list[dict] = []
        prev_loss = loss_fn.loss(y, full_prediction())
        history.append({"sweep": 0, "train_loss": float(prev_loss)})

        # Snapshot of the best (lowest-loss) iterate.  Structure re-discovery
        # makes the sweep a heuristic with no monotonicity guarantee, so we
        # keep the best state rather than trusting the final one.
        best_loss = prev_loss
        best_state = (intercept, list(coefficients), list(terms))

        # ------------------------------------------------------------------
        # Backfitting sweeps
        # ------------------------------------------------------------------
        for sweep in range(1, self.n_sweeps + 1):
            for j in range(len(terms)):
                # Partial residual: target minus every OTHER term's contribution.
                partial = y - intercept
                for i, (c, t) in enumerate(zip(coefficients, terms, strict=False)):
                    if i != j:
                        partial = partial - float(c) * t.predict(X)

                terms[j] = self._fit_term(X, partial, feature_names)

                # Re-solve intercept and all coefficients jointly (stable OLS).
                Phi = jnp.stack([t.predict(X) for t in terms], axis=1)
                intercept, coef_arr = refit_ols(Phi, y)
                coefficients = [float(c) for c in coef_arr]

            sweep_loss = loss_fn.loss(y, full_prediction())
            history.append({"sweep": sweep, "train_loss": float(sweep_loss)})

            if sweep_loss < best_loss:
                best_loss = sweep_loss
                best_state = (intercept, list(coefficients), list(terms))

            if abs(prev_loss - sweep_loss) < self.tol:
                break
            prev_loss = sweep_loss

        intercept, coefficients, terms = best_state
        self.model_ = AdditiveSymbolicModel(
            intercept=float(intercept),
            terms=terms,
            coefficients=coefficients,
            learning_rates=[1.0] * len(terms),
            feature_names=feature_names,
            training_history=history,
        )
        self._is_fitted = True
        return self
