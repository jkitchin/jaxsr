"""
Stagewise (boosting-style) additive symbolic regression.

:class:`StagewiseSymbolicRegressor` builds a model of the form::

    f(x) = intercept + sum_k coefficients[k] * g_k(x)

by repeatedly fitting a small symbolic expression ``g_k`` to the current
residual (pseudo-residual for general losses).  This is conceptually
"gradient boosting, but the weak learners are symbolic expressions instead of
trees".

Once discovered, a term is *frozen* -- its internal structure never changes.
Only the linear weights may be re-estimated (see ``refit_coefficients``).  The
future :class:`~jaxsr.additive.backfitting.BackfittingSymbolicRegressor` will
instead revise terms in place.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .._compat import _SklearnCompatMixin
from ..regressor import SymbolicRegressor, fit_symbolic
from .coefficient_refit import refit_ols
from .ensemble import AdditiveSymbolicModel, additive_predict
from .losses import get_loss


class StagewiseSymbolicRegressor(_SklearnCompatMixin):
    """
    Stagewise additive symbolic regression (symbolic gradient boosting).

    Fits an additive ensemble of small symbolic expressions by iteratively
    fitting each new expression to the residual of the current model.  Old
    terms are frozen; optionally the linear coefficients over all discovered
    terms are refit by least squares after each stage.

    Parameters
    ----------
    n_terms : int
        Maximum number of boosting stages (symbolic terms) to add.
    learning_rate : float
        Shrinkage applied to each stage's contribution when
        ``refit_coefficients=False``.  Ignored when ``refit_coefficients=True``
        (the weights are then chosen by least squares), but still recorded.
    max_complexity : int
        Complexity budget for each stage's symbolic expression, expressed as
        the maximum number of basis terms (passed as ``max_terms`` to the
        underlying :func:`jaxsr.fit_symbolic`).  Keep this small to favour many
        simple, interpretable terms over one large expression.
    refit_coefficients : bool
        If True, after each new term is added, re-solve the intercept and all
        per-term coefficients by ordinary least squares over the discovered
        symbolic features.  If False, use fixed learning-rate-scaled stagewise
        weights.
    loss : str
        Loss function name.  Currently only ``"squared_error"`` is supported.
    early_stopping : bool
        If True, hold out a validation split and stop adding terms once the
        validation loss stops improving.
    validation_fraction : float
        Fraction of the training data held out for early-stopping validation.
        Only used when ``early_stopping=True``.
    patience : int
        Number of consecutive non-improving stages tolerated before stopping.
    min_delta : float
        Minimum decrease in validation loss to count as an improvement.
    max_poly_degree : int
        Maximum polynomial degree available to each stage.
    include_transcendental : bool
        If True, allow ``log``/``exp``/``sqrt``/``inv`` terms in each stage.
    include_ratios : bool
        If True, allow ratio terms ``x_i / x_j`` in each stage.
    strategy : str
        Selection strategy for each stage (see :class:`jaxsr.SymbolicRegressor`).
    information_criterion : str
        Information criterion used to control complexity within each stage:
        ``"aic"``, ``"aicc"``, or ``"bic"``.
    feature_names : list of str, optional
        Names for the input features.  Defaults to ``["x0", "x1", ...]``.
    random_state : int, optional
        Seed controlling the early-stopping validation split.

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
    learning_rates_ : list of float
        Learning rate recorded at each stage.
    training_history_ : list of dict
        Per-stage diagnostics.
    n_terms_ : int
        Number of terms in the fitted model.

    Examples
    --------
    >>> import numpy as np
    >>> from jaxsr.additive import StagewiseSymbolicRegressor
    >>> X = np.random.randn(200, 2)
    >>> y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2
    >>> model = StagewiseSymbolicRegressor(n_terms=5, refit_coefficients=True)
    >>> model.fit(X, y)  # doctest: +SKIP
    >>> print(model)  # doctest: +SKIP
    """

    def __init__(
        self,
        n_terms: int = 10,
        learning_rate: float = 0.1,
        max_complexity: int = 4,
        refit_coefficients: bool = True,
        loss: str = "squared_error",
        early_stopping: bool = False,
        validation_fraction: float = 0.2,
        patience: int = 3,
        min_delta: float = 1e-8,
        max_poly_degree: int = 3,
        include_transcendental: bool = False,
        include_ratios: bool = False,
        strategy: str = "greedy_forward",
        information_criterion: str = "bic",
        feature_names: list[str] | None = None,
        random_state: int | None = None,
    ):
        self.n_terms = n_terms
        self.learning_rate = learning_rate
        self.max_complexity = max_complexity
        self.refit_coefficients = refit_coefficients
        self.loss = loss
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.max_poly_degree = max_poly_degree
        self.include_transcendental = include_transcendental
        self.include_ratios = include_ratios
        self.strategy = strategy
        self.information_criterion = information_criterion
        self.feature_names = feature_names
        self.random_state = random_state

        # Fitted attributes
        self.model_: AdditiveSymbolicModel | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fitted-attribute accessors
    # ------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        """Raise if the model has not been fitted yet."""
        if not self._is_fitted or self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    @property
    def intercept_(self) -> float:
        """Fitted intercept."""
        self._check_is_fitted()
        return self.model_.intercept

    @property
    def coefficients_(self) -> list[float]:
        """Fitted per-term coefficients."""
        self._check_is_fitted()
        return self.model_.coefficients

    @property
    def expressions_(self) -> list[str]:
        """Human-readable expression for each term."""
        self._check_is_fitted()
        return self.model_.expressions

    @property
    def terms_(self) -> list[SymbolicRegressor]:
        """The fitted symbolic terms."""
        self._check_is_fitted()
        return self.model_.terms

    @property
    def learning_rates_(self) -> list[float]:
        """Learning rate recorded at each stage."""
        self._check_is_fitted()
        return self.model_.learning_rates

    @property
    def training_history_(self) -> list[dict]:
        """Per-stage diagnostics."""
        self._check_is_fitted()
        return self.model_.training_history

    @property
    def n_terms_(self) -> int:
        """Number of terms in the fitted model."""
        self._check_is_fitted()
        return self.model_.n_terms

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def _validate_params(self) -> None:
        """Validate constructor parameters."""
        if self.n_terms < 1:
            raise ValueError(f"n_terms must be >= 1, got {self.n_terms}.")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if self.max_complexity < 1:
            raise ValueError(f"max_complexity must be >= 1, got {self.max_complexity}.")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}.")
        if self.early_stopping and not (0.0 < self.validation_fraction < 1.0):
            raise ValueError(
                f"validation_fraction must be in (0, 1), got {self.validation_fraction}."
            )

    def _fit_stage(
        self,
        X: jnp.ndarray,
        residual: jnp.ndarray,
        feature_names: list[str],
    ) -> SymbolicRegressor:
        """
        Fit a single symbolic term to the residual using existing machinery.

        Parameters
        ----------
        X : jnp.ndarray of shape (n_samples, n_features)
            Training inputs.
        residual : jnp.ndarray of shape (n_samples,)
            Pseudo-residual target for this stage.
        feature_names : list of str
            Feature names shared across stages.

        Returns
        -------
        SymbolicRegressor
            A fitted symbolic regressor representing ``g_k``.
        """
        return fit_symbolic(
            X,
            residual,
            feature_names=feature_names,
            max_terms=self.max_complexity,
            max_poly_degree=self.max_poly_degree,
            include_transcendental=self.include_transcendental,
            include_ratios=self.include_ratios,
            strategy=self.strategy,
            information_criterion=self.information_criterion,
        )

    def _train_val_split(
        self, X: jnp.ndarray, y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
        """Split off a validation set for early stopping (if enabled)."""
        if not self.early_stopping:
            return X, y, None, None

        n = X.shape[0]
        n_val = max(1, int(round(n * self.validation_fraction)))
        if n_val >= n:
            raise ValueError("validation_fraction too large: no training samples remain.")

        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)
        val_idx = jnp.asarray(perm[:n_val])
        train_idx = jnp.asarray(perm[n_val:])
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> StagewiseSymbolicRegressor:
        """
        Fit the stagewise additive symbolic model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : StagewiseSymbolicRegressor
            The fitted estimator.

        Raises
        ------
        ValueError
            If parameters are invalid, ``X`` and ``y`` have mismatched sample
            counts, or ``X``/``y`` contain non-finite values.
        """
        self._validate_params()

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

        loss_fn = get_loss(self.loss)
        X_tr, y_tr, X_val, y_val = self._train_val_split(X, y)

        intercept = loss_fn.initial_prediction(y_tr)
        terms: list[SymbolicRegressor] = []
        coefficients: list[float] = []
        learning_rates: list[float] = []
        history: list[dict] = []

        prediction_tr = jnp.full((X_tr.shape[0],), intercept)

        best_val = np.inf
        best_iter = -1
        n_no_improve = 0

        for k in range(self.n_terms):
            residual = loss_fn.negative_gradient(y_tr, prediction_tr)
            term = self._fit_stage(X_tr, residual, feature_names)
            terms.append(term)
            learning_rates.append(self.learning_rate)

            if self.refit_coefficients:
                Phi_tr = jnp.stack([t.predict(X_tr) for t in terms], axis=1)
                intercept, coef_arr = refit_ols(Phi_tr, y_tr)
                coefficients = [float(c) for c in coef_arr]
                prediction_tr = intercept + Phi_tr @ coef_arr
            else:
                coefficients.append(self.learning_rate)
                prediction_tr = prediction_tr + self.learning_rate * term.predict(X_tr)

            train_loss = loss_fn.loss(y_tr, prediction_tr)

            val_loss = None
            if X_val is not None:
                val_pred = additive_predict(X_val, intercept, terms, coefficients)
                val_loss = loss_fn.loss(y_val, val_pred)

            history.append(
                {
                    "n_terms": k + 1,
                    "train_loss": float(train_loss),
                    "val_loss": None if val_loss is None else float(val_loss),
                    "intercept": float(intercept),
                    "coefficients": [float(c) for c in coefficients],
                }
            )

            # Early stopping bookkeeping
            if val_loss is not None:
                if val_loss < best_val - self.min_delta:
                    best_val = val_loss
                    best_iter = k
                    n_no_improve = 0
                else:
                    n_no_improve += 1
                    if n_no_improve >= self.patience:
                        break

        # Roll back to the best validation iteration if early stopping was used.
        if self.early_stopping and best_iter >= 0 and best_iter < len(terms) - 1:
            best_snapshot = history[best_iter]
            terms = terms[: best_iter + 1]
            learning_rates = learning_rates[: best_iter + 1]
            coefficients = list(best_snapshot["coefficients"])
            intercept = float(best_snapshot["intercept"])

        self.model_ = AdditiveSymbolicModel(
            intercept=float(intercept),
            terms=terms,
            coefficients=coefficients,
            learning_rates=learning_rates,
            feature_names=feature_names,
            training_history=history,
        )
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction / interpretation
    # ------------------------------------------------------------------
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict with the fitted additive model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        jnp.ndarray of shape (n_samples,)
            Predicted values.
        """
        self._check_is_fitted()
        return self.model_.predict(X)

    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Compute the R^2 score on ``(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        float
            R^2 score.
        """
        self._check_is_fitted()
        y = jnp.asarray(y).ravel()
        y_pred = self.predict(X)
        ss_res = jnp.sum((y - y_pred) ** 2)
        ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-10))

    def to_expression(self):
        """
        Return the combined model as a single simplified SymPy expression.

        Returns
        -------
        sympy.Expr
            Combined symbolic expression for the whole ensemble.
        """
        self._check_is_fitted()
        return self.model_.to_expression()

    def __repr__(self) -> str:
        """Pretty structural repr when fitted, sklearn-style otherwise."""
        if self._is_fitted and self.model_ is not None:
            return self.model_.describe(name=type(self).__name__)
        return _SklearnCompatMixin.__repr__(self)
