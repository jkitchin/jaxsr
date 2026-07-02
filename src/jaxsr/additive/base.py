"""
Shared base class for additive symbolic regressors.

Both :class:`~jaxsr.additive.stagewise.StagewiseSymbolicRegressor` and
:class:`~jaxsr.additive.backfitting.BackfittingSymbolicRegressor` produce an
:class:`~jaxsr.additive.ensemble.AdditiveSymbolicModel` and share the same
fitted-attribute accessors, prediction, interpretation, and JSON
serialization.  Only the fitting strategy differs, so everything except
``fit`` lives here.
"""

from __future__ import annotations

import json

import jax.numpy as jnp

from .._compat import _SklearnCompatMixin
from ..regressor import SymbolicRegressor
from .ensemble import AdditiveSymbolicModel
from .losses import get_loss, loss_from_config


class _BaseAdditiveRegressor(_SklearnCompatMixin):
    """
    Common machinery for additive symbolic regressors.

    Subclasses must:

    * store their constructor parameters as same-named attributes (including a
      ``loss`` attribute),
    * implement ``fit`` to populate ``self.model_`` with an
      :class:`AdditiveSymbolicModel` and set ``self._is_fitted = True``.
    """

    model_: AdditiveSymbolicModel | None
    _is_fitted: bool

    # ------------------------------------------------------------------
    # Fitted-attribute accessors
    # ------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        """Raise if the model has not been fitted yet."""
        if not getattr(self, "_is_fitted", False) or self.model_ is None:
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
        """Per-term learning rate / step scale recorded during fitting."""
        self._check_is_fitted()
        return self.model_.learning_rates

    @property
    def training_history_(self) -> list[dict]:
        """Per-iteration diagnostics."""
        self._check_is_fitted()
        return self.model_.training_history

    @property
    def n_terms_(self) -> int:
        """Number of terms in the fitted model."""
        self._check_is_fitted()
        return self.model_.n_terms

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

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def _state_dict(self) -> dict:
        """
        Return a JSON-serialisable dictionary of the fitted model state.

        Each symbolic term is serialised via the underlying
        :class:`jaxsr.SymbolicRegressor` state dictionary, which avoids the
        (unpicklable) basis-function closures.

        Returns
        -------
        dict
            Dictionary containing constructor config and fitted state.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_is_fitted()
        config = self.get_params(deep=False)
        # Serialise the loss faithfully: a plain name when it has no parameters,
        # otherwise a {"name", "params"} dict (e.g. quantile / huber).
        loss_config = get_loss(self.loss).to_config()
        config["loss"] = loss_config["name"] if not loss_config["params"] else loss_config
        return {
            "config": config,
            "intercept": float(self.model_.intercept),
            "coefficients": [float(c) for c in self.model_.coefficients],
            "learning_rates": [float(lr) for lr in self.model_.learning_rates],
            "feature_names": list(self.model_.feature_names),
            "training_history": self.model_.training_history,
            "terms": [term._state_dict() for term in self.model_.terms],
        }

    @classmethod
    def _from_dict(cls, data: dict) -> _BaseAdditiveRegressor:
        """
        Reconstruct a fitted regressor from a state dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`_state_dict`.

        Returns
        -------
        _BaseAdditiveRegressor
            The reconstructed fitted model (an instance of ``cls``).
        """
        config = dict(data["config"])
        config["loss"] = loss_from_config(config["loss"])
        model = cls(**config)
        terms = [SymbolicRegressor._from_dict(t) for t in data["terms"]]
        model.model_ = AdditiveSymbolicModel(
            intercept=float(data["intercept"]),
            terms=terms,
            coefficients=[float(c) for c in data["coefficients"]],
            learning_rates=[float(lr) for lr in data["learning_rates"]],
            feature_names=list(data["feature_names"]),
            training_history=data.get("training_history", []),
        )
        model._is_fitted = True
        return model

    def save(self, filepath: str) -> None:
        """
        Save the fitted model to a JSON file.

        Parameters
        ----------
        filepath : str
            Destination path.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        with open(filepath, "w") as f:
            json.dump(self._state_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> _BaseAdditiveRegressor:
        """
        Load a fitted model from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to a file created by :meth:`save`.

        Returns
        -------
        _BaseAdditiveRegressor
            The loaded fitted model (an instance of ``cls``).
        """
        with open(filepath) as f:
            data = json.load(f)
        return cls._from_dict(data)

    def __repr__(self) -> str:
        """Pretty structural repr when fitted, sklearn-style otherwise."""
        if getattr(self, "_is_fitted", False) and self.model_ is not None:
            return self.model_.describe(name=type(self).__name__)
        return _SklearnCompatMixin.__repr__(self)
