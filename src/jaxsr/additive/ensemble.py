"""
Core additive symbolic model representation.

An additive symbolic model has the form::

    f(x) = intercept + sum_j coefficients[j] * terms[j](x)

where each ``terms[j]`` is a small symbolic expression (a fitted
:class:`jaxsr.SymbolicRegressor`) discovered by the existing JAXSR machinery.
This is analogous to gradient boosting, except each weak learner is an
interpretable symbolic expression rather than a decision tree.

The :class:`AdditiveSymbolicModel` is a plain data container: the fitting
strategy (stagewise, backfitting, ...) lives in the regressor classes and
produces one of these objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

if TYPE_CHECKING:
    from ..regressor import SymbolicRegressor


def additive_predict(
    X: jnp.ndarray,
    intercept: float,
    terms: list[SymbolicRegressor],
    coefficients: list[float] | jnp.ndarray,
) -> jnp.ndarray:
    """
    Evaluate an additive model ``intercept + sum_j coef_j * term_j(X)``.

    Parameters
    ----------
    X : jnp.ndarray of shape (n_samples, n_features)
        Input data.
    intercept : float
        Additive intercept.
    terms : list of SymbolicRegressor
        Fitted symbolic terms ``g_j``.
    coefficients : list of float or jnp.ndarray
        Per-term coefficients, aligned with ``terms``.

    Returns
    -------
    jnp.ndarray of shape (n_samples,)
        Predicted values.
    """
    X = jnp.atleast_2d(jnp.asarray(X))
    prediction = jnp.full((X.shape[0],), float(intercept))
    for coef, term in zip(coefficients, terms, strict=False):
        prediction = prediction + float(coef) * term.predict(X)
    return prediction


@dataclass(repr=False)
class AdditiveSymbolicModel:
    """
    Container for an additive symbolic model.

    Prediction is ``intercept + sum_j coefficients[j] * terms[j](X)``.

    Parameters
    ----------
    intercept : float
        Additive intercept ``c``.
    terms : list of SymbolicRegressor
        Fitted symbolic expressions ``g_j`` (the boosting "weak learners").
    coefficients : list of float
        Per-term weights ``eta_j``.  When coefficients are not refit these are
        the learning-rate-scaled stagewise weights; when refit they are the
        least-squares solution over all discovered terms.
    learning_rates : list of float
        Learning rate used at each stage (recorded for reproducibility).
    feature_names : list of str
        Feature names, shared across all terms.
    training_history : list of dict
        Per-stage diagnostics (train loss, validation loss, coefficients, ...).
    """

    intercept: float
    terms: list[SymbolicRegressor]
    coefficients: list[float]
    learning_rates: list[float]
    feature_names: list[str]
    training_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_terms(self) -> int:
        """Number of symbolic terms in the model."""
        return len(self.terms)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict with the additive model.

        Parameters
        ----------
        X : jnp.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        jnp.ndarray of shape (n_samples,)
            Predicted values.

        Raises
        ------
        ValueError
            If ``X`` has a different number of features than the model was
            fit with.
        """
        X = jnp.atleast_2d(jnp.asarray(X))
        n_expected = len(self.feature_names)
        if X.shape[1] != n_expected:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was fit with {n_expected}."
            )
        return additive_predict(X, self.intercept, self.terms, self.coefficients)

    @property
    def expressions(self) -> list[str]:
        """Human-readable expression string for each symbolic term."""
        return [term.expression_ for term in self.terms]

    def to_expression(self):
        """
        Combine all terms into a single simplified SymPy expression.

        Returns
        -------
        sympy.Expr
            ``intercept + sum_j coefficients[j] * g_j`` after simplification.
            Falls back to the unsimplified sum if simplification fails.

        Raises
        ------
        ImportError
            If SymPy is not installed.
        """
        import sympy

        expr = sympy.Float(float(self.intercept))
        for coef, term in zip(self.coefficients, self.terms, strict=False):
            expr = expr + sympy.Float(float(coef)) * term.to_sympy()
        try:
            return sympy.simplify(expr)
        except (TypeError, ValueError, AttributeError):
            return expr

    def describe(self, name: str = "AdditiveSymbolicModel") -> str:
        """
        Return a multi-line human-readable summary of the model.

        Parameters
        ----------
        name : str
            Class/label to show as the heading.

        Returns
        -------
        str
            Pretty-printed model structure.
        """
        lines = [f"{name}("]
        lines.append(f"    intercept = {float(self.intercept):.4g}")
        if self.terms:
            lines.append("    terms =")
            for coef, term in zip(self.coefficients, self.terms, strict=False):
                coef = float(coef)
                sign = "+" if coef >= 0 else "-"
                lines.append(f"        {sign} {abs(coef):.4g} * ({term.expression_})")
        else:
            lines.append("    terms = (none)")
        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Pretty multi-line representation."""
        return self.describe()
