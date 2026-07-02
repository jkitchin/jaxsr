"""
Loss functions for additive symbolic regression.

Additive (boosting-style) symbolic regression fits each new symbolic term to
the *pseudo-residual* of the current ensemble.  For squared error the
pseudo-residual is simply ``y - y_pred``; for other differentiable losses it
is the negative gradient ``-dL/dy_pred``, which is what gradient boosting
fits.  This lets JAXSR learn symbolic models under losses that ordinary
least-squares selection cannot target directly:

* :class:`SquaredError` -- standard regression (mean).
* :class:`AbsoluteError` -- robust to outliers (median).
* :class:`HuberLoss` -- quadratic near zero, linear in the tails (robust).
* :class:`QuantileLoss` -- pinball loss for quantile / interval estimation.

New losses can be added by subclassing :class:`Loss` and registering them in
``_LOSSES``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp


class Loss(ABC):
    """
    Abstract base class for additive-regression loss functions.

    A concrete loss defines three things:

    * :meth:`initial_prediction` -- the constant that minimises the loss and
      is used to initialise the ensemble intercept.
    * :meth:`negative_gradient` -- the pseudo-residual each weak learner fits.
    * :meth:`loss` -- the scalar training/validation loss for reporting and
      early stopping.
    """

    #: Human-readable name used in the loss registry.
    name: str = "loss"

    @abstractmethod
    def initial_prediction(self, y: jnp.ndarray) -> float:
        """
        Return the optimal constant prediction for ``y``.

        Parameters
        ----------
        y : jnp.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        float
            Constant used to initialise the additive model intercept.
        """

    @abstractmethod
    def negative_gradient(self, y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
        """
        Return the pseudo-residual ``-dL/dy_pred`` the next term should fit.

        Parameters
        ----------
        y : jnp.ndarray of shape (n_samples,)
            Target values.
        y_pred : jnp.ndarray of shape (n_samples,)
            Current ensemble prediction.

        Returns
        -------
        jnp.ndarray of shape (n_samples,)
            Pseudo-residuals.
        """

    @abstractmethod
    def loss(self, y: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """
        Return the scalar loss between ``y`` and ``y_pred``.

        Parameters
        ----------
        y : jnp.ndarray of shape (n_samples,)
            Target values.
        y_pred : jnp.ndarray of shape (n_samples,)
            Predicted values.

        Returns
        -------
        float
            Scalar loss value.
        """

    def to_config(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable ``{"name", "params"}`` description.

        Returns
        -------
        dict
            ``name`` is the registry key; ``params`` are the constructor
            keyword arguments needed to rebuild this loss.
        """
        return {"name": self.name, "params": {}}

    def __repr__(self) -> str:
        params = self.to_config()["params"]
        inner = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{type(self).__name__}({inner})"


class SquaredError(Loss):
    """
    Squared-error loss ``L = mean((y - y_pred)**2)``.

    The optimal constant prediction is the mean of ``y`` and the
    pseudo-residual is the ordinary residual ``y - y_pred``.
    """

    name = "squared_error"

    def initial_prediction(self, y: jnp.ndarray) -> float:
        """Return ``mean(y)``."""
        return float(jnp.mean(jnp.asarray(y)))

    def negative_gradient(self, y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
        """Return the residual ``y - y_pred``."""
        return jnp.asarray(y) - jnp.asarray(y_pred)

    def loss(self, y: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """Return the mean squared error."""
        y = jnp.asarray(y)
        y_pred = jnp.asarray(y_pred)
        return float(jnp.mean((y - y_pred) ** 2))


class AbsoluteError(Loss):
    """
    Absolute-error (L1) loss ``L = mean(|y - y_pred|)``.

    Robust to outliers.  The optimal constant is the median and the negative
    gradient is ``sign(y - y_pred)``.
    """

    name = "absolute_error"

    def initial_prediction(self, y: jnp.ndarray) -> float:
        """Return ``median(y)``."""
        return float(jnp.median(jnp.asarray(y)))

    def negative_gradient(self, y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
        """Return ``sign(y - y_pred)``."""
        return jnp.sign(jnp.asarray(y) - jnp.asarray(y_pred))

    def loss(self, y: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """Return the mean absolute error."""
        return float(jnp.mean(jnp.abs(jnp.asarray(y) - jnp.asarray(y_pred))))


class HuberLoss(Loss):
    """
    Huber loss: quadratic for small residuals, linear beyond ``delta``.

    Combines the efficiency of squared error near zero with the robustness of
    absolute error in the tails.

    Parameters
    ----------
    delta : float
        Threshold at which the loss transitions from quadratic to linear.
        Must be positive.

    Raises
    ------
    ValueError
        If ``delta`` is not positive.
    """

    name = "huber"

    def __init__(self, delta: float = 1.35):
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}.")
        self.delta = float(delta)

    def initial_prediction(self, y: jnp.ndarray) -> float:
        """Return ``median(y)`` (a robust location estimate)."""
        return float(jnp.median(jnp.asarray(y)))

    def negative_gradient(self, y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
        """Return ``r`` where ``|r| <= delta`` else ``delta * sign(r)``."""
        r = jnp.asarray(y) - jnp.asarray(y_pred)
        return jnp.where(jnp.abs(r) <= self.delta, r, self.delta * jnp.sign(r))

    def loss(self, y: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """Return the mean Huber loss."""
        r = jnp.asarray(y) - jnp.asarray(y_pred)
        abs_r = jnp.abs(r)
        quad = 0.5 * r**2
        lin = self.delta * (abs_r - 0.5 * self.delta)
        return float(jnp.mean(jnp.where(abs_r <= self.delta, quad, lin)))

    def to_config(self) -> dict[str, Any]:
        """Return ``{"name": "huber", "params": {"delta": delta}}``."""
        return {"name": self.name, "params": {"delta": self.delta}}


class QuantileLoss(Loss):
    """
    Quantile (pinball) loss for estimating the ``quantile``-th conditional
    quantile of the target.

    Useful for asymmetric costs and for building prediction intervals (fit one
    model per quantile).

    Parameters
    ----------
    quantile : float
        Target quantile in the open interval ``(0, 1)``.

    Raises
    ------
    ValueError
        If ``quantile`` is not in ``(0, 1)``.
    """

    name = "quantile"

    def __init__(self, quantile: float = 0.5):
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}.")
        self.quantile = float(quantile)

    def initial_prediction(self, y: jnp.ndarray) -> float:
        """Return the empirical ``quantile``-th quantile of ``y``."""
        return float(jnp.quantile(jnp.asarray(y), self.quantile))

    def negative_gradient(self, y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
        """Return ``q`` where ``y > y_pred`` else ``q - 1``."""
        r = jnp.asarray(y) - jnp.asarray(y_pred)
        return jnp.where(r > 0, self.quantile, self.quantile - 1.0)

    def loss(self, y: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """Return the mean pinball loss."""
        r = jnp.asarray(y) - jnp.asarray(y_pred)
        return float(jnp.mean(jnp.maximum(self.quantile * r, (self.quantile - 1.0) * r)))

    def to_config(self) -> dict[str, Any]:
        """Return ``{"name": "quantile", "params": {"quantile": quantile}}``."""
        return {"name": self.name, "params": {"quantile": self.quantile}}


# Registry of available losses.  Add future losses here.
_LOSSES: dict[str, type[Loss]] = {
    "squared_error": SquaredError,
    "absolute_error": AbsoluteError,
    "huber": HuberLoss,
    "quantile": QuantileLoss,
}


def get_loss(loss: str | Loss) -> Loss:
    """
    Resolve a loss name (or instance) to a :class:`Loss` instance.

    Parameters
    ----------
    loss : str or Loss
        Either a registered loss name (``"squared_error"``, ``"absolute_error"``,
        ``"huber"``, ``"quantile"``) or an already-constructed :class:`Loss`
        instance.  Names build losses with their default parameters; pass an
        instance (e.g. ``QuantileLoss(0.9)``) to customise.

    Returns
    -------
    Loss
        A loss instance.

    Raises
    ------
    ValueError
        If ``loss`` is a string that is not a registered loss name.
    TypeError
        If ``loss`` is neither a string nor a :class:`Loss` instance.
    """
    if isinstance(loss, Loss):
        return loss
    if isinstance(loss, str):
        if loss not in _LOSSES:
            raise ValueError(f"Unknown loss {loss!r}. Available losses: {sorted(_LOSSES)}.")
        return _LOSSES[loss]()
    raise TypeError(f"loss must be a str or Loss instance, got {type(loss).__name__}.")


def loss_from_config(config: str | dict[str, Any] | Loss) -> Loss:
    """
    Rebuild a :class:`Loss` from a name, an instance, or a ``to_config`` dict.

    Parameters
    ----------
    config : str or dict or Loss
        A registry name, a ``{"name", "params"}`` dictionary produced by
        :meth:`Loss.to_config`, or a :class:`Loss` instance.

    Returns
    -------
    Loss
        A loss instance.

    Raises
    ------
    ValueError
        If the config names an unknown loss.
    TypeError
        If ``config`` is not a str, dict, or :class:`Loss`.
    """
    if isinstance(config, (str, Loss)):
        return get_loss(config)
    if isinstance(config, dict):
        name = config["name"]
        if name not in _LOSSES:
            raise ValueError(f"Unknown loss {name!r}. Available losses: {sorted(_LOSSES)}.")
        return _LOSSES[name](**config.get("params", {}))
    raise TypeError(f"config must be a str, dict, or Loss, got {type(config).__name__}.")
