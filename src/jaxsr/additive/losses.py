"""
Loss functions for additive symbolic regression.

Additive (boosting-style) symbolic regression fits each new symbolic term to
the *pseudo-residual* of the current ensemble.  For squared error the
pseudo-residual is simply ``y - y_pred``, but the abstraction below leaves a
clean hook for gradient boosting with other differentiable losses, where the
base learner fits the negative gradient ``-dL/dy_pred``.

Only :class:`SquaredError` is implemented for the first milestone.  Future
losses (absolute error, Huber, Poisson, logistic, quantile) can be added by
subclassing :class:`Loss` and registering them in ``_LOSSES``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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


# Registry of available losses.  Add future losses here.
_LOSSES: dict[str, type[Loss]] = {
    "squared_error": SquaredError,
}


def get_loss(loss: str | Loss) -> Loss:
    """
    Resolve a loss name (or instance) to a :class:`Loss` instance.

    Parameters
    ----------
    loss : str or Loss
        Either a registered loss name (currently only ``"squared_error"``) or
        an already-constructed :class:`Loss` instance.

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
