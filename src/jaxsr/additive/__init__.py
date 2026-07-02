"""
Additive symbolic regression for JAXSR.

Fits models of the form ``f(x) = c + sum_k eta_k * g_k(x)``, where each
``g_k`` is a small symbolic expression discovered by the existing JAXSR
machinery.  This is analogous to gradient boosting, except each weak learner
is an interpretable symbolic expression rather than a decision tree.

Public API
----------
StagewiseSymbolicRegressor
    Boosting-style regressor that fits each new symbolic term to the current
    residual and freezes it.  This is the first-milestone workhorse.
BackfittingSymbolicRegressor
    Scaffold for a future BART/iBART-style regressor that revises terms in
    place (not yet implemented).
AdditiveSymbolicModel
    Plain container for a fitted additive model.
Loss, SquaredError, AbsoluteError, HuberLoss, QuantileLoss, get_loss
    Loss abstraction and registry. Squared error is the default; absolute
    error, Huber, and quantile (pinball) losses enable robust and quantile
    regression via gradient boosting.
refit_ols
    Least-squares refit of intercept and per-term coefficients.
"""

from __future__ import annotations

from .backfitting import BackfittingSymbolicRegressor
from .coefficient_refit import refit_ols
from .ensemble import AdditiveSymbolicModel, additive_predict
from .losses import (
    AbsoluteError,
    HuberLoss,
    Loss,
    QuantileLoss,
    SquaredError,
    get_loss,
    loss_from_config,
)
from .stagewise import StagewiseSymbolicRegressor

__all__ = [
    "AbsoluteError",
    "AdditiveSymbolicModel",
    "BackfittingSymbolicRegressor",
    "HuberLoss",
    "Loss",
    "QuantileLoss",
    "SquaredError",
    "StagewiseSymbolicRegressor",
    "additive_predict",
    "get_loss",
    "loss_from_config",
    "refit_ols",
]
