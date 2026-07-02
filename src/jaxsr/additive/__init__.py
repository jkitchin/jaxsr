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
Loss, SquaredError, get_loss
    Loss abstraction and registry (hooks for future gradient-boosting losses).
refit_ols
    Least-squares refit of intercept and per-term coefficients.
"""

from __future__ import annotations

from .backfitting import BackfittingSymbolicRegressor
from .coefficient_refit import refit_ols
from .ensemble import AdditiveSymbolicModel, additive_predict
from .losses import Loss, SquaredError, get_loss
from .stagewise import StagewiseSymbolicRegressor

__all__ = [
    "AdditiveSymbolicModel",
    "BackfittingSymbolicRegressor",
    "Loss",
    "SquaredError",
    "StagewiseSymbolicRegressor",
    "additive_predict",
    "get_loss",
    "refit_ols",
]
