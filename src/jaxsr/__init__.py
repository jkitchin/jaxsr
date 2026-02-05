"""
JAXSR: JAX-based Symbolic Regression

A fully open-source symbolic regression library built on JAX that discovers
interpretable algebraic expressions from data using sparse optimization techniques.

Inspired by ALAMO (Automated Learning of Algebraic Models for Optimization).
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "John Kitchin"

# Core classes
from .basis import BasisFunction, BasisLibrary
from .regressor import SymbolicRegressor, fit_symbolic
from .constraints import Constraints, Constraint, ConstraintType
from .sampling import AdaptiveSampler, SamplingStrategy

# Selection
from .selection import (
    SelectionResult,
    SelectionPath,
    greedy_forward_selection,
    greedy_backward_elimination,
    exhaustive_search,
    lasso_path_selection,
    compute_pareto_front,
)

# Metrics
from .metrics import (
    compute_aic,
    compute_bic,
    compute_aicc,
    compute_mse,
    compute_rmse,
    compute_mae,
    compute_r2,
    compute_all_metrics,
    cross_validate,
)

# Uncertainty Quantification
from .uncertainty import (
    compute_unbiased_variance,
    compute_coeff_covariance,
    coefficient_intervals,
    prediction_interval,
    ensemble_predict,
    BayesianModelAverage,
    conformal_predict_split,
    conformal_predict_jackknife_plus,
    bootstrap_coefficients,
    bootstrap_predict,
    bootstrap_model_selection,
)

# Simplification
from .simplify import simplify_expression, SimplificationResult

# Sampling utilities
from .sampling import (
    latin_hypercube_sample,
    sobol_sample,
    halton_sample,
    grid_sample,
)

# Plotting (optional import to avoid matplotlib dependency issues)
def _get_plotting():
    from . import plotting
    return plotting

__all__ = [
    # Version
    "__version__",
    # Core classes
    "BasisFunction",
    "BasisLibrary",
    "SymbolicRegressor",
    "fit_symbolic",
    "Constraints",
    "Constraint",
    "ConstraintType",
    "AdaptiveSampler",
    "SamplingStrategy",
    # Selection
    "SelectionResult",
    "SelectionPath",
    "greedy_forward_selection",
    "greedy_backward_elimination",
    "exhaustive_search",
    "lasso_path_selection",
    "compute_pareto_front",
    # Metrics
    "compute_aic",
    "compute_bic",
    "compute_aicc",
    "compute_mse",
    "compute_rmse",
    "compute_mae",
    "compute_r2",
    "compute_all_metrics",
    "cross_validate",
    # Uncertainty Quantification
    "compute_unbiased_variance",
    "compute_coeff_covariance",
    "coefficient_intervals",
    "prediction_interval",
    "ensemble_predict",
    "BayesianModelAverage",
    "conformal_predict_split",
    "conformal_predict_jackknife_plus",
    "bootstrap_coefficients",
    "bootstrap_predict",
    "bootstrap_model_selection",
    # Simplification
    "simplify_expression",
    "SimplificationResult",
    # Sampling
    "latin_hypercube_sample",
    "sobol_sample",
    "halton_sample",
    "grid_sample",
]
