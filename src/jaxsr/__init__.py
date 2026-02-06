"""
JAXSR: JAX-based Symbolic Regression

A fully open-source symbolic regression library built on JAX that discovers
interpretable algebraic expressions from data using sparse optimization techniques.

Inspired by ALAMO (Automated Learning of Algebraic Models for Optimization).
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "John Kitchin"

# Active Learning & Acquisition Functions
from .acquisition import (
    LCB,
    UCB,
    AcquisitionFunction,
    AcquisitionResult,
    ActiveLearner,
    AOptimal,
    BMAUncertainty,
    Composite,
    ConfidenceBandWidth,
    DOptimal,
    EnsembleDisagreement,
    ExpectedImprovement,
    ModelDiscrimination,
    ModelMax,
    ModelMin,
    PredictionVariance,
    ProbabilityOfImprovement,
    ThompsonSampling,
    suggest_points,
)
from .basis import BasisFunction, BasisLibrary
from .constraints import Constraint, Constraints, ConstraintType

# Metrics
from .metrics import (
    ModelComparison,
    compare_models,
    compute_adjusted_r2,
    compute_aic,
    compute_aicc,
    compute_all_metrics,
    compute_bic,
    compute_information_criterion,
    compute_mae,
    compute_mse,
    compute_r2,
    compute_rmse,
    cross_validate,
    format_comparison_table,
)
from .regressor import SymbolicRegressor, fit_symbolic

# Response Surface Methodology
from .rsm import (
    CanonicalAnalysis,
    ResponseSurface,
    box_behnken_design,
    canonical_analysis,
    central_composite_design,
    decode,
    encode,
    factorial_design,
    fractional_factorial_design,
)

# Sampling utilities
from .sampling import (
    AdaptiveSampler,
    SamplingStrategy,
    grid_sample,
    halton_sample,
    latin_hypercube_sample,
    sobol_sample,
)

# Selection
from .selection import (
    SelectionPath,
    SelectionResult,
    compute_pareto_front,
    exhaustive_search,
    greedy_backward_elimination,
    greedy_forward_selection,
    lasso_path_selection,
)

# Simplification
from .simplify import SimplificationResult, simplify_expression
from .study import DOEStudy

# Uncertainty Quantification
from .uncertainty import (
    AnovaResult,
    AnovaRow,
    BayesianModelAverage,
    anova,
    bootstrap_coefficients,
    bootstrap_model_selection,
    bootstrap_predict,
    coefficient_intervals,
    compute_coeff_covariance,
    compute_unbiased_variance,
    conformal_predict_jackknife_plus,
    conformal_predict_split,
    ensemble_predict,
    prediction_interval,
)


# Plotting (optional import to avoid matplotlib dependency issues)
def _get_plotting():
    from . import plotting

    return plotting


__all__ = [
    # Version
    "__version__",
    # Active Learning & Acquisition Functions
    "AcquisitionFunction",
    "AcquisitionResult",
    "ActiveLearner",
    "AOptimal",
    "BMAUncertainty",
    "Composite",
    "ConfidenceBandWidth",
    "DOptimal",
    "EnsembleDisagreement",
    "ExpectedImprovement",
    "LCB",
    "ModelDiscrimination",
    "ModelMax",
    "ModelMin",
    "PredictionVariance",
    "ProbabilityOfImprovement",
    "ThompsonSampling",
    "UCB",
    "suggest_points",
    # Core classes
    "BasisFunction",
    "BasisLibrary",
    "Constraint",
    "Constraints",
    "ConstraintType",
    "DOEStudy",
    "SymbolicRegressor",
    "fit_symbolic",
    # Metrics
    "ModelComparison",
    "compare_models",
    "compute_adjusted_r2",
    "compute_aic",
    "compute_aicc",
    "compute_all_metrics",
    "compute_bic",
    "compute_information_criterion",
    "compute_mae",
    "compute_mse",
    "compute_r2",
    "compute_rmse",
    "cross_validate",
    "format_comparison_table",
    # Response Surface Methodology
    "CanonicalAnalysis",
    "ResponseSurface",
    "box_behnken_design",
    "canonical_analysis",
    "central_composite_design",
    "decode",
    "encode",
    "factorial_design",
    "fractional_factorial_design",
    # Sampling
    "AdaptiveSampler",
    "SamplingStrategy",
    "grid_sample",
    "halton_sample",
    "latin_hypercube_sample",
    "sobol_sample",
    # Selection
    "SelectionPath",
    "SelectionResult",
    "compute_pareto_front",
    "exhaustive_search",
    "greedy_backward_elimination",
    "greedy_forward_selection",
    "lasso_path_selection",
    # Simplification
    "SimplificationResult",
    "simplify_expression",
    # Uncertainty Quantification
    "AnovaResult",
    "AnovaRow",
    "BayesianModelAverage",
    "anova",
    "bootstrap_coefficients",
    "bootstrap_model_selection",
    "bootstrap_predict",
    "coefficient_intervals",
    "compute_coeff_covariance",
    "compute_unbiased_variance",
    "conformal_predict_jackknife_plus",
    "conformal_predict_split",
    "ensemble_predict",
    "prediction_interval",
]
