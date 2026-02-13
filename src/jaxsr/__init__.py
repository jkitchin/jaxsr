"""
JAXSR: JAX-based Symbolic Regression

A fully open-source symbolic regression library built on JAX that discovers
interpretable algebraic expressions from data using sparse optimization techniques.

Inspired by ALAMO (Automated Learning of Algebraic Models for Optimization).
"""

from __future__ import annotations

__version__ = "0.1.1"
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

# Classification
from .classifier import SymbolicClassifier, fit_symbolic_classification
from .constraints import Constraint, Constraints, ConstraintType
from .dynamics import DynamicsResult, discover_dynamics, estimate_derivatives

# Metrics
from .metrics import (
    ModelComparison,
    compare_models,
    compute_accuracy,
    compute_adjusted_r2,
    compute_aic,
    compute_aicc,
    compute_all_classification_metrics,
    compute_all_metrics,
    compute_auc_roc,
    compute_bic,
    compute_classification_ic,
    compute_confusion_matrix,
    compute_f1_score,
    compute_information_criterion,
    compute_log_loss,
    compute_mae,
    compute_matthews_corrcoef,
    compute_mse,
    compute_precision,
    compute_r2,
    compute_recall,
    compute_rmse,
    cross_validate,
    cross_validate_classification,
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
    ClassificationPath,
    ClassificationResult,
    SelectionPath,
    SelectionResult,
    compute_pareto_front,
    compute_pareto_front_classification,
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
    bootstrap_classification_coefficients,
    bootstrap_coefficients,
    bootstrap_model_selection,
    bootstrap_predict,
    calibration_curve,
    classification_coefficient_intervals,
    coefficient_intervals,
    compute_coeff_covariance,
    compute_unbiased_variance,
    conformal_classification_split,
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
    # Classification
    "ClassificationPath",
    "ClassificationResult",
    "SymbolicClassifier",
    "bootstrap_classification_coefficients",
    "calibration_curve",
    "classification_coefficient_intervals",
    "compute_accuracy",
    "compute_all_classification_metrics",
    "compute_auc_roc",
    "compute_classification_ic",
    "compute_confusion_matrix",
    "compute_f1_score",
    "compute_log_loss",
    "compute_matthews_corrcoef",
    "compute_pareto_front_classification",
    "compute_precision",
    "compute_recall",
    "conformal_classification_split",
    "cross_validate_classification",
    "fit_symbolic_classification",
    # Core classes
    "BasisFunction",
    "BasisLibrary",
    "Constraint",
    "Constraints",
    "ConstraintType",
    "DOEStudy",
    "SymbolicRegressor",
    "fit_symbolic",
    # Dynamics / ODE Discovery
    "DynamicsResult",
    "discover_dynamics",
    "estimate_derivatives",
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
