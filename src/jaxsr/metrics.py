"""
Metrics and Model Comparison for JAXSR.

Provides information criteria, cross-validation scores, and model comparison utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from .regressor import SymbolicRegressor


# =============================================================================
# Information Criteria
# =============================================================================


def compute_aic(
    n_samples: int,
    n_params: int,
    mse: float,
    variance: float | None = None,
) -> float:
    """
    Compute Akaike Information Criterion.

    AIC = n * log(MSE) + 2 * k

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_params : int
        Number of model parameters.
    mse : float
        Mean squared error.
    variance : float, optional
        Known error variance (if None, estimated from MSE).

    Returns
    -------
    aic : float
        AIC value (lower is better).
    """
    n = n_samples
    k = n_params

    if mse <= 0:
        return float("inf")

    # AIC = -2 * log_likelihood + 2 * k
    # For Gaussian: log_likelihood = -n/2 * log(2*pi*sigma^2) - n/2
    # Simplified: AIC = n * log(MSE) + 2 * k
    log_lik = -n / 2 * jnp.log(2 * jnp.pi * mse) - n / 2
    return float(-2 * log_lik + 2 * k)


def compute_bic(
    n_samples: int,
    n_params: int,
    mse: float,
    variance: float | None = None,
) -> float:
    """
    Compute Bayesian Information Criterion.

    BIC = n * log(MSE) + k * log(n)

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_params : int
        Number of model parameters.
    mse : float
        Mean squared error.
    variance : float, optional
        Known error variance (if None, estimated from MSE).

    Returns
    -------
    bic : float
        BIC value (lower is better).
    """
    n = n_samples
    k = n_params

    if mse <= 0:
        return float("inf")

    log_lik = -n / 2 * jnp.log(2 * jnp.pi * mse) - n / 2
    return float(-2 * log_lik + k * jnp.log(n))


def compute_aicc(
    n_samples: int,
    n_params: int,
    mse: float,
    variance: float | None = None,
) -> float:
    """
    Compute corrected Akaike Information Criterion (AICc).

    AICc = AIC + 2*k*(k+1) / (n-k-1)

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_params : int
        Number of model parameters.
    mse : float
        Mean squared error.
    variance : float, optional
        Known error variance (if None, estimated from MSE).

    Returns
    -------
    aicc : float
        AICc value (lower is better).

    Notes
    -----
    AICc includes a correction for small sample sizes. It should be preferred
    when n/k < 40.
    """
    n = n_samples
    k = n_params

    aic = compute_aic(n, k, mse, variance)

    if n - k - 1 <= 0:
        return float("inf")

    correction = (2 * k * (k + 1)) / (n - k - 1)
    return aic + correction


def compute_hqc(
    n_samples: int,
    n_params: int,
    mse: float,
) -> float:
    """
    Compute Hannan-Quinn Criterion.

    HQC = n * log(MSE) + 2 * k * log(log(n))

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_params : int
        Number of model parameters.
    mse : float
        Mean squared error.

    Returns
    -------
    hqc : float
        HQC value (lower is better).

    Notes
    -----
    HQC is an alternative to BIC that penalizes complexity less severely.
    """
    n = n_samples
    k = n_params

    if mse <= 0 or n <= 2:
        return float("inf")

    log_log_n = jnp.log(jnp.log(n))
    if log_log_n <= 0:
        return float("inf")

    log_lik = -n / 2 * jnp.log(2 * jnp.pi * mse) - n / 2
    return float(-2 * log_lik + 2 * k * log_log_n)


def compute_mdl(
    n_samples: int,
    n_params: int,
    mse: float,
) -> float:
    """
    Compute Minimum Description Length criterion.

    MDL = n/2 * log(MSE) + k/2 * log(n)

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_params : int
        Number of model parameters.
    mse : float
        Mean squared error.

    Returns
    -------
    mdl : float
        MDL value (lower is better).
    """
    n = n_samples
    k = n_params

    if mse <= 0:
        return float("inf")

    return float(n / 2 * jnp.log(mse) + k / 2 * jnp.log(n))


def compute_information_criterion(
    n_samples: int,
    n_params: int,
    mse: float,
    criterion: str = "bic",
) -> float:
    """
    Compute the specified information criterion.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_params : int
        Number of model parameters.
    mse : float
        Mean squared error.
    criterion : str
        One of "aic", "aicc", "bic", "hqc", "mdl".

    Returns
    -------
    ic : float
        Information criterion value (lower is better).
    """
    criteria = {
        "aic": compute_aic,
        "aicc": compute_aicc,
        "bic": compute_bic,
        "hqc": compute_hqc,
        "mdl": compute_mdl,
    }

    if criterion not in criteria:
        raise ValueError(f"Unknown criterion: {criterion}. Available: {list(criteria.keys())}")

    return criteria[criterion](n_samples, n_params, mse)


# =============================================================================
# Cross-Validation
# =============================================================================


def cross_validate(
    model: SymbolicRegressor,
    X: jnp.ndarray,
    y: jnp.ndarray,
    cv: int = 5,
    scoring: str = "neg_mse",
    random_state: int | None = None,
) -> dict[str, Any]:
    """
    Perform k-fold cross-validation.

    Parameters
    ----------
    model : SymbolicRegressor
        Model to evaluate.
    X : jnp.ndarray
        Feature matrix.
    y : jnp.ndarray
        Target values.
    cv : int
        Number of folds.
    scoring : str
        Scoring metric: "neg_mse", "neg_mae", "r2".
    random_state : int, optional
        Random seed for fold splitting.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - "test_scores": array of test scores for each fold
        - "train_scores": array of train scores for each fold
        - "mean_test_score": mean test score
        - "std_test_score": std of test scores
    """
    n_samples = X.shape[0]
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n_samples)

    fold_size = n_samples // cv
    test_scores = []
    train_scores = []

    scoring_funcs = {
        "neg_mse": lambda y_true, y_pred: -float(jnp.mean((y_true - y_pred) ** 2)),
        "neg_mae": lambda y_true, y_pred: -float(jnp.mean(jnp.abs(y_true - y_pred))),
        "r2": lambda y_true, y_pred: float(
            1
            - jnp.sum((y_true - y_pred) ** 2)
            / jnp.maximum(jnp.sum((y_true - jnp.mean(y_true)) ** 2), 1e-10)
        ),
    }

    if scoring not in scoring_funcs:
        raise ValueError(f"Unknown scoring: {scoring}. Available: {list(scoring_funcs.keys())}")

    score_func = scoring_funcs[scoring]

    for i in range(cv):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < cv - 1 else n_samples

        test_idx = indices[start_idx:end_idx]
        train_idx = np.concatenate([indices[:start_idx], indices[end_idx:]])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone and fit model
        model_clone = model.__class__(
            basis_library=model.basis_library,
            max_terms=model.max_terms,
            strategy=model.strategy,
            information_criterion=model.information_criterion,
            cv_folds=model.cv_folds,
            regularization=model.regularization,
            constraints=model.constraints,
            random_state=model.random_state,
        )
        model_clone.fit(X_train, y_train)

        y_pred_test = model_clone.predict(X_test)
        y_pred_train = model_clone.predict(X_train)

        test_scores.append(score_func(y_test, y_pred_test))
        train_scores.append(score_func(y_train, y_pred_train))

    test_scores = np.array(test_scores)
    train_scores = np.array(train_scores)

    return {
        "test_scores": test_scores,
        "train_scores": train_scores,
        "mean_test_score": float(np.mean(test_scores)),
        "std_test_score": float(np.std(test_scores)),
        "mean_train_score": float(np.mean(train_scores)),
        "std_train_score": float(np.std(train_scores)),
    }


def compute_cv_score(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    cv: int = 5,
    random_state: int | None = None,
) -> float:
    """
    Compute cross-validation MSE for a design matrix.

    This is a lower-level function that works directly with the design matrix.

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        Target values.
    cv : int
        Number of folds.
    random_state : int, optional
        Random seed.

    Returns
    -------
    cv_mse : float
        Mean cross-validation MSE.
    """
    n_samples = Phi.shape[0]
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n_samples)

    fold_size = n_samples // cv
    mse_scores = []

    for i in range(cv):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < cv - 1 else n_samples

        test_idx = indices[start_idx:end_idx]
        train_idx = np.concatenate([indices[:start_idx], indices[end_idx:]])

        Phi_train, Phi_test = Phi[train_idx], Phi[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Solve least squares
        coeffs, _, _, _ = jnp.linalg.lstsq(Phi_train, y_train, rcond=None)
        y_pred = Phi_test @ coeffs
        mse = float(jnp.mean((y_test - y_pred) ** 2))
        mse_scores.append(mse)

    return float(np.mean(mse_scores))


def compute_loo_mse(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    coefficients: jnp.ndarray,
) -> float:
    """
    Compute leave-one-out MSE efficiently using Sherman-Morrison formula.

    This avoids refitting the model n times by using the hat matrix.

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        Target values.
    coefficients : jnp.ndarray
        Fitted coefficients.

    Returns
    -------
    loo_mse : float
        Leave-one-out mean squared error.
    """
    y_pred = Phi @ coefficients
    residuals = y - y_pred

    # Hat matrix diagonal: h_ii = Phi_i @ (Phi.T @ Phi)^-1 @ Phi_i.T
    # Using pseudo-inverse for numerical stability
    Phi_pinv = jnp.linalg.pinv(Phi)
    H = Phi @ Phi_pinv
    h_diag = jnp.diag(H)

    # LOO residual: e_i / (1 - h_ii)
    loo_residuals = residuals / (1 - h_diag + 1e-10)
    loo_mse = jnp.mean(loo_residuals**2)

    return float(loo_mse)


def compute_press(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    coefficients: jnp.ndarray,
) -> float:
    """
    Compute PRESS (Predicted Residual Error Sum of Squares).

    PRESS = sum_i (e_i / (1 - h_ii))^2

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix.
    y : jnp.ndarray
        Target values.
    coefficients : jnp.ndarray
        Fitted coefficients.

    Returns
    -------
    press : float
        PRESS statistic.
    """
    return compute_loo_mse(Phi, y, coefficients) * len(y)


# =============================================================================
# Regression Metrics
# =============================================================================


def compute_mse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Compute mean squared error."""
    return float(jnp.mean((y_true - y_pred) ** 2))


def compute_rmse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Compute root mean squared error."""
    return float(jnp.sqrt(jnp.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Compute mean absolute error."""
    return float(jnp.mean(jnp.abs(y_true - y_pred)))


def compute_r2(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Compute R-squared (coefficient of determination)."""
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def compute_adjusted_r2(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    n_params: int,
) -> float:
    """
    Compute adjusted R-squared.

    Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - k - 1)

    Parameters
    ----------
    y_true : jnp.ndarray
        True values.
    y_pred : jnp.ndarray
        Predicted values.
    n_params : int
        Number of model parameters.

    Returns
    -------
    adj_r2 : float
        Adjusted R-squared.
    """
    n = len(y_true)
    k = n_params
    r2 = compute_r2(y_true, y_pred)

    if n - k - 1 <= 0:
        return float("-inf")

    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def compute_max_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Compute maximum absolute error."""
    return float(jnp.max(jnp.abs(y_true - y_pred)))


def compute_mape(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """
    Compute mean absolute percentage error.

    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    """
    mask = jnp.abs(y_true) > 1e-10
    if not jnp.any(mask):
        return float("inf")
    return float(jnp.mean(jnp.abs((y_true - y_pred) / y_true)[mask]) * 100)


def compute_all_metrics(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    n_params: int,
) -> dict[str, float]:
    """
    Compute all standard regression metrics.

    Parameters
    ----------
    y_true : jnp.ndarray
        True values.
    y_pred : jnp.ndarray
        Predicted values.
    n_params : int
        Number of model parameters.

    Returns
    -------
    metrics : dict
        Dictionary containing all metrics.
    """
    n = len(y_true)
    mse = compute_mse(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
        "adjusted_r2": compute_adjusted_r2(y_true, y_pred, n_params),
        "max_error": compute_max_error(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "aic": compute_aic(n, n_params, mse),
        "aicc": compute_aicc(n, n_params, mse),
        "bic": compute_bic(n, n_params, mse),
    }


# =============================================================================
# Model Comparison
# =============================================================================


@dataclass
class ModelComparison:
    """Container for model comparison results."""

    models: list[SymbolicRegressor]
    names: list[str]
    train_metrics: list[dict[str, float]]
    test_metrics: list[dict[str, float]] | None
    rankings: dict[str, list[int]]


def compare_models(
    models: list[SymbolicRegressor],
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_test: jnp.ndarray | None = None,
    y_test: jnp.ndarray | None = None,
    names: list[str] | None = None,
) -> ModelComparison:
    """
    Compare multiple fitted models.

    Parameters
    ----------
    models : list of SymbolicRegressor
        Fitted models to compare.
    X_train : jnp.ndarray
        Training features.
    y_train : jnp.ndarray
        Training targets.
    X_test : jnp.ndarray, optional
        Test features.
    y_test : jnp.ndarray, optional
        Test targets.
    names : list of str, optional
        Names for each model.

    Returns
    -------
    comparison : ModelComparison
        Comparison results including metrics and rankings.
    """
    if names is None:
        names = [f"Model_{i}" for i in range(len(models))]

    train_metrics = []
    test_metrics = [] if X_test is not None else None

    for model in models:
        y_pred_train = model.predict(X_train)
        n_params = len(model.coefficients_) if model.coefficients_ is not None else 0
        train_metrics.append(compute_all_metrics(y_train, y_pred_train, n_params))

        if X_test is not None and y_test is not None:
            y_pred_test = model.predict(X_test)
            test_metrics.append(compute_all_metrics(y_test, y_pred_test, n_params))

    # Compute rankings for each metric
    rankings = {}
    metrics_to_rank = ["mse", "rmse", "mae", "r2", "bic", "aic"]

    for metric in metrics_to_rank:
        values = [m[metric] for m in train_metrics]
        # Lower is better for error metrics, higher for R2
        reverse = metric == "r2"
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)
        rankings[f"train_{metric}"] = [sorted_indices.index(i) + 1 for i in range(len(values))]

        if test_metrics:
            values = [m[metric] for m in test_metrics]
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)
            rankings[f"test_{metric}"] = [sorted_indices.index(i) + 1 for i in range(len(values))]

    return ModelComparison(
        models=models,
        names=names,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        rankings=rankings,
    )


def format_comparison_table(comparison: ModelComparison) -> str:
    """
    Format model comparison as a text table.

    Parameters
    ----------
    comparison : ModelComparison
        Comparison results.

    Returns
    -------
    table : str
        Formatted table string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Model Comparison")
    lines.append("=" * 80)

    # Header
    metrics = ["mse", "rmse", "r2", "bic", "complexity"]
    header = f"{'Model':<15} | " + " | ".join(f"{m:>10}" for m in metrics)
    lines.append(header)
    lines.append("-" * 80)

    # Training metrics
    lines.append("Training:")
    for i, name in enumerate(comparison.names):
        model = comparison.models[i]
        m = comparison.train_metrics[i]
        complexity = model.complexity_ if hasattr(model, "complexity_") else "N/A"
        row = f"{name:<15} | " + " | ".join(
            f"{m.get(metric, 'N/A'):>10.4f}" if isinstance(m.get(metric), float) else f"{'N/A':>10}"
            for metric in metrics[:-1]
        )
        row += f" | {complexity:>10}"
        lines.append(row)

    # Test metrics
    if comparison.test_metrics:
        lines.append("")
        lines.append("Test:")
        for i, name in enumerate(comparison.names):
            model = comparison.models[i]
            m = comparison.test_metrics[i]
            complexity = model.complexity_ if hasattr(model, "complexity_") else "N/A"
            row = f"{name:<15} | " + " | ".join(
                (
                    f"{m.get(metric, 'N/A'):>10.4f}"
                    if isinstance(m.get(metric), float)
                    else f"{'N/A':>10}"
                )
                for metric in metrics[:-1]
            )
            row += f" | {complexity:>10}"
            lines.append(row)

    lines.append("=" * 80)
    return "\n".join(lines)


# =============================================================================
# Classification Information Criteria
# =============================================================================


def compute_classification_ic(
    n_samples: int,
    n_params: int,
    neg_log_likelihood: float,
    criterion: str = "bic",
) -> float:
    """
    Compute information criterion from Bernoulli negative log-likelihood.

    Unlike regression IC (which starts from MSE/Gaussian likelihood), this
    uses the Bernoulli log-likelihood directly:
        AIC = 2*NLL + 2*k
        BIC = 2*NLL + k*log(n)
        AICc = AIC + 2*k*(k+1)/(n-k-1)

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_params : int
        Number of model parameters.
    neg_log_likelihood : float
        Negative log-likelihood (sum, not mean).
    criterion : str
        One of ``"aic"``, ``"aicc"``, ``"bic"``.

    Returns
    -------
    ic : float
        Information criterion value (lower is better).

    Raises
    ------
    ValueError
        If *criterion* is not one of ``"aic"``, ``"aicc"``, ``"bic"``.
    """
    n = n_samples
    k = n_params
    nll = neg_log_likelihood

    if criterion == "aic":
        return 2 * nll + 2 * k
    elif criterion == "bic":
        return 2 * nll + k * float(jnp.log(n))
    elif criterion == "aicc":
        aic = 2 * nll + 2 * k
        if n - k - 1 <= 0:
            return float("inf")
        return aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Available: ['aic', 'aicc', 'bic']")


# =============================================================================
# Classification Metrics
# =============================================================================


def compute_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : jnp.ndarray
        True class labels.
    y_pred : jnp.ndarray
        Predicted class labels.

    Returns
    -------
    accuracy : float
        Fraction of correct predictions.
    """
    y_true = jnp.asarray(y_true).ravel()
    y_pred = jnp.asarray(y_pred).ravel()
    return float(jnp.mean(y_true == y_pred))


def compute_log_loss(
    y_true: jnp.ndarray,
    y_pred_proba: jnp.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Compute binary or multiclass log-loss (cross-entropy).

    Parameters
    ----------
    y_true : jnp.ndarray
        True class labels of shape ``(n,)``.
    y_pred_proba : jnp.ndarray
        Predicted probabilities. Shape ``(n,)`` for binary or ``(n, K)``
        for multiclass.
    eps : float
        Clipping bound for numerical safety.

    Returns
    -------
    loss : float
        Mean negative log-likelihood per sample.
    """
    y_true = jnp.asarray(y_true).ravel()
    y_pred_proba = jnp.asarray(y_pred_proba)

    if y_pred_proba.ndim == 1:
        # Binary case: y_pred_proba is P(y=1)
        p = jnp.clip(y_pred_proba, eps, 1 - eps)
        nll = -(y_true * jnp.log(p) + (1 - y_true) * jnp.log(1 - p))
    else:
        # Multiclass: y_pred_proba is (n, K)
        p = jnp.clip(y_pred_proba, eps, 1.0)
        n = len(y_true)
        idx = jnp.arange(n)
        y_int = y_true.astype(int)
        nll = -jnp.log(p[idx, y_int])

    return float(jnp.mean(nll))


def compute_precision(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    pos_label: int = 1,
) -> float:
    """
    Compute precision for a binary classification problem.

    Parameters
    ----------
    y_true : jnp.ndarray
        True class labels.
    y_pred : jnp.ndarray
        Predicted class labels.
    pos_label : int
        Label considered as positive.

    Returns
    -------
    precision : float
        TP / (TP + FP). Returns 0.0 when there are no positive predictions.
    """
    y_true = jnp.asarray(y_true).ravel()
    y_pred = jnp.asarray(y_pred).ravel()
    tp = float(jnp.sum((y_pred == pos_label) & (y_true == pos_label)))
    fp = float(jnp.sum((y_pred == pos_label) & (y_true != pos_label)))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def compute_recall(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    pos_label: int = 1,
) -> float:
    """
    Compute recall (sensitivity) for a binary classification problem.

    Parameters
    ----------
    y_true : jnp.ndarray
        True class labels.
    y_pred : jnp.ndarray
        Predicted class labels.
    pos_label : int
        Label considered as positive.

    Returns
    -------
    recall : float
        TP / (TP + FN). Returns 0.0 when there are no positive samples.
    """
    y_true = jnp.asarray(y_true).ravel()
    y_pred = jnp.asarray(y_pred).ravel()
    tp = float(jnp.sum((y_pred == pos_label) & (y_true == pos_label)))
    fn = float(jnp.sum((y_pred != pos_label) & (y_true == pos_label)))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def compute_f1_score(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    pos_label: int = 1,
) -> float:
    """
    Compute F1 score (harmonic mean of precision and recall).

    Parameters
    ----------
    y_true : jnp.ndarray
        True class labels.
    y_pred : jnp.ndarray
        Predicted class labels.
    pos_label : int
        Label considered as positive.

    Returns
    -------
    f1 : float
        F1 score. Returns 0.0 when precision + recall = 0.
    """
    p = compute_precision(y_true, y_pred, pos_label)
    r = compute_recall(y_true, y_pred, pos_label)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_auc_roc(
    y_true: jnp.ndarray,
    y_score: jnp.ndarray,
) -> float:
    """
    Compute Area Under the ROC Curve via the trapezoidal rule.

    Parameters
    ----------
    y_true : jnp.ndarray
        True binary labels (0 or 1).
    y_score : jnp.ndarray
        Predicted scores or probabilities for the positive class.

    Returns
    -------
    auc : float
        AUC-ROC value in ``[0, 1]``.

    Raises
    ------
    ValueError
        If *y_true* contains fewer than two distinct classes.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    if len(np.unique(y_true)) < 2:
        raise ValueError("AUC-ROC requires at least two distinct classes in y_true.")

    # Sort by descending score
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    # Compute TPR and FPR at each threshold
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    total_pos = tps[-1]
    total_neg = fps[-1]

    if total_pos == 0 or total_neg == 0:
        return 0.0

    tpr = np.concatenate([[0], tps / total_pos])
    fpr = np.concatenate([[0], fps / total_neg])

    # Trapezoidal rule
    auc = float(np.trapz(tpr, fpr))
    return auc


def compute_confusion_matrix(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    n_classes: int | None = None,
) -> np.ndarray:
    """
    Compute the confusion matrix.

    Parameters
    ----------
    y_true : jnp.ndarray
        True class labels.
    y_pred : jnp.ndarray
        Predicted class labels.
    n_classes : int, optional
        Number of classes. Inferred from data if ``None``.

    Returns
    -------
    cm : np.ndarray
        Confusion matrix of shape ``(n_classes, n_classes)`` where
        ``cm[i, j]`` is the count of samples with true label *i* and
        predicted label *j*.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred, strict=False):
        cm[t, p] += 1
    return cm


def compute_matthews_corrcoef(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
) -> float:
    """
    Compute Matthews Correlation Coefficient for binary classification.

    Parameters
    ----------
    y_true : jnp.ndarray
        True binary labels.
    y_pred : jnp.ndarray
        Predicted binary labels.

    Returns
    -------
    mcc : float
        MCC in ``[-1, 1]``. Returns 0.0 when the denominator is zero.
    """
    cm = compute_confusion_matrix(y_true, y_pred, n_classes=2)
    tp = float(cm[1, 1])
    tn = float(cm[0, 0])
    fp = float(cm[0, 1])
    fn = float(cm[1, 0])

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def compute_all_classification_metrics(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    y_pred_proba: jnp.ndarray | None = None,
    n_params: int = 0,
) -> dict[str, float]:
    """
    Compute a comprehensive suite of classification metrics.

    Parameters
    ----------
    y_true : jnp.ndarray
        True class labels.
    y_pred : jnp.ndarray
        Predicted class labels.
    y_pred_proba : jnp.ndarray, optional
        Predicted probabilities (enables log-loss and AUC-ROC).
    n_params : int
        Number of model parameters (for IC calculation).

    Returns
    -------
    metrics : dict[str, float]
        Dictionary of metric name to value.
    """
    metrics: dict[str, float] = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred),
        "recall": compute_recall(y_true, y_pred),
        "f1": compute_f1_score(y_true, y_pred),
        "mcc": compute_matthews_corrcoef(y_true, y_pred),
    }

    if y_pred_proba is not None:
        metrics["log_loss"] = compute_log_loss(y_true, y_pred_proba)
        n = len(y_true)
        nll = metrics["log_loss"] * n
        metrics["aic"] = compute_classification_ic(n, n_params, nll, "aic")
        metrics["bic"] = compute_classification_ic(n, n_params, nll, "bic")
        metrics["aicc"] = compute_classification_ic(n, n_params, nll, "aicc")

        y_proba_arr = jnp.asarray(y_pred_proba)
        if y_proba_arr.ndim == 1:
            try:
                metrics["auc_roc"] = compute_auc_roc(y_true, y_pred_proba)
            except ValueError:
                pass

    return metrics


def cross_validate_classification(
    model,
    X: jnp.ndarray,
    y: jnp.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
    random_state: int | None = None,
) -> dict[str, Any]:
    """
    Perform k-fold cross-validation for a classification model.

    Parameters
    ----------
    model : SymbolicClassifier
        Model to evaluate (must implement ``fit`` and ``predict``).
    X : jnp.ndarray
        Feature matrix.
    y : jnp.ndarray
        Target labels.
    cv : int
        Number of folds.
    scoring : str
        Scoring metric: ``"accuracy"``, ``"neg_log_loss"``, ``"f1"``.
    random_state : int, optional
        Random seed for fold splitting.

    Returns
    -------
    results : dict
        Dictionary with keys ``"test_scores"``, ``"train_scores"``,
        ``"mean_test_score"``, ``"std_test_score"``,
        ``"mean_train_score"``, ``"std_train_score"``.

    Raises
    ------
    ValueError
        If *scoring* is not a recognised metric name.
    """
    n_samples = X.shape[0]
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n_samples)

    fold_size = n_samples // cv
    test_scores = []
    train_scores = []

    def _acc(y_t, y_p, _model):
        return compute_accuracy(y_t, y_p)

    def _neg_ll(y_t, _y_p, m):
        return -compute_log_loss(y_t, m.predict_proba(jnp.atleast_2d(jnp.asarray(X))))

    def _f1(y_t, y_p, _model):
        return compute_f1_score(y_t, y_p)

    scoring_funcs = {
        "accuracy": _acc,
        "f1": _f1,
    }

    if scoring == "neg_log_loss":
        # Special handling: needs predict_proba, not predict
        pass
    elif scoring not in scoring_funcs:
        raise ValueError(
            f"Unknown scoring: {scoring}. Available: {list(scoring_funcs.keys()) + ['neg_log_loss']}"
        )

    for i in range(cv):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < cv - 1 else n_samples

        test_idx = indices[start_idx:end_idx]
        train_idx = np.concatenate([indices[:start_idx], indices[end_idx:]])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone and fit model
        from .classifier import SymbolicClassifier

        model_clone = SymbolicClassifier(
            basis_library=model.basis_library,
            max_terms=model.max_terms,
            strategy=model.strategy,
            information_criterion=model.information_criterion,
            regularization=model.regularization,
            constraints=model.constraints,
            random_state=model.random_state,
        )
        model_clone.fit(X_train, y_train)

        y_pred_test = model_clone.predict(X_test)
        y_pred_train = model_clone.predict(X_train)

        if scoring == "neg_log_loss":
            proba_test = model_clone.predict_proba(X_test)
            proba_train = model_clone.predict_proba(X_train)
            if proba_test.ndim == 2:
                proba_test = proba_test[:, 1]
                proba_train = proba_train[:, 1]
            test_scores.append(-compute_log_loss(y_test, proba_test))
            train_scores.append(-compute_log_loss(y_train, proba_train))
        else:
            score_func = scoring_funcs[scoring]
            test_scores.append(score_func(y_test, y_pred_test, model_clone))
            train_scores.append(score_func(y_train, y_pred_train, model_clone))

    test_scores_arr = np.array(test_scores)
    train_scores_arr = np.array(train_scores)

    return {
        "test_scores": test_scores_arr,
        "train_scores": train_scores_arr,
        "mean_test_score": float(np.mean(test_scores_arr)),
        "std_test_score": float(np.std(test_scores_arr)),
        "mean_train_score": float(np.mean(train_scores_arr)),
        "std_train_score": float(np.std(train_scores_arr)),
    }
