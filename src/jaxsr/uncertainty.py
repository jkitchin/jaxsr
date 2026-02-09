"""
Uncertainty Quantification for JAXSR.

Provides classical OLS intervals, Pareto front ensemble predictions,
Bayesian Model Averaging, conformal prediction, and bootstrap methods.

Since JAXSR models are linear-in-parameters (y = Phi @ beta), classical
OLS inference applies directly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from .regressor import SymbolicRegressor


# =============================================================================
# Classical OLS Uncertainty (Phase 1, Strategy 1)
# =============================================================================


def compute_unbiased_variance(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    coefficients: jnp.ndarray,
) -> float:
    """
    Compute unbiased noise variance estimate: s^2 = SSR / (n - p).

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        Target values of shape (n_samples,).
    coefficients : jnp.ndarray
        Fitted coefficients of shape (n_features,).

    Returns
    -------
    sigma_sq : float
        Unbiased variance estimate.
    """
    n, p = Phi.shape
    residuals = y - Phi @ coefficients
    ssr = float(jnp.sum(residuals**2))
    dof = n - p
    if dof <= 0:
        warnings.warn(
            f"Degrees of freedom ({dof}) <= 0. Cannot compute unbiased variance.", stacklevel=2
        )
        return float("inf")
    return ssr / dof


def compute_coeff_covariance(
    Phi: jnp.ndarray,
    sigma_sq: float,
) -> jnp.ndarray:
    """
    Compute coefficient covariance matrix: Cov(beta) = sigma^2 * (Phi^T Phi)^{-1}.

    Uses SVD for numerical stability.

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix of shape (n_samples, n_features).
    sigma_sq : float
        Unbiased noise variance estimate.

    Returns
    -------
    cov : jnp.ndarray
        Covariance matrix of shape (n_features, n_features).
    """
    U, s, Vt = jnp.linalg.svd(Phi, full_matrices=False)
    # (Phi^T Phi)^{-1} = V @ diag(1/s^2) @ V^T
    rcond = jnp.finfo(Phi.dtype).eps * max(Phi.shape)
    cutoff = rcond * jnp.max(s)
    s_inv_sq = jnp.where(s > cutoff, 1.0 / (s**2), 0.0)
    PhiTPhiInv = Vt.T @ jnp.diag(s_inv_sq) @ Vt
    return sigma_sq * PhiTPhiInv


def coefficient_intervals(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    coefficients: jnp.ndarray,
    names: list[str],
    alpha: float = 0.05,
) -> dict[str, tuple[float, float, float, float]]:
    """
    Compute t-based confidence intervals for each coefficient.

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        Target values.
    coefficients : jnp.ndarray
        Fitted coefficients.
    names : list of str
        Names of basis functions.
    alpha : float
        Significance level (default 0.05 for 95% CIs).

    Returns
    -------
    intervals : dict
        {name: (estimate, lower, upper, se)} for each coefficient.
    """
    n, p = Phi.shape
    dof = n - p
    if dof <= 0:
        warnings.warn("Not enough degrees of freedom for coefficient intervals.", stacklevel=2)
        return {
            name: (float(coef), float("nan"), float("nan"), float("nan"))
            for name, coef in zip(names, coefficients, strict=False)
        }

    sigma_sq = compute_unbiased_variance(Phi, y, coefficients)
    cov = compute_coeff_covariance(Phi, sigma_sq)
    se = jnp.sqrt(jnp.diag(cov))

    t_crit = stats.t.ppf(1 - alpha / 2, dof)

    intervals = {}
    for i, name in enumerate(names):
        est = float(coefficients[i])
        std_err = float(se[i])
        lower = est - t_crit * std_err
        upper = est + t_crit * std_err
        intervals[name] = (est, lower, upper, std_err)

    return intervals


def prediction_interval(
    Phi_train: jnp.ndarray,
    y_train: jnp.ndarray,
    coefficients: jnp.ndarray,
    Phi_new: jnp.ndarray,
    alpha: float = 0.05,
) -> dict[str, jnp.ndarray]:
    """
    Compute prediction and confidence intervals for new observations.

    Confidence band: uncertainty in the mean response E[y|x].
    Prediction interval: uncertainty in a new observation y.

    Parameters
    ----------
    Phi_train : jnp.ndarray
        Training design matrix of shape (n_train, p).
    y_train : jnp.ndarray
        Training target values of shape (n_train,).
    coefficients : jnp.ndarray
        Fitted coefficients of shape (p,).
    Phi_new : jnp.ndarray
        Design matrix for new points of shape (n_new, p).
    alpha : float
        Significance level (default 0.05 for 95% intervals).

    Returns
    -------
    result : dict
        Dictionary with keys:
        - "y_pred": predicted values
        - "pred_lower", "pred_upper": prediction interval bounds
        - "conf_lower", "conf_upper": confidence band bounds
        - "pred_se": prediction standard error
        - "conf_se": confidence standard error (mean response)
    """
    n, p = Phi_train.shape
    dof = n - p
    if dof <= 0:
        y_pred = Phi_new @ coefficients
        nan_arr = jnp.full_like(y_pred, float("nan"))
        return {
            "y_pred": y_pred,
            "pred_lower": nan_arr,
            "pred_upper": nan_arr,
            "conf_lower": nan_arr,
            "conf_upper": nan_arr,
            "pred_se": nan_arr,
            "conf_se": nan_arr,
        }

    sigma_sq = compute_unbiased_variance(Phi_train, y_train, coefficients)
    sigma = jnp.sqrt(sigma_sq)

    # Compute (Phi^T Phi)^{-1} via SVD
    U, s, Vt = jnp.linalg.svd(Phi_train, full_matrices=False)
    rcond = jnp.finfo(Phi_train.dtype).eps * max(Phi_train.shape)
    cutoff = rcond * jnp.max(s)
    s_inv_sq = jnp.where(s > cutoff, 1.0 / (s**2), 0.0)
    PhiTPhiInv = Vt.T @ jnp.diag(s_inv_sq) @ Vt

    y_pred = Phi_new @ coefficients

    # Leverage for new points: h(x_new) = x_new^T (Phi^T Phi)^{-1} x_new
    # Vectorized: each row of Phi_new
    h_new = jnp.sum((Phi_new @ PhiTPhiInv) * Phi_new, axis=1)

    t_crit = stats.t.ppf(1 - alpha / 2, dof)

    # Confidence band on E[y|x]: var = sigma^2 * h(x)
    conf_se = sigma * jnp.sqrt(h_new)
    conf_lower = y_pred - t_crit * conf_se
    conf_upper = y_pred + t_crit * conf_se

    # Prediction interval for new y: var = sigma^2 * (1 + h(x))
    pred_se = sigma * jnp.sqrt(1 + h_new)
    pred_lower = y_pred - t_crit * pred_se
    pred_upper = y_pred + t_crit * pred_se

    return {
        "y_pred": y_pred,
        "pred_lower": pred_lower,
        "pred_upper": pred_upper,
        "conf_lower": conf_lower,
        "conf_upper": conf_upper,
        "pred_se": pred_se,
        "conf_se": conf_se,
    }


# =============================================================================
# Pareto Front Ensemble (Phase 1, Strategy 2)
# =============================================================================


def ensemble_predict(
    model: SymbolicRegressor,
    X_new: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """
    Predictions from all Pareto-front models.

    Provides a measure of structural/model uncertainty: how much
    do predictions vary across plausible model complexities?

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model with selection path.
    X_new : jnp.ndarray
        New input data of shape (n_new, n_features).

    Returns
    -------
    result : dict
        Dictionary with keys:
        - "y_mean": mean prediction across Pareto models
        - "y_std": std of predictions across Pareto models
        - "y_min": min prediction across Pareto models
        - "y_max": max prediction across Pareto models
        - "y_all": array of shape (n_models, n_new) with all predictions
        - "models": list of SelectionResult for the Pareto models
    """
    model._check_is_fitted()
    X_new = jnp.atleast_2d(jnp.asarray(X_new))

    pareto = model.pareto_front_

    if len(pareto) == 0:
        y_pred = model.predict(X_new)
        return {
            "y_mean": y_pred,
            "y_std": jnp.zeros_like(y_pred),
            "y_min": y_pred,
            "y_max": y_pred,
            "y_all": y_pred[None, :],
            "models": [model._result],
        }

    predictions = []
    for result in pareto:
        Phi = model.basis_library.evaluate_subset(X_new, result.selected_indices)
        y_pred = Phi @ result.coefficients
        predictions.append(y_pred)

    y_all = jnp.stack(predictions, axis=0)

    return {
        "y_mean": jnp.mean(y_all, axis=0),
        "y_std": jnp.std(y_all, axis=0),
        "y_min": jnp.min(y_all, axis=0),
        "y_max": jnp.max(y_all, axis=0),
        "y_all": y_all,
        "models": pareto,
    }


# =============================================================================
# Bayesian Model Averaging (Phase 2, Strategy 3)
# =============================================================================


class BayesianModelAverage:
    """
    IC-weighted model averaging across selection path or Pareto front.

    Weights are computed as:
        w_k = exp(-0.5 * IC_k) / sum(exp(-0.5 * IC_j))

    BMA variance includes within-model and between-model components.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model with selection path.
    criterion : str
        Information criterion to use for weights: "bic" or "aic".
    use_pareto : bool
        If True, average over Pareto front models only.
        If False, average over all models in the selection path.
    top_k : int, optional
        If specified, only use the top k models by IC value.
    """

    def __init__(
        self,
        model: SymbolicRegressor,
        criterion: str = "bic",
        use_pareto: bool = True,
        top_k: int | None = None,
    ):
        model._check_is_fitted()
        self._model = model

        if use_pareto:
            candidates = model.pareto_front_
        else:
            candidates = model._selection_path.results

        if not candidates:
            raise ValueError("No candidate models available for BMA.")

        # Sort by IC and optionally take top_k
        ic_values = np.array([getattr(r, criterion) for r in candidates])
        sorted_idx = np.argsort(ic_values)
        if top_k is not None:
            sorted_idx = sorted_idx[:top_k]

        self._results = [candidates[i] for i in sorted_idx]
        self._criterion = criterion

        # Compute weights: w_k = exp(-0.5 * delta_IC_k) / sum(...)
        # Use delta-IC (relative to best) for numerical stability
        ic_vals = np.array([getattr(r, criterion) for r in self._results])
        delta_ic = ic_vals - ic_vals.min()
        log_weights = -0.5 * delta_ic
        log_weights -= np.max(log_weights)  # additional stability
        raw_weights = np.exp(log_weights)
        self._weights = raw_weights / raw_weights.sum()

    @property
    def weights(self) -> dict[str, float]:
        """Model weights keyed by expression string."""
        return {
            r.expression(): float(w) for r, w in zip(self._results, self._weights, strict=False)
        }

    @property
    def expressions(self) -> list[str]:
        """Expressions of models in the average."""
        return [r.expression() for r in self._results]

    def predict(self, X: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        BMA prediction with uncertainty.

        Parameters
        ----------
        X : jnp.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        y_mean : jnp.ndarray
            Weighted mean prediction.
        y_std : jnp.ndarray
            BMA standard deviation (within + between model variance).
        """
        X = jnp.atleast_2d(jnp.asarray(X))
        library = self._model.basis_library

        predictions = []
        for result in self._results:
            Phi = library.evaluate_subset(X, result.selected_indices)
            y_pred = Phi @ result.coefficients
            predictions.append(y_pred)

        y_all = jnp.stack(predictions, axis=0)  # (n_models, n_samples)
        weights = jnp.array(self._weights)[:, None]  # (n_models, 1)

        # Weighted mean
        y_mean = jnp.sum(weights * y_all, axis=0)

        # BMA variance = within-model + between-model
        # Between-model: weighted variance of predictions
        sq_diff = (y_all - y_mean[None, :]) ** 2
        between_var = jnp.sum(weights * sq_diff, axis=0)

        # Within-model: weighted average of per-model variances
        # For OLS: var_k = sigma_k^2 (constant per model)
        within_var = jnp.zeros_like(y_mean)
        for i, result in enumerate(self._results):
            within_var = within_var + self._weights[i] * result.mse

        total_var = within_var + between_var
        y_std = jnp.sqrt(total_var)

        return y_mean, y_std

    def predict_interval(
        self, X: jnp.ndarray, alpha: float = 0.05
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        BMA prediction interval using Gaussian approximation.

        Parameters
        ----------
        X : jnp.ndarray
            Input data.
        alpha : float
            Significance level.

        Returns
        -------
        y_pred : jnp.ndarray
            BMA mean prediction.
        lower : jnp.ndarray
            Lower bound.
        upper : jnp.ndarray
            Upper bound.
        """
        y_mean, y_std = self.predict(X)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        lower = y_mean - z_crit * y_std
        upper = y_mean + z_crit * y_std
        return y_mean, lower, upper


# =============================================================================
# Conformal Prediction (Phase 2, Strategy 4)
# =============================================================================


def conformal_predict_split(
    model: SymbolicRegressor,
    X_cal: jnp.ndarray,
    y_cal: jnp.ndarray,
    X_new: jnp.ndarray,
    alpha: float = 0.05,
) -> dict[str, jnp.ndarray]:
    """
    Split conformal prediction intervals.

    Uses a held-out calibration set to construct distribution-free
    prediction intervals with finite-sample coverage guarantee.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model (trained on separate training data).
    X_cal : jnp.ndarray
        Calibration features of shape (n_cal, n_features).
    y_cal : jnp.ndarray
        Calibration targets of shape (n_cal,).
    X_new : jnp.ndarray
        New points for prediction.
    alpha : float
        Significance level (coverage = 1 - alpha).

    Returns
    -------
    result : dict
        Dictionary with keys:
        - "y_pred": point predictions
        - "lower": lower interval bound
        - "upper": upper interval bound
        - "quantile": the conformal quantile used
    """
    model._check_is_fitted()
    X_cal = jnp.atleast_2d(jnp.asarray(X_cal))
    y_cal = jnp.asarray(y_cal).ravel()
    X_new = jnp.atleast_2d(jnp.asarray(X_new))

    # Compute nonconformity scores on calibration set
    y_cal_pred = model.predict(X_cal)
    scores = jnp.abs(y_cal - y_cal_pred)

    # Compute (1-alpha)(1 + 1/n) quantile
    n_cal = len(y_cal)
    q_level = min((1 - alpha) * (1 + 1 / n_cal), 1.0)
    quantile = float(jnp.quantile(scores, q_level))

    # Prediction intervals
    y_pred = model.predict(X_new)
    lower = y_pred - quantile
    upper = y_pred + quantile

    return {
        "y_pred": y_pred,
        "lower": lower,
        "upper": upper,
        "quantile": quantile,
    }


def conformal_predict_jackknife_plus(
    model: SymbolicRegressor,
    X_new: jnp.ndarray,
    alpha: float = 0.05,
) -> dict[str, jnp.ndarray]:
    """
    Jackknife+ conformal prediction intervals.

    Uses the LOO infrastructure to compute prediction intervals
    without a separate calibration set.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model (must have stored training data).
    X_new : jnp.ndarray
        New points for prediction.
    alpha : float
        Significance level (coverage = 1 - alpha).

    Returns
    -------
    result : dict
        Dictionary with keys:
        - "y_pred": point predictions
        - "lower": lower interval bound
        - "upper": upper interval bound
    """
    model._check_is_fitted()
    X_new = jnp.atleast_2d(jnp.asarray(X_new))

    if model._X_train is None or model._y_train is None:
        raise ValueError("Training data not available. Model must be fitted with fit().")

    X_train = model._X_train
    y_train = model._y_train
    n = len(y_train)

    # Compute design matrix for training data
    Phi_train = model.basis_library.evaluate_subset(X_train, model._result.selected_indices)
    coefficients = model._result.coefficients

    # Compute leverage (hat matrix diagonal)
    Phi_pinv = jnp.linalg.pinv(Phi_train)
    H = Phi_train @ Phi_pinv
    h_diag = jnp.diag(H)

    # LOO residuals: e_i / (1 - h_ii)
    y_pred_train = Phi_train @ coefficients
    residuals = y_train - y_pred_train
    loo_residuals = residuals / (1 - h_diag + 1e-10)

    # LOO predictions: y_hat_i^{-i} = y_i - e_i/(1-h_ii) ... wait,
    # y_hat_i^{-i} = y_hat_i - h_ii * e_i / (1 - h_ii)
    #             = y_pred_train - h_diag * loo_residuals
    # Actually: y_hat^{-i}_i = y_i - loo_residual_i
    # No: loo_residual_i = e_i / (1 - h_ii), and y_hat^{-i}_i = y_i - loo_residual_i
    # is wrong. The correct LOO prediction is:
    # y_hat^{-i}_i = y_hat_i - h_ii * residual_i / (1 - h_ii)
    #             = y_pred_train_i - h_diag_i * residuals_i / (1 - h_diag_i)
    # The LOO residual is y_i - y_hat^{-i}_i = residuals_i / (1 - h_diag_i)
    # So nonconformity scores are |loo_residuals|
    scores = jnp.abs(loo_residuals)

    # Jackknife+ quantile
    q_level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    quantile = float(jnp.quantile(scores, q_level))

    # Prediction intervals
    y_pred = model.predict(X_new)
    lower = y_pred - quantile
    upper = y_pred + quantile

    return {
        "y_pred": y_pred,
        "lower": lower,
        "upper": upper,
    }


# =============================================================================
# Bootstrap Methods (Phase 3, Strategies 5 + 6)
# =============================================================================


def bootstrap_coefficients(
    model: SymbolicRegressor,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Residual bootstrap for coefficient uncertainty.

    Resamples residuals, creates y* = y_hat + e*, and refits OLS.
    No Gaussian assumption needed.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model.
    n_bootstrap : int
        Number of bootstrap samples.
    alpha : float
        Significance level for confidence intervals.
    seed : int, optional
        Random seed.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - "coefficients": (n_bootstrap, p) array of bootstrap coefficients
        - "mean": mean of bootstrap coefficients
        - "std": std of bootstrap coefficients
        - "lower": lower CI bound for each coefficient
        - "upper": upper CI bound for each coefficient
        - "names": coefficient names
    """
    model._check_is_fitted()

    if model._X_train is None or model._y_train is None:
        raise ValueError("Training data not available.")

    Phi_train = model.basis_library.evaluate_subset(model._X_train, model._result.selected_indices)
    y_train = model._y_train
    coefficients = model._result.coefficients
    y_hat = Phi_train @ coefficients
    residuals = y_train - y_hat

    rng = np.random.RandomState(seed)
    n = len(y_train)

    # Generate all bootstrap y* at once
    boot_indices = rng.randint(0, n, size=(n_bootstrap, n))
    boot_residuals = np.array(residuals)[boot_indices]  # (n_bootstrap, n)
    y_star = np.array(y_hat)[None, :] + boot_residuals  # (n_bootstrap, n)

    # Vectorized OLS: beta* = (Phi^T Phi)^{-1} Phi^T y*
    # Pre-compute pseudo-inverse
    Phi_np = np.array(Phi_train)
    PhiTPhiInv_PhiT = np.linalg.pinv(Phi_np)  # (p, n)

    # All bootstrap coefficients at once
    boot_coeffs = (PhiTPhiInv_PhiT @ y_star.T).T  # (n_bootstrap, p)

    boot_coeffs = jnp.array(boot_coeffs)
    lower = jnp.percentile(boot_coeffs, 100 * alpha / 2, axis=0)
    upper = jnp.percentile(boot_coeffs, 100 * (1 - alpha / 2), axis=0)

    return {
        "coefficients": boot_coeffs,
        "mean": jnp.mean(boot_coeffs, axis=0),
        "std": jnp.std(boot_coeffs, axis=0),
        "lower": lower,
        "upper": upper,
        "names": model._result.selected_names,
    }


def bootstrap_predict(
    model: SymbolicRegressor,
    X_new: jnp.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, jnp.ndarray]:
    """
    Bootstrap prediction intervals.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model.
    X_new : jnp.ndarray
        New input data.
    n_bootstrap : int
        Number of bootstrap samples.
    alpha : float
        Significance level.
    seed : int, optional
        Random seed.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - "y_pred": point prediction (from original model)
        - "y_mean": mean of bootstrap predictions
        - "y_std": std of bootstrap predictions
        - "lower": lower prediction bound
        - "upper": upper prediction bound
    """
    model._check_is_fitted()
    X_new = jnp.atleast_2d(jnp.asarray(X_new))

    boot_result = bootstrap_coefficients(model, n_bootstrap, alpha, seed)
    boot_coeffs = boot_result["coefficients"]  # (n_bootstrap, p)

    Phi_new = model.basis_library.evaluate_subset(X_new, model._result.selected_indices)

    # All bootstrap predictions: (n_bootstrap, n_new)
    boot_preds = boot_coeffs @ Phi_new.T

    y_pred = model.predict(X_new)
    lower = jnp.percentile(boot_preds, 100 * alpha / 2, axis=0)
    upper = jnp.percentile(boot_preds, 100 * (1 - alpha / 2), axis=0)

    return {
        "y_pred": y_pred,
        "y_mean": jnp.mean(boot_preds, axis=0),
        "y_std": jnp.std(boot_preds, axis=0),
        "lower": lower,
        "upper": upper,
    }


def bootstrap_model_selection(
    model: SymbolicRegressor,
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_bootstrap: int = 100,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Pairs bootstrap for model selection stability.

    Resamples (X_i, y_i) pairs, reruns full model.fit(), and tracks
    which features are selected across bootstrap samples.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model (used as template).
    X : jnp.ndarray
        Training features.
    y : jnp.ndarray
        Training targets.
    n_bootstrap : int
        Number of bootstrap samples.
    seed : int, optional
        Random seed.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - "feature_frequencies": dict mapping feature name to selection frequency
        - "stability_score": fraction of bootstraps selecting the same features
        - "expressions": list of expressions found across bootstraps
    """
    X = jnp.atleast_2d(jnp.asarray(X))
    y = jnp.asarray(y).ravel()
    n = len(y)

    rng = np.random.RandomState(seed)

    feature_counts: dict[str, int] = {}
    original_features = set(model.selected_features_)
    same_count = 0
    expressions = []

    for _b in range(n_bootstrap):
        boot_idx = rng.randint(0, n, size=n)
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]

        # Clone model
        model_boot = model.__class__(
            basis_library=model.basis_library,
            max_terms=model.max_terms,
            strategy=model.strategy,
            information_criterion=model.information_criterion,
            cv_folds=model.cv_folds,
            regularization=model.regularization,
            constraints=model.constraints,
            random_state=model.random_state,
        )
        try:
            model_boot.fit(X_boot, y_boot)
            selected = set(model_boot.selected_features_)
            for feat in selected:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
            if selected == original_features:
                same_count += 1
            expressions.append(model_boot.expression_)
        except Exception:
            continue

    total = max(sum(1 for _ in expressions), 1)
    feature_frequencies = {k: v / total for k, v in feature_counts.items()}
    stability_score = same_count / total

    return {
        "feature_frequencies": feature_frequencies,
        "stability_score": stability_score,
        "expressions": expressions,
    }


# =============================================================================
# ANOVA (Analysis of Variance)
# =============================================================================


@dataclass
class AnovaRow:
    """A single row of the ANOVA table.

    Parameters
    ----------
    source : str
        Name of the source of variation (term name, "Model", "Residual",
        or "Total").
    df : int
        Degrees of freedom.
    sum_sq : float
        Sum of squares.
    mean_sq : float
        Mean sum of squares (``sum_sq / df``).
    f_value : float or None
        F-statistic (``None`` for Residual and Total rows).
    p_value : float or None
        p-value from the F-distribution (``None`` for Residual and Total rows).
    """

    source: str
    df: int
    sum_sq: float
    mean_sq: float
    f_value: float | None = None
    p_value: float | None = None


@dataclass
class AnovaResult:
    """Result of an ANOVA decomposition.

    Parameters
    ----------
    rows : list of AnovaRow
        Per-term rows followed by Model, Residual, and Total summary rows.
    type : str
        ANOVA type: ``"sequential"`` (Type I) or ``"marginal"`` (Type III).
    warnings : list of str
        Diagnostic messages (e.g. when p-values are approximate).
    """

    rows: list[AnovaRow] = field(default_factory=list)
    type: str = "sequential"
    warnings: list[str] = field(default_factory=list)

    # -- helpers for convenient access ------------------------------------

    @property
    def term_names(self) -> list[str]:
        """Names of the individual term rows (excludes summary rows)."""
        summary = {"Model", "Residual", "Total"}
        return [r.source for r in self.rows if r.source not in summary]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the table to a plain dictionary."""
        return {
            "type": self.type,
            "warnings": list(self.warnings),
            "rows": [
                {
                    "source": r.source,
                    "df": r.df,
                    "sum_sq": r.sum_sq,
                    "mean_sq": r.mean_sq,
                    "f_value": r.f_value,
                    "p_value": r.p_value,
                }
                for r in self.rows
            ],
        }

    def __repr__(self) -> str:  # noqa: D105
        hdr = f"ANOVA Table (Type: {self.type})\n"
        line = f"{'Source':<25s} {'df':>4s} {'SS':>12s} {'MS':>12s} {'F':>10s} {'p':>10s}\n"
        sep = "-" * 77 + "\n"
        body = ""
        for r in self.rows:
            f_str = f"{r.f_value:10.4f}" if r.f_value is not None else " " * 10
            p_str = f"{r.p_value:10.4g}" if r.p_value is not None else " " * 10
            body += (
                f"{r.source:<25s} {r.df:4d} {r.sum_sq:12.4f} {r.mean_sq:12.4f} {f_str} {p_str}\n"
            )
        warn = ""
        if self.warnings:
            warn = "\nWarnings:\n" + "\n".join(f"  - {w}" for w in self.warnings) + "\n"
        return hdr + sep + line + sep + body + sep + warn


def anova(
    model: SymbolicRegressor,
    anova_type: str = "sequential",
) -> AnovaResult:
    """Perform ANOVA on a fitted :class:`SymbolicRegressor`.

    Decomposes the total sum of squares into contributions from each
    selected basis-function term plus a residual, and tests each term's
    significance with an F-test.

    Two decomposition types are supported:

    * **``"sequential"``** (Type I): terms are added in the order they
      appear in ``model.selected_features_`` and each term's contribution
      is the *extra* sum of squares beyond the previous terms.
    * **``"marginal"``** (Type III): each term's contribution is the
      extra sum of squares from adding it *last*, after all other terms.

    Parameters
    ----------
    model : SymbolicRegressor
        A fitted model.
    anova_type : str
        ``"sequential"`` (Type I) or ``"marginal"`` (Type III).

    Returns
    -------
    AnovaResult
        ANOVA table with per-term rows plus Model, Residual, and Total
        summary rows.

    Notes
    -----
    The F-test p-values assume independent, normally distributed residuals
    and an unconstrained OLS fit.  They are **approximate** when:

    * The model contains **parametric (nonlinear) basis functions** whose
      parameters were optimised during fitting, because the effective
      degrees of freedom are larger than reported.
    * **Constraints** were applied, since the coefficient estimates are
      no longer ordinary least-squares.

    In these cases the sums-of-squares decomposition is still meaningful,
    but the p-values should be treated as indicative rather than exact.
    The returned :pyclass:`AnovaResult` includes diagnostic warnings when
    these conditions are detected.

    Examples
    --------
    >>> from jaxsr.uncertainty import anova
    >>> table = anova(model)
    >>> print(table)
    >>> for row in table.rows:
    ...     print(row.source, row.f_value, row.p_value)
    """
    model._check_is_fitted()
    anova_type = anova_type.lower()
    if anova_type not in ("sequential", "marginal"):
        raise ValueError(f"anova_type must be 'sequential' or 'marginal', got '{anova_type}'")

    X = model._X_train
    y = model._y_train
    n = len(y)
    selected = list(np.array(model._result.selected_indices))
    names = list(model._result.selected_names)
    p = len(selected)

    # Full model design matrix and residuals
    Phi_full = model.basis_library.evaluate_subset(X, selected)
    y_hat_full = Phi_full @ model._result.coefficients
    ss_res_full = float(jnp.sum((y - y_hat_full) ** 2))
    ss_tot = float(jnp.sum((y - jnp.mean(y)) ** 2))
    ss_model = ss_tot - ss_res_full
    df_res = n - p
    ms_res = ss_res_full / df_res if df_res > 0 else float("inf")

    # -- Diagnostics / warnings -------------------------------------------
    warn: list[str] = []
    if model.basis_library.has_parametric:
        warn.append(
            "Model contains parametric (nonlinear) basis functions. "
            "F-test p-values are approximate because the effective "
            "degrees of freedom do not account for nonlinear parameter "
            "optimisation."
        )
    if model.constraints is not None:
        warn.append(
            "Constrained coefficients were used. F-test p-values are "
            "approximate because the fit is not unconstrained OLS."
        )

    # -- Per-term SS computation ------------------------------------------
    rows: list[AnovaRow] = []

    if anova_type == "sequential":
        # Type I: add terms one at a time in order
        ss_prev = ss_tot  # residual SS of the "null" (intercept-free) model
        for k in range(p):
            subset = selected[: k + 1]
            Phi_k = model.basis_library.evaluate_subset(X, subset)
            beta_k = jnp.linalg.lstsq(Phi_k, y, rcond=None)[0]
            ss_res_k = float(jnp.sum((y - Phi_k @ beta_k) ** 2))
            ss_term = ss_prev - ss_res_k  # extra SS explained by this term
            ss_term = max(ss_term, 0.0)
            ms_term = ss_term  # df = 1 per term
            f_val = ms_term / ms_res if ms_res > 0 else float("inf")
            p_val = float(1.0 - stats.f.cdf(f_val, 1, df_res)) if df_res > 0 else float("nan")
            rows.append(
                AnovaRow(
                    source=names[k],
                    df=1,
                    sum_sq=ss_term,
                    mean_sq=ms_term,
                    f_value=f_val,
                    p_value=p_val,
                )
            )
            ss_prev = ss_res_k
    else:
        # Type III: each term's contribution is the extra SS when adding
        # it last (i.e. removing it from the full model).
        for k in range(p):
            subset_minus_k = [s for j, s in enumerate(selected) if j != k]
            if len(subset_minus_k) == 0:
                # Only one term — its marginal SS is the whole model SS
                ss_term = ss_model
            else:
                Phi_mk = model.basis_library.evaluate_subset(X, subset_minus_k)
                beta_mk = jnp.linalg.lstsq(Phi_mk, y, rcond=None)[0]
                ss_res_mk = float(jnp.sum((y - Phi_mk @ beta_mk) ** 2))
                ss_term = ss_res_mk - ss_res_full
            ss_term = max(ss_term, 0.0)
            ms_term = ss_term
            f_val = ms_term / ms_res if ms_res > 0 else float("inf")
            p_val = float(1.0 - stats.f.cdf(f_val, 1, df_res)) if df_res > 0 else float("nan")
            rows.append(
                AnovaRow(
                    source=names[k],
                    df=1,
                    sum_sq=ss_term,
                    mean_sq=ms_term,
                    f_value=f_val,
                    p_value=p_val,
                )
            )

    # -- Summary rows -----------------------------------------------------
    # SS_total is about the mean (df = n-1), so df_model = n - 1 - df_res
    # which equals p - 1 when the model includes an intercept.
    df_model = n - 1 - df_res
    df_model = max(df_model, 1)
    ms_model = ss_model / df_model if df_model > 0 else 0.0
    f_model = ms_model / ms_res if ms_res > 0 else float("inf")
    p_model = float(1.0 - stats.f.cdf(f_model, df_model, df_res)) if df_res > 0 else float("nan")

    rows.append(
        AnovaRow(
            source="Model",
            df=df_model,
            sum_sq=ss_model,
            mean_sq=ms_model,
            f_value=f_model,
            p_value=p_model,
        )
    )
    rows.append(
        AnovaRow(
            source="Residual",
            df=df_res,
            sum_sq=ss_res_full,
            mean_sq=ms_res,
        )
    )
    rows.append(
        AnovaRow(
            source="Total",
            df=n - 1,
            sum_sq=ss_tot,
            mean_sq=ss_tot / (n - 1) if n > 1 else 0.0,
        )
    )

    return AnovaResult(rows=rows, type=anova_type, warnings=warn)


# =============================================================================
# Classification Uncertainty Quantification
# =============================================================================


def classification_coefficient_intervals(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    coefficients: jnp.ndarray,
    names: list[str],
    alpha: float = 0.05,
) -> dict[str, tuple[float, float, float, float]]:
    """
    Wald confidence intervals for logistic regression coefficients.

    Uses the Fisher information matrix
    ``I(w) = Phi^T diag(mu*(1-mu)) Phi`` to compute asymptotic
    standard errors, then applies Normal quantiles (MLE asymptotics).

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix of shape ``(n, p)``.
    y : jnp.ndarray
        Binary labels of shape ``(n,)``.
    coefficients : jnp.ndarray
        Fitted logistic regression coefficients.
    names : list of str
        Coefficient names.
    alpha : float
        Significance level (default 0.05 for 95% CIs).

    Returns
    -------
    intervals : dict
        ``{name: (estimate, lower, upper, se)}`` for each coefficient.
    """
    from .selection import _sigmoid

    eta = Phi @ coefficients
    mu = _sigmoid(eta)
    mu = jnp.clip(mu, 1e-10, 1.0 - 1e-10)
    W_diag = mu * (1.0 - mu)

    # Fisher information matrix: I = Phi^T W Phi
    # Use SVD of sqrt(W)*Phi for stability
    sqrt_W = jnp.sqrt(W_diag)
    Phi_w = Phi * sqrt_W[:, None]
    U, s, Vt = jnp.linalg.svd(Phi_w, full_matrices=False)

    rcond = jnp.finfo(Phi.dtype).eps * max(Phi.shape)
    cutoff = rcond * jnp.max(s)
    s_inv_sq = jnp.where(s > cutoff, 1.0 / (s**2), 0.0)
    fisher_inv = Vt.T @ jnp.diag(s_inv_sq) @ Vt

    se = jnp.sqrt(jnp.diag(fisher_inv))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    intervals = {}
    for i, name in enumerate(names):
        est = float(coefficients[i])
        std_err = float(se[i])
        lower = est - z_crit * std_err
        upper = est + z_crit * std_err
        intervals[name] = (est, lower, upper, std_err)

    return intervals


def bootstrap_classification_coefficients(
    model,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Pairs bootstrap for logistic regression coefficient uncertainty.

    Resamples ``(X_i, y_i)`` pairs, refits IRLS on each bootstrap
    sample, and collects coefficient distributions.

    Parameters
    ----------
    model : SymbolicClassifier
        Fitted binary classifier.
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        Significance level for confidence intervals.
    seed : int, optional
        Random seed.

    Returns
    -------
    result : dict
        Dictionary with keys:

        - ``"coefficients"``: ``(n_bootstrap, p)`` array
        - ``"mean"``: mean of bootstrap coefficients
        - ``"std"``: std of bootstrap coefficients
        - ``"lower"``: lower CI bound per coefficient
        - ``"upper"``: upper CI bound per coefficient
        - ``"names"``: coefficient names
    """
    from .selection import fit_irls

    model._check_is_fitted()
    X = model._X_train
    y = model._y_train
    n = len(y)

    Phi_train = model.basis_library.evaluate_subset(X, model._result.selected_indices)

    rng = np.random.RandomState(seed)
    boot_coeffs_list = []

    for _b in range(n_bootstrap):
        boot_idx = rng.randint(0, n, size=n)
        Phi_boot = Phi_train[boot_idx]
        y_boot = y[boot_idx]

        try:
            coeffs, _nll, _nit, _conv = fit_irls(
                Phi_boot,
                y_boot,
                regularization=model.regularization,
            )
            boot_coeffs_list.append(np.array(coeffs))
        except Exception:
            continue

    if not boot_coeffs_list:
        p = len(model._result.coefficients)
        return {
            "coefficients": jnp.zeros((0, p)),
            "mean": jnp.zeros(p),
            "std": jnp.zeros(p),
            "lower": jnp.zeros(p),
            "upper": jnp.zeros(p),
            "names": model._result.selected_names,
        }

    boot_coeffs = jnp.array(np.stack(boot_coeffs_list))
    lower = jnp.percentile(boot_coeffs, 100 * alpha / 2, axis=0)
    upper = jnp.percentile(boot_coeffs, 100 * (1 - alpha / 2), axis=0)

    return {
        "coefficients": boot_coeffs,
        "mean": jnp.mean(boot_coeffs, axis=0),
        "std": jnp.std(boot_coeffs, axis=0),
        "lower": lower,
        "upper": upper,
        "names": model._result.selected_names,
    }


def conformal_classification_split(
    model,
    X_cal: jnp.ndarray,
    y_cal: jnp.ndarray,
    X_new: jnp.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Split conformal prediction sets for classification.

    Returns prediction sets (sets of labels) with marginal coverage
    guarantee.  Uses the nonconformity score ``1 - P(y_true | x)``.

    Parameters
    ----------
    model : SymbolicClassifier
        Fitted classification model.
    X_cal : jnp.ndarray
        Calibration features of shape ``(n_cal, p)``.
    y_cal : jnp.ndarray
        Calibration labels of shape ``(n_cal,)``.
    X_new : jnp.ndarray
        New features of shape ``(n_new, p)``.
    alpha : float
        Significance level (coverage = 1 - alpha).

    Returns
    -------
    result : dict
        Dictionary with keys:

        - ``"prediction_sets"``: list of sets of predicted labels
        - ``"quantile"``: the conformal quantile used
        - ``"y_pred"``: point predictions (most likely class)
    """
    model._check_is_fitted()
    X_cal = jnp.atleast_2d(jnp.asarray(X_cal))
    y_cal = jnp.asarray(y_cal).ravel()
    X_new = jnp.atleast_2d(jnp.asarray(X_new))

    proba_cal = model.predict_proba(X_cal)
    n_cal = len(y_cal)

    # Nonconformity scores: 1 - P(y_true | x)
    if proba_cal.ndim == 1:
        # Binary: proba_cal is P(y=1)
        p_true = jnp.where(y_cal == 1, proba_cal, 1 - proba_cal)
    else:
        idx = jnp.arange(n_cal)
        y_int = y_cal.astype(int)
        p_true = proba_cal[idx, y_int]

    scores = 1.0 - p_true

    # Quantile with finite-sample correction
    q_level = min((1 - alpha) * (1 + 1 / n_cal), 1.0)
    quantile = float(jnp.quantile(scores, q_level))

    # Construct prediction sets
    proba_new = model.predict_proba(X_new)
    classes = np.asarray(model.classes_)
    prediction_sets = []

    if proba_new.ndim == 1:
        # Binary
        for i in range(len(X_new)):
            pset = set()
            if 1 - float(proba_new[i]) <= quantile:
                pset.add(int(classes[1]))
            if float(proba_new[i]) <= quantile:
                pset.add(int(classes[0]))
            if not pset:
                pset.add(int(classes[1]) if float(proba_new[i]) >= 0.5 else int(classes[0]))
            prediction_sets.append(pset)
    else:
        for i in range(len(X_new)):
            pset = set()
            for k in range(len(classes)):
                if 1 - float(proba_new[i, k]) <= quantile:
                    pset.add(int(classes[k]))
            if not pset:
                pset.add(int(classes[int(jnp.argmax(proba_new[i]))]))
            prediction_sets.append(pset)

    y_pred = model.predict(X_new)

    return {
        "prediction_sets": prediction_sets,
        "quantile": quantile,
        "y_pred": y_pred,
    }


def calibration_curve(
    y_true: jnp.ndarray,
    y_prob: jnp.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """
    Compute reliability diagram data for binary classification.

    Bins predicted probabilities and computes the observed fraction of
    positives (``fraction_of_positives``) and the average predicted
    probability (``mean_predicted_value``) in each bin.

    Parameters
    ----------
    y_true : jnp.ndarray
        True binary labels.
    y_prob : jnp.ndarray
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins.

    Returns
    -------
    result : dict
        Dictionary with keys:

        - ``"fraction_of_positives"``: observed positive rate per bin
        - ``"mean_predicted_value"``: mean predicted probability per bin
        - ``"bin_counts"``: number of samples per bin
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    fractions = []
    means = []
    counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = mask.sum()
        if count > 0:
            fractions.append(y_true[mask].mean())
            means.append(y_prob[mask].mean())
            counts.append(count)

    return {
        "fraction_of_positives": np.array(fractions),
        "mean_predicted_value": np.array(means),
        "bin_counts": np.array(counts),
    }
