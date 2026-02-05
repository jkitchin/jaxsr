"""
Uncertainty Quantification for JAXSR.

Provides classical OLS intervals, Pareto front ensemble predictions,
Bayesian Model Averaging, conformal prediction, and bootstrap methods.

Since JAXSR models are linear-in-parameters (y = Phi @ beta), classical
OLS inference applies directly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import jax
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
    ssr = float(jnp.sum(residuals ** 2))
    dof = n - p
    if dof <= 0:
        warnings.warn(
            f"Degrees of freedom ({dof}) <= 0. Cannot compute unbiased variance."
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
    s_inv_sq = jnp.where(s > cutoff, 1.0 / (s ** 2), 0.0)
    PhiTPhiInv = Vt.T @ jnp.diag(s_inv_sq) @ Vt
    return sigma_sq * PhiTPhiInv


def coefficient_intervals(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    coefficients: jnp.ndarray,
    names: List[str],
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float, float, float]]:
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
        warnings.warn("Not enough degrees of freedom for coefficient intervals.")
        return {
            name: (float(coef), float("nan"), float("nan"), float("nan"))
            for name, coef in zip(names, coefficients)
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
) -> Dict[str, jnp.ndarray]:
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
    s_inv_sq = jnp.where(s > cutoff, 1.0 / (s ** 2), 0.0)
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
) -> Dict[str, jnp.ndarray]:
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
        top_k: Optional[int] = None,
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
    def weights(self) -> Dict[str, float]:
        """Model weights keyed by expression string."""
        return {
            r.expression(): float(w)
            for r, w in zip(self._results, self._weights)
        }

    @property
    def expressions(self) -> List[str]:
        """Expressions of models in the average."""
        return [r.expression() for r in self._results]

    def predict(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
) -> Dict[str, jnp.ndarray]:
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
) -> Dict[str, jnp.ndarray]:
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
        raise ValueError(
            "Training data not available. Model must be fitted with fit()."
        )

    X_train = model._X_train
    y_train = model._y_train
    n = len(y_train)

    # Compute design matrix for training data
    Phi_train = model.basis_library.evaluate_subset(
        X_train, model._result.selected_indices
    )
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
    seed: Optional[int] = None,
) -> Dict[str, Any]:
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

    Phi_train = model.basis_library.evaluate_subset(
        model._X_train, model._result.selected_indices
    )
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
    PhiT = Phi_np.T
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
    seed: Optional[int] = None,
) -> Dict[str, jnp.ndarray]:
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

    Phi_new = model.basis_library.evaluate_subset(
        X_new, model._result.selected_indices
    )

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
    seed: Optional[int] = None,
) -> Dict[str, Any]:
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

    feature_counts: Dict[str, int] = {}
    original_features = set(model.selected_features_)
    same_count = 0
    expressions = []

    for b in range(n_bootstrap):
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
