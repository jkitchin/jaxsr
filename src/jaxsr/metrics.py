"""
Metrics and Model Comparison for JAXSR.

Provides information criteria, cross-validation scores, and model comparison utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from .utils import validate_sample_weight, weighted_mean, whiten

if TYPE_CHECKING:
    from .regressor import SymbolicRegressor

# NumPy 2.0 renamed trapz -> trapezoid
_np_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


# =============================================================================
# Information Criteria
# =============================================================================
#
# Effective sample size under sample weights
# ------------------------------------------
# JAXSR treats ``sample_weight`` as *relative precision*: observation ``i`` is
# assumed to have variance ``sigma^2 / w_i``.  Weights are normalised to sum to
# ``n`` (see :func:`jaxsr.utils.validate_sample_weight`) and the ``n_samples``
# passed to every criterion below stays the nominal ``n`` -- the number of
# observations actually measured.  Consequences worth knowing:
#
# * The criteria are invariant to the overall scale of the weights, so ``w`` and
#   ``1000 * w`` select the same model.
# * Weighting does not manufacture or destroy observations.  ``k * log(n)`` in
#   BIC penalises complexity by how much data was collected, not by how much of
#   it happened to be precise.  This is why row duplication is *not* a valid way
#   to emulate weights: it inflates ``n`` and shifts every criterion.
# * The MSE fed to the criteria is the weighted MSE ``sum_i w_i r_i^2 / n``.
#
# If most of the weight sits on a handful of rows, the nominal ``n`` overstates
# how much information the comparison rests on;
# :func:`jaxsr.utils.effective_sample_size` reports the Kish effective sample
# size for that diagnosis.

# Smallest MSE that still produces a finite log-likelihood.
_MSE_FLOOR = float(np.finfo(float).tiny)


def _floor_mse(mse: float) -> float:
    """
    Clamp an MSE away from zero so that ``log(mse)`` stays finite.

    A perfect fit has ``mse == 0`` and is the *best* possible model, so it must
    not be scored ``+inf`` -- every information criterion here is
    lower-is-better, and ``+inf`` would make selection reject the exact model.
    Exact zeros are routine in float64, where ``selection.py`` computes MSE via
    the closed form ``(yTy - c @ rhs) / n``, which cancels to 0.0 on a perfect
    fit (in float32 the residual-based path returns a small positive number
    instead, which is why this only shows up at higher precision).

    Parameters
    ----------
    mse : float
        Mean squared error. Negative values are treated as zero.

    Returns
    -------
    float
        ``mse`` clamped to at least the smallest positive normal float.
    """
    return max(float(mse), _MSE_FLOOR)


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

    mse = _floor_mse(mse)

    # AIC = -2 * log_likelihood + 2 * k
    # For Gaussian: log_likelihood = -n/2 * log(2*pi*sigma^2) - n/2
    # Simplified: AIC = n * log(MSE) + 2 * k
    log_lik = -n / 2 * math.log(2 * math.pi * mse) - n / 2
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

    mse = _floor_mse(mse)

    log_lik = -n / 2 * math.log(2 * math.pi * mse) - n / 2
    return float(-2 * log_lik + k * math.log(n))


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

    if n <= 2:
        return float("inf")
    mse = _floor_mse(mse)

    log_log_n = math.log(math.log(n))
    if log_log_n <= 0:
        return float("inf")

    log_lik = -n / 2 * math.log(2 * math.pi * mse) - n / 2
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

    mse = _floor_mse(mse)

    return float(n / 2 * math.log(mse) + k / 2 * math.log(n))


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

    Notes
    -----
    Under sample weights, pass the *nominal* number of observations as
    ``n_samples`` and the weighted MSE ``sum_i w_i r_i^2 / n`` as ``mse``.
    See the effective-sample-size note at the top of this module.
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

CV_STRATEGIES = ("kfold", "group-kfold", "leave-one-group-out")


def _group_label(value: Any) -> Any:
    """Convert a numpy scalar group label to a plain Python object."""
    return value.item() if hasattr(value, "item") else value


def group_indices(groups: Any) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Split row indices by group label.

    Parameters
    ----------
    groups : array-like of shape (n_samples,)
        Group label for each row. Labels may be integers, floats, or strings.

    Returns
    -------
    unique : np.ndarray
        Unique group labels, sorted.
    row_indices : list of np.ndarray
        ``row_indices[i]`` holds the row indices belonging to ``unique[i]``.

    Raises
    ------
    ValueError
        If ``groups`` is empty or not one-dimensional after raveling.
    """
    groups_arr = np.asarray(groups).ravel()
    if groups_arr.size == 0:
        raise ValueError("groups must contain at least one label.")
    unique = np.unique(groups_arr)
    return unique, [np.flatnonzero(groups_arr == g) for g in unique]


def _kfold_splits(
    n_samples: int, cv: int, rng: np.random.RandomState
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build shuffled k-fold (train, test) index pairs over rows."""
    indices = rng.permutation(n_samples)
    fold_size = n_samples // cv
    splits = []
    for i in range(cv):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < cv - 1 else n_samples
        test_idx = indices[start_idx:end_idx]
        train_idx = np.concatenate([indices[:start_idx], indices[end_idx:]])
        splits.append((train_idx, test_idx))
    return splits


def _group_kfold_splits(
    row_indices: list[np.ndarray], cv: int
) -> list[tuple[np.ndarray, np.ndarray, list[int]]]:
    """
    Assign whole groups to ``cv`` folds, balancing rows per fold.

    Groups are placed largest-first into the fold that currently holds the
    fewest rows, so no group is split across the train/test boundary.
    """
    fold_groups: list[list[int]] = [[] for _ in range(cv)]
    fold_sizes = np.zeros(cv, dtype=int)
    order = sorted(range(len(row_indices)), key=lambda g: (-len(row_indices[g]), g))
    for g in order:
        target = int(np.argmin(fold_sizes))
        fold_groups[target].append(g)
        fold_sizes[target] += len(row_indices[g])

    all_rows = np.arange(sum(len(idx) for idx in row_indices))
    splits = []
    for members in fold_groups:
        test_idx = np.sort(np.concatenate([row_indices[g] for g in members]))
        train_idx = np.setdiff1d(all_rows, test_idx)
        splits.append((train_idx, test_idx, sorted(members)))
    return splits


def _logo_splits(row_indices: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, list[int]]]:
    """Build leave-one-group-out (train, test, held-out groups) triples."""
    all_rows = np.arange(sum(len(idx) for idx in row_indices))
    splits = []
    for g, test_idx in enumerate(row_indices):
        splits.append((np.setdiff1d(all_rows, test_idx), test_idx, [g]))
    return splits


def cross_validate(
    model: SymbolicRegressor,
    X: jnp.ndarray,
    y: jnp.ndarray,
    cv: int = 5,
    scoring: str = "neg_mse",
    random_state: int | None = None,
    groups: Any | None = None,
    strategy: str = "kfold",
    sample_weight: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """
    Perform cross-validation, optionally holding out whole groups.

    When rows are not independent observations -- replicates of one
    experimental condition, points along one measured curve, samples from one
    subject -- splitting at the row level leaks information from the training
    set into the test set and reports an optimistic score. Passing ``groups``
    keeps every row of a group on the same side of the split.

    Parameters
    ----------
    model : SymbolicRegressor
        Model to evaluate. Cloned (unfitted) for every fold.
    X : jnp.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs).
    cv : int
        Number of folds. Ignored when ``strategy="leave-one-group-out"``.
    scoring : str
        Scoring metric: "neg_mse", "neg_mae", "r2".
    random_state : int, optional
        Random seed for fold splitting (used by ``strategy="kfold"`` only;
        the group strategies are deterministic).
    groups : array-like of shape (n_samples,), optional
        Group label for each row. Required by the group strategies. If given
        with the default ``strategy="kfold"``, the strategy is promoted to
        ``"group-kfold"``.
    strategy : str
        One of ``"kfold"`` (random rows), ``"group-kfold"`` (whole groups
        distributed over ``cv`` folds), or ``"leave-one-group-out"`` (one fold
        per group).
    sample_weight : jnp.ndarray, optional
        Per-sample weights.  Weights follow their rows into the folds under
        every strategy: each fold is fitted on its training weights and scored
        with its test weights, so down-weighted observations neither drive the
        fit nor dominate the score.  Per-group scores are weighted the same
        way.

    Returns
    -------
    results : dict
        Dictionary with keys:

        - "test_scores": array of test scores for each fold
        - "train_scores": array of train scores for each fold
        - "mean_test_score", "std_test_score": summary of the test scores
        - "mean_train_score", "std_train_score": summary of the train scores
        - "strategy": the strategy actually used
        - "n_splits": number of folds evaluated
        - "groups_out": list of the held-out group labels per fold (empty
          lists for ``"kfold"``)
        - "per_group_scores": dict mapping group label to the score on that
          group's rows while it was held out (empty for ``"kfold"``)
        - "edge_groups": the lowest and highest group labels when labels are
          numeric, else an empty list. These are the extrapolation cases --
          read their entries in "per_group_scores" separately from the mean,
          since interpolation and extrapolation carry different risk.
        - "edge_group_scores": "per_group_scores" restricted to "edge_groups"

    Raises
    ------
    ValueError
        If ``scoring`` or ``strategy`` is unknown, if ``cv < 2`` for a k-fold
        strategy, if ``groups`` is missing for a group strategy or has the
        wrong length, if there are fewer groups than folds, if
        ``sample_weight`` is invalid, or if a fold ends up with zero total
        weight on either side of its split.

    Examples
    --------
    >>> result = cross_validate(model, X, y, cv=5)  # doctest: +SKIP
    >>> result = cross_validate(  # doctest: +SKIP
    ...     model, X, y, groups=temperature_id, strategy="leave-one-group-out"
    ... )
    """
    from .regressor import _clone_estimator

    scoring_funcs = {
        "neg_mse": lambda y_true, y_pred, w: -compute_mse(y_true, y_pred, w),
        "neg_mae": lambda y_true, y_pred, w: -compute_mae(y_true, y_pred, w),
        "r2": lambda y_true, y_pred, w: compute_r2(y_true, y_pred, w),
    }
    if scoring not in scoring_funcs:
        raise ValueError(f"Unknown scoring: {scoring}. Available: {list(scoring_funcs.keys())}")
    score_func = scoring_funcs[scoring]

    if strategy not in CV_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(CV_STRATEGIES)}")
    if groups is not None and strategy == "kfold":
        strategy = "group-kfold"
    if groups is None and strategy != "kfold":
        raise ValueError(f"strategy={strategy!r} requires groups.")

    n_samples = X.shape[0]
    weights = validate_sample_weight(sample_weight, n_samples)
    rng = np.random.RandomState(random_state)

    unique_groups: np.ndarray | None = None
    if groups is None:
        if cv < 2:
            raise ValueError(f"cv must be at least 2, got {cv}.")
        if cv > n_samples:
            raise ValueError(f"cv={cv} exceeds the number of samples ({n_samples}).")
        splits = [(tr, te, []) for tr, te in _kfold_splits(n_samples, cv, rng)]
    else:
        unique_groups, row_indices = group_indices(groups)
        n_grouped_rows = sum(len(idx) for idx in row_indices)
        if n_grouped_rows != n_samples:
            raise ValueError(
                f"groups has {np.asarray(groups).ravel().size} labels "
                f"but X has {n_samples} rows."
            )
        if len(unique_groups) < 2:
            raise ValueError("Group-based cross-validation requires at least 2 distinct groups.")
        if strategy == "leave-one-group-out":
            splits = _logo_splits(row_indices)
        else:
            if cv < 2:
                raise ValueError(f"cv must be at least 2, got {cv}.")
            if cv > len(unique_groups):
                raise ValueError(
                    f"cv={cv} exceeds the number of distinct groups ({len(unique_groups)}). "
                    f"Use a smaller cv or strategy='leave-one-group-out'."
                )
            splits = _group_kfold_splits(row_indices, cv)

    test_scores = []
    train_scores = []
    groups_out: list[list[Any]] = []
    per_group_scores: dict[Any, float] = {}

    for fold, (train_idx, test_idx, held_out) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = None if weights is None else weights[train_idx]
        w_test = None if weights is None else weights[test_idx]
        for split, w_split in (("train", w_train), ("test", w_test)):
            if w_split is not None and float(jnp.sum(w_split)) <= 0:
                raise ValueError(
                    f"Fold {fold} has zero total sample_weight in its {split} split; "
                    "cross-validation cannot score it. Drop the zero-weight rows, "
                    "use fewer folds, or group differently."
                )

        # Clone and fit model
        model_clone = _clone_estimator(model)
        model_clone.fit(X_train, y_train, sample_weight=w_train)

        y_pred_test = model_clone.predict(X_test)
        y_pred_train = model_clone.predict(X_train)

        test_scores.append(score_func(y_test, y_pred_test, w_test))
        train_scores.append(score_func(y_train, y_pred_train, w_train))

        fold_labels = []
        for g in held_out:
            label = _group_label(unique_groups[g])
            fold_labels.append(label)
            # Score each held-out group on its own rows, so a group that the
            # model extrapolates badly to is not averaged away by its fold.
            rows = np.flatnonzero(np.isin(test_idx, row_indices[g]))
            w_rows = None if w_test is None else w_test[rows]
            if w_rows is not None and float(jnp.sum(w_rows)) <= 0:
                # Every row of this group was zero-weighted; scoring it would
                # divide by zero, and reporting 0.0 would read as a real score.
                per_group_scores[label] = float("nan")
            else:
                per_group_scores[label] = score_func(y_test[rows], y_pred_test[rows], w_rows)
        groups_out.append(fold_labels)

    test_scores = np.array(test_scores)
    train_scores = np.array(train_scores)

    edge_groups: list[Any] = []
    if unique_groups is not None and np.issubdtype(unique_groups.dtype, np.number):
        edge_groups = [_group_label(unique_groups[0]), _group_label(unique_groups[-1])]

    return {
        "test_scores": test_scores,
        "train_scores": train_scores,
        "mean_test_score": float(np.mean(test_scores)),
        "std_test_score": float(np.std(test_scores)),
        "mean_train_score": float(np.mean(train_scores)),
        "std_train_score": float(np.std(train_scores)),
        "strategy": strategy,
        "n_splits": len(splits),
        "groups_out": groups_out,
        "per_group_scores": per_group_scores,
        "edge_groups": edge_groups,
        "edge_group_scores": {g: per_group_scores[g] for g in edge_groups if g in per_group_scores},
    }


def compute_cv_score(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    cv: int = 5,
    random_state: int | None = None,
    sample_weight: jnp.ndarray | None = None,
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
    sample_weight : jnp.ndarray, optional
        Per-sample weights.  Each fold is fitted by weighted least squares on
        its training weights and scored by weighted MSE on its test weights.

    Returns
    -------
    cv_mse : float
        Mean cross-validation MSE.

    Raises
    ------
    ValueError
        If ``sample_weight`` is invalid.
    """
    n_samples = Phi.shape[0]
    weights = validate_sample_weight(sample_weight, n_samples)
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
        w_train = None if weights is None else weights[train_idx]
        w_test = None if weights is None else weights[test_idx]

        # Solve (weighted) least squares
        coeffs, _, _, _ = jnp.linalg.lstsq(
            whiten(Phi_train, w_train), whiten(y_train, w_train), rcond=None
        )
        y_pred = Phi_test @ coeffs
        mse_scores.append(compute_mse(y_test, y_pred, w_test))

    return float(np.mean(mse_scores))


def compute_loo_mse(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    coefficients: jnp.ndarray,
    sample_weight: jnp.ndarray | None = None,
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
    sample_weight : jnp.ndarray, optional
        Per-sample weights.  The leverages become the weighted-least-squares
        leverages ``w_i * phi_i^T (Phi^T W Phi)^-1 phi_i`` and the LOO
        residuals are averaged with the same weights.

    Returns
    -------
    loo_mse : float
        Leave-one-out mean squared error.

    Raises
    ------
    ValueError
        If ``sample_weight`` is invalid.
    """
    w = validate_sample_weight(sample_weight, len(y))
    residuals = y - Phi @ coefficients

    # Hat matrix diagonal of the whitened problem:
    #   h_ii = w_i * phi_i^T (Phi^T W Phi)^-1 phi_i
    # which reduces to the ordinary leverage when the weights are all 1.
    Phi_w = whiten(Phi, w)
    h_diag = jnp.sum(Phi_w * jnp.linalg.pinv(Phi_w).T, axis=1)

    # LOO residual: e_i / (1 - h_ii)
    loo_residuals = residuals / (1 - h_diag + 1e-10)

    return float(weighted_mean(loo_residuals**2, w))


def compute_press(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    coefficients: jnp.ndarray,
    sample_weight: jnp.ndarray | None = None,
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
    sample_weight : jnp.ndarray, optional
        Per-sample weights; see :func:`compute_loo_mse`.

    Returns
    -------
    press : float
        PRESS statistic.

    Raises
    ------
    ValueError
        If ``sample_weight`` is invalid.
    """
    return compute_loo_mse(Phi, y, coefficients, sample_weight) * len(y)


# =============================================================================
# Regression Metrics
# =============================================================================


def compute_mse(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: jnp.ndarray | None = None,
) -> float:
    """
    Compute mean squared error, optionally weighted.

    Parameters
    ----------
    y_true : jnp.ndarray
        True values of shape ``(n_samples,)``.
    y_pred : jnp.ndarray
        Predicted values of shape ``(n_samples,)``.
    sample_weight : jnp.ndarray, optional
        Per-sample weights.  Weights are normalised to sum to ``n_samples``,
        so the result is ``sum_i w_i r_i^2 / n``.

    Returns
    -------
    float
        (Weighted) mean squared error.

    Raises
    ------
    ValueError
        If ``sample_weight`` has the wrong length or contains negative or
        non-finite values.
    """
    w = validate_sample_weight(sample_weight, len(y_true))
    return float(weighted_mean((y_true - y_pred) ** 2, w))


def compute_rmse(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: jnp.ndarray | None = None,
) -> float:
    """
    Compute root mean squared error, optionally weighted.

    Parameters
    ----------
    y_true : jnp.ndarray
        True values.
    y_pred : jnp.ndarray
        Predicted values.
    sample_weight : jnp.ndarray, optional
        Per-sample weights.

    Returns
    -------
    float
        (Weighted) root mean squared error.

    Raises
    ------
    ValueError
        If ``sample_weight`` is invalid.
    """
    return float(np.sqrt(compute_mse(y_true, y_pred, sample_weight)))


def compute_mae(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: jnp.ndarray | None = None,
) -> float:
    """
    Compute mean absolute error, optionally weighted.

    Parameters
    ----------
    y_true : jnp.ndarray
        True values.
    y_pred : jnp.ndarray
        Predicted values.
    sample_weight : jnp.ndarray, optional
        Per-sample weights.

    Returns
    -------
    float
        (Weighted) mean absolute error.

    Raises
    ------
    ValueError
        If ``sample_weight`` is invalid.
    """
    w = validate_sample_weight(sample_weight, len(y_true))
    return float(weighted_mean(jnp.abs(y_true - y_pred), w))


def compute_r2(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: jnp.ndarray | None = None,
) -> float:
    """
    Compute R-squared (coefficient of determination), optionally weighted.

    Parameters
    ----------
    y_true : jnp.ndarray
        True values.
    y_pred : jnp.ndarray
        Predicted values.
    sample_weight : jnp.ndarray, optional
        Per-sample weights.  Both the residual and the total sum of squares
        are weighted, and the total is taken about the *weighted* mean of
        ``y_true``.

    Returns
    -------
    float
        (Weighted) R-squared.

    Raises
    ------
    ValueError
        If ``sample_weight`` is invalid.
    """
    w = validate_sample_weight(sample_weight, len(y_true))
    weights = jnp.ones_like(y_true) if w is None else w
    y_bar = weighted_mean(y_true, w)
    ss_res = jnp.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = jnp.sum(weights * (y_true - y_bar) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def compute_adjusted_r2(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    n_params: int,
    sample_weight: jnp.ndarray | None = None,
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
    sample_weight : jnp.ndarray, optional
        Per-sample weights.  ``n`` remains the nominal sample count; only R²
        itself is weighted.

    Returns
    -------
    adj_r2 : float
        Adjusted R-squared.

    Raises
    ------
    ValueError
        If ``sample_weight`` is invalid.
    """
    n = len(y_true)
    k = n_params
    r2 = compute_r2(y_true, y_pred, sample_weight)

    if n - k - 1 <= 0:
        return float("-inf")

    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def compute_max_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Compute maximum absolute error (unweighted -- a max is not an average)."""
    return float(jnp.max(jnp.abs(y_true - y_pred)))


def compute_mape(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: jnp.ndarray | None = None,
) -> float:
    """
    Compute mean absolute percentage error, optionally weighted.

    MAPE = mean(|y_true - y_pred| / |y_true|) * 100

    Parameters
    ----------
    y_true : jnp.ndarray
        True values.  Entries with ``|y_true| <= 1e-10`` are skipped.
    y_pred : jnp.ndarray
        Predicted values.
    sample_weight : jnp.ndarray, optional
        Per-sample weights.

    Returns
    -------
    float
        (Weighted) MAPE in percent, or ``inf`` if every target is ~0.

    Raises
    ------
    ValueError
        If ``sample_weight`` is invalid.
    """
    w = validate_sample_weight(sample_weight, len(y_true))
    mask = jnp.abs(y_true) > 1e-10
    if not bool(jnp.any(mask)):
        return float("inf")
    pct = jnp.abs((y_true - y_pred) / y_true)[mask]
    w_masked = None if w is None else w[mask]
    if w_masked is not None and float(jnp.sum(w_masked)) <= 0:
        return float("inf")
    return float(weighted_mean(pct, w_masked) * 100)


def compute_all_metrics(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    n_params: int,
    sample_weight: jnp.ndarray | None = None,
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
    sample_weight : jnp.ndarray, optional
        Per-sample weights.  Every averaged metric is weighted;
        ``"max_error"`` is taken over the samples with non-zero weight, since
        a maximum has no weighted analogue.

    Returns
    -------
    metrics : dict
        Dictionary containing all metrics.

    Raises
    ------
    ValueError
        If ``sample_weight`` is invalid.
    """
    n = len(y_true)
    w = validate_sample_weight(sample_weight, n)
    mse = compute_mse(y_true, y_pred, w)

    if w is None:
        max_error = compute_max_error(y_true, y_pred)
    else:
        nonzero = w > 0
        max_error = (
            compute_max_error(y_true[nonzero], y_pred[nonzero])
            if bool(jnp.any(nonzero))
            else float("nan")
        )

    return {
        "mse": mse,
        "rmse": compute_rmse(y_true, y_pred, w),
        "mae": compute_mae(y_true, y_pred, w),
        "r2": compute_r2(y_true, y_pred, w),
        "adjusted_r2": compute_adjusted_r2(y_true, y_pred, n_params, w),
        "max_error": max_error,
        "mape": compute_mape(y_true, y_pred, w),
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
    auc = float(_np_trapezoid(tpr, fpr))
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
