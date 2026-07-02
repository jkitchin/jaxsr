"""
Bootstrap structural uncertainty for additive symbolic models.

Refitting an additive symbolic regressor on bootstrap resamples of the training
data quantifies how stable the *discovered structure* is:

* **Inclusion probabilities** -- how often each basis function is selected across
  resamples (a cheap, frequentist approximation to a posterior inclusion
  probability).  When these are all near 0 or 1 the structure is identifiable
  and a single expression is trustworthy; diffuse values (e.g. under collinear
  features) signal genuine structural uncertainty, where no single expression is
  well determined.
* **Predictive ensemble** -- the spread of predictions across resamples, giving
  intervals that reflect structural variability, not just coefficient noise.

This is a model-agnostic stand-in for a full Bayesian treatment: it works for
any additive regressor (stagewise or backfitting) and reuses the estimator's own
fitting machinery.  It also serves as a decision gate -- if the inclusion
probabilities are already crisp, a heavier Bayesian model buys little.
"""

from __future__ import annotations

import warnings
from collections import Counter

import jax.numpy as jnp
import numpy as np


def _clone(estimator):
    """Return an unfitted copy of ``estimator`` with the same configuration."""
    return type(estimator)(**estimator.get_params(deep=False))


def bootstrap_additive(
    estimator,
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_bootstrap: int = 100,
    random_state: int | None = None,
) -> dict:
    """
    Refit an additive regressor on bootstrap resamples to assess structure.

    Parameters
    ----------
    estimator : StagewiseSymbolicRegressor or BackfittingSymbolicRegressor
        A configured (fitted or unfitted) additive regressor.  It is cloned via
        its constructor parameters and refit on each resample; the original is
        not modified.
    X : array-like of shape (n_samples, n_features)
        Training inputs.
    y : array-like of shape (n_samples,)
        Target values.
    n_bootstrap : int
        Number of bootstrap resamples.
    random_state : int, optional
        Seed for the resampling for reproducibility.

    Returns
    -------
    dict
        Plain dictionary with keys:

        ``"inclusion_probabilities"`` : dict[str, float]
            Basis-function name -> fraction of resamples selecting it (in any
            term), sorted descending.
        ``"n_terms"`` : numpy.ndarray
            Number of terms in each successful resample fit.
        ``"models"`` : list
            The fitted bootstrap estimators (use with
            :func:`bootstrap_predict_additive`).
        ``"n_bootstrap"`` : int
            Number of resamples that fit successfully.

    Raises
    ------
    ValueError
        If ``X``/``y`` are mismatched or ``n_bootstrap < 1``.
    RuntimeError
        If every bootstrap fit fails.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")

    X = np.asarray(jnp.atleast_2d(jnp.asarray(X)))
    y = np.asarray(jnp.asarray(y).ravel())
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X: {X.shape[0]}, y: {y.shape[0]}."
        )

    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    counts: Counter = Counter()
    n_terms: list[int] = []
    models: list = []
    n_failed = 0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        model = _clone(estimator)
        try:
            model.fit(X[idx], y[idx])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            n_failed += 1
            continue
        names: set[str] = set()
        for term in model.terms_:
            names.update(term.selected_features_)
        counts.update(names)
        n_terms.append(model.n_terms_)
        models.append(model)

    n_ok = len(models)
    if n_ok == 0:
        raise RuntimeError("All bootstrap fits failed.")
    if n_failed:
        warnings.warn(
            f"{n_failed}/{n_bootstrap} bootstrap fits failed and were skipped.",
            stacklevel=2,
        )

    inclusion = {name: counts[name] / n_ok for name in counts}
    inclusion = dict(sorted(inclusion.items(), key=lambda kv: -kv[1]))

    return {
        "inclusion_probabilities": inclusion,
        "n_terms": np.array(n_terms),
        "models": models,
        "n_bootstrap": n_ok,
    }


def bootstrap_predict_additive(
    models: list,
    X: jnp.ndarray,
    alpha: float = 0.1,
) -> dict:
    """
    Predictive ensemble from a bootstrap set of additive models.

    Parameters
    ----------
    models : list
        Fitted additive estimators, e.g. the ``"models"`` entry returned by
        :func:`bootstrap_additive`.
    X : array-like of shape (n_samples, n_features)
        Inputs to predict.
    alpha : float
        Significance level; the interval covers ``1 - alpha`` (default 0.1 for
        a 90% interval).

    Returns
    -------
    dict
        Plain dictionary with keys ``"mean"``, ``"std"``, ``"median"``,
        ``"lower"``, ``"upper"`` (each of shape ``(n_samples,)``) and
        ``"predictions"`` of shape ``(n_models, n_samples)``.

    Raises
    ------
    ValueError
        If ``models`` is empty or ``alpha`` is not in ``(0, 1)``.
    """
    if not models:
        raise ValueError("models must be a non-empty list of fitted estimators.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    preds = np.stack([np.asarray(m.predict(X)) for m in models])
    return {
        "mean": preds.mean(axis=0),
        "std": preds.std(axis=0),
        "median": np.median(preds, axis=0),
        "lower": np.quantile(preds, alpha / 2, axis=0),
        "upper": np.quantile(preds, 1 - alpha / 2, axis=0),
        "predictions": preds,
    }
