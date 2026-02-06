"""
Active Learning Acquisition Functions for JAXSR.

Provides a composable framework for design-of-experiment (DOE) and Bayesian
optimization-style active learning on top of JAXSR symbolic regression models.

Because JAXSR models are linear-in-parameters (y = Phi @ beta), we have
**closed-form** expressions for prediction variance, leverage, and coefficient
covariance.  This means acquisition functions that rely on posterior uncertainty
are exact---no GP approximations needed.

Classes
-------
AcquisitionFunction
    Abstract base class.  Subclasses implement ``score()``.
PredictionVariance
    Pure exploration: sigma^2(x) from OLS posterior.
ConfidenceBandWidth
    Width of the confidence band on E[y|x].
EnsembleDisagreement
    Standard deviation across Pareto-front model predictions.
BMAUncertainty
    Bayesian Model Averaging uncertainty (within + between variance).
ModelDiscrimination
    Maximum pairwise disagreement among top Pareto models.
UCB / LCB
    Upper / Lower Confidence Bound: y_hat +/- kappa * sigma.
ExpectedImprovement
    EI(x) = E[max(0, f_best - y_hat(x))]  (Gaussian).
ProbabilityOfImprovement
    PI(x) = P(y(x) < f_best - xi).
ThompsonSampling
    Draw beta ~ N(beta_hat, Cov), score = sampled prediction.
AOptimal
    Minimize tr(Cov) after adding a candidate point.
DOptimal
    Maximize det(Phi^T Phi) after adding a candidate point.
Composite
    Weighted combination of acquisition functions.
ActiveLearner
    Orchestrator that combines candidate generation, scoring, batch
    selection, and the iterative update loop.

Example
-------
>>> from jaxsr.acquisition import ActiveLearner, UCB, ExpectedImprovement
>>> learner = ActiveLearner(model, bounds=[(0, 5)], acquisition=UCB(kappa=2.0))
>>> result = learner.suggest(n_points=5)
>>> # run experiments on result.points ...
>>> learner.update(result.points, y_new)
"""

from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
)

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from scipy.stats import qmc

if TYPE_CHECKING:
    from .regressor import SymbolicRegressor


# =========================================================================
# Data Structures
# =========================================================================


@dataclass
class AcquisitionResult:
    """Result returned by :meth:`ActiveLearner.suggest`.

    Attributes
    ----------
    points : jnp.ndarray
        Suggested points, shape ``(n_points, n_features)``.
    scores : jnp.ndarray
        Acquisition score for each suggested point.
    acquisition : str
        Name of the acquisition function used.
    metadata : dict
        Extra info (e.g. ``y_pred``, ``sigma``, batch diversity stats).
    """

    points: jnp.ndarray
    scores: jnp.ndarray
    acquisition: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BatchStrategy(Enum):
    """Strategies for selecting diverse batches."""

    GREEDY = "greedy"
    PENALIZED = "penalized"
    KRIGING_BELIEVER = "kriging_believer"
    D_OPTIMAL = "d_optimal"


# =========================================================================
# Helpers (shared by many acquisition functions)
# =========================================================================


def _get_pred_and_sigma(
    model: SymbolicRegressor,
    X: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (y_pred, sigma) for *X* using the OLS posterior.

    sigma(x) = sigma_hat * sqrt( phi(x)^T (Phi^T Phi)^{-1} phi(x) )
    """
    from .uncertainty import compute_unbiased_variance

    X = jnp.atleast_2d(jnp.asarray(X))
    model._check_is_fitted()

    Phi_train = model.basis_library.evaluate_subset(model._X_train, model._result.selected_indices)
    sigma_sq = compute_unbiased_variance(Phi_train, model._y_train, model._result.coefficients)

    # (Phi^T Phi)^{-1} via SVD for stability
    U, s, Vt = jnp.linalg.svd(Phi_train, full_matrices=False)
    rcond = jnp.finfo(Phi_train.dtype).eps * max(Phi_train.shape)
    cutoff = rcond * jnp.max(s)
    s_inv_sq = jnp.where(s > cutoff, 1.0 / (s**2), 0.0)
    PhiTPhiInv = Vt.T @ jnp.diag(s_inv_sq) @ Vt

    Phi_new = model.basis_library.evaluate_subset(X, model._result.selected_indices)
    y_pred = Phi_new @ model._result.coefficients

    # Vectorised leverage: h_ii = phi_i^T @ (Phi^T Phi)^{-1} @ phi_i
    h = jnp.sum((Phi_new @ PhiTPhiInv) * Phi_new, axis=1)
    sigma = jnp.sqrt(jnp.maximum(sigma_sq * h, 0.0))

    return y_pred, sigma


def _get_PhiTPhiInv(model: SymbolicRegressor) -> jnp.ndarray:
    """Return (Phi^T Phi)^{-1} for the training design matrix."""
    Phi_train = model.basis_library.evaluate_subset(model._X_train, model._result.selected_indices)
    U, s, Vt = jnp.linalg.svd(Phi_train, full_matrices=False)
    rcond = jnp.finfo(Phi_train.dtype).eps * max(Phi_train.shape)
    cutoff = rcond * jnp.max(s)
    s_inv_sq = jnp.where(s > cutoff, 1.0 / (s**2), 0.0)
    return Vt.T @ jnp.diag(s_inv_sq) @ Vt


# =========================================================================
# Base class
# =========================================================================


class AcquisitionFunction(abc.ABC):
    """Abstract base for all acquisition functions.

    Subclasses must implement :meth:`score`.  The convention is
    **higher score = more desirable to sample**.

    Supports composition via ``+`` and scalar ``*``::

        combined = 0.7 * UCB(kappa=2) + 0.3 * PredictionVariance()
    """

    # -- Composition helpers -----------------------------------------------

    def __add__(self, other: AcquisitionFunction) -> Composite:
        if isinstance(other, Composite):
            return Composite(functions=[(1.0, self)] + other.functions)
        return Composite(functions=[(1.0, self), (1.0, other)])

    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplemented

    def __mul__(self, scalar: float) -> Composite:
        return Composite(functions=[(float(scalar), self)])

    def __rmul__(self, scalar: float) -> Composite:
        return self.__mul__(scalar)

    # -- Interface ---------------------------------------------------------

    @abc.abstractmethod
    def score(
        self,
        X_candidates: jnp.ndarray,
        model: SymbolicRegressor,
    ) -> jnp.ndarray:
        """Score each candidate (higher = more desirable).

        Parameters
        ----------
        X_candidates : jnp.ndarray, shape (n_candidates, n_features)
            Candidate points to evaluate.
        model : SymbolicRegressor
            Fitted symbolic regression model.

        Returns
        -------
        scores : jnp.ndarray, shape (n_candidates,)
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__


# =========================================================================
# Exploration: Uncertainty-Based Acquisition Functions
# =========================================================================


class PredictionVariance(AcquisitionFunction):
    r"""Score = prediction variance sigma^2(x).

    For an OLS model, the prediction variance at a new point *x* is

    .. math::
        \sigma^2(x) = \hat\sigma^2 \, \varphi(x)^\top
        (\Phi^\top\Phi)^{-1} \varphi(x)

    where :math:`\hat\sigma^2` is the unbiased noise variance estimate.

    **When to use:** Default choice for pure exploration.  Sample where the
    model is least certain about the mean prediction.  Good for improving
    model accuracy uniformly across the input space.
    """

    def score(self, X_candidates, model):
        _, sigma = _get_pred_and_sigma(model, X_candidates)
        return sigma**2


class ConfidenceBandWidth(AcquisitionFunction):
    r"""Score = width of the ``1-alpha`` confidence band on E[y|x].

    The width is :math:`2 \, t_{\alpha/2,\nu} \, \sigma(x)`.

    Parameters
    ----------
    alpha : float
        Significance level (default 0.05 for 95% band).

    **When to use:** When you care about the width of the frequentist
    confidence band specifically (e.g. you have a coverage target).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def score(self, X_candidates, model):
        _, sigma = _get_pred_and_sigma(model, X_candidates)

        Phi_train = model.basis_library.evaluate_subset(
            model._X_train, model._result.selected_indices
        )
        n, p = Phi_train.shape
        dof = n - p
        if dof <= 0:
            return sigma  # fallback
        t_crit = stats.t.ppf(1 - self.alpha / 2, dof)
        return 2.0 * t_crit * sigma


class EnsembleDisagreement(AcquisitionFunction):
    """Score = standard deviation across Pareto-front model predictions.

    This captures *structural* (model-form) uncertainty: where do models of
    different complexity disagree?

    **When to use:** When you suspect model misspecification and want to
    identify regions where the choice of model complexity matters most.
    Helps resolve "which model is right?" rather than "how noisy is the
    data?".
    """

    def score(self, X_candidates, model):
        from .uncertainty import ensemble_predict

        result = ensemble_predict(model, X_candidates)
        return result["y_std"]


class BMAUncertainty(AcquisitionFunction):
    """Score = BMA posterior standard deviation.

    Combines within-model variance (noise) and between-model variance
    (structural uncertainty) via IC-weighted Bayesian Model Averaging.

    Parameters
    ----------
    criterion : str
        Information criterion for weights (``"bic"`` or ``"aic"``).
    top_k : int or None
        Number of top models to include in the average.

    **When to use:** The most comprehensive uncertainty measure available.
    Use when you want a single number that accounts for both noise and
    model selection uncertainty.  More expensive than
    :class:`PredictionVariance` because it evaluates multiple models.
    """

    def __init__(self, criterion: str = "bic", top_k: int | None = None):
        self.criterion = criterion
        self.top_k = top_k

    def score(self, X_candidates, model):
        from .uncertainty import BayesianModelAverage

        bma = BayesianModelAverage(model, criterion=self.criterion, top_k=self.top_k)
        _, y_std = bma.predict(X_candidates)
        return y_std


class ModelDiscrimination(AcquisitionFunction):
    """Score = maximum pairwise disagreement among Pareto-front models.

    For each candidate *x*, computes the max absolute difference between
    any pair of Pareto-front model predictions.  Points with high scores
    are the most informative for deciding *which model structure is correct*.

    Parameters
    ----------
    top_k : int or None
        Only compare the *top_k* Pareto models.  ``None`` uses all.

    **When to use:** You have several competing model forms on the Pareto
    front and want to efficiently determine which complexity level is
    correct.  Particularly useful in early stages when model structure is
    still uncertain.
    """

    def __init__(self, top_k: int | None = None):
        self.top_k = top_k

    def score(self, X_candidates, model):
        from .uncertainty import ensemble_predict

        result = ensemble_predict(model, X_candidates)
        y_all = result["y_all"]  # (n_models, n_candidates)

        if self.top_k is not None and y_all.shape[0] > self.top_k:
            y_all = y_all[: self.top_k]

        if y_all.shape[0] <= 1:
            return jnp.zeros(X_candidates.shape[0])

        return result["y_max"] - result["y_min"]


# =========================================================================
# Exploitation: Model-Based Optimisation
# =========================================================================


class ModelMin(AcquisitionFunction):
    """Score = -y_hat(x).  Finds the minimum of the surrogate.

    **When to use:** Pure exploitation.  You fully trust the model and want
    to sample at its predicted optimum.  Combine with an exploration term
    (e.g. ``0.8 * ModelMin() + 0.2 * PredictionVariance()``) to avoid
    getting stuck.
    """

    def score(self, X_candidates, model):
        y_pred = model.predict(X_candidates)
        return -y_pred


class ModelMax(AcquisitionFunction):
    """Score = y_hat(x).  Finds the maximum of the surrogate.

    **When to use:** Pure exploitation when maximising the response.
    """

    def score(self, X_candidates, model):
        return model.predict(X_candidates)


# =========================================================================
# Exploration-Exploitation Tradeoffs
# =========================================================================


class UCB(AcquisitionFunction):
    r"""Upper Confidence Bound.

    .. math::
        \text{UCB}(x) = \hat y(x) + \kappa \, \sigma(x)

    Parameters
    ----------
    kappa : float
        Exploration weight.  Larger values favour exploration.
        A common starting point is ``kappa=2.0``.

    **When to use:** You want to **maximise** the response while exploring.
    The ``kappa`` parameter directly controls the exploration-exploitation
    balance.

    - ``kappa = 0``: pure exploitation (equivalent to :class:`ModelMax`).
    - ``kappa ~ 2``: balanced.
    - ``kappa > 3``: heavily exploratory.
    """

    def __init__(self, kappa: float = 2.0):
        self.kappa = kappa

    def score(self, X_candidates, model):
        y_pred, sigma = _get_pred_and_sigma(model, X_candidates)
        return y_pred + self.kappa * sigma


class LCB(AcquisitionFunction):
    r"""Lower Confidence Bound.

    .. math::
        \text{LCB}(x) = -\hat y(x) + \kappa \, \sigma(x)

    Score convention: **higher is more desirable**, so the sign of the
    predicted mean is flipped.  Minimising y corresponds to maximising
    ``-y_hat + kappa * sigma``.

    Parameters
    ----------
    kappa : float
        Exploration weight (same semantics as :class:`UCB`).

    **When to use:** You want to **minimise** the response while exploring.
    Mirror image of UCB.
    """

    def __init__(self, kappa: float = 2.0):
        self.kappa = kappa

    def score(self, X_candidates, model):
        y_pred, sigma = _get_pred_and_sigma(model, X_candidates)
        return -y_pred + self.kappa * sigma


class ExpectedImprovement(AcquisitionFunction):
    r"""Expected Improvement over the current best.

    .. math::
        \text{EI}(x) = (\hat y_\text{best} - \hat y(x))\,
        \Phi(z) + \sigma(x)\,\phi(z),
        \quad z = \frac{\hat y_\text{best} - \hat y(x) - \xi}{\sigma(x)}

    for **minimisation** (flip signs for maximisation).

    Parameters
    ----------
    y_best : float or None
        Current best observed value.  If ``None``, the best value from the
        training set is used automatically.
    xi : float
        Jitter for encouraging exploration (default 0.01).
    minimize : bool
        If ``True`` (default), seek points that *reduce* y.  If ``False``,
        seek points that *increase* y.

    **When to use:** The gold standard for Bayesian optimisation.  Naturally
    balances exploration (high sigma) and exploitation (predicted
    improvement over best).  Less sensitive to hyperparameters than UCB.
    """

    def __init__(
        self,
        y_best: float | None = None,
        xi: float = 0.01,
        minimize: bool = True,
    ):
        self.y_best = y_best
        self.xi = xi
        self.minimize = minimize

    def score(self, X_candidates, model):
        y_pred, sigma = _get_pred_and_sigma(model, X_candidates)

        if self.y_best is not None:
            y_best = self.y_best
        else:
            # Use best observed training value
            if self.minimize:
                y_best = float(jnp.min(model._y_train))
            else:
                y_best = float(jnp.max(model._y_train))

        # Avoid division by zero
        sigma_safe = jnp.maximum(sigma, 1e-10)

        if self.minimize:
            improvement = y_best - y_pred - self.xi
        else:
            improvement = y_pred - y_best - self.xi

        z = improvement / sigma_safe
        ei = improvement * _norm_cdf(z) + sigma_safe * _norm_pdf(z)

        # Zero out where sigma is negligible, and clamp to non-negative
        # (EI is non-negative by definition; small negatives are numerical noise)
        ei = jnp.where(sigma > 1e-10, ei, 0.0)
        return jnp.maximum(ei, 0.0)


class ProbabilityOfImprovement(AcquisitionFunction):
    r"""Probability of improving over the current best.

    .. math::
        \text{PI}(x) = \Phi\!\left(
            \frac{\hat y_\text{best} - \hat y(x) - \xi}{\sigma(x)}
        \right)

    Parameters
    ----------
    y_best : float or None
        Current best observed value.  ``None`` = auto from training data.
    xi : float
        Improvement threshold (default 0.01).
    minimize : bool
        If ``True``, seek reduction in y.

    **When to use:** Simpler than EI.  Useful when you care about the
    *probability* of beating a threshold rather than the *magnitude* of
    improvement.  Tends to be more exploitative than EI for the same ``xi``
    because it ignores improvement magnitude.
    """

    def __init__(
        self,
        y_best: float | None = None,
        xi: float = 0.01,
        minimize: bool = True,
    ):
        self.y_best = y_best
        self.xi = xi
        self.minimize = minimize

    def score(self, X_candidates, model):
        y_pred, sigma = _get_pred_and_sigma(model, X_candidates)

        if self.y_best is not None:
            y_best = self.y_best
        else:
            if self.minimize:
                y_best = float(jnp.min(model._y_train))
            else:
                y_best = float(jnp.max(model._y_train))

        sigma_safe = jnp.maximum(sigma, 1e-10)

        if self.minimize:
            z = (y_best - y_pred - self.xi) / sigma_safe
        else:
            z = (y_pred - y_best - self.xi) / sigma_safe

        pi = _norm_cdf(z)
        pi = jnp.where(sigma > 1e-10, pi, 0.0)
        return pi


class ThompsonSampling(AcquisitionFunction):
    r"""Thompson Sampling via posterior coefficient draw.

    Draws :math:`\beta \sim \mathcal{N}(\hat\beta, \text{Cov}(\hat\beta))`
    and scores each candidate with the sampled model.

    Parameters
    ----------
    minimize : bool
        If ``True``, score = -y_sampled (prefer smaller predicted y).
    seed : int or None
        Random seed for the posterior draw.

    **When to use:** A randomised acquisition that naturally explores.
    Each call to ``score()`` draws a *different* model from the posterior,
    so repeated calls produce diverse batches without explicit
    diversification.  Theoretically elegant; matches Bayes-optimal
    one-step-ahead strategy.
    """

    def __init__(self, minimize: bool = True, seed: int | None = None):
        self.minimize = minimize
        self.seed = seed
        self._call_count = 0

    def score(self, X_candidates, model):
        from .uncertainty import compute_coeff_covariance, compute_unbiased_variance

        model._check_is_fitted()
        X_candidates = jnp.atleast_2d(jnp.asarray(X_candidates))

        Phi_train = model.basis_library.evaluate_subset(
            model._X_train, model._result.selected_indices
        )
        sigma_sq = compute_unbiased_variance(Phi_train, model._y_train, model._result.coefficients)
        cov = compute_coeff_covariance(Phi_train, sigma_sq)

        beta_hat = np.array(model._result.coefficients)
        cov_np = np.array(cov)

        # Make covariance PSD (add small diagonal if needed)
        eigvals = np.linalg.eigvalsh(cov_np)
        if np.min(eigvals) < 0:
            cov_np += (abs(np.min(eigvals)) + 1e-8) * np.eye(len(beta_hat))

        # Draw a sample from the posterior
        seed = self.seed if self.seed is not None else None
        if seed is not None:
            seed = seed + self._call_count
        rng = np.random.RandomState(seed)
        beta_sample = rng.multivariate_normal(beta_hat, cov_np)
        self._call_count += 1

        Phi_new = model.basis_library.evaluate_subset(X_candidates, model._result.selected_indices)
        y_sample = Phi_new @ jnp.array(beta_sample)

        if self.minimize:
            return -y_sample
        return y_sample


# =========================================================================
# Optimal Experimental Design
# =========================================================================


class AOptimal(AcquisitionFunction):
    r"""A-Optimal design: reduce average parameter uncertainty.

    Score for adding candidate *x* is the reduction in
    :math:`\text{tr}(\text{Cov}(\hat\beta))`.

    .. math::
        \Delta\text{tr} = \frac{
            \varphi(x)^\top (\Phi^\top\Phi)^{-1}
            \text{Cov}(\hat\beta) (\Phi^\top\Phi)^{-1} \varphi(x)
        }{1 + \varphi(x)^\top (\Phi^\top\Phi)^{-1} \varphi(x)}

    **When to use:** When your goal is to tighten the confidence intervals
    on *all* coefficients.  Good for model building where every parameter
    matters (e.g. reporting a physics equation).
    """

    def score(self, X_candidates, model):
        from .uncertainty import compute_unbiased_variance

        model._check_is_fitted()
        X_candidates = jnp.atleast_2d(jnp.asarray(X_candidates))

        Phi_train = model.basis_library.evaluate_subset(
            model._X_train, model._result.selected_indices
        )
        sigma_sq = compute_unbiased_variance(Phi_train, model._y_train, model._result.coefficients)
        PhiTPhiInv = _get_PhiTPhiInv(model)
        cov = sigma_sq * PhiTPhiInv

        Phi_new = model.basis_library.evaluate_subset(X_candidates, model._result.selected_indices)

        # Vectorised: for each candidate phi_i
        # numerator_i = phi_i^T @ PhiTPhiInv @ Cov @ PhiTPhiInv @ phi_i
        # denominator_i = 1 + phi_i^T @ PhiTPhiInv @ phi_i
        M = PhiTPhiInv @ cov @ PhiTPhiInv  # (p, p)
        numer = jnp.sum((Phi_new @ M) * Phi_new, axis=1)
        denom = 1.0 + jnp.sum((Phi_new @ PhiTPhiInv) * Phi_new, axis=1)

        return numer / denom


class DOptimal(AcquisitionFunction):
    r"""D-Optimal design: maximise information gain.

    Score = leverage :math:`h(x) = \varphi(x)^\top (\Phi^\top\Phi)^{-1}
    \varphi(x)`, which is proportional to the increase in
    :math:`\det(\Phi^\top\Phi)` from adding the point.

    **When to use:** When you want maximum information per experiment.
    Classic choice for building precise models with minimal data.  Tends
    to place points at the "edges" of the design space.
    """

    def score(self, X_candidates, model):
        model._check_is_fitted()
        X_candidates = jnp.atleast_2d(jnp.asarray(X_candidates))

        PhiTPhiInv = _get_PhiTPhiInv(model)
        Phi_new = model.basis_library.evaluate_subset(X_candidates, model._result.selected_indices)
        # h = phi^T (Phi^T Phi)^{-1} phi
        return jnp.sum((Phi_new @ PhiTPhiInv) * Phi_new, axis=1)


# =========================================================================
# Composite (weighted combination)
# =========================================================================


class Composite(AcquisitionFunction):
    """Weighted combination of acquisition functions.

    Typically created via operator overloading rather than directly::

        acq = 0.7 * UCB(kappa=2) + 0.3 * PredictionVariance()

    Each component is scored independently and the weighted sum is
    returned after min-max normalisation of each component.

    Attributes
    ----------
    functions : list of (weight, AcquisitionFunction)
    """

    def __init__(self, functions: list[tuple[float, AcquisitionFunction]]):
        self.functions = functions

    def __add__(self, other: AcquisitionFunction) -> Composite:
        if isinstance(other, Composite):
            return Composite(functions=self.functions + other.functions)
        return Composite(functions=self.functions + [(1.0, other)])

    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplemented

    def __mul__(self, scalar: float) -> Composite:
        return Composite(functions=[(w * float(scalar), f) for w, f in self.functions])

    def __rmul__(self, scalar: float) -> Composite:
        return self.__mul__(scalar)

    @property
    def name(self) -> str:
        parts = []
        for w, f in self.functions:
            parts.append(f"{w:.2g}*{f.name}")
        return " + ".join(parts)

    def score(self, X_candidates, model):
        total = jnp.zeros(X_candidates.shape[0])
        for weight, func in self.functions:
            raw = func.score(X_candidates, model)
            # Min-max normalise to [0, 1] so weights are meaningful
            rmin = jnp.min(raw)
            rmax = jnp.max(raw)
            denom = rmax - rmin
            normalised = jnp.where(denom > 1e-10, (raw - rmin) / denom, 0.5)
            total = total + weight * normalised
        return total


# =========================================================================
# Normal distribution helpers
# =========================================================================


def _norm_cdf(z: jnp.ndarray) -> jnp.ndarray:
    """Standard normal CDF using JAX-compatible erfc."""
    return 0.5 * (1.0 + jax.lax.erf(z / jnp.sqrt(2.0)))


def _norm_pdf(z: jnp.ndarray) -> jnp.ndarray:
    """Standard normal PDF."""
    return jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * jnp.pi)


# =========================================================================
# ActiveLearner
# =========================================================================


class ActiveLearner:
    """Orchestrator for iterative active learning / DOE.

    Combines candidate generation, acquisition scoring, batch selection,
    and model updating into a single workflow.

    Parameters
    ----------
    model : SymbolicRegressor
        A **fitted** model to improve.
    bounds : list of (float, float)
        Input bounds ``[(lo, hi), ...]`` for each feature.
    acquisition : AcquisitionFunction
        The acquisition function (or composite) to use for scoring.
    n_candidates : int
        Number of space-filling candidates to generate internally.
    candidate_method : str
        Candidate generation method: ``"lhs"`` (default), ``"sobol"``,
        ``"halton"``, or ``"random"``.
    random_state : int or None
        Seed for reproducibility.

    Attributes
    ----------
    history_X : list of jnp.ndarray
        Points suggested at each iteration.
    history_y : list of jnp.ndarray
        Observations received at each iteration.
    iteration : int
        Number of suggest-update cycles completed.

    Examples
    --------
    >>> learner = ActiveLearner(model, bounds=[(0, 5)],
    ...     acquisition=UCB(kappa=2))
    >>> for _ in range(10):
    ...     result = learner.suggest(n_points=3)
    ...     y_new = oracle(result.points)
    ...     learner.update(result.points, y_new)
    """

    def __init__(
        self,
        model: SymbolicRegressor,
        bounds: list[tuple[float, float]],
        acquisition: AcquisitionFunction,
        n_candidates: int = 1000,
        candidate_method: str = "lhs",
        random_state: int | None = None,
    ):
        model._check_is_fitted()

        self.model = model
        self.bounds = bounds
        self.acquisition = acquisition
        self.n_candidates = n_candidates
        self.candidate_method = candidate_method
        self.random_state = random_state

        self._bounds_array = np.array(bounds)
        self._rng = np.random.RandomState(random_state)

        if len(bounds) != model.basis_library.n_features:
            raise ValueError(
                f"Number of bounds ({len(bounds)}) must match "
                f"number of features ({model.basis_library.n_features})"
            )

        # Tracking
        self.history_X: list[jnp.ndarray] = []
        self.history_y: list[jnp.ndarray] = []
        self.iteration: int = 0

    # -- Public API --------------------------------------------------------

    def suggest(
        self,
        n_points: int = 5,
        batch_strategy: str = "greedy",
        min_distance: float = 0.01,
        candidates: jnp.ndarray | None = None,
    ) -> AcquisitionResult:
        """Suggest the next batch of points to evaluate.

        Parameters
        ----------
        n_points : int
            Number of points to return.
        batch_strategy : str
            How to select a batch: ``"greedy"`` (top-k by score),
            ``"penalized"`` (top-1, penalise nearby, repeat),
            ``"kriging_believer"`` (top-1, fantasise observation, repeat),
            ``"d_optimal"`` (select batch maximising det(Phi^T Phi)).
        min_distance : float
            Minimum normalised distance from training data.
        candidates : jnp.ndarray, optional
            Pre-specified candidate pool.  If ``None``, candidates are
            generated internally.

        Returns
        -------
        AcquisitionResult
            With ``.points``, ``.scores``, and ``.metadata``.
        """
        strategy = BatchStrategy(batch_strategy)

        if candidates is None:
            candidates = self._generate_candidates(self.n_candidates)

        candidates = jnp.atleast_2d(jnp.asarray(candidates))

        # Filter candidates too close to training data
        if self.model._X_train is not None:
            candidates = self._filter_by_distance(candidates, self.model._X_train, min_distance)

        if len(candidates) == 0:
            warnings.warn("All candidates filtered.  Returning random points.", stacklevel=2)
            candidates = self._generate_candidates(n_points)

        if len(candidates) < n_points:
            extra = self._generate_candidates(self.n_candidates * 2)
            candidates = jnp.vstack([candidates, extra])
            if self.model._X_train is not None:
                candidates = self._filter_by_distance(candidates, self.model._X_train, min_distance)

        # Dispatch to batch strategy
        if strategy == BatchStrategy.GREEDY:
            return self._select_greedy(candidates, n_points)
        elif strategy == BatchStrategy.PENALIZED:
            return self._select_penalized(candidates, n_points)
        elif strategy == BatchStrategy.KRIGING_BELIEVER:
            return self._select_kriging_believer(candidates, n_points)
        elif strategy == BatchStrategy.D_OPTIMAL:
            return self._select_d_optimal(candidates, n_points)
        else:
            raise ValueError(f"Unknown batch_strategy: {batch_strategy}")

    def update(
        self,
        X_new: jnp.ndarray,
        y_new: jnp.ndarray,
        refit: bool = True,
    ) -> None:
        """Add new observations and refit the model.

        Parameters
        ----------
        X_new : jnp.ndarray
            New input points.
        y_new : jnp.ndarray
            Observed responses.
        refit : bool
            If ``True``, rerun full feature selection.
            If ``False``, keep the same basis terms and refit coefficients.
        """
        X_new = jnp.atleast_2d(jnp.asarray(X_new))
        y_new = jnp.asarray(y_new).ravel()

        self.history_X.append(X_new)
        self.history_y.append(y_new)
        self.iteration += 1

        self.model.update(X_new, y_new, refit=refit)

    def converged(
        self,
        tol: float = 1e-3,
        window: int = 3,
        metric: str = "mse",
    ) -> bool:
        """Check whether model improvement has stalled.

        Parameters
        ----------
        tol : float
            Minimum relative improvement to consider "not converged".
        window : int
            Number of recent iterations to compare.
        metric : str
            Metric to track: ``"mse"`` or ``"r2"``.

        Returns
        -------
        bool
            ``True`` if improvement over the last *window* iterations is
            below *tol*.
        """
        if self.iteration < window + 1:
            return False

        # Recompute metric at each iteration?  We use the current model's
        # metric as a proxy since we refit after each update.
        current = self.model.metrics_[metric]

        # Rough heuristic: compare current to a few iterations ago by
        # looking at prediction quality on all data so far.
        if metric == "mse":
            return current < tol
        elif metric == "r2":
            return current > 1.0 - tol
        return False

    @property
    def best_y(self) -> float:
        """Best observed y in the training set."""
        return float(jnp.min(self.model._y_train))

    @property
    def best_X(self) -> jnp.ndarray:
        """Input corresponding to the best observed y."""
        idx = int(jnp.argmin(self.model._y_train))
        return self.model._X_train[idx]

    @property
    def n_observations(self) -> int:
        """Total number of observations in the training set."""
        return len(self.model._y_train)

    # -- Batch Selection Strategies ----------------------------------------

    def _select_greedy(self, candidates: jnp.ndarray, n_points: int) -> AcquisitionResult:
        """Select top-k candidates by acquisition score."""
        scores = self.acquisition.score(candidates, self.model)

        if len(candidates) <= n_points:
            idx = jnp.arange(len(candidates))
        else:
            idx = jnp.argsort(scores)[-n_points:]

        return AcquisitionResult(
            points=candidates[idx],
            scores=scores[idx],
            acquisition=self.acquisition.name,
        )

    def _select_penalized(self, candidates: jnp.ndarray, n_points: int) -> AcquisitionResult:
        """Greedy selection with distance-based penalisation.

        After selecting the best candidate, nearby candidates are penalised
        proportionally to their proximity, then the next best is selected.
        This encourages spatial diversity in the batch.
        """
        scores = np.array(self.acquisition.score(candidates, self.model))
        candidates_np = np.array(candidates)

        lower = self._bounds_array[:, 0]
        upper = self._bounds_array[:, 1]
        scale = upper - lower
        scale = np.where(scale > 0, scale, 1.0)
        candidates_norm = (candidates_np - lower) / scale

        selected_idx = []
        for _ in range(min(n_points, len(candidates))):
            best = int(np.argmax(scores))
            selected_idx.append(best)

            # Penalise nearby candidates
            dists = np.linalg.norm(candidates_norm - candidates_norm[best], axis=1)
            # Gaussian penalty: penalty peaks at 1 for dist=0, decays with
            # bandwidth ~ 1/sqrt(n_points) to spread the batch out.
            bandwidth = 1.0 / max(np.sqrt(n_points), 1.0)
            penalty = np.exp(-0.5 * (dists / bandwidth) ** 2)
            scores *= 1.0 - penalty

        selected_idx = np.array(selected_idx)
        return AcquisitionResult(
            points=jnp.array(candidates_np[selected_idx]),
            scores=jnp.array(scores[selected_idx]),
            acquisition=self.acquisition.name,
            metadata={"batch_strategy": "penalized"},
        )

    def _select_kriging_believer(self, candidates: jnp.ndarray, n_points: int) -> AcquisitionResult:
        """Kriging Believer batch selection.

        1. Score all candidates.
        2. Select the best candidate x*.
        3. "Fantasise" that y(x*) = y_hat(x*) and temporarily update the
           model's training data.
        4. Re-score remaining candidates and repeat.

        This accounts for the information gain within a batch: later
        selections adjust based on what the earlier selections would teach.
        """
        candidates_np = np.array(candidates)
        remaining_mask = np.ones(len(candidates_np), dtype=bool)

        # Save original state
        X_orig = self.model._X_train
        y_orig = self.model._y_train
        result_orig = self.model._result

        selected_indices = []
        selected_scores = []

        try:
            for _ in range(min(n_points, len(candidates_np))):
                active = jnp.array(candidates_np[remaining_mask])
                if len(active) == 0:
                    break

                scores = self.acquisition.score(active, self.model)
                best_local = int(jnp.argmax(scores))
                selected_scores.append(float(scores[best_local]))

                # Map back to global index
                global_indices = np.where(remaining_mask)[0]
                best_global = global_indices[best_local]
                selected_indices.append(best_global)
                remaining_mask[best_global] = False

                # Fantasy update: pretend we observed y_hat at x*
                x_star = jnp.atleast_2d(jnp.array(candidates_np[best_global]))
                y_star = self.model.predict(x_star)
                self.model._X_train = jnp.vstack([self.model._X_train, x_star])
                self.model._y_train = jnp.concatenate([self.model._y_train, y_star])

                # Quick coefficient refit (not full selection)
                Phi = self.model.basis_library.evaluate_subset(
                    self.model._X_train, self.model._result.selected_indices
                )
                coeffs = jnp.linalg.lstsq(Phi, self.model._y_train, rcond=None)[0]
                from .selection import SelectionResult

                self.model._result = SelectionResult(
                    coefficients=coeffs,
                    selected_indices=result_orig.selected_indices,
                    selected_names=result_orig.selected_names,
                    mse=float(jnp.mean((self.model._y_train - Phi @ coeffs) ** 2)),
                    complexity=result_orig.complexity,
                    aic=result_orig.aic,
                    bic=result_orig.bic,
                    aicc=result_orig.aicc,
                    n_samples=len(self.model._y_train),
                )
        finally:
            # Restore original model state
            self.model._X_train = X_orig
            self.model._y_train = y_orig
            self.model._result = result_orig

        selected_indices = np.array(selected_indices)
        return AcquisitionResult(
            points=jnp.array(candidates_np[selected_indices]),
            scores=jnp.array(selected_scores),
            acquisition=self.acquisition.name,
            metadata={"batch_strategy": "kriging_believer"},
        )

    def _select_d_optimal(self, candidates: jnp.ndarray, n_points: int) -> AcquisitionResult:
        """D-Optimal batch selection: maximise det(Phi^T Phi) for the batch.

        Uses the existing ``d_optimal_select`` from ``sampling.py`` but
        wraps it to include acquisition scores.
        """
        from .sampling import d_optimal_select

        # d_optimal_select uses list indexing internally; convert to numpy
        # arrays to avoid JAX deprecation issues with list-based indexing.
        candidates_np = np.array(candidates)
        selected_idx = d_optimal_select(
            candidates=jnp.array(candidates_np),
            n_select=n_points,
            basis_library=self.model.basis_library,
            selected_indices=self.model.selected_indices_,
            random_state=self._rng.randint(2**31),
        )
        selected_idx = np.array(selected_idx, dtype=int)

        # Also score for informational purposes
        scores = self.acquisition.score(candidates, self.model)

        return AcquisitionResult(
            points=jnp.array(candidates_np[selected_idx]),
            scores=scores[selected_idx],
            acquisition=self.acquisition.name,
            metadata={"batch_strategy": "d_optimal"},
        )

    # -- Candidate Generation ----------------------------------------------

    def _generate_candidates(self, n: int) -> jnp.ndarray:
        """Generate space-filling candidate points."""
        lower = self._bounds_array[:, 0]
        upper = self._bounds_array[:, 1]
        d = len(self.bounds)

        if self.candidate_method == "lhs":
            sampler = qmc.LatinHypercube(d=d, seed=self._rng)
            samples = sampler.random(n)
        elif self.candidate_method == "sobol":
            sampler = qmc.Sobol(d=d, scramble=True, seed=self._rng)
            samples = sampler.random(n)
        elif self.candidate_method == "halton":
            sampler = qmc.Halton(d=d, scramble=True, seed=self._rng)
            samples = sampler.random(n)
        elif self.candidate_method == "random":
            samples = self._rng.random((n, d))
        else:
            raise ValueError(f"Unknown candidate_method: {self.candidate_method}")

        samples = qmc.scale(samples, lower, upper)
        return jnp.array(samples)

    def _filter_by_distance(
        self,
        candidates: jnp.ndarray,
        existing: jnp.ndarray,
        min_distance: float,
    ) -> jnp.ndarray:
        """Filter candidates too close to existing points."""
        lower = self._bounds_array[:, 0]
        upper = self._bounds_array[:, 1]
        scale = upper - lower
        scale = np.where(scale > 0, scale, 1.0)

        cand_np = np.array(candidates)
        exist_np = np.array(existing)

        cand_norm = (cand_np - lower) / scale
        exist_norm = (exist_np - lower) / scale

        keep = np.ones(len(cand_np), dtype=bool)
        for i, c in enumerate(cand_norm):
            dists = np.linalg.norm(exist_norm - c, axis=1)
            if np.min(dists) < min_distance:
                keep[i] = False

        return candidates[keep]


# =========================================================================
# Convenience: quick-access functions
# =========================================================================


def suggest_points(
    model: SymbolicRegressor,
    bounds: list[tuple[float, float]],
    acquisition: AcquisitionFunction,
    n_points: int = 5,
    batch_strategy: str = "greedy",
    n_candidates: int = 1000,
    random_state: int | None = None,
) -> AcquisitionResult:
    """One-shot convenience: suggest points without creating an ActiveLearner.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model.
    bounds : list of (float, float)
        Feature bounds.
    acquisition : AcquisitionFunction
        Acquisition function to use.
    n_points : int
        Number of points.
    batch_strategy : str
        Batch strategy.
    n_candidates : int
        Candidate pool size.
    random_state : int or None
        Seed.

    Returns
    -------
    AcquisitionResult
    """
    learner = ActiveLearner(
        model=model,
        bounds=bounds,
        acquisition=acquisition,
        n_candidates=n_candidates,
        random_state=random_state,
    )
    return learner.suggest(n_points=n_points, batch_strategy=batch_strategy)
