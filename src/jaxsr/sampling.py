"""
Adaptive Sampling Strategies for JAXSR.

Implements various strategies for suggesting new data points to query,
inspired by ALAMO's adaptive sampling approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

if TYPE_CHECKING:
    from .regressor import SymbolicRegressor


class SamplingStrategy(Enum):
    """Available sampling strategies."""

    UNCERTAINTY = "uncertainty"
    ERROR = "error"
    LEVERAGE = "leverage"
    GRADIENT = "gradient"
    SPACE_FILLING = "space_filling"
    RANDOM = "random"


@dataclass
class SamplingResult:
    """
    Result of adaptive sampling suggestion.

    Parameters
    ----------
    points : jnp.ndarray
        Suggested points of shape (n_points, n_features).
    scores : jnp.ndarray
        Acquisition scores for each point.
    strategy : str
        Strategy used.
    """

    points: jnp.ndarray
    scores: jnp.ndarray
    strategy: str


class AdaptiveSampler:
    """
    Adaptive sampling for iterative model improvement.

    Suggests new data points to query based on various strategies
    that identify regions of high uncertainty or potential improvement.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model to improve.
    bounds : list of tuple
        Bounds (lower, upper) for each feature.
    strategy : str
        Sampling strategy: "uncertainty", "error", "leverage",
        "gradient", "space_filling", "random".
    batch_size : int
        Number of points to suggest per call.
    n_candidates : int
        Number of candidate points to evaluate.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> sampler = AdaptiveSampler(
    ...     model=model,
    ...     bounds=[(300, 500), (1, 10)],
    ...     strategy="uncertainty",
    ... )
    >>> X_new = sampler.suggest(n_points=5)
    >>> # Query oracle/experiment for y_new
    >>> model.update(X_new, y_new)
    """

    def __init__(
        self,
        model: SymbolicRegressor,
        bounds: list[tuple[float, float]],
        strategy: str = "uncertainty",
        batch_size: int = 5,
        n_candidates: int = 1000,
        random_state: int | None = None,
        discrete_dims: dict[int, list] | None = None,
    ):
        self.model = model
        self.bounds = bounds
        self.strategy = SamplingStrategy(strategy)
        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.random_state = random_state
        self.discrete_dims = discrete_dims or {}

        self._rng = np.random.RandomState(random_state)
        self._bounds_array = np.array(bounds)

        if len(bounds) != model.basis_library.n_features:
            raise ValueError(
                f"Number of bounds ({len(bounds)}) must match "
                f"number of features ({model.basis_library.n_features})"
            )

    def suggest(
        self,
        n_points: int | None = None,
        exclude_points: jnp.ndarray | None = None,
        min_distance: float = 0.01,
    ) -> SamplingResult:
        """
        Suggest new points to query.

        Parameters
        ----------
        n_points : int, optional
            Number of points to suggest. Defaults to batch_size.
        exclude_points : jnp.ndarray, optional
            Points to exclude (e.g., already queried).
        min_distance : float
            Minimum normalized distance from existing points.

        Returns
        -------
        result : SamplingResult
            Suggested points and their scores.
        """
        n_points = n_points or self.batch_size

        # Generate candidate points
        candidates = self._generate_candidates(self.n_candidates)

        # Exclude points too close to existing data or excluded points
        if exclude_points is not None:
            candidates = self._filter_by_distance(candidates, exclude_points, min_distance)

        if self.model._X_train is not None:
            candidates = self._filter_by_distance(candidates, self.model._X_train, min_distance)

        if len(candidates) < n_points:
            # Generate more candidates if needed
            additional = self._generate_candidates(self.n_candidates * 2)
            candidates = jnp.vstack([candidates, additional])
            if exclude_points is not None:
                candidates = self._filter_by_distance(candidates, exclude_points, min_distance)
            if self.model._X_train is not None:
                candidates = self._filter_by_distance(candidates, self.model._X_train, min_distance)

        # Compute acquisition scores
        scores = self._compute_scores(candidates)

        # Select top points
        if len(candidates) <= n_points:
            selected_idx = jnp.arange(len(candidates))
        else:
            selected_idx = jnp.argsort(scores)[-n_points:]

        selected_points = candidates[selected_idx]
        selected_scores = scores[selected_idx]

        return SamplingResult(
            points=selected_points,
            scores=selected_scores,
            strategy=self.strategy.value,
        )

    def _generate_candidates(self, n: int) -> jnp.ndarray:
        """Generate candidate points using space-filling design."""
        # Use Latin Hypercube Sampling for better coverage
        sampler = qmc.LatinHypercube(d=len(self.bounds), seed=self._rng)
        samples = sampler.random(n)

        # Scale to bounds
        lower = self._bounds_array[:, 0]
        upper = self._bounds_array[:, 1]
        candidates = np.array(qmc.scale(samples, lower, upper))

        # Snap discrete dimensions to valid values
        for dim_idx, values in self.discrete_dims.items():
            values_arr = np.array(values, dtype=float)
            for row in range(len(candidates)):
                dists = np.abs(values_arr - candidates[row, dim_idx])
                candidates[row, dim_idx] = values_arr[np.argmin(dists)]

        return jnp.array(candidates)

    def _filter_by_distance(
        self,
        candidates: jnp.ndarray,
        existing: jnp.ndarray,
        min_distance: float,
    ) -> jnp.ndarray:
        """Filter candidates that are too close to existing points."""
        # Normalize to [0, 1]
        lower = self._bounds_array[:, 0]
        upper = self._bounds_array[:, 1]
        scale = upper - lower

        candidates_norm = (candidates - lower) / scale
        existing_norm = (np.array(existing) - lower) / scale

        # Compute distances
        keep_mask = np.ones(len(candidates), dtype=bool)

        for i, cand in enumerate(candidates_norm):
            distances = np.linalg.norm(existing_norm - cand, axis=1)
            if np.min(distances) < min_distance:
                keep_mask[i] = False

        return candidates[keep_mask]

    def _compute_scores(self, candidates: jnp.ndarray) -> jnp.ndarray:
        """Compute acquisition scores for candidates."""
        if self.strategy == SamplingStrategy.UNCERTAINTY:
            return self._score_uncertainty(candidates)
        elif self.strategy == SamplingStrategy.ERROR:
            return self._score_error(candidates)
        elif self.strategy == SamplingStrategy.LEVERAGE:
            return self._score_leverage(candidates)
        elif self.strategy == SamplingStrategy.GRADIENT:
            return self._score_gradient(candidates)
        elif self.strategy == SamplingStrategy.SPACE_FILLING:
            return self._score_space_filling(candidates)
        else:  # RANDOM
            return jnp.array(self._rng.random(len(candidates)))

    def _score_uncertainty(self, candidates: jnp.ndarray) -> jnp.ndarray:
        """
        Score based on prediction variance (approximated).

        Uses leverage as a proxy for uncertainty since we're fitting
        a linear model in the basis function space.
        """
        # Compute leverage for candidates
        return self._score_leverage(candidates)

    def _score_error(self, candidates: jnp.ndarray) -> jnp.ndarray:
        """
        Score based on cross-validation residuals.

        Uses distance from training residual patterns.
        """
        if self.model._X_train is None or self.model._y_train is None:
            return jnp.ones(len(candidates))

        # Get training residuals
        y_pred_train = self.model.predict(self.model._X_train)
        residuals = jnp.abs(self.model._y_train - y_pred_train)

        # Score based on similarity to high-residual regions
        scores = jnp.zeros(len(candidates))

        for i, cand in enumerate(candidates):
            # Distance-weighted residual score
            distances = jnp.linalg.norm(self.model._X_train - cand, axis=1)
            weights = jnp.exp(-distances)
            scores = scores.at[i].set(jnp.sum(weights * residuals) / (jnp.sum(weights) + 1e-10))

        return scores

    def _score_leverage(self, candidates: jnp.ndarray) -> jnp.ndarray:
        """
        Score based on leverage (influence on model).

        High leverage points have more influence on the fit.
        """
        if self.model._X_train is None:
            return jnp.ones(len(candidates))

        # Get design matrix for training data
        Phi_train = self.model.basis_library.evaluate_subset(
            self.model._X_train,
            self.model.selected_indices_,
        )

        # Compute (Phi.T @ Phi)^-1 for leverage calculation
        try:
            Phi_pinv = jnp.linalg.pinv(Phi_train)
            M = Phi_pinv @ Phi_pinv.T
        except Exception:
            return jnp.ones(len(candidates))

        # Compute leverage for each candidate
        scores = jnp.zeros(len(candidates))

        for i, cand in enumerate(candidates):
            phi_cand = self.model.basis_library.evaluate_subset(
                cand.reshape(1, -1),
                self.model.selected_indices_,
            ).ravel()

            # h = phi @ (Phi.T @ Phi)^-1 @ phi.T
            leverage = phi_cand @ M @ phi_cand
            scores = scores.at[i].set(leverage)

        return scores

    def _score_gradient(self, candidates: jnp.ndarray) -> jnp.ndarray:
        """
        Score based on gradient magnitude.

        Regions with high gradient may have rapid function changes.
        """
        eps = 1e-5
        n_features = len(self.bounds)

        scores = jnp.zeros(len(candidates))

        for i, cand in enumerate(candidates):
            grad_norm = 0.0

            for j in range(n_features):
                cand_plus = cand.at[j].add(eps)
                cand_minus = cand.at[j].add(-eps)

                y_plus = self.model.predict(cand_plus.reshape(1, -1))[0]
                y_minus = self.model.predict(cand_minus.reshape(1, -1))[0]

                grad_j = (y_plus - y_minus) / (2 * eps)
                grad_norm += grad_j**2

            scores = scores.at[i].set(jnp.sqrt(grad_norm))

        return scores

    def _score_space_filling(self, candidates: jnp.ndarray) -> jnp.ndarray:
        """
        Score based on distance from existing points.

        Encourages exploration of unsampled regions.
        """
        if self.model._X_train is None:
            return jnp.ones(len(candidates))

        # Normalize to [0, 1]
        lower = self._bounds_array[:, 0]
        upper = self._bounds_array[:, 1]
        scale = upper - lower

        candidates_norm = (np.array(candidates) - lower) / scale
        existing_norm = (np.array(self.model._X_train) - lower) / scale

        scores = jnp.zeros(len(candidates))

        for i, cand in enumerate(candidates_norm):
            # Minimum distance to existing points
            distances = np.linalg.norm(existing_norm - cand, axis=1)
            min_dist = np.min(distances)
            scores = scores.at[i].set(min_dist)

        return scores


# =============================================================================
# Space-Filling Designs
# =============================================================================


def _snap_discrete(samples: np.ndarray, discrete_dims: dict[int, list]) -> np.ndarray:
    """Snap discrete dimensions to their nearest valid values."""
    if not discrete_dims:
        return samples
    samples = np.array(samples)
    for dim_idx, values in discrete_dims.items():
        values_arr = np.array(values, dtype=float)
        for row in range(len(samples)):
            dists = np.abs(values_arr - samples[row, dim_idx])
            samples[row, dim_idx] = values_arr[np.argmin(dists)]
    return samples


def latin_hypercube_sample(
    n_samples: int,
    bounds: list[tuple[float, float]],
    random_state: int | None = None,
    discrete_dims: dict[int, list] | None = None,
) -> jnp.ndarray:
    """
    Generate Latin Hypercube samples.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    bounds : list of tuple
        Bounds (lower, upper) for each dimension.
    random_state : int, optional
        Random seed.
    discrete_dims : dict, optional
        Mapping of dimension index to list of valid discrete values.
        Continuous dimensions are sampled normally; discrete dimensions
        are snapped to the nearest valid value.

    Returns
    -------
    samples : jnp.ndarray
        Sample points of shape (n_samples, n_dims).
    """
    sampler = qmc.LatinHypercube(d=len(bounds), seed=random_state)
    samples = sampler.random(n_samples)

    bounds_array = np.array(bounds)
    lower = bounds_array[:, 0]
    upper = bounds_array[:, 1]

    samples = qmc.scale(samples, lower, upper)
    samples = _snap_discrete(samples, discrete_dims or {})
    return jnp.array(samples)


def sobol_sample(
    n_samples: int,
    bounds: list[tuple[float, float]],
    random_state: int | None = None,
    discrete_dims: dict[int, list] | None = None,
) -> jnp.ndarray:
    """
    Generate Sobol sequence samples.

    Parameters
    ----------
    n_samples : int
        Number of samples (rounded to power of 2).
    bounds : list of tuple
        Bounds (lower, upper) for each dimension.
    random_state : int, optional
        Random seed for scrambling.
    discrete_dims : dict, optional
        Mapping of dimension index to list of valid discrete values.

    Returns
    -------
    samples : jnp.ndarray
        Sample points.
    """
    sampler = qmc.Sobol(d=len(bounds), scramble=True, seed=random_state)
    samples = sampler.random(n_samples)

    bounds_array = np.array(bounds)
    lower = bounds_array[:, 0]
    upper = bounds_array[:, 1]

    samples = qmc.scale(samples, lower, upper)
    samples = _snap_discrete(samples, discrete_dims or {})
    return jnp.array(samples)


def halton_sample(
    n_samples: int,
    bounds: list[tuple[float, float]],
    random_state: int | None = None,
    discrete_dims: dict[int, list] | None = None,
) -> jnp.ndarray:
    """
    Generate Halton sequence samples.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    bounds : list of tuple
        Bounds (lower, upper) for each dimension.
    random_state : int, optional
        Random seed for scrambling.
    discrete_dims : dict, optional
        Mapping of dimension index to list of valid discrete values.

    Returns
    -------
    samples : jnp.ndarray
        Sample points.
    """
    sampler = qmc.Halton(d=len(bounds), scramble=True, seed=random_state)
    samples = sampler.random(n_samples)

    bounds_array = np.array(bounds)
    lower = bounds_array[:, 0]
    upper = bounds_array[:, 1]

    samples = qmc.scale(samples, lower, upper)
    samples = _snap_discrete(samples, discrete_dims or {})
    return jnp.array(samples)


def grid_sample(
    n_per_dim: int,
    bounds: list[tuple[float, float]],
    discrete_dims: dict[int, list] | None = None,
) -> jnp.ndarray:
    """
    Generate grid samples.

    Parameters
    ----------
    n_per_dim : int
        Number of samples per dimension (for continuous dims).
    bounds : list of tuple
        Bounds (lower, upper) for each dimension.
    discrete_dims : dict, optional
        Mapping of dimension index to list of valid discrete values.
        Discrete dimensions use their exact values instead of linspace.

    Returns
    -------
    samples : jnp.ndarray
        Sample points.
    """
    discrete_dims = discrete_dims or {}
    grids = []
    for i, (lower, upper) in enumerate(bounds):
        if i in discrete_dims:
            grids.append(np.array(discrete_dims[i], dtype=float))
        else:
            grids.append(np.linspace(lower, upper, n_per_dim))
    mesh = np.meshgrid(*grids, indexing="ij")
    samples = np.stack([m.ravel() for m in mesh], axis=1)
    return jnp.array(samples)


# =============================================================================
# Optimal Experimental Design
# =============================================================================


def d_optimal_select(
    candidates: jnp.ndarray,
    n_select: int,
    basis_library,
    selected_indices: jnp.ndarray,
    random_state: int | None = None,
) -> jnp.ndarray:
    """
    Select D-optimal points from candidates.

    Maximizes det(Phi.T @ Phi) for selected design matrix.

    Parameters
    ----------
    candidates : jnp.ndarray
        Candidate points of shape (n_candidates, n_features).
    n_select : int
        Number of points to select.
    basis_library : BasisLibrary
        Basis function library.
    selected_indices : jnp.ndarray
        Indices of selected basis functions.
    random_state : int, optional
        Random seed for initialization.

    Returns
    -------
    selected : jnp.ndarray
        Indices of selected candidates.
    """
    rng = np.random.RandomState(random_state)
    n_candidates = len(candidates)

    # Initialize with random selection
    selected = set(rng.choice(n_candidates, size=min(n_select, n_candidates), replace=False))
    available = set(range(n_candidates)) - selected

    # Greedy improvement
    max_iter = 100
    for _ in range(max_iter):
        improved = False

        for i in list(selected):
            best_det = -np.inf
            best_swap = None

            # Current design matrix
            current_idx = list(selected)
            Phi_current = basis_library.evaluate_subset(
                candidates[jnp.array(current_idx)], selected_indices
            )
            current_det = np.linalg.det(Phi_current.T @ Phi_current)

            # Try swapping i with each available point
            for j in available:
                test_idx = [k if k != i else j for k in current_idx]
                Phi_test = basis_library.evaluate_subset(
                    candidates[jnp.array(test_idx)], selected_indices
                )
                test_det = np.linalg.det(Phi_test.T @ Phi_test)

                if test_det > best_det:
                    best_det = test_det
                    best_swap = (i, j)

            if best_swap and best_det > current_det * 1.001:
                selected.remove(best_swap[0])
                selected.add(best_swap[1])
                available.add(best_swap[0])
                available.remove(best_swap[1])
                improved = True

        if not improved:
            break

    return jnp.array(list(selected))
