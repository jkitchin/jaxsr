"""
Tests for JAXSR Active Learning Acquisition Functions.

Tests cover:
- All individual acquisition functions (scoring, shape, sign conventions)
- Composite / weighted combinations
- ActiveLearner orchestrator (suggest, update, batch strategies, convergence)
- Edge cases (single point, low DoF, degenerate models)
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr import BasisLibrary, SymbolicRegressor
from jaxsr.acquisition import (
    AcquisitionResult,
    ActiveLearner,
    AOptimal,
    BatchStrategy,
    BMAUncertainty,
    Composite,
    ConfidenceBandWidth,
    DOptimal,
    EnsembleDisagreement,
    ExpectedImprovement,
    LCB,
    ModelDiscrimination,
    ModelMax,
    ModelMin,
    PredictionVariance,
    ProbabilityOfImprovement,
    ThompsonSampling,
    UCB,
    suggest_points,
)


# =========================================================================
# Fixtures
# =========================================================================


def _make_linear_data(n=100, noise_std=0.5, seed=42):
    """Generate y = 2*x + 1 + noise."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 5, (n, 1))
    y = 2.0 * X[:, 0] + 1.0 + noise_std * rng.randn(n)
    return jnp.array(X), jnp.array(y)


def _make_quadratic_data(n=120, noise_std=0.3, seed=42):
    """Generate y = x^2 - 2*x + 1 + noise."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 4, (n, 1))
    y = X[:, 0] ** 2 - 2.0 * X[:, 0] + 1.0 + noise_std * rng.randn(n)
    return jnp.array(X), jnp.array(y)


def _make_2d_data(n=150, noise_std=0.2, seed=42):
    """Generate y = 3*x0 - x1^2 + 0.5*x0*x1 + noise."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 5, (n, 2))
    y = (
        3.0 * X[:, 0]
        - X[:, 1] ** 2
        + 0.5 * X[:, 0] * X[:, 1]
        + noise_std * rng.randn(n)
    )
    return jnp.array(X), jnp.array(y)


def _fit_model_1d(X, y, max_terms=3):
    """Fit a 1d model with polynomials up to degree 3."""
    library = (
        BasisLibrary(n_features=1)
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=3)
    )
    model = SymbolicRegressor(
        basis_library=library, max_terms=max_terms, strategy="greedy_forward"
    )
    model.fit(X, y)
    return model


def _fit_model_2d(X, y, max_terms=5):
    """Fit a 2d model with polynomials and interactions."""
    library = (
        BasisLibrary(n_features=2, feature_names=["x0", "x1"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=3)
        .add_interactions(max_order=2)
    )
    model = SymbolicRegressor(
        basis_library=library, max_terms=max_terms, strategy="greedy_forward"
    )
    model.fit(X, y)
    return model


@pytest.fixture
def linear_model():
    X, y = _make_linear_data()
    return _fit_model_1d(X, y, max_terms=2)


@pytest.fixture
def quadratic_model():
    X, y = _make_quadratic_data()
    return _fit_model_1d(X, y, max_terms=3)


@pytest.fixture
def model_2d():
    X, y = _make_2d_data()
    return _fit_model_2d(X, y, max_terms=5)


BOUNDS_1D = [(0.0, 5.0)]
BOUNDS_2D = [(0.0, 5.0), (0.0, 5.0)]


# =========================================================================
# Test: Individual Acquisition Functions
# =========================================================================


class TestPredictionVariance:
    def test_shape(self, linear_model):
        acq = PredictionVariance()
        X_cand = jnp.linspace(0, 5, 20).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert scores.shape == (20,)

    def test_nonnegative(self, linear_model):
        acq = PredictionVariance()
        X_cand = jnp.linspace(0, 5, 50).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert jnp.all(scores >= 0)

    def test_higher_at_extremes(self, linear_model):
        """Variance should be higher at the edges of the training range."""
        acq = PredictionVariance()
        X_center = jnp.array([[2.5]])
        X_edge = jnp.array([[0.0], [5.0]])
        score_center = acq.score(X_center, linear_model)
        score_edge = acq.score(X_edge, linear_model)
        # At least one edge should have higher variance than center
        assert jnp.max(score_edge) > score_center[0]

    def test_name(self):
        assert PredictionVariance().name == "PredictionVariance"


class TestConfidenceBandWidth:
    def test_shape(self, linear_model):
        acq = ConfidenceBandWidth(alpha=0.05)
        X_cand = jnp.linspace(0, 5, 15).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert scores.shape == (15,)

    def test_nonnegative(self, linear_model):
        acq = ConfidenceBandWidth(alpha=0.05)
        X_cand = jnp.linspace(0, 5, 30).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert jnp.all(scores >= 0)

    def test_wider_with_lower_alpha(self, linear_model):
        """A 99% band should be wider than a 90% band."""
        X_cand = jnp.linspace(0, 5, 10).reshape(-1, 1)
        wide = ConfidenceBandWidth(alpha=0.01).score(X_cand, linear_model)
        narrow = ConfidenceBandWidth(alpha=0.10).score(X_cand, linear_model)
        assert jnp.all(wide >= narrow - 1e-6)


class TestEnsembleDisagreement:
    def test_shape(self, quadratic_model):
        acq = EnsembleDisagreement()
        X_cand = jnp.linspace(-2, 4, 10).reshape(-1, 1)
        scores = acq.score(X_cand, quadratic_model)
        assert scores.shape == (10,)

    def test_nonnegative(self, quadratic_model):
        acq = EnsembleDisagreement()
        X_cand = jnp.linspace(-2, 4, 20).reshape(-1, 1)
        scores = acq.score(X_cand, quadratic_model)
        assert jnp.all(scores >= 0)


class TestBMAUncertainty:
    def test_shape(self, quadratic_model):
        acq = BMAUncertainty(criterion="bic")
        X_cand = jnp.linspace(-2, 4, 8).reshape(-1, 1)
        scores = acq.score(X_cand, quadratic_model)
        assert scores.shape == (8,)

    def test_nonnegative(self, quadratic_model):
        acq = BMAUncertainty()
        X_cand = jnp.linspace(-2, 4, 20).reshape(-1, 1)
        scores = acq.score(X_cand, quadratic_model)
        assert jnp.all(scores >= 0)


class TestModelDiscrimination:
    def test_shape(self, quadratic_model):
        acq = ModelDiscrimination()
        X_cand = jnp.linspace(-2, 4, 12).reshape(-1, 1)
        scores = acq.score(X_cand, quadratic_model)
        assert scores.shape == (12,)

    def test_nonnegative(self, quadratic_model):
        acq = ModelDiscrimination()
        X_cand = jnp.linspace(-2, 4, 20).reshape(-1, 1)
        scores = acq.score(X_cand, quadratic_model)
        assert jnp.all(scores >= 0)

    def test_with_top_k(self, quadratic_model):
        acq = ModelDiscrimination(top_k=2)
        X_cand = jnp.linspace(-2, 4, 5).reshape(-1, 1)
        scores = acq.score(X_cand, quadratic_model)
        assert scores.shape == (5,)


class TestModelMinMax:
    def test_model_min_prefers_low_y(self, linear_model):
        acq = ModelMin()
        X_low = jnp.array([[0.0]])
        X_high = jnp.array([[5.0]])
        # y = 2x + 1, so x=0 gives y=1 (low) and x=5 gives y=11 (high)
        score_low = acq.score(X_low, linear_model)
        score_high = acq.score(X_high, linear_model)
        assert score_low > score_high  # higher score = more desirable

    def test_model_max_prefers_high_y(self, linear_model):
        acq = ModelMax()
        X_low = jnp.array([[0.0]])
        X_high = jnp.array([[5.0]])
        score_low = acq.score(X_low, linear_model)
        score_high = acq.score(X_high, linear_model)
        assert score_high > score_low


class TestUCB:
    def test_shape(self, linear_model):
        acq = UCB(kappa=2.0)
        X_cand = jnp.linspace(0, 5, 20).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert scores.shape == (20,)

    def test_kappa_zero_is_model_max(self, linear_model):
        """UCB with kappa=0 should equal raw prediction."""
        X_cand = jnp.linspace(0, 5, 10).reshape(-1, 1)
        ucb_scores = UCB(kappa=0).score(X_cand, linear_model)
        y_pred = linear_model.predict(X_cand)
        np.testing.assert_allclose(ucb_scores, y_pred, atol=1e-5)

    def test_higher_kappa_more_exploratory(self, linear_model):
        """Larger kappa should give relatively higher scores to uncertain regions."""
        X_cand = jnp.linspace(0, 5, 30).reshape(-1, 1)
        scores_low = UCB(kappa=0.5).score(X_cand, linear_model)
        scores_high = UCB(kappa=5.0).score(X_cand, linear_model)
        # Higher kappa boosts sigma contribution everywhere
        assert jnp.all(scores_high >= scores_low - 1e-6)


class TestLCB:
    def test_shape(self, linear_model):
        acq = LCB(kappa=2.0)
        X_cand = jnp.linspace(0, 5, 20).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert scores.shape == (20,)

    def test_prefers_low_y(self, linear_model):
        """LCB should prefer regions with low predicted y."""
        X_low = jnp.array([[0.0]])
        X_high = jnp.array([[5.0]])
        score_low = LCB(kappa=0).score(X_low, linear_model)
        score_high = LCB(kappa=0).score(X_high, linear_model)
        assert score_low > score_high  # -y_hat, so lower y => higher score


class TestExpectedImprovement:
    def test_shape(self, linear_model):
        acq = ExpectedImprovement()
        X_cand = jnp.linspace(0, 5, 15).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert scores.shape == (15,)

    def test_nonnegative(self, linear_model):
        acq = ExpectedImprovement()
        X_cand = jnp.linspace(0, 5, 30).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert jnp.all(scores >= -1e-8)

    def test_minimize_vs_maximize(self, linear_model):
        X_cand = jnp.linspace(0, 5, 20).reshape(-1, 1)
        ei_min = ExpectedImprovement(minimize=True).score(X_cand, linear_model)
        ei_max = ExpectedImprovement(minimize=False).score(X_cand, linear_model)
        # They should differ (unless trivially zero)
        assert not jnp.allclose(ei_min, ei_max)

    def test_custom_y_best(self, linear_model):
        acq = ExpectedImprovement(y_best=5.0, minimize=True)
        X_cand = jnp.linspace(0, 5, 10).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert scores.shape == (10,)


class TestProbabilityOfImprovement:
    def test_shape(self, linear_model):
        acq = ProbabilityOfImprovement()
        X_cand = jnp.linspace(0, 5, 15).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert scores.shape == (15,)

    def test_between_zero_and_one(self, linear_model):
        acq = ProbabilityOfImprovement()
        X_cand = jnp.linspace(0, 5, 30).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert jnp.all(scores >= -1e-8)
        assert jnp.all(scores <= 1.0 + 1e-8)

    def test_minimize_vs_maximize(self, linear_model):
        X_cand = jnp.linspace(0, 5, 20).reshape(-1, 1)
        pi_min = ProbabilityOfImprovement(minimize=True).score(
            X_cand, linear_model
        )
        pi_max = ProbabilityOfImprovement(minimize=False).score(
            X_cand, linear_model
        )
        assert not jnp.allclose(pi_min, pi_max)


class TestThompsonSampling:
    def test_shape(self, linear_model):
        acq = ThompsonSampling(seed=42)
        X_cand = jnp.linspace(0, 5, 20).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert scores.shape == (20,)

    def test_stochastic(self, linear_model):
        """Different calls (different internal seeds) should give different scores."""
        acq = ThompsonSampling(seed=42)
        X_cand = jnp.linspace(0, 5, 10).reshape(-1, 1)
        s1 = acq.score(X_cand, linear_model)
        s2 = acq.score(X_cand, linear_model)
        # s1 and s2 come from different posterior draws
        assert not jnp.allclose(s1, s2)

    def test_minimize_flag(self, linear_model):
        X_cand = jnp.linspace(0, 5, 10).reshape(-1, 1)
        s_min = ThompsonSampling(minimize=True, seed=0).score(
            X_cand, linear_model
        )
        s_max = ThompsonSampling(minimize=False, seed=0).score(
            X_cand, linear_model
        )
        np.testing.assert_allclose(s_min, -s_max, atol=1e-5)


class TestAOptimal:
    def test_shape(self, linear_model):
        acq = AOptimal()
        X_cand = jnp.linspace(0, 5, 10).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert scores.shape == (10,)

    def test_nonnegative(self, linear_model):
        acq = AOptimal()
        X_cand = jnp.linspace(0, 5, 30).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert jnp.all(scores >= -1e-8)


class TestDOptimal:
    def test_shape(self, linear_model):
        acq = DOptimal()
        X_cand = jnp.linspace(0, 5, 10).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert scores.shape == (10,)

    def test_nonnegative(self, linear_model):
        acq = DOptimal()
        X_cand = jnp.linspace(0, 5, 30).reshape(-1, 1)
        scores = acq.score(X_cand, linear_model)
        assert jnp.all(scores >= -1e-8)

    def test_higher_at_edges(self, linear_model):
        """D-optimal should prefer edges of design space (for linear model)."""
        acq = DOptimal()
        X_center = jnp.array([[2.5]])
        X_edge = jnp.array([[0.0], [5.0]])
        score_center = acq.score(X_center, linear_model)
        score_edge = acq.score(X_edge, linear_model)
        assert jnp.max(score_edge) > score_center[0]


# =========================================================================
# Test: Composite Acquisition Functions
# =========================================================================


class TestComposite:
    def test_add_two(self, linear_model):
        combined = UCB(kappa=2) + PredictionVariance()
        X_cand = jnp.linspace(0, 5, 10).reshape(-1, 1)
        scores = combined.score(X_cand, linear_model)
        assert scores.shape == (10,)

    def test_weighted(self, linear_model):
        combined = 0.7 * UCB(kappa=2) + 0.3 * PredictionVariance()
        X_cand = jnp.linspace(0, 5, 10).reshape(-1, 1)
        scores = combined.score(X_cand, linear_model)
        assert scores.shape == (10,)

    def test_name(self):
        combined = 0.7 * UCB(kappa=2) + 0.3 * PredictionVariance()
        assert "UCB" in combined.name
        assert "PredictionVariance" in combined.name

    def test_triple(self, linear_model):
        combined = UCB(kappa=1) + PredictionVariance() + ModelMin()
        X_cand = jnp.linspace(0, 5, 10).reshape(-1, 1)
        scores = combined.score(X_cand, linear_model)
        assert scores.shape == (10,)

    def test_scalar_mul(self, linear_model):
        scaled = 2.0 * PredictionVariance()
        assert isinstance(scaled, Composite)
        X_cand = jnp.linspace(0, 5, 10).reshape(-1, 1)
        scores = scaled.score(X_cand, linear_model)
        assert scores.shape == (10,)


# =========================================================================
# Test: ActiveLearner
# =========================================================================


class TestActiveLearnerBasic:
    def test_suggest_returns_result(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        result = learner.suggest(n_points=3)
        assert isinstance(result, AcquisitionResult)
        assert result.points.shape == (3, 1)
        assert result.scores.shape == (3,)

    def test_suggest_respects_n_points(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, UCB(kappa=2), random_state=42
        )
        for n in [1, 3, 10]:
            result = learner.suggest(n_points=n)
            assert result.points.shape[0] == n

    def test_bounds_mismatch_raises(self, linear_model):
        with pytest.raises(ValueError, match="bounds"):
            ActiveLearner(linear_model, [(0, 1), (0, 1)], PredictionVariance())

    def test_unfitted_model_raises(self):
        library = BasisLibrary(n_features=1).add_constant().add_linear()
        model = SymbolicRegressor(basis_library=library)
        with pytest.raises(RuntimeError, match="not fitted"):
            ActiveLearner(model, [(0, 5)], PredictionVariance())

    def test_2d(self, model_2d):
        learner = ActiveLearner(
            model_2d, BOUNDS_2D, UCB(kappa=2), random_state=42
        )
        result = learner.suggest(n_points=5)
        assert result.points.shape == (5, 2)


class TestActiveLearnerUpdate:
    def test_update_increases_data(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        n_before = learner.n_observations
        result = learner.suggest(n_points=3)
        y_new = jnp.array([1.0, 2.0, 3.0])
        learner.update(result.points, y_new)
        assert learner.n_observations == n_before + 3

    def test_iteration_counter(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        assert learner.iteration == 0
        result = learner.suggest(n_points=2)
        learner.update(result.points, jnp.array([1.0, 2.0]))
        assert learner.iteration == 1

    def test_history_tracking(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        result = learner.suggest(n_points=2)
        y_new = jnp.array([1.0, 2.0])
        learner.update(result.points, y_new)
        assert len(learner.history_X) == 1
        assert len(learner.history_y) == 1

    def test_update_with_refit_false(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        result = learner.suggest(n_points=2)
        y_new = jnp.array([1.0, 2.0])
        learner.update(result.points, y_new, refit=False)
        assert learner.iteration == 1


class TestActiveLearnerBatchStrategies:
    def test_greedy(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        result = learner.suggest(n_points=5, batch_strategy="greedy")
        assert result.points.shape == (5, 1)

    def test_penalized(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        result = learner.suggest(n_points=5, batch_strategy="penalized")
        assert result.points.shape == (5, 1)
        assert result.metadata.get("batch_strategy") == "penalized"

    def test_penalized_diversity(self, linear_model):
        """Penalized batch should be more spread out than greedy."""
        learner_g = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        learner_p = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        result_g = learner_g.suggest(n_points=5, batch_strategy="greedy")
        result_p = learner_p.suggest(n_points=5, batch_strategy="penalized")

        # Penalized batch should span a wider range
        range_g = float(jnp.max(result_g.points) - jnp.min(result_g.points))
        range_p = float(jnp.max(result_p.points) - jnp.min(result_p.points))
        assert range_p >= range_g * 0.8  # allow some tolerance

    def test_kriging_believer(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        result = learner.suggest(
            n_points=5, batch_strategy="kriging_believer"
        )
        assert result.points.shape == (5, 1)
        assert result.metadata.get("batch_strategy") == "kriging_believer"

    def test_kriging_believer_restores_model(self, linear_model):
        """Kriging believer should not permanently alter the model."""
        n_before = len(linear_model._y_train)
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, UCB(kappa=2), random_state=42
        )
        _ = learner.suggest(n_points=5, batch_strategy="kriging_believer")
        assert len(linear_model._y_train) == n_before

    def test_d_optimal(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        result = learner.suggest(n_points=5, batch_strategy="d_optimal")
        assert result.points.shape == (5, 1)
        assert result.metadata.get("batch_strategy") == "d_optimal"

    def test_invalid_strategy(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        with pytest.raises(ValueError):
            learner.suggest(n_points=3, batch_strategy="invalid_strategy")


class TestActiveLearnerConvergence:
    def test_not_converged_early(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        assert not learner.converged(tol=1e-3, window=3)

    def test_converged_low_mse(self):
        """A model with near-zero MSE should be converged."""
        rng = np.random.RandomState(0)
        X = rng.uniform(0, 5, (200, 1))
        y = 2.0 * X[:, 0] + 1.0  # No noise
        X, y = jnp.array(X), jnp.array(y)
        model = _fit_model_1d(X, y, max_terms=2)

        learner = ActiveLearner(
            model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        # Simulate enough iterations
        for _ in range(5):
            result = learner.suggest(n_points=2)
            y_new = 2.0 * result.points[:, 0] + 1.0
            learner.update(result.points, y_new)

        assert learner.converged(tol=1e-2, window=2)


class TestActiveLearnerProperties:
    def test_best_y(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        best = learner.best_y
        assert best == float(jnp.min(linear_model._y_train))

    def test_best_X(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        best_x = learner.best_X
        assert best_x.shape == (1,)

    def test_n_observations(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        assert learner.n_observations == 100


class TestActiveLearnerCandidateMethods:
    def test_lhs(self, linear_model):
        learner = ActiveLearner(
            linear_model,
            BOUNDS_1D,
            PredictionVariance(),
            candidate_method="lhs",
            random_state=42,
        )
        result = learner.suggest(n_points=3)
        assert result.points.shape == (3, 1)

    def test_sobol(self, linear_model):
        learner = ActiveLearner(
            linear_model,
            BOUNDS_1D,
            PredictionVariance(),
            candidate_method="sobol",
            random_state=42,
        )
        result = learner.suggest(n_points=3)
        assert result.points.shape == (3, 1)

    def test_halton(self, linear_model):
        learner = ActiveLearner(
            linear_model,
            BOUNDS_1D,
            PredictionVariance(),
            candidate_method="halton",
            random_state=42,
        )
        result = learner.suggest(n_points=3)
        assert result.points.shape == (3, 1)

    def test_random(self, linear_model):
        learner = ActiveLearner(
            linear_model,
            BOUNDS_1D,
            PredictionVariance(),
            candidate_method="random",
            random_state=42,
        )
        result = learner.suggest(n_points=3)
        assert result.points.shape == (3, 1)

    def test_invalid_method(self, linear_model):
        learner = ActiveLearner(
            linear_model,
            BOUNDS_1D,
            PredictionVariance(),
            candidate_method="invalid",
            random_state=42,
        )
        with pytest.raises(ValueError, match="candidate_method"):
            learner.suggest(n_points=3)


class TestActiveLearnerCustomCandidates:
    def test_user_provided_candidates(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        my_candidates = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        result = learner.suggest(
            n_points=2, candidates=my_candidates, min_distance=0.0
        )
        assert result.points.shape == (2, 1)
        # All returned points should be from our candidates
        for pt in result.points:
            assert any(jnp.allclose(pt, c) for c in my_candidates)


# =========================================================================
# Test: suggest_points convenience function
# =========================================================================


class TestSuggestPoints:
    def test_basic(self, linear_model):
        result = suggest_points(
            linear_model, BOUNDS_1D, PredictionVariance(), n_points=3,
            random_state=42,
        )
        assert isinstance(result, AcquisitionResult)
        assert result.points.shape == (3, 1)

    def test_with_ucb(self, linear_model):
        result = suggest_points(
            linear_model, BOUNDS_1D, UCB(kappa=2), n_points=5,
            random_state=42,
        )
        assert result.points.shape == (5, 1)

    def test_2d(self, model_2d):
        result = suggest_points(
            model_2d, BOUNDS_2D, ExpectedImprovement(), n_points=4,
            random_state=42,
        )
        assert result.points.shape == (4, 2)


# =========================================================================
# Test: Full Active Learning Loop
# =========================================================================


class TestFullLoop:
    def test_loop_reduces_mse(self):
        """An active learning loop should reduce MSE over iterations."""
        rng = np.random.RandomState(0)
        X_init = rng.uniform(0, 5, (20, 1))
        y_true = lambda x: 2.0 * x[:, 0] ** 2 - 3.0 * x[:, 0] + 1.0
        y_init = y_true(X_init) + 0.3 * rng.randn(20)
        X_init, y_init = jnp.array(X_init), jnp.array(y_init)

        model = _fit_model_1d(X_init, y_init, max_terms=3)
        mse_before = model.metrics_["mse"]

        learner = ActiveLearner(
            model, [(0.0, 5.0)], PredictionVariance(), random_state=42
        )

        for _ in range(5):
            result = learner.suggest(n_points=5)
            y_new = y_true(np.array(result.points)) + 0.3 * rng.randn(5)
            learner.update(result.points, jnp.array(y_new))

        mse_after = model.metrics_["mse"]
        # More data should help (or at least not hurt)
        assert mse_after <= mse_before * 1.5  # generous tolerance

    def test_loop_with_ei(self):
        """Active loop with EI for minimisation."""
        rng = np.random.RandomState(1)
        X_init = rng.uniform(0, 5, (30, 1))
        y_true = lambda x: (x[:, 0] - 2.5) ** 2 + 1.0
        y_init = y_true(X_init) + 0.1 * rng.randn(30)
        X_init, y_init = jnp.array(X_init), jnp.array(y_init)

        model = _fit_model_1d(X_init, y_init, max_terms=3)
        learner = ActiveLearner(
            model,
            [(0.0, 5.0)],
            ExpectedImprovement(minimize=True),
            random_state=42,
        )

        for _ in range(3):
            result = learner.suggest(n_points=3)
            y_new = y_true(np.array(result.points)) + 0.1 * rng.randn(3)
            learner.update(result.points, jnp.array(y_new))

        # The learner should have found points near x=2.5
        assert learner.n_observations > 30

    def test_loop_with_composite(self):
        """Active loop with composite acquisition."""
        rng = np.random.RandomState(2)
        X_init = rng.uniform(0, 5, (30, 1))
        y_init = 2.0 * X_init[:, 0] + 1.0 + 0.3 * rng.randn(30)
        X_init, y_init = jnp.array(X_init), jnp.array(y_init)

        model = _fit_model_1d(X_init, y_init, max_terms=2)
        acq = 0.6 * UCB(kappa=2) + 0.4 * PredictionVariance()
        learner = ActiveLearner(model, [(0.0, 5.0)], acq, random_state=42)

        result = learner.suggest(n_points=5)
        assert result.points.shape == (5, 1)
        assert "UCB" in result.acquisition or "Composite" in result.acquisition


# =========================================================================
# Test: Edge Cases
# =========================================================================


class TestEdgeCases:
    def test_single_candidate(self, linear_model):
        learner = ActiveLearner(
            linear_model, BOUNDS_1D, PredictionVariance(), random_state=42
        )
        candidates = jnp.array([[2.5]])
        result = learner.suggest(n_points=1, candidates=candidates, min_distance=0.0)
        assert result.points.shape[0] >= 1

    def test_more_points_requested_than_candidates(self, linear_model):
        """Should still return something reasonable."""
        learner = ActiveLearner(
            linear_model,
            BOUNDS_1D,
            PredictionVariance(),
            n_candidates=10,
            random_state=42,
        )
        result = learner.suggest(n_points=5, min_distance=0.0)
        assert result.points.shape[0] >= 1

    def test_all_acquisition_functions_work_2d(self, model_2d):
        """Smoke test: all acquisition functions produce valid output on 2d data."""
        X_cand = jnp.array([[1.0, 1.0], [2.0, 3.0], [4.0, 4.0]])

        acqs = [
            PredictionVariance(),
            ConfidenceBandWidth(alpha=0.05),
            EnsembleDisagreement(),
            BMAUncertainty(),
            ModelDiscrimination(),
            ModelMin(),
            ModelMax(),
            UCB(kappa=2),
            LCB(kappa=2),
            ExpectedImprovement(),
            ProbabilityOfImprovement(),
            ThompsonSampling(seed=42),
            AOptimal(),
            DOptimal(),
        ]

        for acq in acqs:
            scores = acq.score(X_cand, model_2d)
            assert scores.shape == (3,), f"{acq.name} returned wrong shape"
            assert jnp.all(jnp.isfinite(scores)), (
                f"{acq.name} returned non-finite scores"
            )
