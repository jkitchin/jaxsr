"""
Tests for JAXSR Uncertainty Quantification.

Tests cover:
- Classical OLS intervals (coefficient CIs, prediction/confidence intervals)
- Pareto front ensemble predictions
- Bayesian Model Averaging
- Conformal prediction (split and jackknife+)
- Residual bootstrap
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr import BasisLibrary, SymbolicRegressor
from jaxsr.uncertainty import (
    BayesianModelAverage,
    bootstrap_coefficients,
    bootstrap_predict,
    compute_unbiased_variance,
)

# =============================================================================
# Fixtures
# =============================================================================


def _make_linear_data(n=100, noise_std=0.5, seed=42):
    """Generate y = 2*x + 1 + noise."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 5, (n, 1))
    y = 2.0 * X[:, 0] + 1.0 + noise_std * rng.randn(n)
    return jnp.array(X), jnp.array(y)


def _make_quadratic_data(n=100, noise_std=0.5, seed=42):
    """Generate y = 1.5*x^2 - 0.5*x + 2 + noise."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 3, (n, 1))
    y = 1.5 * X[:, 0] ** 2 - 0.5 * X[:, 0] + 2.0 + noise_std * rng.randn(n)
    return jnp.array(X), jnp.array(y)


def _fit_model(X, y, max_terms=3):
    """Fit a basic model."""
    library = (
        BasisLibrary(n_features=X.shape[1])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=3)
    )
    model = SymbolicRegressor(basis_library=library, max_terms=max_terms, strategy="greedy_forward")
    model.fit(X, y)
    return model


# =============================================================================
# Phase 1: Classical OLS Tests
# =============================================================================


class TestUnbiasedVariance:
    def test_basic(self):
        """Unbiased variance should be close to true noise variance."""
        X, y = _make_linear_data(n=200, noise_std=0.5, seed=0)
        model = _fit_model(X, y, max_terms=2)
        Phi = model._get_Phi_train()
        sigma_sq = compute_unbiased_variance(Phi, y, model.coefficients_)
        # True variance is 0.25; allow reasonable tolerance
        assert 0.1 < sigma_sq < 0.6

    def test_insufficient_dof(self):
        """Warn when degrees of freedom are insufficient."""
        X = jnp.array([[1.0], [2.0]])
        y = jnp.array([3.0, 5.0])
        Phi = jnp.column_stack([jnp.ones(2), X[:, 0]])
        coeffs = jnp.array([1.0, 2.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_unbiased_variance(Phi, y, coeffs)
            assert result == float("inf")
            assert len(w) == 1


class TestCoefficientCovariance:
    def test_shape(self):
        """Covariance matrix should be (p, p)."""
        X, y = _make_linear_data()
        model = _fit_model(X, y, max_terms=2)
        cov = model.covariance_matrix_
        p = len(model.coefficients_)
        assert cov.shape == (p, p)

    def test_positive_diagonal(self):
        """Diagonal elements (variances) should be positive."""
        X, y = _make_linear_data()
        model = _fit_model(X, y, max_terms=2)
        cov = model.covariance_matrix_
        assert jnp.all(jnp.diag(cov) > 0)

    def test_symmetric(self):
        """Covariance matrix should be symmetric."""
        X, y = _make_linear_data()
        model = _fit_model(X, y, max_terms=2)
        cov = model.covariance_matrix_
        np.testing.assert_allclose(np.array(cov), np.array(cov.T), atol=1e-6)


class TestCoefficientIntervals:
    def test_coverage_simulation(self):
        """
        Across many seeds, ~95% of true coefficients should fall in 95% CIs.

        Uses y = 2*x + 1. True coefficients: intercept=1, slope=2.
        """
        n_seeds = 100
        alpha = 0.05
        true_intercept = 1.0
        true_slope = 2.0
        covers_intercept = 0
        covers_slope = 0

        for seed in range(n_seeds):
            X, y = _make_linear_data(n=50, noise_std=0.5, seed=seed)
            model = _fit_model(X, y, max_terms=2)
            intervals = model.coefficient_intervals(alpha)

            # Find intercept and slope in the intervals
            for name, (_est, lo, hi, _se) in intervals.items():
                if name == "1":
                    if lo <= true_intercept <= hi:
                        covers_intercept += 1
                elif name == "x0":
                    if lo <= true_slope <= hi:
                        covers_slope += 1

        # Coverage should be approximately 95%, allow range [80%, 100%]
        assert (
            covers_intercept / n_seeds >= 0.80
        ), f"Intercept coverage {covers_intercept}/{n_seeds} too low"
        assert covers_slope / n_seeds >= 0.80, f"Slope coverage {covers_slope}/{n_seeds} too low"

    def test_returns_correct_keys(self):
        X, y = _make_linear_data()
        model = _fit_model(X, y, max_terms=2)
        intervals = model.coefficient_intervals()
        for name in model.selected_features_:
            assert name in intervals
            est, lo, hi, se = intervals[name]
            assert lo < est < hi
            assert se > 0


class TestPredictionInterval:
    def test_coverage_simulation(self):
        """~95% of test points should fall within 95% prediction interval."""
        n_covers = 0
        n_total = 0
        alpha = 0.05

        for seed in range(50):
            rng = np.random.RandomState(seed)
            X_train = jnp.array(rng.uniform(0, 5, (80, 1)))
            y_train = jnp.array(2.0 * np.array(X_train[:, 0]) + 1.0 + 0.5 * rng.randn(80))
            X_test = jnp.array(rng.uniform(0, 5, (20, 1)))
            y_test = jnp.array(2.0 * np.array(X_test[:, 0]) + 1.0 + 0.5 * rng.randn(20))

            model = _fit_model(X_train, y_train, max_terms=2)
            y_pred, lower, upper = model.predict_interval(X_test, alpha)

            covered = (y_test >= lower) & (y_test <= upper)
            n_covers += int(jnp.sum(covered))
            n_total += len(y_test)

        coverage = n_covers / n_total
        # Should be ~95%, allow [85%, 100%]
        assert coverage >= 0.85, f"Prediction interval coverage {coverage:.2f} too low"

    def test_confidence_band_narrower_than_prediction(self):
        """Confidence band should be strictly inside prediction band."""
        X, y = _make_linear_data(n=100)
        model = _fit_model(X, y, max_terms=2)
        X_new = jnp.linspace(0, 5, 30).reshape(-1, 1)

        y_pred_p, pred_lo, pred_hi = model.predict_interval(X_new)
        y_pred_c, conf_lo, conf_hi = model.confidence_band(X_new)

        # Confidence band should be narrower
        np.testing.assert_array_less(np.array(conf_lo), np.array(conf_hi))
        np.testing.assert_array_less(np.array(pred_lo), np.array(conf_lo))
        np.testing.assert_array_less(np.array(conf_hi), np.array(pred_hi))

        # Predictions should be the same
        np.testing.assert_allclose(np.array(y_pred_p), np.array(y_pred_c), atol=1e-5)


class TestSigmaProperty:
    def test_sigma_matches_manual(self):
        """sigma_ should match sqrt(SSR/(n-p))."""
        X, y = _make_linear_data(n=100, noise_std=0.5)
        model = _fit_model(X, y, max_terms=2)

        Phi = model._get_Phi_train()
        residuals = y - Phi @ model.coefficients_
        n, p = Phi.shape
        expected_sigma = float(jnp.sqrt(jnp.sum(residuals**2) / (n - p)))

        assert abs(model.sigma_ - expected_sigma) < 1e-6

    def test_sigma_close_to_true(self):
        """sigma_ should be close to true noise level."""
        X, y = _make_linear_data(n=500, noise_std=0.5, seed=123)
        model = _fit_model(X, y, max_terms=2)
        # Should be close to 0.5
        assert 0.3 < model.sigma_ < 0.8


# =============================================================================
# Phase 1: Ensemble Tests
# =============================================================================


class TestEnsemblePredict:
    def test_basic_output(self):
        """Ensemble predict should return expected keys."""
        X, y = _make_quadratic_data(n=100)
        model = _fit_model(X, y, max_terms=3)
        X_new = jnp.linspace(-2, 3, 20).reshape(-1, 1)
        result = model.predict_ensemble(X_new)

        assert "y_mean" in result
        assert "y_std" in result
        assert "y_min" in result
        assert "y_max" in result
        assert "y_all" in result
        assert len(result["y_mean"]) == 20

    def test_single_model_zero_std(self):
        """If only one Pareto model, std should be 0."""
        X, y = _make_linear_data(n=50)
        # Very simple model likely produces just 1 Pareto model with 1 term
        library = BasisLibrary(n_features=1).add_constant().add_linear()
        model = SymbolicRegressor(basis_library=library, max_terms=1, strategy="greedy_forward")
        model.fit(X, y)
        result = model.predict_ensemble(X)
        # With max_terms=1 and greedy forward, there's only 1 model on the path
        # so Pareto front has 1 model → std = 0
        if len(result["models"]) == 1:
            np.testing.assert_allclose(np.array(result["y_std"]), 0.0, atol=1e-6)


# =============================================================================
# Phase 2: BMA Tests
# =============================================================================


class TestBayesianModelAverage:
    def test_weights_sum_to_one(self):
        """BMA weights should sum to 1."""
        X, y = _make_quadratic_data(n=100)
        model = _fit_model(X, y, max_terms=3)
        bma = BayesianModelAverage(model, criterion="bic")
        total = sum(bma.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_predict_shape(self):
        """BMA predict should return correct shapes."""
        X, y = _make_quadratic_data(n=100)
        model = _fit_model(X, y, max_terms=3)
        X_new = jnp.linspace(-2, 3, 15).reshape(-1, 1)
        bma = BayesianModelAverage(model, criterion="bic")
        y_mean, y_std = bma.predict(X_new)
        assert y_mean.shape == (15,)
        assert y_std.shape == (15,)

    def test_predict_interval(self):
        """BMA interval should have lower < mean < upper."""
        X, y = _make_quadratic_data(n=100)
        model = _fit_model(X, y, max_terms=3)
        X_new = jnp.linspace(-2, 3, 15).reshape(-1, 1)
        y_pred, lower, upper = model.predict_bma(X_new)
        assert jnp.all(lower <= y_pred)
        assert jnp.all(y_pred <= upper)


# =============================================================================
# Phase 2: Conformal Prediction Tests
# =============================================================================


class TestConformalSplit:
    def test_coverage(self):
        """Split conformal should achieve approximate coverage."""
        rng = np.random.RandomState(42)
        X_all = jnp.array(rng.uniform(0, 5, (300, 1)))
        y_all = jnp.array(2.0 * np.array(X_all[:, 0]) + 1.0 + 0.5 * rng.randn(300))

        X_train, y_train = X_all[:150], y_all[:150]
        X_cal, y_cal = X_all[150:250], y_all[150:250]
        X_test, y_test = X_all[250:], y_all[250:]

        model = _fit_model(X_train, y_train, max_terms=2)
        y_pred, lower, upper = model.predict_conformal(
            X_test, alpha=0.1, method="split", X_cal=X_cal, y_cal=y_cal
        )

        covered = (y_test >= lower) & (y_test <= upper)
        coverage = float(jnp.mean(covered))
        # Should be ~90%, allow [75%, 100%]
        assert coverage >= 0.75, f"Split conformal coverage {coverage:.2f} too low"

    def test_requires_calibration_data(self):
        """Split conformal should raise if X_cal/y_cal not provided."""
        X, y = _make_linear_data()
        model = _fit_model(X, y, max_terms=2)
        with pytest.raises(ValueError, match="X_cal and y_cal"):
            model.predict_conformal(X[:5], method="split")


class TestConformalJackknifePlus:
    def test_coverage(self):
        """Jackknife+ should achieve approximate coverage."""
        rng = np.random.RandomState(42)
        X_train = jnp.array(rng.uniform(0, 5, (100, 1)))
        y_train = jnp.array(2.0 * np.array(X_train[:, 0]) + 1.0 + 0.5 * rng.randn(100))
        X_test = jnp.array(rng.uniform(0, 5, (50, 1)))
        y_test = jnp.array(2.0 * np.array(X_test[:, 0]) + 1.0 + 0.5 * rng.randn(50))

        model = _fit_model(X_train, y_train, max_terms=2)
        y_pred, lower, upper = model.predict_conformal(X_test, alpha=0.1)

        covered = (y_test >= lower) & (y_test <= upper)
        coverage = float(jnp.mean(covered))
        assert coverage >= 0.70, f"Jackknife+ coverage {coverage:.2f} too low"

    def test_output_shape(self):
        X, y = _make_linear_data(n=60)
        model = _fit_model(X, y, max_terms=2)
        X_new = jnp.linspace(0, 5, 10).reshape(-1, 1)
        y_pred, lower, upper = model.predict_conformal(X_new)
        assert y_pred.shape == (10,)
        assert lower.shape == (10,)
        assert upper.shape == (10,)


# =============================================================================
# Phase 3: Bootstrap Tests
# =============================================================================


class TestBootstrapCoefficients:
    def test_basic_output(self):
        X, y = _make_linear_data(n=100)
        model = _fit_model(X, y, max_terms=2)
        result = bootstrap_coefficients(model, n_bootstrap=100, seed=42)

        assert "coefficients" in result
        assert "mean" in result
        assert "std" in result
        assert "lower" in result
        assert "upper" in result
        assert result["coefficients"].shape[0] == 100

    def test_coverage(self):
        """Bootstrap CIs should have reasonable coverage."""
        n_covers_intercept = 0
        n_covers_slope = 0
        n_seeds = 30

        for seed in range(n_seeds):
            X, y = _make_linear_data(n=80, noise_std=0.5, seed=seed)
            model = _fit_model(X, y, max_terms=2)
            result = bootstrap_coefficients(model, n_bootstrap=500, alpha=0.05, seed=seed)
            names = result["names"]
            for i, name in enumerate(names):
                lo = float(result["lower"][i])
                hi = float(result["upper"][i])
                if name == "1" and lo <= 1.0 <= hi:
                    n_covers_intercept += 1
                elif name == "x0" and lo <= 2.0 <= hi:
                    n_covers_slope += 1

        # Allow somewhat lower coverage for bootstrap (it's approximate)
        assert n_covers_intercept / n_seeds >= 0.70
        assert n_covers_slope / n_seeds >= 0.70


class TestBootstrapPredict:
    def test_output_shape(self):
        X, y = _make_linear_data(n=80)
        model = _fit_model(X, y, max_terms=2)
        X_new = jnp.linspace(0, 5, 15).reshape(-1, 1)
        result = bootstrap_predict(model, X_new, n_bootstrap=50, seed=42)
        assert result["y_pred"].shape == (15,)
        assert result["lower"].shape == (15,)
        assert result["upper"].shape == (15,)

    def test_interval_ordering(self):
        """Lower < mean < upper."""
        X, y = _make_linear_data(n=80)
        model = _fit_model(X, y, max_terms=2)
        X_new = jnp.linspace(0.5, 4.5, 10).reshape(-1, 1)
        result = bootstrap_predict(model, X_new, n_bootstrap=200, seed=42)
        assert jnp.all(result["lower"] <= result["y_mean"])
        assert jnp.all(result["y_mean"] <= result["upper"])


# =============================================================================
# Regressor Convenience Method Tests
# =============================================================================


class TestRegressorUQMethods:
    def test_predict_interval_shape(self):
        X, y = _make_linear_data(n=80)
        model = _fit_model(X, y, max_terms=2)
        X_new = jnp.linspace(0, 5, 10).reshape(-1, 1)
        y_pred, lo, hi = model.predict_interval(X_new)
        assert y_pred.shape == (10,)
        assert lo.shape == (10,)
        assert hi.shape == (10,)

    def test_confidence_band_shape(self):
        X, y = _make_linear_data(n=80)
        model = _fit_model(X, y, max_terms=2)
        X_new = jnp.linspace(0, 5, 10).reshape(-1, 1)
        y_pred, lo, hi = model.confidence_band(X_new)
        assert y_pred.shape == (10,)
        assert lo.shape == (10,)
        assert hi.shape == (10,)

    def test_not_fitted_raises(self):
        library = BasisLibrary(n_features=1).add_constant().add_linear()
        model = SymbolicRegressor(basis_library=library)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.sigma_

    def test_constrained_warns(self):
        """Should warn if constraints are active."""
        from jaxsr import Constraints

        X, y = _make_linear_data(n=80)
        library = BasisLibrary(n_features=1).add_constant().add_linear()
        model = SymbolicRegressor(
            basis_library=library,
            max_terms=2,
            constraints=Constraints(),
        )
        model.fit(X, y)
        # Accessing sigma_ with constraints should emit a warning
        # (empty Constraints may not trigger constraint path but
        # the warning checks self.constraints is not None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = model.sigma_
            assert any("Constraints are active" in str(wi.message) for wi in w)

    def test_regularized_warns(self):
        """Should warn if regularization is active."""
        X, y = _make_linear_data(n=80)
        library = BasisLibrary(n_features=1).add_constant().add_linear()
        model = SymbolicRegressor(
            basis_library=library,
            max_terms=2,
            regularization=0.01,
        )
        model.fit(X, y)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = model.sigma_
            assert any("Regularization is active" in str(wi.message) for wi in w)


# =============================================================================
# Unfitted Model Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_prediction_interval_on_training_data(self):
        """Prediction interval on training data should contain most points."""
        X, y = _make_linear_data(n=100, noise_std=0.5)
        model = _fit_model(X, y, max_terms=2)
        y_pred, lo, hi = model.predict_interval(X)
        covered = (y >= lo) & (y <= hi)
        coverage = float(jnp.mean(covered))
        assert coverage >= 0.85

    def test_wide_intervals_small_n(self):
        """With very few data points, intervals should be wide."""
        X, y = _make_linear_data(n=10, noise_std=0.5)
        model = _fit_model(X, y, max_terms=2)
        X_new = jnp.array([[2.5]])
        y_pred, lo, hi = model.predict_interval(X_new)
        width = float(hi[0] - lo[0])
        assert width > 1.0, f"Interval width {width} suspiciously narrow for n=10"
