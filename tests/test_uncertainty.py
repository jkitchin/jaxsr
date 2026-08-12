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
    anova,
    bootstrap_coefficients,
    bootstrap_model_selection,
    bootstrap_predict,
    compute_unbiased_variance,
    summarize_selection_replicates,
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


# =============================================================================
# ANOVA Tests
# =============================================================================


class TestAnovaSequential:
    """Tests for Type I (sequential) ANOVA."""

    def test_basic_structure(self):
        """ANOVA table should have per-term + 3 summary rows."""
        X, y = _make_quadratic_data(n=100, noise_std=0.3)
        model = _fit_model(X, y, max_terms=3)
        result = anova(model, anova_type="sequential")
        n_terms = len(model.selected_features_)
        # per-term rows + Model + Residual + Total
        assert len(result.rows) == n_terms + 3
        assert result.type == "sequential"

    def test_ss_decomposition(self):
        """Per-term SS should sum to SS_model (Type I property)."""
        X, y = _make_quadratic_data(n=100, noise_std=0.3)
        model = _fit_model(X, y, max_terms=3)
        result = anova(model, anova_type="sequential")

        summary = {r.source: r for r in result.rows}
        term_ss_sum = sum(
            r.sum_sq for r in result.rows if r.source not in ("Model", "Residual", "Total")
        )
        np.testing.assert_allclose(term_ss_sum, summary["Model"].sum_sq, rtol=1e-4)

    def test_ss_total_equals_model_plus_residual(self):
        """SS_total = SS_model + SS_residual."""
        X, y = _make_quadratic_data(n=100, noise_std=0.3)
        model = _fit_model(X, y, max_terms=3)
        result = anova(model, anova_type="sequential")
        summary = {r.source: r for r in result.rows}
        np.testing.assert_allclose(
            summary["Total"].sum_sq,
            summary["Model"].sum_sq + summary["Residual"].sum_sq,
            rtol=1e-4,
        )

    def test_df_decomposition(self):
        """df_total = df_model + df_residual."""
        X, y = _make_quadratic_data(n=100, noise_std=0.3)
        model = _fit_model(X, y, max_terms=3)
        result = anova(model, anova_type="sequential")
        summary = {r.source: r for r in result.rows}
        assert summary["Total"].df == summary["Model"].df + summary["Residual"].df

    def test_significant_terms_have_small_p(self):
        """For a strong signal the quadratic term should be significant."""
        X, y = _make_quadratic_data(n=200, noise_std=0.1)
        model = _fit_model(X, y, max_terms=3)
        result = anova(model, anova_type="sequential")
        # At least one term should have p < 0.05
        term_p = [r.p_value for r in result.rows if r.p_value is not None and r.source != "Model"]
        assert any(p < 0.05 for p in term_p), f"No significant terms found: {term_p}"

    def test_model_f_significant(self):
        """Overall model F-test should be significant."""
        X, y = _make_quadratic_data(n=200, noise_std=0.1)
        model = _fit_model(X, y, max_terms=3)
        result = anova(model, anova_type="sequential")
        model_row = next(r for r in result.rows if r.source == "Model")
        assert model_row.p_value < 0.01


class TestAnovaMarginal:
    """Tests for Type III (marginal) ANOVA."""

    def test_basic_structure(self):
        """Marginal ANOVA should have same number of rows."""
        X, y = _make_quadratic_data(n=100, noise_std=0.3)
        model = _fit_model(X, y, max_terms=3)
        result = anova(model, anova_type="marginal")
        n_terms = len(model.selected_features_)
        assert len(result.rows) == n_terms + 3
        assert result.type == "marginal"

    def test_marginal_ss_nonneg(self):
        """Each marginal SS should be non-negative."""
        X, y = _make_quadratic_data(n=100, noise_std=0.3)
        model = _fit_model(X, y, max_terms=3)
        result = anova(model, anova_type="marginal")
        for r in result.rows:
            assert r.sum_sq >= 0, f"Negative SS for {r.source}: {r.sum_sq}"

    def test_summary_rows_match_sequential(self):
        """Model, Residual, and Total summary rows should be the same."""
        X, y = _make_quadratic_data(n=100, noise_std=0.3)
        model = _fit_model(X, y, max_terms=3)
        seq = anova(model, anova_type="sequential")
        marg = anova(model, anova_type="marginal")
        for name in ("Model", "Residual", "Total"):
            r_seq = next(r for r in seq.rows if r.source == name)
            r_marg = next(r for r in marg.rows if r.source == name)
            np.testing.assert_allclose(r_seq.sum_sq, r_marg.sum_sq, rtol=1e-6)
            assert r_seq.df == r_marg.df


class TestAnovaHelpers:
    """Tests for AnovaResult helper methods."""

    def test_to_dict(self):
        """to_dict should return a plain dict with all fields."""
        X, y = _make_linear_data(n=50, noise_std=0.5)
        model = _fit_model(X, y, max_terms=2)
        result = anova(model)
        d = result.to_dict()
        assert "rows" in d
        assert "type" in d
        assert isinstance(d["rows"], list)
        assert all("source" in r for r in d["rows"])

    def test_repr(self):
        """__repr__ should produce a readable table string."""
        X, y = _make_linear_data(n=50, noise_std=0.5)
        model = _fit_model(X, y, max_terms=2)
        result = anova(model)
        text = repr(result)
        assert "ANOVA Table" in text
        assert "Residual" in text
        assert "Total" in text

    def test_term_names(self):
        """term_names should list only per-term sources."""
        X, y = _make_quadratic_data(n=100, noise_std=0.3)
        model = _fit_model(X, y, max_terms=3)
        result = anova(model)
        assert "Model" not in result.term_names
        assert "Residual" not in result.term_names
        assert "Total" not in result.term_names
        assert len(result.term_names) == len(model.selected_features_)

    def test_invalid_type_raises(self):
        """Passing an invalid anova_type should raise ValueError."""
        X, y = _make_linear_data(n=50, noise_std=0.5)
        model = _fit_model(X, y, max_terms=2)
        with pytest.raises(ValueError, match="anova_type"):
            anova(model, anova_type="type_ii")


class TestAnovaWarnings:
    """Tests that ANOVA emits warnings for parametric / constrained models."""

    def test_parametric_warning(self):
        """Parametric models should trigger a warning about approximate p-values."""
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 5, (80, 1))
        y = 3.0 * np.exp(-0.5 * X[:, 0]) + 1.0 + 0.2 * rng.randn(80)
        X, y = jnp.array(X), jnp.array(y)

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_parametric(
                name="exp(-a*x)",
                func=lambda X, a: jnp.exp(-a * X[:, 0]),
                param_bounds={"a": (0.01, 5.0)},
                feature_indices=(0,),
            )
        )
        model = SymbolicRegressor(basis_library=library, max_terms=3, strategy="greedy_forward")
        model.fit(X, y)
        result = anova(model)
        assert any("parametric" in w.lower() for w in result.warnings)


# =============================================================================
# Resampling Level Tests (grouped / pipeline bootstrap)
# =============================================================================


class _RecordingRegressor(SymbolicRegressor):
    """SymbolicRegressor that records the design matrix of every fit call.

    ``_clone_estimator`` rebuilds the estimator via ``type(est)(**params)``, so
    a subclass survives cloning and lets a test inspect what each replicate
    was actually trained on.
    """

    fit_calls: list = []

    def fit(self, X, y, sample_weight=None):
        type(self).fit_calls.append(np.asarray(X))
        return super().fit(X, y, sample_weight=sample_weight)


def _make_grouped_data(n_groups=6, per_group=10, seed=0):
    """Rows sharing a group label share an offset, so groups are not exchangeable."""
    rng = np.random.RandomState(seed)
    X_parts, y_parts, group_parts = [], [], []
    for g in range(n_groups):
        x = rng.uniform(0, 5, (per_group, 1))
        offset = 3.0 * g
        y = 2.0 * x[:, 0] + offset + 0.1 * rng.randn(per_group)
        X_parts.append(x)
        y_parts.append(y)
        group_parts.append(np.full(per_group, g))
    return (
        jnp.array(np.vstack(X_parts)),
        jnp.array(np.concatenate(y_parts)),
        np.concatenate(group_parts),
    )


class TestGroupedBootstrapModelSelection:
    def test_replicates_contain_whole_groups_only(self):
        """Every group appears 0, 1, 2, ... times over -- never partially."""
        X, y, groups = _make_grouped_data(n_groups=5, per_group=8)
        library = BasisLibrary(n_features=1).add_constant().add_linear()
        model = SymbolicRegressor(basis_library=library, max_terms=2)
        model.fit(X, y)

        _RecordingRegressor.fit_calls = []
        template = _RecordingRegressor(basis_library=library, max_terms=2)
        template.fit(X, y)
        _RecordingRegressor.fit_calls = []

        bootstrap_model_selection(template, X, y, n_bootstrap=10, seed=0, groups=groups)

        assert len(_RecordingRegressor.fit_calls) == 10
        X_np = np.asarray(X)
        per_group = 8
        row_to_group = {tuple(row): int(g) for row, g in zip(X_np, groups, strict=True)}
        for X_boot in _RecordingRegressor.fit_calls:
            assert X_boot.shape[0] == X_np.shape[0]
            counts: dict[int, int] = {}
            for row in X_boot:
                g = row_to_group[tuple(row)]
                counts[g] = counts.get(g, 0) + 1
            # A group is drawn whole or not at all, never split.
            assert all(c % per_group == 0 for c in counts.values())

    def test_reports_group_resampling(self):
        X, y, groups = _make_grouped_data()
        model = _fit_model(X, y, max_terms=2)
        result = bootstrap_model_selection(model, X, y, n_bootstrap=8, seed=1, groups=groups)
        assert result["resampling"] == "groups"
        assert result["n_replicates"] + result["n_failed"] == 8

    def test_length_mismatch_raises(self):
        X, y, groups = _make_grouped_data()
        model = _fit_model(X, y, max_terms=2)
        with pytest.raises(ValueError, match="labels"):
            bootstrap_model_selection(model, X, y, n_bootstrap=2, groups=groups[:-3])

    def test_single_group_raises(self):
        X, y = _make_linear_data(n=40)
        model = _fit_model(X, y, max_terms=2)
        with pytest.raises(ValueError, match="at least 2 distinct groups"):
            bootstrap_model_selection(model, X, y, n_bootstrap=2, groups=np.zeros(40))

    def test_groups_and_resample_fn_conflict(self):
        X, y, groups = _make_grouped_data()
        model = _fit_model(X, y, max_terms=2)
        with pytest.raises(ValueError, match="not both"):
            bootstrap_model_selection(
                model, X, y, n_bootstrap=2, groups=groups, resample_fn=lambda rng: (X, y)
            )


class TestPipelineBootstrapModelSelection:
    def test_resample_fn_drives_every_replicate(self):
        X, y = _make_linear_data(n=60, seed=3)
        model = _fit_model(X, y, max_terms=2)

        calls = []

        def regenerate(rng):
            # Stands in for re-running an upstream stage (smoother, simulation).
            calls.append(rng.randint(0, 1000))
            noise = rng.randn(len(y))
            return X, y + 0.1 * noise

        result = bootstrap_model_selection(
            model, None, None, n_bootstrap=6, seed=7, resample_fn=regenerate
        )
        assert len(calls) == 6
        assert result["resampling"] == "pipeline"
        assert result["n_replicates"] == 6

    def test_seed_makes_pipeline_replicates_reproducible(self):
        X, y = _make_linear_data(n=60, seed=3)
        model = _fit_model(X, y, max_terms=2)

        def regenerate(rng):
            return X, y + 0.2 * rng.randn(len(y))

        a = bootstrap_model_selection(
            model, None, None, n_bootstrap=5, seed=11, resample_fn=regenerate
        )
        b = bootstrap_model_selection(
            model, None, None, n_bootstrap=5, seed=11, resample_fn=regenerate
        )
        assert a["expressions"] == b["expressions"]

    def test_bad_resample_fn_return_raises(self):
        X, y = _make_linear_data(n=40)
        model = _fit_model(X, y, max_terms=2)
        with pytest.raises(TypeError, match=r"\(X_b, y_b\)"):
            bootstrap_model_selection(model, None, None, n_bootstrap=1, resample_fn=lambda rng: X)

    def test_missing_data_without_resample_fn_raises(self):
        X, y = _make_linear_data(n=40)
        model = _fit_model(X, y, max_terms=2)
        with pytest.raises(ValueError, match="X and y are required"):
            bootstrap_model_selection(model, None, None, n_bootstrap=2)

    def test_invalid_n_bootstrap_raises(self):
        X, y = _make_linear_data(n=40)
        model = _fit_model(X, y, max_terms=2)
        with pytest.raises(ValueError, match="n_bootstrap"):
            bootstrap_model_selection(model, X, y, n_bootstrap=0)


class TestRowBootstrapBackwardCompatibility:
    def test_legacy_keys_present(self):
        X, y = _make_linear_data(n=60)
        model = _fit_model(X, y, max_terms=2)
        result = bootstrap_model_selection(model, X, y, n_bootstrap=10, seed=0)
        assert set(result) >= {"feature_frequencies", "stability_score", "expressions"}
        assert result["resampling"] == "rows"
        assert 0.0 <= result["stability_score"] <= 1.0
        assert all(0.0 <= f <= 1.0 for f in result["feature_frequencies"].values())


class TestSummarizeSelectionReplicates:
    def test_counts_distinct_structures(self):
        replicates = [["a", "b"], ["b", "a"], ["a"], ["a", "c"]]
        summary = summarize_selection_replicates(replicates)
        assert summary["n_replicates"] == 4
        # ("a", "b") and ("b", "a") are the same structure.
        assert summary["n_distinct_structures"] == 3
        assert summary["structures"][0]["features"] == ["a", "b"]
        assert summary["structures"][0]["count"] == 2
        assert summary["feature_frequencies"]["a"] == 1.0
        assert summary["feature_frequencies"]["b"] == 0.5

    def test_stability_is_modal_frequency_without_reference(self):
        summary = summarize_selection_replicates([["a"], ["a"], ["b"], ["c"]])
        assert summary["stability_score"] == 0.5
        assert summary["reference_features"] is None

    def test_stability_measured_against_reference(self):
        summary = summarize_selection_replicates([["a"], ["a"], ["b"], ["c"]], reference=["b"])
        assert summary["stability_score"] == 0.25
        assert summary["reference_features"] == ["b"]

    def test_accepts_mappings_and_models(self):
        X, y = _make_linear_data(n=50)
        model = _fit_model(X, y, max_terms=2)
        summary = summarize_selection_replicates(
            [model, {"features": model.selected_features_, "expression": "custom"}],
            reference=model,
        )
        assert summary["stability_score"] == 1.0
        assert summary["expressions"][1] == "custom"

    def test_records_resampling_label(self):
        summary = summarize_selection_replicates([["a"]], resampling="pipeline")
        assert summary["resampling"] == "pipeline"

    def test_empty_replicates(self):
        summary = summarize_selection_replicates([])
        assert summary["n_replicates"] == 0
        assert summary["stability_score"] == 0.0
        assert summary["feature_frequencies"] == {}

    def test_unsupported_replicate_type_raises(self):
        with pytest.raises(TypeError, match="Cannot read selected features"):
            summarize_selection_replicates([42])


class TestParametricSelectionStability:
    """Selection stability for libraries with parametric basis functions."""

    def _setup(self, n=120, seed=0):
        """Data from y = 3*exp(-0.5x) + 1 and a library containing that form."""
        rng = np.random.RandomState(seed)
        X_np = rng.uniform(0, 5, (n, 1))
        y_np = 3.0 * np.exp(-0.5 * X_np[:, 0]) + 1.0 + 0.05 * rng.randn(n)
        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
            .add_parametric(
                name="exp(-a*x)",
                func=lambda X, a: jnp.exp(-a * X[:, 0]),
                param_bounds={"a": (0.05, 3.0)},
                feature_indices=(0,),
            )
        )
        model = SymbolicRegressor(
            basis_library=library,
            max_terms=2,
            strategy="exhaustive",
            information_criterion="bic",
        )
        model.fit(jnp.array(X_np), jnp.array(y_np))
        return model, jnp.array(X_np), jnp.array(y_np)

    def test_parametric_basis_aggregates_under_one_key(self):
        """A parametric basis is counted once, not once per fitted value.

        Regression test for keying on the rendered name, which gave every
        replicate a unique feature and a stability score of 0.
        """
        model, X, y = self._setup()
        result = bootstrap_model_selection(model, X, y, n_bootstrap=6, seed=0)

        assert result["feature_frequencies"]["exp(-a*x)"] == 1.0
        # No key carries a substituted value.
        assert not any(
            k.startswith("exp(-") and k != "exp(-a*x)" for k in result["feature_frequencies"]
        )
        assert result["n_distinct_structures"] == 1
        assert result["stability_score"] == 1.0

    def test_parameter_distribution_reported(self):
        """The fitted nonlinear parameter comes back as a distribution."""
        model, X, y = self._setup()
        result = bootstrap_model_selection(model, X, y, n_bootstrap=6, seed=0)

        dists = result["parameter_distributions"]
        assert set(dists) == {"exp(-a*x)"}
        summary = dists["exp(-a*x)"]["a"]
        assert set(summary) == {"mean", "sd", "q05", "q95", "n"}
        assert summary["n"] == 6
        assert 0.3 < summary["mean"] < 0.7  # true value is 0.5
        assert summary["q05"] <= summary["mean"] <= summary["q95"]
        assert summary["sd"] >= 0.0

    def test_no_distributions_without_parametric_bases(self):
        """An ordinary library reports an empty parameter_distributions."""
        X, y = _make_linear_data(n=60)
        model = _fit_model(X, y, max_terms=2)
        result = bootstrap_model_selection(model, X, y, n_bootstrap=5, seed=0)
        assert result["parameter_distributions"] == {}

    def test_does_not_mutate_the_original_model(self):
        """Bootstrapping leaves the caller's model and library untouched."""
        model, X, y = self._setup()
        expression_before = model.expression_
        names_before = list(model.basis_library.names)
        pred_before = np.asarray(model.predict(X))

        bootstrap_model_selection(model, X, y, n_bootstrap=4, seed=0)

        assert model.expression_ == expression_before
        assert model.basis_library.names == names_before
        assert np.allclose(np.asarray(model.predict(X)), pred_before)

    def test_summarize_replicates_reads_canonical_names(self):
        """Passing fitted models straight to the summariser keys them the same."""
        model, X, y = self._setup()
        summary = summarize_selection_replicates([model], reference=model)
        assert "exp(-a*x)" in summary["feature_frequencies"]
        assert summary["stability_score"] == 1.0
        assert set(summary["parameter_distributions"]) == {"exp(-a*x)"}
