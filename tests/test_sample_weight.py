"""Tests for ``sample_weight`` support across the library.

The invariants exercised here are the ones that make weights trustworthy:
uniform weights change nothing, the overall scale of the weights changes
nothing, a zero weight is equivalent to deleting the row from the fit, and
every derived quantity (MSE, information criteria, R², intervals, ANOVA)
is computed from the same weighted objective the coefficients came from.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr import (
    BasisLibrary,
    Constraints,
    MultiOutputSymbolicRegressor,
    SymbolicRegressor,
    cross_validate,
    fit_symbolic,
)
from jaxsr.constraints import fit_constrained_ols
from jaxsr.metrics import (
    compute_information_criterion,
    compute_mae,
    compute_mse,
    compute_r2,
    compute_rmse,
)
from jaxsr.selection import (
    exhaustive_search,
    fit_ols,
    fit_ridge,
    greedy_backward_elimination,
    greedy_forward_selection,
    lasso_path_selection,
    select_features,
)
from jaxsr.uncertainty import (
    anova,
    bootstrap_coefficients,
    bootstrap_model_selection,
    coefficient_intervals,
    conformal_predict_jackknife_plus,
)
from jaxsr.utils import effective_sample_size, validate_sample_weight, whiten


def _library():
    """Small polynomial library in one variable."""
    return (
        BasisLibrary(n_features=1, feature_names=["x"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=3)
    )


def _corrupted_half():
    """The reproducer from the issue: half the data is corrupt and down-weighted."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    y = 3 * X[:, 0] + 0.1 * rng.normal(size=200)
    y[:100] += 50  # corrupt the first half
    w = np.ones(200)
    w[:100] = 1e-6  # ...and down-weight exactly that half
    return jnp.array(X), jnp.array(y), jnp.array(w)


class TestValidateSampleWeight:
    """Tests for weight validation and normalisation."""

    def test_none_passes_through(self):
        assert validate_sample_weight(None, 10) is None

    def test_normalised_to_sum_n(self):
        w = validate_sample_weight(np.array([1e-3, 1e-3, 2e-3, 2e-3]), 4)
        assert float(jnp.sum(w)) == pytest.approx(4.0)
        # ratios preserved
        assert float(w[2] / w[0]) == pytest.approx(2.0)

    def test_idempotent(self):
        w1 = validate_sample_weight(np.array([1.0, 3.0, 4.0]), 3)
        w2 = validate_sample_weight(w1, 3)
        np.testing.assert_allclose(np.asarray(w1), np.asarray(w2))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="entries but there are"):
            validate_sample_weight(np.ones(5), 4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_sample_weight(np.array([1.0, -1.0]), 2)

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            validate_sample_weight(np.array([1.0, np.nan]), 2)

    def test_all_zero_raises(self):
        with pytest.raises(ValueError, match="positive sum"):
            validate_sample_weight(np.zeros(4), 4)

    def test_whiten_matches_manual(self):
        w = jnp.array([4.0, 1.0, 0.0])
        Phi = jnp.arange(6.0).reshape(3, 2)
        np.testing.assert_allclose(
            np.asarray(whiten(Phi, w)), np.asarray(Phi) * np.sqrt([[4.0], [1.0], [0.0]])
        )

    def test_effective_sample_size(self):
        assert effective_sample_size(None, 7) == 7.0
        assert effective_sample_size(jnp.ones(7), 7) == pytest.approx(7.0)
        # half the rows carrying essentially no weight halves the ESS
        w = validate_sample_weight(np.r_[np.full(50, 1e-9), np.ones(50)], 100)
        assert effective_sample_size(w, 100) == pytest.approx(50.0, rel=1e-3)


class TestIssueReproducer:
    """The exact scenario reported: weights must change the answer."""

    def test_weighted_fit_recovers_clean_model(self):
        X, y, w = _corrupted_half()
        library = _library()

        unweighted = SymbolicRegressor(basis_library=library, max_terms=2).fit(X, y)
        weighted = SymbolicRegressor(basis_library=library, max_terms=2).fit(X, y, sample_weight=w)
        clean = SymbolicRegressor(basis_library=library, max_terms=2).fit(X[100:], y[100:])

        # The bug was that these two were identical.
        assert weighted.expression_ != unweighted.expression_
        # The weighted fit agrees with a fit on the uncorrupted half alone.
        assert weighted.selected_features_ == clean.selected_features_
        np.testing.assert_allclose(
            np.asarray(weighted.coefficients_),
            np.asarray(clean.coefficients_),
            rtol=1e-3,
        )

    def test_effective_sample_size_reports_the_loss(self):
        X, y, w = _corrupted_half()
        model = SymbolicRegressor(basis_library=_library(), max_terms=2).fit(X, y, sample_weight=w)
        assert model.effective_sample_size_ == pytest.approx(100.0, rel=1e-3)
        # ...but n itself is unchanged, which is what the criteria use.
        assert model._result.n_samples == 200


class TestInvariants:
    """Properties that must hold for any weighting scheme."""

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 1))
        y = 2.0 * X[:, 0] + 0.5 + 0.05 * rng.normal(size=60)
        return jnp.array(X), jnp.array(y)

    @pytest.mark.parametrize(
        "strategy", ["greedy_forward", "greedy_backward", "exhaustive", "lasso_path"]
    )
    def test_uniform_weights_match_unweighted(self, data, strategy):
        X, y = data
        library = _library()
        kw = {"basis_library": library, "max_terms": 3, "strategy": strategy}

        plain = SymbolicRegressor(**kw).fit(X, y)
        uniform = SymbolicRegressor(**kw).fit(X, y, sample_weight=jnp.ones(len(y)))

        assert uniform.selected_features_ == plain.selected_features_
        np.testing.assert_allclose(
            np.asarray(uniform.coefficients_), np.asarray(plain.coefficients_), rtol=1e-6
        )
        assert uniform.metrics_["bic"] == pytest.approx(plain.metrics_["bic"], rel=1e-9)

    def test_scale_invariance(self, data):
        """Only weight ratios matter -- w and 1000*w must agree exactly."""
        X, y = data
        w = jnp.array(np.linspace(0.5, 2.0, len(y)))
        library = _library()

        small = SymbolicRegressor(basis_library=library, max_terms=3).fit(X, y, sample_weight=w)
        large = SymbolicRegressor(basis_library=library, max_terms=3).fit(
            X, y, sample_weight=1000.0 * w
        )

        assert small.selected_features_ == large.selected_features_
        # Only float32 rounding of the rescaling separates the two.
        np.testing.assert_allclose(
            np.asarray(small.coefficients_), np.asarray(large.coefficients_), rtol=1e-5
        )
        assert small.metrics_["mse"] == pytest.approx(large.metrics_["mse"], rel=1e-5)
        assert small.metrics_["aic"] == pytest.approx(large.metrics_["aic"], rel=1e-5)

    def test_zero_weight_equals_dropping_the_row(self, data):
        """A weight of 0 removes a row from the fit (though not from n)."""
        X, y = data
        w = np.ones(len(y))
        w[:10] = 0.0
        library = _library()

        zeroed = SymbolicRegressor(basis_library=library, max_terms=2).fit(
            X, y, sample_weight=jnp.array(w)
        )
        dropped = SymbolicRegressor(basis_library=library, max_terms=2).fit(X[10:], y[10:])

        assert zeroed.selected_features_ == dropped.selected_features_
        np.testing.assert_allclose(
            np.asarray(zeroed.coefficients_), np.asarray(dropped.coefficients_), rtol=1e-5
        )

    def test_integer_weights_match_row_duplication_in_coefficients_only(self, data):
        """Doubling a weight fits like duplicating the row -- but n must not move."""
        X, y = data
        w = np.ones(len(y))
        w[:20] = 2.0
        library = _library()

        weighted = SymbolicRegressor(basis_library=library, max_terms=2).fit(
            X, y, sample_weight=jnp.array(w)
        )
        duplicated = SymbolicRegressor(basis_library=library, max_terms=2).fit(
            jnp.concatenate([X, X[:20]]), jnp.concatenate([y, y[:20]])
        )

        np.testing.assert_allclose(
            np.asarray(weighted.coefficients_), np.asarray(duplicated.coefficients_), rtol=1e-5
        )
        # ...and this is exactly why duplication is not a valid emulation:
        assert weighted._result.n_samples == len(y)
        assert duplicated._result.n_samples == len(y) + 20
        assert weighted.metrics_["bic"] != pytest.approx(duplicated.metrics_["bic"])


class TestReportedQuantities:
    """MSE, information criteria and R² must come from the weighted objective."""

    @pytest.fixture
    def fitted(self):
        X, y, w = _corrupted_half()
        model = SymbolicRegressor(basis_library=_library(), max_terms=2).fit(X, y, sample_weight=w)
        return model, X, y, w

    def test_mse_is_the_weighted_mse(self, fitted):
        model, X, y, w = fitted
        expected = compute_mse(y, model.predict(X), w)
        assert model.metrics_["mse"] == pytest.approx(expected, rel=1e-6)

    def test_information_criteria_use_weighted_mse_and_nominal_n(self, fitted):
        model, X, y, w = fitted
        k = len(model.coefficients_)
        for criterion in ("aic", "bic", "aicc"):
            expected = compute_information_criterion(len(y), k, model.metrics_["mse"], criterion)
            assert model.metrics_[criterion] == pytest.approx(expected, rel=1e-9)

    def test_r2_is_weighted(self, fitted):
        model, X, y, w = fitted
        expected = compute_r2(y, model.predict(X), w)
        assert model.metrics_["r2"] == pytest.approx(expected, rel=1e-6)
        # The unweighted R2 on corrupted data is far worse; weighting must not
        # silently report that number.
        assert model.metrics_["r2"] > compute_r2(y, model.predict(X))

    def test_score_accepts_its_own_weights(self, fitted):
        model, X, y, w = fitted
        assert model.score(X, y, sample_weight=w) == pytest.approx(
            compute_r2(y, model.predict(X), w), rel=1e-6
        )
        assert model.score(X, y) != pytest.approx(model.score(X, y, sample_weight=w))

    def test_sample_weight_property_is_normalised(self, fitted):
        model, _X, y, _w = fitted
        assert float(jnp.sum(model.sample_weight_)) == pytest.approx(len(y))

    def test_unweighted_model_reports_none(self):
        X, y, _w = _corrupted_half()
        model = SymbolicRegressor(basis_library=_library(), max_terms=2).fit(X, y)
        assert model.sample_weight_ is None
        assert model.effective_sample_size_ == pytest.approx(len(y))


class TestMetricFunctions:
    """Weighted metrics against hand-computed values."""

    def test_weighted_mse_and_mae(self):
        y = jnp.array([0.0, 0.0, 0.0, 0.0])
        y_pred = jnp.array([1.0, 1.0, 3.0, 3.0])
        w = jnp.array([3.0, 3.0, 1.0, 1.0])
        # normalised weights are [1.5, 1.5, 0.5, 0.5]
        assert compute_mse(y, y_pred, w) == pytest.approx((1.5 + 1.5 + 4.5 + 4.5) / 4)
        assert compute_rmse(y, y_pred, w) == pytest.approx(np.sqrt(3.0))
        assert compute_mae(y, y_pred, w) == pytest.approx((1.5 + 1.5 + 1.5 + 1.5) / 4)

    def test_weighted_r2_uses_weighted_mean(self):
        y = jnp.array([0.0, 2.0, 100.0, 200.0])
        y_pred = jnp.array([0.0, 2.0, 0.0, 0.0])
        # With effectively all the weight on the first two points -- which the
        # model fits exactly -- R2 is 1 even though it misses the other two by
        # a mile.  The total sum of squares is taken about the weighted mean
        # (1.0 here), not the plain mean (75.5).
        w = jnp.array([1.0, 1.0, 1e-12, 1e-12])
        assert compute_r2(y, y_pred, w) == pytest.approx(1.0, abs=1e-6)
        assert compute_r2(y, y_pred) < 0.1

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            compute_mse(jnp.zeros(3), jnp.zeros(3), jnp.array([1.0, -1.0, 1.0]))


class TestFittingPrimitives:
    """The low-level solvers."""

    def test_fit_ols_matches_normal_equations(self):
        rng = np.random.default_rng(5)
        Phi = rng.normal(size=(30, 3))
        y = rng.normal(size=30)
        w = np.abs(rng.normal(size=30)) + 0.1

        coeffs, mse = fit_ols(jnp.array(Phi), jnp.array(y), sample_weight=jnp.array(w))

        W = np.diag(w)
        expected = np.linalg.solve(Phi.T @ W @ Phi, Phi.T @ W @ y)
        np.testing.assert_allclose(np.asarray(coeffs), expected, rtol=1e-5)

        w_norm = w * (len(w) / w.sum())
        resid = y - Phi @ np.asarray(coeffs)
        assert mse == pytest.approx(float(np.sum(w_norm * resid**2) / len(y)), rel=1e-5)

    def test_fit_ridge_is_weighted(self):
        rng = np.random.default_rng(6)
        Phi = rng.normal(size=(30, 3))
        y = rng.normal(size=30)
        w = np.abs(rng.normal(size=30)) + 0.1
        w_norm = w * (len(w) / w.sum())

        coeffs, _ = fit_ridge(jnp.array(Phi), jnp.array(y), 0.5, sample_weight=jnp.array(w))

        W = np.diag(w_norm)
        expected = np.linalg.solve(Phi.T @ W @ Phi + 0.5 * np.eye(3), Phi.T @ W @ y)
        np.testing.assert_allclose(np.asarray(coeffs), expected, rtol=1e-5)

    def test_unweighted_calls_are_unchanged(self):
        Phi = jnp.column_stack([jnp.ones(4), jnp.array([1.0, 2.0, 3.0, 4.0])])
        y = jnp.array([3.0, 5.0, 7.0, 9.0])
        coeffs, mse = fit_ols(Phi, y)
        np.testing.assert_allclose(np.asarray(coeffs), [1.0, 2.0], atol=1e-5)
        assert mse < 1e-10


class TestSelectionStrategies:
    """Weights must steer term selection, not only the final coefficients."""

    @pytest.fixture
    def problem(self):
        X, y, w = _corrupted_half()
        library = _library()
        Phi = library.evaluate(X)
        return Phi, y, w, library

    @pytest.mark.parametrize(
        "func",
        [
            greedy_forward_selection,
            greedy_backward_elimination,
            exhaustive_search,
            lasso_path_selection,
        ],
    )
    def test_strategy_accepts_weights(self, problem, func):
        Phi, y, w, library = problem
        path = func(
            Phi,
            y,
            library.names,
            library.complexities,
            max_terms=2,
            sample_weight=w,
        )
        # The clean signal is y = 3x; the corrupted, unweighted fit picks up a
        # large constant instead.
        assert "x" in path.best.selected_names

    def test_select_features_forwards_weights(self, problem):
        Phi, y, w, library = problem
        weighted = select_features(
            Phi, y, library.names, library.complexities, max_terms=2, sample_weight=w
        )
        plain = select_features(Phi, y, library.names, library.complexities, max_terms=2)
        assert weighted.best.selected_names != plain.best.selected_names


class TestConstraints:
    """Constrained refits must honour the weights too."""

    def test_constrained_refit_is_weighted(self):
        X, y, w = _corrupted_half()
        constraints = Constraints().add_sign_constraint("x", "positive")

        model = SymbolicRegressor(
            basis_library=_library(), max_terms=2, constraints=constraints
        ).fit(X, y, sample_weight=w)

        assert "x" in model.selected_features_
        idx = model.selected_features_.index("x")
        assert float(model.coefficients_[idx]) == pytest.approx(3.0, rel=0.05)

    def test_fit_constrained_ols_weighted_matches_wls(self):
        rng = np.random.default_rng(9)
        Phi = rng.normal(size=(40, 2))
        y = rng.normal(size=40)
        w = np.abs(rng.normal(size=40)) + 0.1

        coeffs, mse = fit_constrained_ols(
            Phi=jnp.array(Phi),
            y=jnp.array(y),
            constraints=Constraints(),  # no constraints -> plain WLS fast path
            basis_names=["a", "b"],
            feature_names=["x"],
            X=jnp.array(rng.normal(size=(40, 1))),
            sample_weight=jnp.array(w),
        )

        W = np.diag(w)
        expected = np.linalg.solve(Phi.T @ W @ Phi, Phi.T @ W @ y)
        np.testing.assert_allclose(np.asarray(coeffs), expected, rtol=1e-4)
        assert mse > 0


class TestUncertainty:
    """Intervals, bootstrap and ANOVA under weights."""

    @pytest.fixture
    def fitted(self):
        X, y, w = _corrupted_half()
        model = SymbolicRegressor(basis_library=_library(), max_terms=2).fit(X, y, sample_weight=w)
        return model, X, y, w

    def test_coefficient_intervals_are_weighted(self, fitted):
        model, _X, _y, _w = fitted
        intervals = model.coefficient_intervals()
        est, lo, hi, se = intervals["x"]
        assert lo < est < hi
        # The corrupted rows carry ~no weight, so the SE reflects the clean half.
        assert se < 0.05

    def test_coefficient_intervals_function_respects_weights(self, fitted):
        model, X, y, w = fitted
        Phi = model.basis_library.evaluate_subset(X, model.selected_indices_)
        weighted = coefficient_intervals(
            Phi, y, model.coefficients_, model.selected_features_, sample_weight=w
        )
        unweighted = coefficient_intervals(Phi, y, model.coefficients_, model.selected_features_)
        assert weighted["x"][3] < unweighted["x"][3]

    def test_sigma_reflects_clean_rows(self, fitted):
        model, _X, _y, _w = fitted
        # noise sigma of the clean half is 0.1
        assert model.sigma_ == pytest.approx(0.1, rel=0.3)

    def test_prediction_interval_is_finite_and_tight(self, fitted):
        model, _X, _y, _w = fitted
        X_new = jnp.array([[0.0], [1.0]])
        y_pred, lo, hi = model.predict_interval(X_new)
        assert bool(jnp.all(jnp.isfinite(lo))) and bool(jnp.all(jnp.isfinite(hi)))
        assert bool(jnp.all(hi - lo < 2.0))

    def test_bootstrap_coefficients_uses_weights(self, fitted):
        model, _X, _y, _w = fitted
        boot = bootstrap_coefficients(model, n_bootstrap=100, seed=0)
        i = boot["names"].index("x")
        assert float(boot["mean"][i]) == pytest.approx(3.0, rel=0.05)
        assert float(boot["std"][i]) < 0.05

    def test_bootstrap_model_selection_inherits_weights(self, fitted):
        model, X, y, _w = fitted
        result = bootstrap_model_selection(model, X, y, n_bootstrap=5, seed=0)
        assert result["feature_frequencies"].get("x", 0.0) > 0.5

    def test_bootstrap_model_selection_weights_follow_groups(self, fitted):
        model, X, y, w = fitted
        groups = np.repeat(np.arange(20), 10)
        result = bootstrap_model_selection(
            model, X, y, n_bootstrap=5, seed=0, groups=groups, sample_weight=w
        )
        assert result["resampling"] == "groups"
        assert result["feature_frequencies"].get("x", 0.0) > 0.5

    def test_bootstrap_model_selection_rejects_weights_with_resample_fn(self, fitted):
        model, X, y, w = fitted
        with pytest.raises(ValueError, match="cannot be combined with resample_fn"):
            bootstrap_model_selection(
                model,
                None,
                None,
                n_bootstrap=2,
                resample_fn=lambda rng: (X, y),
                sample_weight=w,
            )

    def test_jackknife_conformal_runs_weighted(self, fitted):
        model, _X, _y, _w = fitted
        result = conformal_predict_jackknife_plus(model, jnp.array([[0.5]]), alpha=0.1)
        assert bool(jnp.all(jnp.isfinite(result["lower"])))
        assert float(result["upper"][0] - result["lower"][0]) < 2.0

    def test_anova_sums_of_squares_are_weighted(self, fitted):
        model, _X, _y, _w = fitted
        table = anova(model)
        rows = {r.source: r for r in table.rows}
        assert rows["Residual"].sum_sq == pytest.approx(
            model.metrics_["mse"] * model._result.n_samples, rel=1e-4
        )
        assert rows["Total"].sum_sq == pytest.approx(
            rows["Model"].sum_sq + rows["Residual"].sum_sq, rel=1e-6
        )


class TestAcquisition:
    """Active learning must score candidates against the weighted posterior."""

    def test_uncertainty_reflects_weights(self):
        """A region whose data was down-weighted must still look uncertain."""
        from jaxsr.acquisition import PredictionVariance

        rng = np.random.default_rng(17)
        X = jnp.array(np.sort(rng.uniform(-3, 3, size=(60, 1)), axis=0))
        x = np.asarray(X)[:, 0]
        # An intercept as well as a slope, so the design has a centre and the
        # leverage is not symmetric about x = 0.
        y = jnp.array(5.0 + 2.0 * x + 0.1 * rng.normal(size=60))

        library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()
        # Trust only the left half of the domain.
        w = np.where(x < 0, 1.0, 1e-6)

        weighted = SymbolicRegressor(basis_library=library, max_terms=2).fit(
            X, y, sample_weight=jnp.array(w)
        )
        plain = SymbolicRegressor(basis_library=library, max_terms=2).fit(X, y)

        candidates = jnp.array([[-2.0], [2.0]])
        acq = PredictionVariance()
        w_scores = np.asarray(acq.score(candidates, weighted))
        p_scores = np.asarray(acq.score(candidates, plain))

        # Unweighted, the two ends of the domain look comparably informed.
        # Weighted, the untrusted right end is far more uncertain, which is
        # where the next experiment should go.
        assert w_scores[1] / w_scores[0] > 5 * (p_scores[1] / p_scores[0])

    def test_kriging_believer_batch_runs_weighted(self):
        from jaxsr.acquisition import ActiveLearner, PredictionVariance

        X, y, w = _corrupted_half()
        model = SymbolicRegressor(basis_library=_library(), max_terms=2).fit(X, y, sample_weight=w)
        learner = ActiveLearner(model, bounds=[(-3.0, 3.0)], acquisition=PredictionVariance())
        result = learner.suggest(n_points=3, batch_strategy="kriging_believer")

        assert result.points.shape == (3, 1)
        # The fantasy updates must leave the model exactly as they found it.
        assert len(model.sample_weight_) == len(y)
        assert model._X_train.shape[0] == len(y)


class TestCrossValidation:
    """Weights follow their rows into the folds."""

    def test_cross_validate_with_weights(self):
        X, y, w = _corrupted_half()
        model = SymbolicRegressor(basis_library=_library(), max_terms=2)
        weighted = cross_validate(model, X, y, cv=3, sample_weight=w, random_state=0)
        plain = cross_validate(model, X, y, cv=3, random_state=0)
        # Down-weighting the corrupted half makes the folds far more predictable.
        assert weighted["mean_test_score"] > plain["mean_test_score"]

    def test_weights_compose_with_grouped_cv(self):
        """Weights and group-aware splitting are independent choices."""
        X, y, w = _corrupted_half()
        groups = np.repeat(np.arange(10), 20)
        model = SymbolicRegressor(basis_library=_library(), max_terms=2)

        result = cross_validate(model, X, y, cv=5, groups=groups, sample_weight=w, random_state=0)
        assert result["strategy"] == "group-kfold"
        assert result["n_splits"] == 5
        assert np.isfinite(result["mean_test_score"])
        # Groups 0-4 are entirely corrupted-and-down-weighted; their per-group
        # scores must still be finite numbers, not silent zeros.
        assert all(np.isfinite(v) for v in result["per_group_scores"].values())

    def test_zero_weight_group_scores_nan_not_zero(self):
        """A group with no weight gets NaN, not a number that reads as a score."""
        X, y, _w = _corrupted_half()
        groups = np.repeat(np.arange(10), 20)
        w = np.ones(200)
        w[:20] = 0.0  # group 0 carries no weight at all
        model = SymbolicRegressor(basis_library=_library(), max_terms=2)

        result = cross_validate(
            model, X, y, cv=5, groups=groups, sample_weight=jnp.array(w), random_state=0
        )
        # Group 0 shares its fold with a weighted group, so the fold is scorable
        # but group 0 itself is not.
        assert np.isnan(result["per_group_scores"][0])
        assert np.isfinite(result["mean_test_score"])

    def test_zero_weight_fold_raises_rather_than_scoring_nothing(self):
        """Leave-one-group-out on an unweighted group has nothing to score."""
        X, y, _w = _corrupted_half()
        groups = np.repeat(np.arange(10), 20)
        w = np.ones(200)
        w[:20] = 0.0
        model = SymbolicRegressor(basis_library=_library(), max_terms=2)

        with pytest.raises(ValueError, match="zero total sample_weight"):
            cross_validate(
                model,
                X,
                y,
                groups=groups,
                strategy="leave-one-group-out",
                sample_weight=jnp.array(w),
            )


class TestOtherEntryPoints:
    """The remaining public surfaces that take data."""

    def test_fit_symbolic_forwards_weights(self):
        X, y, w = _corrupted_half()
        model = fit_symbolic(X, y, max_terms=2, include_transcendental=False, sample_weight=w)
        assert "x0" in model.selected_features_

    def test_multioutput_shares_weights(self):
        X, y, w = _corrupted_half()
        Y = jnp.column_stack([y, y])
        mo = MultiOutputSymbolicRegressor(
            estimator=SymbolicRegressor(basis_library=_library(), max_terms=2)
        ).fit(X, Y, sample_weight=w)
        for est in mo.estimators_:
            assert "x" in est.selected_features_

    def test_update_keeps_weights(self):
        X, y, w = _corrupted_half()
        model = SymbolicRegressor(basis_library=_library(), max_terms=2).fit(X, y, sample_weight=w)
        model.update(jnp.array([[0.0]]), jnp.array([0.0]), refit=False)
        assert model.sample_weight_ is not None
        assert len(model.sample_weight_) == 201
        # the appended point enters at unit weight, the corrupt rows stay tiny
        assert float(model.sample_weight_[-1]) > float(model.sample_weight_[0]) * 100

    def test_update_with_explicit_new_weights(self):
        rng = np.random.default_rng(11)
        X = jnp.array(rng.normal(size=(30, 1)))
        y = jnp.array(2.0 * np.asarray(X)[:, 0])
        model = SymbolicRegressor(basis_library=_library(), max_terms=2).fit(X, y)
        model.update(
            jnp.array([[5.0]]),
            jnp.array([100.0]),
            refit=False,
            sample_weight_new=jnp.array([0.0]),
        )
        # A zero-weight outlier must not move the fit.
        assert float(model.predict(jnp.array([[1.0]]))[0]) == pytest.approx(2.0, rel=1e-3)


class TestParametricBasis:
    """Parametric columns are rebuilt from X and must be whitened to match."""

    def test_parametric_fit_respects_weights(self):
        rng = np.random.default_rng(13)
        X = jnp.array(rng.uniform(0.5, 3.0, size=(60, 1)))
        x = np.asarray(X)[:, 0]
        y = np.exp(-2.0 * x)
        y[:30] += 5.0  # corrupt half
        w = np.ones(60)
        w[:30] = 1e-8

        library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant()
        library.add_parametric(
            name="exp(-a*x)",
            func=lambda X, a: jnp.exp(-a * X[:, 0]),
            param_bounds={"a": (0.1, 10.0)},
        )

        model = SymbolicRegressor(basis_library=library, max_terms=2).fit(
            X, jnp.array(y), sample_weight=jnp.array(w)
        )
        # The optimised decay constant should come from the clean half.
        assert model.metrics_["mse"] < 1e-3
