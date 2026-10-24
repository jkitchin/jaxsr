"""Tests for information criteria and cross-validation in ``jaxsr.metrics``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jaxsr import BasisLibrary, SymbolicRegressor
from jaxsr.metrics import (
    _group_kfold_splits,
    _logo_splits,
    compute_aic,
    compute_aicc,
    compute_bic,
    compute_hqc,
    compute_information_criterion,
    compute_mdl,
    cross_validate,
    group_indices,
)

# Every information criterion here is lower-is-better and takes (n, k, mse).
_IC_FUNCS = [compute_aic, compute_aicc, compute_bic, compute_hqc, compute_mdl]


class TestPerfectFitScoring:
    """A zero-MSE model is the best fit and must score better than any other.

    Regression test: these functions previously returned ``+inf`` for
    ``mse <= 0``, which inverted the meaning and made model selection reject
    an exact model.  Exact zeros are routine in float64, where ``selection.py``
    computes MSE via the closed form ``(yTy - c @ rhs) / n``.
    """

    @pytest.mark.parametrize("ic", _IC_FUNCS, ids=lambda f: f.__name__)
    def test_zero_mse_is_finite(self, ic):
        """A perfect fit yields a finite score, not inf or nan."""
        value = ic(n_samples=100, n_params=3, mse=0.0)
        assert math.isfinite(value)

    @pytest.mark.parametrize("ic", _IC_FUNCS, ids=lambda f: f.__name__)
    def test_zero_mse_beats_imperfect_fit(self, ic):
        """A perfect fit scores better (lower) than an imperfect one."""
        assert ic(n_samples=100, n_params=3, mse=0.0) < ic(n_samples=100, n_params=3, mse=1.0)

    @pytest.mark.parametrize("ic", _IC_FUNCS, ids=lambda f: f.__name__)
    def test_ties_broken_by_parameter_count(self, ic):
        """Among equally perfect fits, fewer parameters wins."""
        assert ic(n_samples=100, n_params=2, mse=0.0) < ic(n_samples=100, n_params=5, mse=0.0)

    @pytest.mark.parametrize("ic", _IC_FUNCS, ids=lambda f: f.__name__)
    def test_negative_mse_treated_as_zero(self, ic):
        """A slightly negative MSE from cancellation is treated as a perfect fit."""
        assert ic(n_samples=100, n_params=3, mse=-1e-18) == ic(n_samples=100, n_params=3, mse=0.0)


class TestInformationCriteriaBasics:
    """Ordering and monotonicity properties shared by the criteria."""

    @pytest.mark.parametrize("ic", _IC_FUNCS, ids=lambda f: f.__name__)
    def test_worse_mse_scores_worse(self, ic):
        """Increasing MSE increases (worsens) the criterion."""
        assert ic(n_samples=50, n_params=3, mse=0.1) < ic(n_samples=50, n_params=3, mse=1.0)

    @pytest.mark.parametrize("ic", _IC_FUNCS, ids=lambda f: f.__name__)
    def test_more_params_penalized(self, ic):
        """At equal MSE, more parameters is penalized."""
        assert ic(n_samples=50, n_params=2, mse=0.5) < ic(n_samples=50, n_params=6, mse=0.5)

    def test_aic_matches_closed_form(self):
        """AIC equals n*log(2*pi*mse) + n + 2k."""
        n, k, mse = 40, 3, 0.25
        expected = n * math.log(2 * math.pi * mse) + n + 2 * k
        assert compute_aic(n, k, mse) == pytest.approx(expected)

    def test_bic_penalizes_more_than_aic_for_large_n(self):
        """BIC's log(n) penalty exceeds AIC's factor of 2 once n > e^2."""
        assert compute_bic(100, 4, 0.5) > compute_aic(100, 4, 0.5)

    def test_aicc_exceeds_aic(self):
        """The AICc small-sample correction is strictly positive."""
        assert compute_aicc(30, 4, 0.5) > compute_aic(30, 4, 0.5)

    def test_aicc_infinite_when_correction_undefined(self):
        """AICc is undefined when n - k - 1 <= 0."""
        assert compute_aicc(n_samples=5, n_params=4, mse=0.5) == float("inf")

    def test_hqc_infinite_for_tiny_samples(self):
        """HQC needs log(log(n)) > 0, so n must exceed e."""
        assert compute_hqc(n_samples=2, n_params=1, mse=0.5) == float("inf")


class TestComputeInformationCriterion:
    """The dispatcher used throughout selection."""

    @pytest.mark.parametrize("name", ["aic", "aicc", "bic"])
    def test_supported_criteria_match_direct_calls(self, name):
        """The dispatcher agrees with the individual functions."""
        direct = {"aic": compute_aic, "aicc": compute_aicc, "bic": compute_bic}[name]
        assert compute_information_criterion(60, 3, 0.4, name) == pytest.approx(direct(60, 3, 0.4))

    def test_unsupported_criterion_raises(self):
        """Only aic/aicc/bic are supported; 'cv' in particular is not."""
        with pytest.raises(ValueError):
            compute_information_criterion(60, 3, 0.4, "cv")

    def test_zero_mse_finite_through_dispatcher(self):
        """The perfect-fit fix reaches callers going through the dispatcher."""
        assert np.isfinite(compute_information_criterion(60, 3, 0.0, "aicc"))


# =============================================================================
# Cross-Validation Splitting
# =============================================================================


def _make_grouped_data(n_groups=6, per_group=8, seed=0):
    """Rows within a group share an offset, so rows are not independent."""
    rng = np.random.RandomState(seed)
    X_parts, y_parts, group_parts = [], [], []
    for g in range(n_groups):
        x = rng.uniform(0, 5, (per_group, 1))
        y = 2.0 * x[:, 0] + 3.0 * g + 0.1 * rng.randn(per_group)
        X_parts.append(x)
        y_parts.append(y)
        group_parts.append(np.full(per_group, g))
    return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(group_parts)


def _make_model():
    library = BasisLibrary(n_features=1).add_constant().add_linear()
    return SymbolicRegressor(basis_library=library, max_terms=2)


class TestGroupSplitters:
    def test_group_kfold_never_splits_a_group(self):
        row_indices = [np.arange(0, 5), np.arange(5, 12), np.arange(12, 15), np.arange(15, 25)]
        splits = _group_kfold_splits(row_indices, cv=2)
        assert len(splits) == 2
        for train_idx, test_idx, held_out in splits:
            assert set(train_idx).isdisjoint(test_idx)
            for g in held_out:
                assert set(row_indices[g]).issubset(test_idx)
                assert set(row_indices[g]).isdisjoint(train_idx)
        # Every group is held out exactly once.
        held = sorted(g for _, _, members in splits for g in members)
        assert held == [0, 1, 2, 3]

    def test_group_kfold_balances_fold_sizes(self):
        row_indices = [np.arange(0, 10), np.arange(10, 20), np.arange(20, 22), np.arange(22, 24)]
        splits = _group_kfold_splits(row_indices, cv=2)
        sizes = sorted(len(test_idx) for _, test_idx, _ in splits)
        assert sizes == [12, 12]

    def test_logo_holds_out_one_group_per_fold(self):
        row_indices = [np.arange(0, 4), np.arange(4, 9), np.arange(9, 11)]
        splits = _logo_splits(row_indices)
        assert len(splits) == 3
        for g, (train_idx, test_idx, held_out) in enumerate(splits):
            assert held_out == [g]
            assert list(test_idx) == list(row_indices[g])
            assert set(train_idx).isdisjoint(test_idx)

    def test_group_indices_sorted_and_complete(self):
        unique, row_indices = group_indices([2, 1, 2, 1, 3])
        assert list(unique) == [1, 2, 3]
        assert list(row_indices[0]) == [1, 3]
        assert list(row_indices[1]) == [0, 2]
        assert list(row_indices[2]) == [4]

    def test_group_indices_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one label"):
            group_indices([])


class TestGroupedCrossValidate:
    def test_leave_one_group_out(self):
        X, y, groups = _make_grouped_data(n_groups=5)
        result = cross_validate(_make_model(), X, y, groups=groups, strategy="leave-one-group-out")
        assert result["strategy"] == "leave-one-group-out"
        assert result["n_splits"] == 5
        assert result["groups_out"] == [[0], [1], [2], [3], [4]]
        assert sorted(result["per_group_scores"]) == [0, 1, 2, 3, 4]

    def test_edge_groups_reported_for_numeric_labels(self):
        X, y, groups = _make_grouped_data(n_groups=4)
        result = cross_validate(_make_model(), X, y, groups=groups, strategy="leave-one-group-out")
        assert result["edge_groups"] == [0, 3]
        assert sorted(result["edge_group_scores"]) == [0, 3]

    def test_edge_groups_empty_for_non_numeric_labels(self):
        X, y, groups = _make_grouped_data(n_groups=3)
        labels = np.array(["low", "mid", "high"])[groups.astype(int)]
        result = cross_validate(_make_model(), X, y, groups=labels, strategy="leave-one-group-out")
        assert result["edge_groups"] == []
        assert sorted(result["per_group_scores"]) == ["high", "low", "mid"]

    def test_groups_promote_default_strategy(self):
        X, y, groups = _make_grouped_data(n_groups=4)
        result = cross_validate(_make_model(), X, y, cv=2, groups=groups)
        assert result["strategy"] == "group-kfold"
        assert result["n_splits"] == 2
        # Each group is held out exactly once across the folds.
        held = sorted(g for fold in result["groups_out"] for g in fold)
        assert held == [0, 1, 2, 3]

    def test_grouped_cv_is_pessimistic_when_groups_shift(self):
        """Holding out whole groups exposes the between-group offset row CV hides."""
        X, y, groups = _make_grouped_data(n_groups=5, per_group=12)
        model = _make_model()
        rowwise = cross_validate(model, X, y, cv=5, scoring="neg_mse", random_state=0)
        grouped = cross_validate(model, X, y, groups=groups, strategy="leave-one-group-out")
        assert grouped["mean_test_score"] < rowwise["mean_test_score"]

    def test_plain_kfold_unchanged(self):
        X, y, _ = _make_grouped_data()
        result = cross_validate(_make_model(), X, y, cv=4, random_state=0)
        assert result["strategy"] == "kfold"
        assert result["n_splits"] == 4
        assert result["groups_out"] == [[], [], [], []]
        assert result["per_group_scores"] == {}
        assert result["edge_groups"] == []
        assert result["test_scores"].shape == (4,)


class TestCrossValidateValidation:
    def test_unknown_strategy_raises(self):
        X, y, groups = _make_grouped_data()
        with pytest.raises(ValueError, match="Unknown strategy"):
            cross_validate(_make_model(), X, y, groups=groups, strategy="loo")

    def test_group_strategy_without_groups_raises(self):
        X, y, _ = _make_grouped_data()
        with pytest.raises(ValueError, match="requires groups"):
            cross_validate(_make_model(), X, y, strategy="leave-one-group-out")

    def test_groups_length_mismatch_raises(self):
        X, y, groups = _make_grouped_data()
        with pytest.raises(ValueError, match="labels"):
            cross_validate(_make_model(), X, y, groups=groups[:-2])

    def test_more_folds_than_groups_raises(self):
        X, y, groups = _make_grouped_data(n_groups=3)
        with pytest.raises(ValueError, match="exceeds the number of distinct groups"):
            cross_validate(_make_model(), X, y, cv=5, groups=groups)

    def test_single_group_raises(self):
        X, y, _ = _make_grouped_data(n_groups=1, per_group=20)
        with pytest.raises(ValueError, match="at least 2 distinct groups"):
            cross_validate(_make_model(), X, y, groups=np.zeros(20))

    def test_unknown_scoring_raises(self):
        X, y, _ = _make_grouped_data()
        with pytest.raises(ValueError, match="Unknown scoring"):
            cross_validate(_make_model(), X, y, scoring="rmse")

    def test_too_few_folds_raises(self):
        X, y, _ = _make_grouped_data()
        with pytest.raises(ValueError, match="cv must be at least 2"):
            cross_validate(_make_model(), X, y, cv=1)
