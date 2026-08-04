"""Tests for information criteria in ``jaxsr.metrics``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jaxsr.metrics import (
    compute_aic,
    compute_aicc,
    compute_bic,
    compute_hqc,
    compute_information_criterion,
    compute_mdl,
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
