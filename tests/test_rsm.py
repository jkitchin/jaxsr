"""
Tests for JAXSR Response Surface Methodology module.

Covers:
- Design generators (factorial, fractional factorial, CCD, Box-Behnken)
- Variable coding / decoding
- Canonical analysis
- ResponseSurface convenience class
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr import BasisLibrary, SymbolicRegressor
from jaxsr.rsm import (
    CanonicalAnalysis,
    ResponseSurface,
    box_behnken_design,
    canonical_analysis,
    central_composite_design,
    decode,
    encode,
    factorial_design,
    fractional_factorial_design,
)

# =========================================================================
# Helpers
# =========================================================================


def _quadratic_oracle(X, noise=0.0, seed=42):
    """y = 10 - 2*x1^2 - 3*x2^2 + x1*x2 + noise.

    Maximum at roughly x1=0, x2=0 in coded units.
    """
    rng = np.random.RandomState(seed)
    x1, x2 = X[:, 0], X[:, 1]
    y = 10.0 - 2.0 * x1**2 - 3.0 * x2**2 + 1.0 * x1 * x2
    if noise > 0:
        y = y + noise * rng.randn(len(y))
    return y


BOUNDS_2 = [(-5.0, 5.0), (-5.0, 5.0)]
BOUNDS_3 = [(0.0, 10.0), (1.0, 5.0), (100.0, 500.0)]


# =========================================================================
# Design Generators
# =========================================================================


class TestFactorialDesign:
    def test_two_level(self):
        X = factorial_design(levels=2, n_factors=3)
        assert X.shape == (8, 3)
        assert set(np.unique(X)) == {-1.0, 1.0}

    def test_three_level(self):
        X = factorial_design(levels=3, n_factors=2)
        assert X.shape == (9, 2)
        np.testing.assert_allclose(sorted(np.unique(X[:, 0])), [-1, 0, 1])

    def test_with_bounds(self):
        X = factorial_design(levels=2, n_factors=2, bounds=BOUNDS_2)
        assert X.shape == (4, 2)
        np.testing.assert_allclose(sorted(np.unique(X[:, 0])), [-5, 5])

    def test_levels_list(self):
        X = factorial_design(levels=[2, 3])
        assert X.shape == (6, 2)

    def test_n_factors_required(self):
        with pytest.raises(ValueError, match="n_factors"):
            factorial_design(levels=2)

    def test_levels_mismatch(self):
        with pytest.raises(ValueError, match="n_factors"):
            factorial_design(levels=[2, 3], n_factors=3)


class TestFractionalFactorial:
    def test_basic(self):
        X = fractional_factorial_design(4, resolution=3)
        # Should have fewer runs than 2^4 = 16 for res III
        assert X.shape[1] == 4
        assert X.shape[0] <= 16
        # All entries should be +/-1
        assert set(np.unique(X)) == {-1.0, 1.0}

    def test_with_bounds(self):
        X = fractional_factorial_design(3, resolution=3, bounds=BOUNDS_3)
        assert X.shape[1] == 3
        # Should be within bounds
        for i in range(3):
            assert np.min(X[:, i]) >= BOUNDS_3[i][0] - 1e-10
            assert np.max(X[:, i]) <= BOUNDS_3[i][1] + 1e-10

    def test_high_resolution_gives_more_runs(self):
        X3 = fractional_factorial_design(5, resolution=3)
        X5 = fractional_factorial_design(5, resolution=5)
        assert X5.shape[0] >= X3.shape[0]


class TestCentralCompositeDesign:
    def test_face_centered(self):
        X = central_composite_design(2, alpha="face")
        # 2^2 cube + 2*2 star + 1 center = 4 + 4 + 1 = 9
        assert X.shape == (9, 2)
        # Face centered: all values in [-1, 1]
        assert np.all(X >= -1 - 1e-10)
        assert np.all(X <= 1 + 1e-10)

    def test_rotatable(self):
        X = central_composite_design(3, alpha="rotatable", center_points=2)
        # 2^3 cube + 2*3 star + 2 center = 8 + 6 + 2 = 16
        assert X.shape == (16, 3)
        # Rotatable alpha for k=3: (2^3)^0.25 = 8^0.25 ≈ 1.68
        alpha_val = (2**3) ** 0.25
        assert np.max(np.abs(X)) == pytest.approx(alpha_val, abs=1e-10)

    def test_with_bounds(self):
        X = central_composite_design(2, alpha="face", bounds=BOUNDS_2)
        assert X.shape[1] == 2
        np.testing.assert_allclose(np.min(X[:, 0]), -5.0)
        np.testing.assert_allclose(np.max(X[:, 0]), 5.0)

    def test_custom_alpha(self):
        X = central_composite_design(2, alpha=1.5)
        assert np.max(np.abs(X)) == pytest.approx(1.5, abs=1e-10)

    def test_unknown_alpha_raises(self):
        with pytest.raises(ValueError, match="Unknown alpha"):
            central_composite_design(2, alpha="unknown_preset")

    def test_center_replicates(self):
        X3 = central_composite_design(2, alpha="face", center_points=3)
        X1 = central_composite_design(2, alpha="face", center_points=1)
        assert X3.shape[0] == X1.shape[0] + 2


class TestBoxBehnkenDesign:
    def test_three_factors(self):
        X = box_behnken_design(3, center_points=1)
        # C(3,2)*4 + 1 = 3*4 + 1 = 13
        assert X.shape == (13, 3)
        # No corner points: at most 2 factors at extremes
        for row in X:
            n_extreme = np.sum(np.abs(row) > 0.5)
            assert n_extreme <= 2

    def test_four_factors(self):
        X = box_behnken_design(4, center_points=2)
        # C(4,2)*4 + 2 = 6*4 + 2 = 26
        assert X.shape == (26, 4)

    def test_with_bounds(self):
        X = box_behnken_design(3, bounds=BOUNDS_3)
        for i in range(3):
            assert np.min(X[:, i]) >= BOUNDS_3[i][0] - 1e-10
            assert np.max(X[:, i]) <= BOUNDS_3[i][1] + 1e-10

    def test_too_few_factors_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            box_behnken_design(2)


# =========================================================================
# Variable Coding
# =========================================================================


class TestCoding:
    def test_encode_decode_roundtrip(self):
        X = np.array([[0, 1, 100], [10, 5, 500], [5, 3, 300]])
        X_coded = encode(X, BOUNDS_3)
        X_back = decode(X_coded, BOUNDS_3)
        np.testing.assert_allclose(X_back, X, atol=1e-10)

    def test_center_encodes_to_zero(self):
        centers = np.array([[(lo + hi) / 2 for lo, hi in BOUNDS_3]])
        coded = encode(centers, BOUNDS_3)
        np.testing.assert_allclose(coded, 0.0, atol=1e-10)

    def test_bounds_encode_to_pm1(self):
        low = np.array([[lo for lo, hi in BOUNDS_3]])
        high = np.array([[hi for lo, hi in BOUNDS_3]])
        np.testing.assert_allclose(encode(low, BOUNDS_3), -1.0, atol=1e-10)
        np.testing.assert_allclose(encode(high, BOUNDS_3), 1.0, atol=1e-10)


# =========================================================================
# Canonical Analysis
# =========================================================================


class TestCanonicalAnalysis:
    def _fit_quadratic_model(self, noise=0.05):
        """Fit a model on a CCD of y = 10 - 2x1^2 - 3x2^2 + x1*x2."""
        X = central_composite_design(2, alpha="face", center_points=3)
        y = _quadratic_oracle(X, noise=noise)

        library = (
            BasisLibrary(n_features=2, feature_names=["x1", "x2"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
            .add_interactions(max_order=2)
        )
        model = SymbolicRegressor(
            basis_library=library,
            max_terms=6,
            strategy="greedy_forward",
        )
        model.fit(jnp.array(X), jnp.array(y))
        return model

    def test_nature_is_maximum(self):
        """Function has a maximum (all negative second-order eigenvalues)."""
        model = self._fit_quadratic_model(noise=0.0)
        ca = canonical_analysis(model)
        assert ca.nature == "maximum"
        assert all(ev < 0 for ev in ca.eigenvalues)

    def test_stationary_point_near_origin(self):
        """True maximum is near coded origin for this function."""
        model = self._fit_quadratic_model(noise=0.0)
        ca = canonical_analysis(model)
        # Stationary point should be close to (0, 0) in coded
        np.testing.assert_allclose(ca.stationary_point, [0, 0], atol=0.5)

    def test_stationary_response_near_max(self):
        """Predicted response at stationary point should be near 10."""
        model = self._fit_quadratic_model(noise=0.0)
        ca = canonical_analysis(model)
        assert ca.stationary_response == pytest.approx(10.0, abs=1.0)

    def test_B_matrix_symmetric(self):
        model = self._fit_quadratic_model()
        ca = canonical_analysis(model)
        np.testing.assert_allclose(ca.B_matrix, ca.B_matrix.T)

    def test_bounds_converts_stationary_to_natural(self):
        model = self._fit_quadratic_model(noise=0.0)
        ca_coded = canonical_analysis(model)
        ca_natural = canonical_analysis(model, bounds=BOUNDS_2)
        # Natural units: center of bounds is 0 (since bounds are [-5,5])
        # so natural ≈ coded * 5 + 0
        expected = ca_coded.stationary_point * 5.0
        np.testing.assert_allclose(ca_natural.stationary_point, expected, atol=0.5)

    def test_non_quadratic_warning(self):
        """A model with only linear terms should warn."""
        rng = np.random.RandomState(0)
        X = rng.uniform(-1, 1, (30, 2))
        y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 1.0 + 0.01 * rng.randn(30)

        library = BasisLibrary(n_features=2, feature_names=["x1", "x2"]).add_constant().add_linear()
        model = SymbolicRegressor(
            basis_library=library,
            max_terms=3,
            strategy="greedy_forward",
        )
        model.fit(jnp.array(X), jnp.array(y))
        ca = canonical_analysis(model)
        assert "no curvature" in ca.nature.lower() or len(ca.warnings) > 0

    def test_saddle_point(self):
        """y = x1^2 - x2^2 should produce a saddle."""
        X = central_composite_design(2, alpha="face", center_points=3)
        y = X[:, 0] ** 2 - X[:, 1] ** 2

        library = (
            BasisLibrary(n_features=2, feature_names=["x1", "x2"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
            .add_interactions(max_order=2)
        )
        model = SymbolicRegressor(
            basis_library=library,
            max_terms=6,
            strategy="greedy_forward",
        )
        model.fit(jnp.array(X), jnp.array(y))
        ca = canonical_analysis(model)
        assert ca.nature == "saddle"

    def test_repr(self):
        model = self._fit_quadratic_model()
        ca = canonical_analysis(model)
        text = repr(ca)
        assert "Canonical Analysis" in text
        assert ca.nature in text


# =========================================================================
# ResponseSurface Class
# =========================================================================


class TestResponseSurface:
    def test_full_workflow(self):
        """CCD -> fit -> anova -> canonical round-trip."""
        rs = ResponseSurface(
            n_factors=2,
            bounds=BOUNDS_2,
            factor_names=["x1", "x2"],
        )
        X = rs.ccd(alpha="face", center_points=3)
        y = _quadratic_oracle(X, noise=0.1)
        rs.fit(X, y)

        # Predictions should work
        y_pred = rs.predict(X)
        assert y_pred.shape == (len(X),)

        # ANOVA
        table = rs.anova()
        assert len(table.rows) > 0

        # Canonical analysis
        ca = rs.canonical()
        assert isinstance(ca, CanonicalAnalysis)
        assert ca.nature in ("maximum", "saddle", "minimum", "stationary ridge")

    def test_box_behnken_workflow(self):
        rs = ResponseSurface(
            n_factors=3,
            bounds=BOUNDS_3,
            factor_names=["T", "P", "C"],
        )
        X = rs.box_behnken(center_points=2)
        # Just check design shape
        assert X.shape[1] == 3
        assert X.shape[0] == 14  # C(3,2)*4 + 2

    def test_factorial_workflow(self):
        rs = ResponseSurface(n_factors=2, bounds=BOUNDS_2)
        X = rs.factorial(levels=3)
        assert X.shape == (9, 2)

    def test_fractional_factorial_workflow(self):
        rs = ResponseSurface(n_factors=4, bounds=[(0, 1)] * 4)
        X = rs.fractional_factorial(resolution=3)
        assert X.shape[1] == 4
        assert X.shape[0] < 16

    def test_encode_decode(self):
        rs = ResponseSurface(n_factors=2, bounds=BOUNDS_2)
        X = np.array([[0.0, 0.0], [5.0, -5.0]])
        coded = rs.encode(X)
        np.testing.assert_allclose(coded, [[0, 0], [1, -1]], atol=1e-10)
        back = rs.decode(coded)
        np.testing.assert_allclose(back, X, atol=1e-10)

    def test_bounds_mismatch_raises(self):
        with pytest.raises(ValueError, match="n_factors"):
            ResponseSurface(n_factors=3, bounds=BOUNDS_2)

    def test_allow_transcendental(self):
        rs = ResponseSurface(
            n_factors=2,
            bounds=BOUNDS_2,
            allow_transcendental=True,
        )
        names = rs.model.basis_library.names
        assert any("exp" in n for n in names)

    def test_summary(self):
        rs = ResponseSurface(n_factors=2, bounds=BOUNDS_2)
        X = rs.ccd(alpha="face", center_points=2)
        y = _quadratic_oracle(X, noise=0.05)
        rs.fit(X, y)
        text = rs.summary()
        assert "Response Surface Summary" in text
        assert "Expression" in text
        assert "Canonical Analysis" in text

    def test_marginal_anova(self):
        rs = ResponseSurface(n_factors=2, bounds=BOUNDS_2)
        X = rs.ccd(alpha="face", center_points=3)
        y = _quadratic_oracle(X, noise=0.1)
        rs.fit(X, y)
        table = rs.anova(anova_type="marginal")
        assert table.type == "marginal"


class TestResponseSurfacePlotting:
    """Smoke tests for plotting (just verify they don't error)."""

    def _setup_fitted_rs(self):
        rs = ResponseSurface(n_factors=2, bounds=BOUNDS_2, factor_names=["x1", "x2"])
        X = rs.ccd(alpha="face", center_points=2)
        y = _quadratic_oracle(X, noise=0.05)
        rs.fit(X, y)
        return rs

    def test_plot_contour(self):
        import matplotlib

        matplotlib.use("Agg")
        rs = self._setup_fitted_rs()
        ax = rs.plot_contour()
        assert ax is not None

    def test_plot_contour_unfilled(self):
        import matplotlib

        matplotlib.use("Agg")
        rs = self._setup_fitted_rs()
        ax = rs.plot_contour(filled=False, show_design=False)
        assert ax is not None

    def test_plot_surface_3d(self):
        import matplotlib

        matplotlib.use("Agg")
        rs = self._setup_fitted_rs()
        ax = rs.plot_surface()
        assert ax is not None

    def test_plot_contour_three_factors(self):
        """With 3 factors, fixed values should appear in the title."""
        import matplotlib

        matplotlib.use("Agg")
        rs = ResponseSurface(n_factors=3, bounds=BOUNDS_3, factor_names=["T", "P", "C"])
        X = rs.ccd(alpha="face", center_points=2)
        rng = np.random.RandomState(0)
        y = X[:, 0] + X[:, 1] ** 2 - X[:, 2] + rng.randn(len(X)) * 0.1
        rs.fit(X, y)
        ax = rs.plot_contour(factors=(0, 1), fixed={2: 300.0})
        assert "C=300" in ax.get_title()
