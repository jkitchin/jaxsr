"""Tests for multivariate (N-D) derivative estimation."""

import numpy as np
import pytest

from jaxsr.derivatives import SurfaceDerivatives, estimate_partial_derivatives

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def smooth_surface():
    """f(x, y) = sin(2x) * exp(y/2) on a rectangular grid, with exact partials."""
    x = np.linspace(0.0, 2.0, 25)
    y = np.linspace(-1.0, 1.0, 20)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    values = np.sin(2 * xx) * np.exp(0.5 * yy)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    exact = {
        (1, 0): (2 * np.cos(2 * xx) * np.exp(0.5 * yy)).ravel(),
        (0, 1): (0.5 * np.sin(2 * xx) * np.exp(0.5 * yy)).ravel(),
        (2, 0): (-4 * np.sin(2 * xx) * np.exp(0.5 * yy)).ravel(),
    }
    return x, y, values, coords, exact


@pytest.fixture()
def bilinear_surface():
    """f(x, y) = 3x + 2y + x*y: exactly representable, so partials are exact."""
    x = np.linspace(0.0, 1.0, 12)
    y = np.linspace(0.0, 2.0, 11)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    values = 3 * xx + 2 * yy + xx * yy
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    return x, y, values, coords


def interior_mask(coords, frac=0.12):
    """Mask selecting points away from the boundary, where smoothers are weakest."""
    mask = np.ones(coords.shape[0], dtype=bool)
    for dim in range(coords.shape[1]):
        lo, hi = coords[:, dim].min(), coords[:, dim].max()
        pad = frac * (hi - lo)
        mask &= (coords[:, dim] >= lo + pad) & (coords[:, dim] <= hi - pad)
    return mask


# ===========================================================================
# TestInputHandling
# ===========================================================================


class TestInputHandling:
    """Tests for coordinate, value, order, and sigma normalization."""

    def test_grid_and_scattered_forms_agree(self, bilinear_surface):
        """Gridded axes + N-D values give the same fit as flat (n, d) coordinates."""
        x, y, values, coords = bilinear_surface
        grid_fit = SurfaceDerivatives().fit([x, y], values)
        flat_fit = SurfaceDerivatives().fit(coords, values.ravel())
        _, d_grid = grid_fit.derivatives(coords, order=[(1, 0)])
        _, d_flat = flat_fit.derivatives(coords, order=[(1, 0)])
        np.testing.assert_allclose(d_grid, d_flat, atol=1e-8)
        np.testing.assert_allclose(grid_fit.coords_, coords)

    def test_grid_shape_mismatch_raises(self, bilinear_surface):
        """Grid axes inconsistent with the values array raise ValueError."""
        x, y, values, _ = bilinear_surface
        with pytest.raises(ValueError, match="does not match the grid"):
            SurfaceDerivatives().fit([x, y[:-1]], values)

    def test_length_mismatch_raises(self):
        """Mismatched coordinate and value counts raise ValueError."""
        coords = np.random.default_rng(0).uniform(size=(20, 2))
        with pytest.raises(ValueError, match="must match the number"):
            SurfaceDerivatives().fit(coords, np.zeros(19))

    def test_non_finite_values_raise(self):
        """Non-finite values raise ValueError."""
        coords = np.random.default_rng(0).uniform(size=(20, 2))
        values = np.zeros(20)
        values[3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            SurfaceDerivatives().fit(coords, values)

    def test_too_few_points_raise(self):
        """Fewer than four points raise ValueError."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
        with pytest.raises(ValueError, match="at least 4 data points"):
            SurfaceDerivatives().fit(coords, np.zeros(3))

    def test_single_order_tuple_gives_one_column(self, bilinear_surface):
        """A single order tuple yields a (n_query, 1) partials array."""
        x, y, values, coords = bilinear_surface
        est = SurfaceDerivatives().fit([x, y], values)
        vals, partials = est.derivatives(coords, order=(1, 0))
        assert vals.shape == (coords.shape[0],)
        assert partials.shape == (coords.shape[0], 1)

    def test_wrong_order_length_raises(self, bilinear_surface):
        """Derivative orders must have one entry per dimension."""
        x, y, values, coords = bilinear_surface
        est = SurfaceDerivatives().fit([x, y], values)
        with pytest.raises(ValueError, match="one per dimension"):
            est.derivatives(coords, order=[(1, 0, 0)])

    def test_negative_order_raises(self, bilinear_surface):
        """Negative derivative orders raise ValueError."""
        x, y, values, coords = bilinear_surface
        est = SurfaceDerivatives().fit([x, y], values)
        with pytest.raises(ValueError, match="non-negative"):
            est.derivatives(coords, order=[(-1, 0)])

    def test_order_above_spline_degree_raises(self, bilinear_surface):
        """Requesting a derivative above the spline degree raises ValueError."""
        x, y, values, coords = bilinear_surface
        est = SurfaceDerivatives(degree=2).fit([x, y], values)
        with pytest.raises(ValueError, match="exceeds the spline degree"):
            est.derivatives(coords, order=[(3, 0)])

    def test_order_above_local_poly_degree_raises(self, bilinear_surface):
        """Local polynomial degree bounds the total derivative order."""
        x, y, values, coords = bilinear_surface
        est = SurfaceDerivatives(method="local_poly", degree=2).fit([x, y], values)
        with pytest.raises(ValueError, match="exceeds the local polynomial degree"):
            est.derivatives(coords, order=[(2, 1)])

    def test_query_dimension_mismatch_raises(self, bilinear_surface):
        """Query coordinates must have the fitted number of dimensions."""
        x, y, values, _ = bilinear_surface
        est = SurfaceDerivatives().fit([x, y], values)
        with pytest.raises(ValueError, match=r"shape \(n_query, 2\)"):
            est.derivatives(np.zeros((5, 3)), order=[(1, 0)])

    def test_unfitted_raises(self):
        """Calling derivatives(), predict(), or summary() before fit() raises."""
        est = SurfaceDerivatives()
        with pytest.raises(RuntimeError, match="must be fitted"):
            est.derivatives(np.zeros((2, 2)), order=[(1, 0)])
        with pytest.raises(RuntimeError, match="must be fitted"):
            est.predict(np.zeros((2, 2)))
        with pytest.raises(RuntimeError, match="must be fitted"):
            est.summary()

    def test_invalid_constructor_arguments(self):
        """Invalid constructor arguments raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            SurfaceDerivatives(method="bogus")
        with pytest.raises(ValueError, match="degree must be"):
            SurfaceDerivatives(degree=0)
        with pytest.raises(ValueError, match="smoothing must be"):
            SurfaceDerivatives(smoothing="cv")
        with pytest.raises(ValueError, match="smoothing_scale"):
            SurfaceDerivatives(smoothing_scale=0.0)
        with pytest.raises(ValueError, match="Numeric smoothing"):
            SurfaceDerivatives(smoothing=-1.0)

    def test_sigma_validation(self, bilinear_surface):
        """sigma must be positive and either scalar or per point."""
        x, y, values, coords = bilinear_surface
        with pytest.raises(ValueError, match="strictly positive"):
            SurfaceDerivatives().fit(coords, values.ravel(), sigma=0.0)
        with pytest.raises(ValueError, match="scalar or have"):
            SurfaceDerivatives().fit(coords, values.ravel(), sigma=np.ones(3))

    def test_smoothing_sigma_requires_sigma(self, bilinear_surface):
        """smoothing='sigma' without a supplied sigma raises ValueError."""
        x, y, values, _ = bilinear_surface
        with pytest.raises(ValueError, match="requires the sigma"):
            SurfaceDerivatives(smoothing="sigma").fit([x, y], values)


# ===========================================================================
# TestTensorSpline
# ===========================================================================


class TestTensorSpline:
    """Tests for the penalized tensor-product B-spline smoother."""

    def test_exact_on_bilinear(self, bilinear_surface):
        """A bilinear surface is in the spline space, so partials are essentially exact."""
        x, y, values, coords = bilinear_surface
        est = SurfaceDerivatives(smoothing=1e-8).fit([x, y], values)
        _, partials = est.derivatives(coords, order=[(1, 0), (0, 1), (1, 1)])
        np.testing.assert_allclose(partials[:, 0], 3 + coords[:, 1], atol=1e-6)
        np.testing.assert_allclose(partials[:, 1], 2 + coords[:, 0], atol=1e-6)
        np.testing.assert_allclose(partials[:, 2], 1.0, atol=1e-6)

    def test_first_partials_on_noisy_surface(self, smooth_surface):
        """Both first partials are recovered from one noisy surface."""
        x, y, values, coords, exact = smooth_surface
        rng = np.random.default_rng(0)
        noisy = values + rng.normal(0, 0.01, values.shape)
        est = SurfaceDerivatives().fit([x, y], noisy, sigma=0.01)
        _, partials = est.derivatives(coords, order=[(1, 0), (0, 1)])
        mask = interior_mask(coords)
        assert np.abs(partials[mask, 0] - exact[(1, 0)][mask]).max() < 0.15
        assert np.abs(partials[mask, 1] - exact[(0, 1)][mask]).max() < 0.15

    def test_second_partial(self, smooth_surface):
        """Second partials track the analytic result on clean data."""
        x, y, values, coords, exact = smooth_surface
        est = SurfaceDerivatives().fit([x, y], values)
        _, partials = est.derivatives(coords, order=[(2, 0)])
        mask = interior_mask(coords)
        rel = np.abs(partials[mask, 0] - exact[(2, 0)][mask]).max() / np.abs(exact[(2, 0)]).max()
        assert rel < 0.1

    def test_predict_matches_zero_order(self, smooth_surface):
        """predict() equals the order-zero output of derivatives()."""
        x, y, values, coords, _ = smooth_surface
        est = SurfaceDerivatives().fit([x, y], values)
        vals, _ = est.derivatives(coords, order=[(1, 0)])
        np.testing.assert_allclose(est.predict(coords), vals)

    def test_predict_return_std(self, smooth_surface):
        """predict(return_std=True) returns positive, correctly shaped standard errors."""
        x, y, values, coords, _ = smooth_surface
        rng = np.random.default_rng(1)
        est = SurfaceDerivatives().fit([x, y], values + rng.normal(0, 0.02, values.shape))
        mean, std = est.predict(coords, return_std=True)
        assert mean.shape == std.shape == (coords.shape[0],)
        assert np.all(std > 0)

    def test_std_grows_with_noise(self, smooth_surface):
        """Derivative standard errors increase with the measurement noise."""
        x, y, values, coords, _ = smooth_surface
        rng = np.random.default_rng(2)
        quiet = SurfaceDerivatives().fit([x, y], values + rng.normal(0, 0.005, values.shape))
        loud = SurfaceDerivatives().fit([x, y], values + rng.normal(0, 0.05, values.shape))
        _, _, std_quiet = quiet.derivatives(coords, order=[(1, 0)], return_std=True)
        _, _, std_loud = loud.derivatives(coords, order=[(1, 0)], return_std=True)
        assert std_loud.mean() > std_quiet.mean()

    def test_gcv_selects_smoothing(self, smooth_surface):
        """With smoothing='auto' the penalty is chosen by GCV and reported."""
        x, y, values, _, _ = smooth_surface
        rng = np.random.default_rng(3)
        est = SurfaceDerivatives().fit([x, y], values + rng.normal(0, 0.02, values.shape))
        assert est.smoothing_source_ == "gcv"
        assert est.smoothing_ > 0
        assert 0 < est.effective_dof_ < values.size

    def test_sigma_criterion_matches_residuals(self, smooth_surface):
        """smoothing='sigma' picks a penalty whose residual scatter matches sigma."""
        x, y, values, _, _ = smooth_surface
        rng = np.random.default_rng(4)
        sigma = 0.02
        est = SurfaceDerivatives(smoothing="sigma").fit(
            [x, y], values + rng.normal(0, sigma, values.shape), sigma=sigma
        )
        assert est.smoothing_source_ == "sigma"
        assert 0.5 * sigma < est.residual_std_ < 2.0 * sigma

    def test_smoothing_scale_increases_smoothing(self, smooth_surface):
        """smoothing_scale multiplies the selected penalty and lowers the effective dof."""
        x, y, values, _, _ = smooth_surface
        rng = np.random.default_rng(5)
        noisy = values + rng.normal(0, 0.02, values.shape)
        base = SurfaceDerivatives().fit([x, y], noisy)
        scaled = SurfaceDerivatives(smoothing_scale=5.0).fit([x, y], noisy)
        assert scaled.smoothing_ == pytest.approx(5.0 * base.smoothing_)
        assert scaled.effective_dof_ < base.effective_dof_

    def test_fixed_smoothing_is_reported(self, bilinear_surface):
        """A numeric smoothing value is used verbatim and reported as fixed."""
        x, y, values, _ = bilinear_surface
        est = SurfaceDerivatives(smoothing=0.5).fit([x, y], values)
        assert est.smoothing_ == pytest.approx(0.5)
        assert est.smoothing_source_ == "fixed"

    def test_scattered_data(self, smooth_surface):
        """Scattered (non-gridded) samples are supported."""
        _, _, _, _, _ = smooth_surface
        rng = np.random.default_rng(6)
        coords = rng.uniform([0, -1], [2, 1], size=(500, 2))
        values = np.sin(2 * coords[:, 0]) * np.exp(0.5 * coords[:, 1])
        est = SurfaceDerivatives().fit(coords, values)
        _, partials = est.derivatives(coords, order=[(1, 0)])
        exact = 2 * np.cos(2 * coords[:, 0]) * np.exp(0.5 * coords[:, 1])
        mask = interior_mask(coords, frac=0.15)
        assert np.abs(partials[mask, 0] - exact[mask]).max() < 0.3

    def test_three_dimensional(self):
        """Three-dimensional surfaces are supported."""
        axis = np.linspace(0.0, 1.0, 8)
        aa, bb, cc = np.meshgrid(axis, axis, axis, indexing="ij")
        values = aa**2 + 2 * bb * cc
        est = SurfaceDerivatives(smoothing=1e-8).fit([axis, axis, axis], values)
        _, partials = est.derivatives(est.coords_, order=[(1, 0, 0), (0, 1, 1)])
        np.testing.assert_allclose(partials[:, 0], 2 * est.coords_[:, 0], atol=1e-5)
        np.testing.assert_allclose(partials[:, 1], 2.0, atol=1e-5)

    def test_one_dimensional(self):
        """One-dimensional data reduces to a smoothing spline derivative."""
        t = np.linspace(0.0, 2 * np.pi, 200).reshape(-1, 1)
        values = np.sin(t.ravel())
        est = SurfaceDerivatives().fit(t, values)
        _, partials = est.derivatives(t, order=(1,))
        mask = interior_mask(t)
        assert np.abs(partials[mask, 0] - np.cos(t.ravel())[mask]).max() < 0.05

    def test_explicit_n_basis(self, smooth_surface):
        """n_basis controls the per-dimension basis size."""
        x, y, values, coords, _ = smooth_surface
        est = SurfaceDerivatives(n_basis=[8, 6]).fit([x, y], values)
        assert est._spline_sizes == [8, 6]
        assert "basis per dim     : [8, 6]" in est.summary()

    def test_n_basis_too_large_raises(self, bilinear_surface):
        """A basis larger than max_basis raises ValueError."""
        x, y, values, _ = bilinear_surface
        with pytest.raises(ValueError, match="above max_basis"):
            SurfaceDerivatives(n_basis=30, max_basis=100).fit([x, y], values)

    def test_n_basis_below_degree_raises(self, bilinear_surface):
        """Fewer basis functions than degree + 1 raises ValueError."""
        x, y, values, _ = bilinear_surface
        with pytest.raises(ValueError, match="at least degree"):
            SurfaceDerivatives(degree=3, n_basis=3).fit([x, y], values)

    def test_summary_reports_smoothing(self, smooth_surface):
        """summary() reports the method, smoothing level, and how it was chosen."""
        x, y, values, _, _ = smooth_surface
        est = SurfaceDerivatives().fit([x, y], values)
        text = est.summary()
        assert "tensor_spline" in text
        assert "penalty lambda" in text
        assert "effective dof" in text
        assert "chosen by" in text


# ===========================================================================
# TestLocalPoly
# ===========================================================================


class TestLocalPoly:
    """Tests for the local polynomial smoother."""

    def test_first_partials(self, smooth_surface):
        """Both first partials are recovered on the interior."""
        x, y, values, coords, exact = smooth_surface
        est = SurfaceDerivatives(method="local_poly").fit([x, y], values)
        _, partials = est.derivatives(coords, order=[(1, 0), (0, 1)])
        mask = interior_mask(coords)
        assert np.abs(partials[mask, 0] - exact[(1, 0)][mask]).max() < 0.15
        assert np.abs(partials[mask, 1] - exact[(0, 1)][mask]).max() < 0.15

    def test_exact_on_quadratic(self):
        """A quadratic is reproduced exactly by a degree-2 local fit."""
        rng = np.random.default_rng(7)
        coords = rng.uniform(-1, 1, size=(200, 2))
        values = 1.0 + 2 * coords[:, 0] - 3 * coords[:, 1] + 0.5 * coords[:, 0] * coords[:, 1]
        est = SurfaceDerivatives(method="local_poly", degree=2).fit(coords, values)
        _, partials = est.derivatives(coords, order=[(1, 0), (0, 1), (1, 1)])
        np.testing.assert_allclose(partials[:, 0], 2 + 0.5 * coords[:, 1], atol=1e-6)
        np.testing.assert_allclose(partials[:, 1], -3 + 0.5 * coords[:, 0], atol=1e-6)
        np.testing.assert_allclose(partials[:, 2], 0.5, atol=1e-6)

    def test_std_is_reported(self, smooth_surface):
        """Standard errors are positive and shaped like the partials."""
        x, y, values, coords, _ = smooth_surface
        rng = np.random.default_rng(8)
        est = SurfaceDerivatives(method="local_poly").fit(
            [x, y], values + rng.normal(0, 0.02, values.shape)
        )
        _, partials, std = est.derivatives(coords[:50], order=[(1, 0), (0, 1)], return_std=True)
        assert std.shape == partials.shape
        assert np.all(std > 0)

    def test_bandwidth_selected_by_gcv(self, smooth_surface):
        """The bandwidth is chosen by GCV and reported."""
        x, y, values, _, _ = smooth_surface
        rng = np.random.default_rng(9)
        est = SurfaceDerivatives(method="local_poly").fit(
            [x, y], values + rng.normal(0, 0.02, values.shape)
        )
        assert est.smoothing_source_ == "gcv"
        assert est.smoothing_ > 0
        assert "bandwidth" in est.summary()

    def test_fixed_bandwidth(self, smooth_surface):
        """A numeric smoothing value is used as the bandwidth verbatim."""
        x, y, values, _, _ = smooth_surface
        est = SurfaceDerivatives(method="local_poly", smoothing=0.7).fit([x, y], values)
        assert est.smoothing_ == pytest.approx(0.7)
        assert est.smoothing_source_ == "fixed"

    def test_too_few_points_for_degree(self):
        """A local degree needing more terms than there are points raises ValueError."""
        rng = np.random.default_rng(10)
        coords = rng.uniform(size=(8, 2))
        with pytest.raises(ValueError, match="needs more than"):
            SurfaceDerivatives(method="local_poly", degree=3).fit(coords, np.zeros(8))


# ===========================================================================
# TestGaussianProcess
# ===========================================================================


class TestGaussianProcess:
    """Tests for the Gaussian process smoother."""

    def test_first_partials(self):
        """Both first partials are recovered from a noisy GP fit."""
        rng = np.random.default_rng(11)
        x = np.linspace(0.0, 2.0, 16)
        y = np.linspace(-1.0, 1.0, 14)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        values = np.sin(2 * xx) * np.exp(0.5 * yy)
        est = SurfaceDerivatives(method="gp").fit(
            [x, y], values + rng.normal(0, 0.01, values.shape), sigma=0.01
        )
        _, partials = est.derivatives(est.coords_, order=[(1, 0), (0, 1)])
        exact_x = (2 * np.cos(2 * xx) * np.exp(0.5 * yy)).ravel()
        exact_y = (0.5 * np.sin(2 * xx) * np.exp(0.5 * yy)).ravel()
        mask = interior_mask(est.coords_)
        assert np.abs(partials[mask, 0] - exact_x[mask]).max() < 0.2
        assert np.abs(partials[mask, 1] - exact_y[mask]).max() < 0.2

    def test_derivative_uncertainty(self):
        """The GP reports positive derivative uncertainty that grows away from data."""
        rng = np.random.default_rng(12)
        coords = rng.uniform(-1, 1, size=(120, 2))
        values = np.sin(coords[:, 0]) + coords[:, 1] ** 2
        est = SurfaceDerivatives(method="gp").fit(coords, values, sigma=0.01)
        _, _, std_in = est.derivatives(coords, order=[(1, 0)], return_std=True)
        _, _, std_out = est.derivatives(np.array([[4.0, 4.0]]), order=[(1, 0)], return_std=True)
        assert np.all(std_in > 0)
        assert std_out[0, 0] > std_in.mean()

    def test_fixed_length_scale(self):
        """A supplied length_scale is used instead of being learned."""
        rng = np.random.default_rng(13)
        coords = rng.uniform(-1, 1, size=(80, 2))
        values = np.sin(coords[:, 0]) + coords[:, 1]
        est = SurfaceDerivatives(method="gp", length_scale=[1.0, 2.0]).fit(coords, values)
        np.testing.assert_allclose(est._gp_length_scale, [1.0, 2.0])
        assert "length scales" in est.summary()

    def test_bad_length_scale_raises(self):
        """Invalid length scales raise ValueError."""
        rng = np.random.default_rng(14)
        coords = rng.uniform(-1, 1, size=(40, 2))
        values = coords[:, 0]
        with pytest.raises(ValueError, match="length_scale must be"):
            SurfaceDerivatives(method="gp", length_scale=[1.0, 2.0, 3.0]).fit(coords, values)
        with pytest.raises(ValueError, match="strictly positive"):
            SurfaceDerivatives(method="gp", length_scale=-1.0).fit(coords, values)

    def test_max_points_guard(self):
        """Exceeding max_points raises a ValueError that names the cap."""
        rng = np.random.default_rng(15)
        coords = rng.uniform(size=(60, 2))
        with pytest.raises(ValueError, match="max_points"):
            SurfaceDerivatives(method="gp", max_points=50).fit(coords, np.zeros(60))

    def test_marginal_likelihood_without_sigma(self):
        """Without sigma the noise level is learned by marginal likelihood."""
        rng = np.random.default_rng(16)
        coords = rng.uniform(-1, 1, size=(100, 2))
        values = np.sin(coords[:, 0]) + rng.normal(0, 0.05, 100)
        est = SurfaceDerivatives(method="gp").fit(coords, values)
        assert est.smoothing_source_ == "marginal_likelihood"
        assert est.smoothing_ > 0


# ===========================================================================
# TestEstimatePartialDerivatives
# ===========================================================================


class TestEstimatePartialDerivatives:
    """Tests for the convenience wrapper."""

    def test_returns_values_and_partials(self, bilinear_surface):
        """The wrapper evaluates at the sample locations by default."""
        x, y, values, coords = bilinear_surface
        vals, partials = estimate_partial_derivatives(
            [x, y], values, order=[(1, 0), (0, 1)], smoothing=1e-8
        )
        assert vals.shape == (coords.shape[0],)
        assert partials.shape == (coords.shape[0], 2)
        np.testing.assert_allclose(partials[:, 0], 3 + coords[:, 1], atol=1e-6)

    def test_query_and_return_std(self, bilinear_surface):
        """Explicit query points and standard errors are supported."""
        x, y, values, _ = bilinear_surface
        query = np.array([[0.5, 1.0], [0.25, 0.5]])
        vals, partials, std = estimate_partial_derivatives(
            [x, y], values, order=[(0, 1)], query=query, return_std=True
        )
        assert vals.shape == (2,)
        assert partials.shape == std.shape == (2, 1)

    def test_method_passthrough(self, bilinear_surface):
        """The method and extra keyword arguments reach the estimator."""
        x, y, values, coords = bilinear_surface
        _, partials = estimate_partial_derivatives(
            coords, values.ravel(), order=[(1, 0)], method="local_poly", degree=2
        )
        np.testing.assert_allclose(partials[:, 0], 3 + coords[:, 1], atol=1e-6)


# ===========================================================================
# TestShiftLawRecovery
# ===========================================================================


class TestShiftLawRecovery:
    """The motivating application: a time-temperature superposition shift law.

    ``y(x, T) = f(x + s(T))`` implies ``y_T = s'(T) * y_x``, so both partials must
    come from one smoothed surface.
    """

    @staticmethod
    def _surface(noise, seed):
        gas_r = 8.314e-3  # kJ/mol/K
        energy = 55.85  # kJ/mol
        t_ref = 350.0
        x = np.linspace(-2.0, 4.0, 20)
        temps = np.linspace(320.0, 400.0, 12)
        xx, tt = np.meshgrid(x, temps, indexing="ij")
        shift = -(energy / (gas_r * np.log(10.0))) * (1.0 / tt - 1.0 / t_ref)
        values = 2.0 + 1.5 * np.tanh(0.8 * (xx + shift - 1.0))
        if noise:
            values = values + np.random.default_rng(seed).normal(0, noise, values.shape)
        return x, temps, values, energy, gas_r

    @staticmethod
    def _effective_energy(est, gas_r):
        _, partials = est.derivatives(est.coords_, order=[(1, 0), (0, 1)])
        y_x, y_t = partials[:, 0], partials[:, 1]
        keep = np.abs(y_x) > 0.15 * np.abs(y_x).max()
        temps = est.coords_[keep, 1]
        return float(np.median(y_t[keep] / y_x[keep] * gas_r * np.log(10.0) * temps**2))

    @pytest.mark.parametrize("method", ["tensor_spline", "local_poly", "gp"])
    def test_clean_surface(self, method):
        """All smoothers recover the activation energy from noise-free data."""
        x, temps, values, energy, gas_r = self._surface(0.0, 0)
        est = SurfaceDerivatives(method=method).fit([x, temps], values)
        assert self._effective_energy(est, gas_r) == pytest.approx(energy, rel=0.02)

    def test_noisy_surface(self):
        """The recovered activation energy stays close under 1% noise."""
        x, temps, values, energy, gas_r = self._surface(0.01, 17)
        est = SurfaceDerivatives(smoothing="sigma").fit([x, temps], values, sigma=0.01)
        assert self._effective_energy(est, gas_r) == pytest.approx(energy, rel=0.05)

    def test_oversmoothing_is_visible(self):
        """Deliberate over-smoothing lowers the effective dof and is reported."""
        x, temps, values, _, _ = self._surface(0.01, 18)
        base = SurfaceDerivatives().fit([x, temps], values, sigma=0.01)
        rough = SurfaceDerivatives(smoothing_scale=20.0).fit([x, temps], values, sigma=0.01)
        assert rough.effective_dof_ < base.effective_dof_
        assert rough.smoothing_ > base.smoothing_
        assert f"{rough.smoothing_:.6g}" in rough.summary()
