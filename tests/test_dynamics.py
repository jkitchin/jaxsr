"""Tests for dynamics (SINDy-style ODE discovery) module."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from jaxsr.basis import BasisLibrary
from jaxsr.dynamics import DynamicsResult, discover_dynamics, estimate_derivatives
from jaxsr.regressor import SymbolicRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sinusoidal_data():
    """Sinusoidal test data with known derivatives."""
    t = np.linspace(0, 2 * np.pi, 200)
    X = np.column_stack([np.sin(t), np.cos(t)])
    # Exact derivatives: [cos(t), -sin(t)]
    dXdt_exact = np.column_stack([np.cos(t), -np.sin(t)])
    return t, X, dXdt_exact


@pytest.fixture()
def decay_data():
    """Exponential decay: dx/dt = -0.5*x."""
    t = np.linspace(0, 10, 300)
    X = np.exp(-0.5 * t).reshape(-1, 1)
    return t, X


@pytest.fixture()
def lotka_volterra_data():
    """Lotka-Volterra system: dx/dt = x - 0.5*x*y, dy/dt = -0.75*y + 0.25*x*y."""

    def rhs(t, z):
        x, y = z
        return [x - 0.5 * x * y, -0.75 * y + 0.25 * x * y]

    sol = solve_ivp(rhs, [0, 10], [2.0, 1.0], t_eval=np.linspace(0, 10, 500), rtol=1e-10)
    return sol.t, sol.y.T


# ===========================================================================
# TestEstimateDerivatives
# ===========================================================================


class TestEstimateDerivatives:
    """Tests for estimate_derivatives."""

    def test_finite_difference_accuracy(self, sinusoidal_data):
        """Finite difference gives reasonable accuracy on smooth data."""
        t, X, dXdt_exact = sinusoidal_data
        dXdt = estimate_derivatives(X, t, method="finite_difference")
        # Exclude boundary points where gradient is less accurate
        np.testing.assert_allclose(dXdt[5:-5], dXdt_exact[5:-5], atol=0.05)

    def test_savgol_accuracy(self, sinusoidal_data):
        """Savitzky-Golay gives better accuracy than finite difference."""
        t, X, dXdt_exact = sinusoidal_data
        dXdt = estimate_derivatives(X, t, method="savgol", window_length=11, polyorder=3)
        np.testing.assert_allclose(dXdt[5:-5], dXdt_exact[5:-5], atol=0.02)

    def test_spline_accuracy(self, sinusoidal_data):
        """Spline gives high accuracy on smooth data."""
        t, X, dXdt_exact = sinusoidal_data
        dXdt = estimate_derivatives(X, t, method="spline", smooth=0.0)
        np.testing.assert_allclose(dXdt, dXdt_exact, atol=0.01)

    def test_finite_difference_nonuniform(self):
        """Finite difference handles non-uniform time spacing."""
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0, 2 * np.pi, 200))
        X = np.sin(t).reshape(-1, 1)
        dXdt = estimate_derivatives(X, t, method="finite_difference")
        # Just check it runs and has the right shape
        assert dXdt.shape == X.shape

    def test_savgol_rejects_nonuniform(self):
        """Savgol raises ValueError on non-uniform spacing."""
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0, 2 * np.pi, 200))
        X = np.sin(t).reshape(-1, 1)
        with pytest.raises(ValueError, match="uniformly spaced"):
            estimate_derivatives(X, t, method="savgol")

    def test_invalid_method(self):
        """Unknown method raises ValueError."""
        t = np.linspace(0, 1, 10)
        X = np.ones((10, 1))
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_derivatives(X, t, method="unknown")

    def test_shape_mismatch(self):
        """Mismatched t and X shapes raise ValueError."""
        t = np.linspace(0, 1, 10)
        X = np.ones((15, 2))
        with pytest.raises(ValueError, match="must match"):
            estimate_derivatives(X, t)

    def test_output_shape(self, sinusoidal_data):
        """Output shape matches input shape."""
        t, X, _ = sinusoidal_data
        dXdt = estimate_derivatives(X, t)
        assert dXdt.shape == X.shape

    def test_1d_input(self):
        """Single state variable (1-D X) is handled correctly."""
        t = np.linspace(0, 2 * np.pi, 100)
        X = np.sin(t)  # 1-D
        dXdt = estimate_derivatives(X, t)
        assert dXdt.shape == (100, 1)
        # Should approximate cos(t)
        np.testing.assert_allclose(dXdt[5:-5, 0], np.cos(t[5:-5]), atol=0.05)

    def test_monotonic_time_required(self):
        """Non-monotonic time raises ValueError."""
        t = np.array([0.0, 1.0, 0.5, 2.0])
        X = np.ones((4, 1))
        with pytest.raises(ValueError, match="monotonically increasing"):
            estimate_derivatives(X, t)


# ===========================================================================
# TestDynamicsResult
# ===========================================================================


class TestDynamicsResult:
    """Tests for DynamicsResult dataclass."""

    def test_summary_format(self):
        """Summary contains all state names and equations."""
        result = DynamicsResult(
            models={},
            equations={
                "x": "d(x)/dt = 1.0*x",
                "y": "d(y)/dt = -0.5*y",
            },
            derivatives=np.zeros((10, 2)),
            state_names=["x", "y"],
            metrics={
                "x": {"r2": 0.99, "mse": 1e-3, "aic": 10.0, "bic": 12.0, "aicc": 11.0},
                "y": {"r2": 0.95, "mse": 2e-3, "aic": 15.0, "bic": 17.0, "aicc": 16.0},
            },
        )
        s = result.summary()
        assert "Discovered ODEs" in s
        assert "d(x)/dt = 1.0*x" in s
        assert "d(y)/dt = -0.5*y" in s
        assert "R²" in s


# ===========================================================================
# TestDiscoverDynamics
# ===========================================================================


class TestDiscoverDynamics:
    """Tests for discover_dynamics end-to-end."""

    def test_linear_decay_accuracy(self, decay_data):
        """Discovers dx/dt = -0.5*x with good R²."""
        t, X = decay_data
        result = discover_dynamics(
            X,
            t,
            state_names=["x"],
            max_terms=3,
            strategy="greedy_forward",
            information_criterion="bic",
        )
        assert result.metrics["x"]["r2"] > 0.9

    def test_lotka_volterra_accuracy(self, lotka_volterra_data):
        """Discovers Lotka-Volterra equations with reasonable accuracy."""
        t, X = lotka_volterra_data
        result = discover_dynamics(
            X,
            t,
            state_names=["x", "y"],
            max_terms=5,
            strategy="greedy_forward",
            information_criterion="bic",
        )
        # Both equations should fit well
        assert result.metrics["x"]["r2"] > 0.9
        assert result.metrics["y"]["r2"] > 0.9

    def test_custom_library_passthrough(self, decay_data):
        """Custom basis library is used instead of the default."""
        t, X = decay_data
        lib = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_polynomials(2)
        result = discover_dynamics(X, t, state_names=["x"], basis_library=lib, max_terms=3)
        assert "x" in result.models

    def test_feature_mismatch_error(self, decay_data):
        """Wrong n_features in basis library raises ValueError."""
        t, X = decay_data
        lib = BasisLibrary(n_features=3, feature_names=["a", "b", "c"]).add_constant()
        with pytest.raises(ValueError, match="must match"):
            discover_dynamics(X, t, basis_library=lib)

    def test_default_state_names(self, decay_data):
        """Default state names are x0, x1, ..."""
        t, X = decay_data
        result = discover_dynamics(X, t, max_terms=3)
        assert result.state_names == ["x0"]

    def test_result_structure(self, decay_data):
        """Result has correct types and keys."""
        t, X = decay_data
        result = discover_dynamics(X, t, state_names=["x"], max_terms=3)
        assert isinstance(result, DynamicsResult)
        assert isinstance(result.models, dict)
        assert isinstance(result.equations, dict)
        assert isinstance(result.metrics, dict)
        assert result.derivatives.shape == X.shape
        assert "x" in result.models
        assert "x" in result.equations
        assert "x" in result.metrics

    def test_models_are_fitted(self, decay_data):
        """Each returned model is fitted and callable."""
        t, X = decay_data
        result = discover_dynamics(X, t, state_names=["x"], max_terms=3)
        model = result.models["x"]
        assert isinstance(model, SymbolicRegressor)
        # Should be able to predict without error
        y_pred = model.predict(X)
        assert y_pred.shape == (X.shape[0],)

    def test_equations_format(self, decay_data):
        """Equations have the expected d(...)/dt = ... format."""
        t, X = decay_data
        result = discover_dynamics(X, t, state_names=["x"], max_terms=3)
        assert result.equations["x"].startswith("d(x)/dt = ")
