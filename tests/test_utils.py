"""Tests for utils module."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr.utils import feature_derivative


class TestFeatureDerivative:
    """Tests for the forward-mode feature derivative helper."""

    @pytest.fixture
    def X(self):
        rng = np.random.default_rng(0)
        return jnp.array(rng.uniform(-1.0, 1.0, size=(20, 2)))

    def test_first_derivative_is_exact(self, X):
        """d/dx0 of exp(3 x0) * x1 matches the analytic derivative."""

        def fn(X):
            return jnp.exp(3 * X[:, 0]) * X[:, 1]

        got = np.asarray(feature_derivative(fn, X, 0, order=1))
        want = 3 * np.exp(3 * np.asarray(X[:, 0])) * np.asarray(X[:, 1])
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_second_derivative_is_exact(self, X):
        """A central second difference with step 1e-2 is off by ~7.5e-5 here; AD is not."""

        def fn(X):
            return jnp.exp(3 * X[:, 0]) + X[:, 1] ** 3

        got = np.asarray(feature_derivative(fn, X, 0, order=2))
        want = 9 * np.exp(3 * np.asarray(X[:, 0]))
        np.testing.assert_allclose(got, want, rtol=1e-5)

        got_x1 = np.asarray(feature_derivative(fn, X, 1, order=2))
        np.testing.assert_allclose(got_x1, 6 * np.asarray(X[:, 1]), rtol=1e-5, atol=1e-5)

    def test_matrix_valued_function(self, X):
        """A design matrix gives one derivative column per basis function."""

        def design(X):
            return jnp.column_stack([jnp.ones(X.shape[0]), X[:, 0], X[:, 0] ** 2, X[:, 1]])

        got = np.asarray(feature_derivative(design, X, 0, order=1))
        x0 = np.asarray(X[:, 0])
        want = np.column_stack([np.zeros_like(x0), np.ones_like(x0), 2 * x0, np.zeros_like(x0)])
        assert got.shape == (20, 4)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_integer_input_is_promoted(self):
        """Integer X is promoted to float instead of failing with an integer tangent."""
        X = jnp.array([[1, 2], [3, 4]])
        got = np.asarray(feature_derivative(lambda X: X[:, 0] ** 2, X, 0))
        np.testing.assert_allclose(got, [2.0, 6.0], rtol=1e-5)

    @pytest.mark.parametrize(
        ("X", "feature_idx", "order", "match"),
        [
            (jnp.ones(3), 0, 1, "2-D"),
            (jnp.ones((3, 2)), 2, 1, "out of range"),
            (jnp.ones((3, 2)), -1, 1, "out of range"),
            (jnp.ones((3, 2)), 0, 3, "order"),
        ],
    )
    def test_invalid_input(self, X, feature_idx, order, match):
        with pytest.raises(ValueError, match=match):
            feature_derivative(lambda X: X[:, 0], X, feature_idx, order=order)
