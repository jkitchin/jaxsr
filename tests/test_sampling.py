"""Tests for sampling module."""

import jax.numpy as jnp
import numpy as np

from jaxsr import AdaptiveSampler, BasisLibrary, SymbolicRegressor


class _AnalyticModel:
    """Stand-in model with a known gradient."""

    basis_library = BasisLibrary(n_features=2)
    _X_train = None

    def predict(self, X):
        X = jnp.atleast_2d(jnp.asarray(X))
        return X[:, 0] ** 2 + 3 * jnp.sin(X[:, 1])


class TestGradientScore:
    """Tests for the gradient-magnitude sampling strategy."""

    def test_score_matches_analytic_gradient_norm(self):
        sampler = AdaptiveSampler(
            _AnalyticModel(), bounds=[(-2.0, 2.0), (-2.0, 2.0)], strategy="gradient"
        )
        rng = np.random.default_rng(1)
        candidates = jnp.array(rng.uniform(-2, 2, size=(50, 2)))

        scores = np.asarray(sampler._score_gradient(candidates))

        c = np.asarray(candidates)
        want = np.sqrt((2 * c[:, 0]) ** 2 + (3 * np.cos(c[:, 1])) ** 2)
        np.testing.assert_allclose(scores, want, rtol=1e-5, atol=1e-6)

    def test_suggest_with_fitted_model(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 2, size=(40, 2))
        y = X[:, 0] ** 2 + 0.5 * X[:, 1]
        library = (
            BasisLibrary(n_features=2, feature_names=["a", "b"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
        )
        model = SymbolicRegressor(basis_library=library, max_terms=3).fit(X, y)

        sampler = AdaptiveSampler(
            model, bounds=[(0.0, 2.0), (0.0, 2.0)], strategy="gradient", random_state=0
        )
        result = sampler.suggest(n_points=3)

        assert result.points.shape == (3, 2)
        assert np.all(np.isfinite(np.asarray(result.scores)))
        # |grad| = sqrt(4 a^2 + 0.25) grows with a, so suggestions lean toward large a.
        assert float(np.mean(np.asarray(result.points)[:, 0])) > 1.0
