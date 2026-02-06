"""Integration tests for JAXSR."""

import jax.numpy as jnp
import numpy as np

from jaxsr import (
    AdaptiveSampler,
    BasisLibrary,
    Constraints,
    SymbolicRegressor,
    fit_symbolic,
)


class TestPolynomialRecovery:
    """Test recovery of polynomial expressions."""

    def test_linear(self):
        """Recover linear expression: y = 2x + 3."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 2.0 * X[:, 0] + 3.0 + np.random.randn(100) * 0.01

        model = fit_symbolic(
            jnp.array(X),
            jnp.array(y),
            max_terms=5,
            max_poly_degree=3,
        )

        assert model.score(jnp.array(X), jnp.array(y)) > 0.99

    def test_quadratic(self):
        """Recover quadratic: y = x^2 - 2x + 1."""
        np.random.seed(42)
        X = np.random.randn(200, 1)
        y = X[:, 0] ** 2 - 2.0 * X[:, 0] + 1.0 + np.random.randn(200) * 0.01

        model = fit_symbolic(
            jnp.array(X),
            jnp.array(y),
            max_terms=5,
            max_poly_degree=3,
        )

        # Should find x^2, x, and constant
        assert model.score(jnp.array(X), jnp.array(y)) > 0.99

    def test_multivariate(self):
        """Recover multivariate: y = 2x + 3y - xy."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] - X[:, 0] * X[:, 1]
        y += np.random.randn(200) * 0.05

        library = (
            BasisLibrary(n_features=2, feature_names=["a", "b"])
            .add_constant()
            .add_linear()
            .add_interactions(max_order=2)
            .add_polynomials(max_degree=2)
        )

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=6,
            strategy="greedy_forward",
        )
        model.fit(jnp.array(X), jnp.array(y))

        assert model.score(jnp.array(X), jnp.array(y)) > 0.95


class TestTranscendentalRecovery:
    """Test recovery of transcendental expressions."""

    def test_exponential(self):
        """Recover exponential: y = exp(x)."""
        np.random.seed(42)
        X = np.random.uniform(-1, 1, (100, 1))
        y = np.exp(X[:, 0]) + np.random.randn(100) * 0.01

        model = fit_symbolic(
            jnp.array(X),
            jnp.array(y),
            max_terms=5,
            include_transcendental=True,
        )

        r2 = model.score(jnp.array(X), jnp.array(y))
        assert r2 > 0.95

    def test_logarithmic(self):
        """Recover logarithmic: y = log(x)."""
        np.random.seed(42)
        X = np.random.uniform(0.1, 3.0, (100, 1))
        y = np.log(X[:, 0]) + np.random.randn(100) * 0.01

        model = fit_symbolic(
            jnp.array(X),
            jnp.array(y),
            max_terms=5,
            include_transcendental=True,
        )

        r2 = model.score(jnp.array(X), jnp.array(y))
        assert r2 > 0.95


class TestConstrainedRegression:
    """Test regression with physical constraints."""

    def test_positive_output(self):
        """Test non-negative output constraint."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        # True model has positive outputs
        y = np.abs(2.0 * X[:, 0] + 1.0) + np.random.randn(100) * 0.1

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
        )

        constraints = Constraints().add_bounds("y", lower=0.0)

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=5,
            constraints=constraints,
        )
        model.fit(jnp.array(X), jnp.array(y))

        # Model should still fit reasonably
        assert model.score(jnp.array(X), jnp.array(y)) > 0.5

    def test_coefficient_sign(self):
        """Test coefficient sign constraint."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 2.0 * X[:, 0] + np.random.randn(100) * 0.1

        library = BasisLibrary(n_features=1, feature_names=["x"]).add_linear()

        constraints = Constraints().add_sign_constraint("x", sign="positive", hard=True)

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=2,
            constraints=constraints,
        )
        model.fit(jnp.array(X), jnp.array(y))

        # Coefficient should be positive
        if "x" in model.selected_features_:
            idx = model.selected_features_.index("x")
            assert model.coefficients_[idx] >= 0


class TestAdaptiveSampling:
    """Test adaptive sampling functionality."""

    def test_suggest_points(self):
        """Test that sampler suggests valid points."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = 2.0 * X[:, 0] + X[:, 1] ** 2

        library = (
            BasisLibrary(n_features=2, feature_names=["x", "y"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
        )

        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(jnp.array(X), jnp.array(y))

        sampler = AdaptiveSampler(
            model=model,
            bounds=[(-3, 3), (-3, 3)],
            strategy="space_filling",
        )

        result = sampler.suggest(n_points=5)

        assert result.points.shape == (5, 2)
        # Points should be within bounds
        assert jnp.all(result.points[:, 0] >= -3)
        assert jnp.all(result.points[:, 0] <= 3)
        assert jnp.all(result.points[:, 1] >= -3)
        assert jnp.all(result.points[:, 1] <= 3)

    def test_different_strategies(self):
        """Test different sampling strategies."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]

        library = BasisLibrary(n_features=2).add_constant().add_linear()
        model = SymbolicRegressor(basis_library=library, max_terms=3)
        model.fit(jnp.array(X), jnp.array(y))

        for strategy in ["space_filling", "random", "leverage"]:
            sampler = AdaptiveSampler(
                model=model,
                bounds=[(-2, 2), (-2, 2)],
                strategy=strategy,
            )
            result = sampler.suggest(n_points=3)
            assert result.points.shape[0] == 3
            assert result.strategy == strategy


class TestModelComparison:
    """Test model comparison utilities."""

    def test_pareto_front(self):
        """Test Pareto front computation."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2.0 * X[:, 0] + X[:, 1] ** 2 + np.random.randn(100) * 0.1

        library = (
            BasisLibrary(n_features=2)
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=3)
            .add_interactions(max_order=2)
        )

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=6,
            strategy="greedy_forward",
        )
        model.fit(jnp.array(X), jnp.array(y))

        pareto = model.pareto_front_

        # Pareto front should have multiple models
        assert len(pareto) >= 1

        # Should be sorted by complexity
        complexities = [r.complexity for r in pareto]
        assert complexities == sorted(complexities)

        # MSE should decrease as complexity increases
        for i in range(1, len(pareto)):
            assert pareto[i].mse <= pareto[i - 1].mse + 1e-6


class TestEndToEnd:
    """End-to-end workflow tests."""

    def test_chemical_kinetics(self):
        """Test discovering rate law from kinetic data."""
        np.random.seed(42)

        # Generate data from Langmuir-Hinshelwood kinetics
        # r = k * C_A * C_B / (1 + K * C_A)
        n_samples = 100
        C_A = np.random.uniform(0.1, 2.0, n_samples)
        C_B = np.random.uniform(0.1, 2.0, n_samples)
        k, K = 2.5, 1.2
        r_true = k * C_A * C_B / (1 + K * C_A)
        r = r_true + np.random.randn(n_samples) * 0.05

        X = jnp.column_stack([C_A, C_B])
        y = jnp.array(r)

        # Build library with rational terms
        library = (
            BasisLibrary(n_features=2, feature_names=["C_A", "C_B"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
            .add_interactions(max_order=2)
            .add_ratios()
        )

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=6,
            strategy="greedy_forward",
        )
        model.fit(X, y)

        # Should achieve good fit
        r2 = model.score(X, y)
        assert r2 > 0.9

    def test_full_workflow(self):
        """Test complete workflow: fit, evaluate, export, reload."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2.0 * X[:, 0] - 1.5 * X[:, 1] ** 2 + 0.5

        # Fit model
        model = fit_symbolic(
            jnp.array(X),
            jnp.array(y),
            feature_names=["a", "b"],
            max_terms=5,
            include_transcendental=False,
        )

        # Evaluate
        r2 = model.score(jnp.array(X), jnp.array(y))
        assert r2 > 0.95

        # Get expression
        expr = model.expression_
        assert "a" in expr or "b" in expr

        # Get callable
        predict_fn = model.to_callable()
        y_pred = predict_fn(X)
        assert y_pred.shape == y.shape

        # Summary
        summary = model.summary()
        assert "R²" in summary
