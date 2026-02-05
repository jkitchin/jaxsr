"""Tests for SymbolicRegressor."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr import BasisLibrary, SymbolicRegressor, fit_symbolic, Constraints


class TestSymbolicRegressor:
    """Tests for SymbolicRegressor class."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2.5 * X[:, 0] + 1.2 * X[:, 0] * X[:, 1] - 0.8 * X[:, 1] ** 2
        y += np.random.randn(100) * 0.1
        return jnp.array(X), jnp.array(y)

    @pytest.fixture
    def library(self):
        """Create test basis library."""
        return (
            BasisLibrary(n_features=2, feature_names=["x", "y"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=3)
            .add_interactions(max_order=2)
        )

    def test_basic_fit(self, simple_data, library):
        """Test basic model fitting."""
        X, y = simple_data

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=5,
            strategy="greedy_forward",
        )
        model.fit(X, y)

        assert model._is_fitted
        assert model.expression_ is not None
        assert len(model.coefficients_) > 0
        assert model.complexity_ > 0

    def test_predict(self, simple_data, library):
        """Test prediction."""
        X, y = simple_data

        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == y.shape

        # Should have reasonable R²
        r2 = model.score(X, y)
        assert r2 > 0.9

    def test_metrics(self, simple_data, library):
        """Test metrics computation."""
        X, y = simple_data

        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(X, y)

        metrics = model.metrics_
        assert "mse" in metrics
        assert "r2" in metrics
        assert "aic" in metrics
        assert "bic" in metrics

        assert metrics["mse"] >= 0
        assert metrics["r2"] <= 1.0

    def test_pareto_front(self, simple_data, library):
        """Test Pareto front computation."""
        X, y = simple_data

        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(X, y)

        pareto = model.pareto_front_
        assert len(pareto) > 0
        # Pareto front should be sorted by complexity
        complexities = [r.complexity for r in pareto]
        assert complexities == sorted(complexities)

    def test_strategies(self, simple_data, library):
        """Test different selection strategies."""
        X, y = simple_data

        for strategy in ["greedy_forward", "exhaustive"]:
            model = SymbolicRegressor(
                basis_library=library,
                max_terms=3,
                strategy=strategy,
            )
            model.fit(X, y)
            assert model._is_fitted
            assert model.score(X, y) > 0.5

    def test_information_criteria(self, simple_data, library):
        """Test different information criteria."""
        X, y = simple_data

        for ic in ["aic", "bic", "aicc"]:
            model = SymbolicRegressor(
                basis_library=library,
                max_terms=5,
                information_criterion=ic,
            )
            model.fit(X, y)
            assert model._is_fitted

    def test_unfitted_error(self, library):
        """Test error when accessing attributes on unfitted model."""
        model = SymbolicRegressor(basis_library=library)

        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.expression_

        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(jnp.ones((10, 2)))

    def test_feature_mismatch_error(self, library):
        """Test error when data has wrong number of features."""
        model = SymbolicRegressor(basis_library=library)
        X = jnp.ones((10, 3))  # 3 features, library expects 2
        y = jnp.ones(10)

        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_sample_count_mismatch_error(self, library):
        """Test error when X and y have different sample counts."""
        model = SymbolicRegressor(basis_library=library)
        X = jnp.ones((10, 2))
        y = jnp.ones(15)  # Different number of samples

        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_no_library_error(self):
        """Test error when no basis library specified."""
        model = SymbolicRegressor()

        with pytest.raises(ValueError, match="basis_library"):
            model.fit(jnp.ones((10, 2)), jnp.ones(10))

    def test_update(self, simple_data, library):
        """Test incremental update."""
        X, y = simple_data

        # Fit on first half
        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(X[:50], y[:50])
        r2_initial = model.score(X[:50], y[:50])

        # Update with second half
        model.update(X[50:], y[50:], refit=False)

        # Should still work
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape

    def test_to_callable(self, simple_data, library):
        """Test conversion to NumPy callable."""
        X, y = simple_data

        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(X, y)

        # Get callable
        predict_fn = model.to_callable()

        # Should work with NumPy arrays
        X_numpy = np.array(X)
        y_pred = predict_fn(X_numpy)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y.shape

    def test_save_load(self, simple_data, library, tmp_path):
        """Test model serialization."""
        X, y = simple_data

        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(X, y)

        # Save
        filepath = tmp_path / "model.json"
        model.save(str(filepath))

        # Load
        loaded = SymbolicRegressor.load(str(filepath))

        # Should produce same predictions
        y_pred_original = model.predict(X)
        y_pred_loaded = loaded.predict(X)

        np.testing.assert_array_almost_equal(y_pred_original, y_pred_loaded)

    def test_summary(self, simple_data, library):
        """Test model summary."""
        X, y = simple_data

        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(X, y)

        summary = model.summary()
        assert "Expression:" in summary
        assert "R²:" in summary
        assert "MSE:" in summary

    def test_repr(self, library):
        """Test string representation."""
        model = SymbolicRegressor(basis_library=library, max_terms=5)
        repr_str = repr(model)
        assert "SymbolicRegressor" in repr_str
        assert "not fitted" in repr_str


class TestFitSymbolic:
    """Tests for fit_symbolic convenience function."""

    def test_basic_usage(self):
        """Test basic usage of fit_symbolic."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2.0 * X[:, 0] + X[:, 1] ** 2

        model = fit_symbolic(
            jnp.array(X),
            jnp.array(y),
            feature_names=["a", "b"],
            max_terms=5,
        )

        assert model._is_fitted
        assert model.score(jnp.array(X), jnp.array(y)) > 0.9

    def test_with_transcendental(self):
        """Test with transcendental functions."""
        np.random.seed(42)
        X = np.random.uniform(0.1, 2.0, (100, 1))
        y = np.log(X[:, 0])

        model = fit_symbolic(
            jnp.array(X),
            jnp.array(y),
            max_terms=5,
            include_transcendental=True,
        )

        assert model._is_fitted
        assert "log" in model.expression_.lower() or model.score(jnp.array(X), jnp.array(y)) > 0.9


class TestConstrainedFitting:
    """Tests for constrained fitting."""

    def test_sign_constraint(self):
        """Test sign constraint on coefficient."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2.0 * X[:, 0] + 1.0 * X[:, 1]

        library = (
            BasisLibrary(n_features=2, feature_names=["x", "y"])
            .add_linear()
        )

        # Constraint: x coefficient must be positive
        constraints = Constraints().add_sign_constraint("x", sign="positive")

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=3,
            constraints=constraints,
        )
        model.fit(jnp.array(X), jnp.array(y))

        # Find coefficient for x
        if "x" in model.selected_features_:
            idx = model.selected_features_.index("x")
            assert model.coefficients_[idx] >= 0

    def test_fixed_coefficient(self):
        """Test fixed coefficient constraint."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 3.0 * X[:, 0]  # No intercept

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
        )

        # Fix intercept to 0
        constraints = Constraints().add_known_coefficient("1", value=0.0)

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=2,
            constraints=constraints,
        )
        model.fit(jnp.array(X), jnp.array(y))

        # Check that model doesn't have a large intercept
        if "1" in model.selected_features_:
            idx = model.selected_features_.index("1")
            assert abs(model.coefficients_[idx]) < 0.1


class TestSymPyExport:
    """Tests for SymPy export functionality."""

    def test_to_sympy(self):
        """Test conversion to SymPy expression."""
        pytest.importorskip("sympy")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2.0 * X[:, 0] + 1.0

        library = (
            BasisLibrary(n_features=2, feature_names=["x", "y"])
            .add_constant()
            .add_linear()
        )

        model = SymbolicRegressor(basis_library=library, max_terms=3)
        model.fit(jnp.array(X), jnp.array(y))

        sympy_expr = model.to_sympy()
        assert sympy_expr is not None

    def test_to_latex(self):
        """Test conversion to LaTeX."""
        pytest.importorskip("sympy")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2.0 * X[:, 0] + X[:, 1] ** 2

        library = (
            BasisLibrary(n_features=2, feature_names=["x", "y"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
        )

        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(jnp.array(X), jnp.array(y))

        latex = model.to_latex()
        assert isinstance(latex, str)
        assert len(latex) > 0
