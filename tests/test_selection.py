"""Tests for model selection algorithms."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr.basis import BasisLibrary
from jaxsr.selection import (
    SelectionResult,
    _fit_subset_gram,
    _precompute_gram,
    _solve_subset_gram,
    compute_pareto_front,
    exhaustive_search,
    fit_ols,
    fit_ridge,
    fit_subset,
    greedy_backward_elimination,
    greedy_forward_selection,
    lasso_path_selection,
    select_features,
)


class TestFitOLS:
    """Tests for OLS fitting."""

    def test_basic_fit(self):
        """Test basic least squares fit."""
        # y = 2*x + 1
        X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        y = jnp.array([3.0, 5.0, 7.0, 9.0])

        Phi = jnp.column_stack([jnp.ones(4), X])
        coeffs, mse = fit_ols(Phi, y)

        np.testing.assert_array_almost_equal(coeffs, jnp.array([1.0, 2.0]), decimal=5)
        assert mse < 1e-10

    def test_overdetermined(self):
        """Test overdetermined system."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_coeffs = np.array([1.5, -2.0, 0.5])
        y = X @ true_coeffs + np.random.randn(100) * 0.1

        coeffs, mse = fit_ols(jnp.array(X), jnp.array(y))

        np.testing.assert_array_almost_equal(coeffs, true_coeffs, decimal=1)
        assert mse < 0.1


class TestFitSubset:
    """Tests for subset fitting."""

    def test_fit_subset(self):
        """Test fitting with subset of basis functions."""
        library = (
            BasisLibrary(n_features=2, feature_names=["x", "y"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
        )

        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = 1.0 + 2.0 * X[:, 0] + 3.0 * X[:, 1]  # True model uses indices 0, 1, 2

        Phi = library.evaluate(jnp.array(X))

        result = fit_subset(
            Phi,
            jnp.array(y),
            [0, 1, 2],  # constant, x, y
            library.names,
            library.complexities,
        )

        assert isinstance(result, SelectionResult)
        assert len(result.coefficients) == 3
        assert result.mse < 1e-10
        assert result.n_terms == 3


class TestGreedyForwardSelection:
    """Tests for greedy forward selection."""

    def test_finds_true_model(self):
        """Test that forward selection finds the correct terms."""
        library = (
            BasisLibrary(n_features=2, feature_names=["x", "y"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=3)
        )

        np.random.seed(42)
        X = np.random.randn(100, 2)
        # True model: y = 2.5*x + 1.2*x*y - 0.8*y^2
        # But since we don't have interactions, use: y = 2.5*x - 0.8*y^2
        y = 2.5 * X[:, 0] - 0.8 * X[:, 1] ** 2 + np.random.randn(100) * 0.01

        Phi = library.evaluate(jnp.array(X))

        path = greedy_forward_selection(
            Phi=Phi,
            y=jnp.array(y),
            basis_names=library.names,
            complexities=library.complexities,
            max_terms=5,
            information_criterion="bic",
        )

        # Should select x and y^2
        best = path.best
        assert "x" in best.selected_names
        assert "y^2" in best.selected_names
        assert best.mse < 0.1

    def test_early_stopping(self):
        """Test that selection stops when IC stops improving."""
        library = (
            BasisLibrary(n_features=2).add_constant().add_linear().add_polynomials(max_degree=4)
        )

        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = 1.0 + 2.0 * X[:, 0]  # Simple linear model

        Phi = library.evaluate(jnp.array(X))

        path = greedy_forward_selection(
            Phi=Phi,
            y=jnp.array(y),
            basis_names=library.names,
            complexities=library.complexities,
            max_terms=10,
            early_stop=True,
        )

        # Should stop early, not use all 10 terms
        assert path.best.n_terms < 10


class TestGreedyBackwardElimination:
    """Tests for greedy backward elimination."""

    def test_eliminates_irrelevant(self):
        """Test that backward elimination removes irrelevant terms."""
        library = (
            BasisLibrary(n_features=2).add_constant().add_linear().add_polynomials(max_degree=2)
        )

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 1.0 + 2.0 * X[:, 0]  # Only uses constant and x0

        Phi = library.evaluate(jnp.array(X))

        path = greedy_backward_elimination(
            Phi=Phi,
            y=jnp.array(y),
            basis_names=library.names,
            complexities=library.complexities,
            min_terms=1,
        )

        best = path.best
        # Should keep constant and x0, possibly a few weakly-correlated terms
        assert "1" in best.selected_names or "x0" in best.selected_names
        assert best.n_terms <= 4


class TestExhaustiveSearch:
    """Tests for exhaustive search."""

    def test_finds_optimal(self):
        """Test that exhaustive search finds optimal model."""
        library = (
            BasisLibrary(n_features=2).add_constant().add_linear().add_polynomials(max_degree=2)
        )

        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1] ** 2

        Phi = library.evaluate(jnp.array(X))

        path = exhaustive_search(
            Phi=Phi,
            y=jnp.array(y),
            basis_names=library.names,
            complexities=library.complexities,
            max_terms=4,
        )

        # Should find model with constant, x0, and x1^2
        best = path.best
        assert best.mse < 0.1

    def test_max_combinations_limit(self):
        """Test that exhaustive search respects combination limit."""
        library = BasisLibrary(n_features=5).build_default(max_poly_degree=3)

        X = jnp.ones((10, 5))
        y = jnp.ones(10)
        Phi = library.evaluate(X)

        with pytest.raises(ValueError, match="exceeding limit"):
            exhaustive_search(
                Phi=Phi,
                y=y,
                basis_names=library.names,
                complexities=library.complexities,
                max_terms=10,
                max_combinations=100,
            )


class TestLassoPathSelection:
    """Tests for LASSO path selection."""

    def test_identifies_sparse_model(self):
        """Test that LASSO path identifies sparse solutions."""
        library = (
            BasisLibrary(n_features=2).add_constant().add_linear().add_polynomials(max_degree=3)
        )

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2.0 * X[:, 0] + np.random.randn(100) * 0.1

        Phi = library.evaluate(jnp.array(X))

        path = lasso_path_selection(
            Phi=Phi,
            y=jnp.array(y),
            basis_names=library.names,
            complexities=library.complexities,
            max_terms=5,
        )

        # Should find sparse model containing x0
        best = path.best
        assert "x0" in best.selected_names


class TestParetoFront:
    """Tests for Pareto front computation."""

    def test_pareto_front(self):
        """Test Pareto front extraction."""
        results = [
            SelectionResult(
                coefficients=jnp.array([1.0]),
                selected_indices=jnp.array([0]),
                selected_names=["1"],
                mse=1.0,
                complexity=1,
                aic=10.0,
                bic=10.0,
                aicc=10.0,
                n_samples=100,
            ),
            SelectionResult(
                coefficients=jnp.array([1.0, 2.0]),
                selected_indices=jnp.array([0, 1]),
                selected_names=["1", "x"],
                mse=0.5,
                complexity=2,
                aic=8.0,
                bic=8.0,
                aicc=8.0,
                n_samples=100,
            ),
            SelectionResult(
                coefficients=jnp.array([1.0, 2.0, 3.0]),
                selected_indices=jnp.array([0, 1, 2]),
                selected_names=["1", "x", "x^2"],
                mse=0.1,
                complexity=4,
                aic=6.0,
                bic=6.0,
                aicc=6.0,
                n_samples=100,
            ),
            # This one is dominated (worse MSE at same complexity)
            SelectionResult(
                coefficients=jnp.array([1.0, 2.0]),
                selected_indices=jnp.array([0, 1]),
                selected_names=["1", "x"],
                mse=0.8,
                complexity=2,
                aic=9.0,
                bic=9.0,
                aicc=9.0,
                n_samples=100,
            ),
        ]

        pareto = compute_pareto_front(results)

        # Should have 3 points on Pareto front
        assert len(pareto) == 3
        # Should be sorted by complexity
        assert pareto[0].complexity <= pareto[1].complexity <= pareto[2].complexity


class TestSelectFeatures:
    """Tests for the unified select_features function."""

    def test_strategy_dispatch(self):
        """Test that select_features dispatches to correct strategy."""
        library = BasisLibrary(n_features=2).add_constant().add_linear()

        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = jnp.array([1.0, 2.0, 3.0])
        Phi = library.evaluate(X)

        for strategy in ["greedy_forward", "exhaustive"]:
            path = select_features(
                Phi=Phi,
                y=y,
                basis_names=library.names,
                complexities=library.complexities,
                strategy=strategy,
                max_terms=3,
            )
            assert path.strategy == strategy

    def test_unknown_strategy_error(self):
        """Test error for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            select_features(
                Phi=jnp.ones((10, 3)),
                y=jnp.ones(10),
                basis_names=["a", "b", "c"],
                complexities=jnp.array([1, 1, 1]),
                strategy="unknown_strategy",
            )


class TestGramPrecompute:
    """Tests for Gram matrix precomputation acceleration."""

    def test_gram_ols_matches_lstsq(self):
        """Verify _solve_subset_gram matches fit_ols on random subsets."""
        np.random.seed(123)
        n, B = 200, 10
        Phi = jnp.array(np.random.randn(n, B))
        true_w = np.zeros(B)
        true_w[:3] = [1.5, -2.0, 0.5]
        y = Phi @ jnp.array(true_w) + jnp.array(np.random.randn(n) * 0.1)

        PhiTPhi, PhiTy, yTy, Phi_ret, y_ret = _precompute_gram(Phi, y)

        # Test several random subsets
        # Note: jnp.linalg.solve (normal equations) and jnp.linalg.lstsq (QR/SVD)
        # use different algorithms, so float32 results differ at ~1e-3 level.
        for indices in [[0, 1, 2], [0, 3, 7], [1, 4, 5, 8], [2]]:
            coeffs_gram, mse_gram = _solve_subset_gram(PhiTPhi, PhiTy, yTy, Phi_ret, y_ret, indices)
            coeffs_ols, mse_ols = fit_ols(Phi[:, jnp.array(indices)], y)

            np.testing.assert_allclose(
                np.array(coeffs_gram), np.array(coeffs_ols), rtol=5e-3, atol=1e-4
            )
            np.testing.assert_allclose(mse_gram, mse_ols, rtol=1e-3, atol=1e-6)

    def test_gram_ridge_matches_standard(self):
        """Verify Gram ridge matches fit_ridge with alpha=0.1."""
        np.random.seed(456)
        n, B = 100, 8
        Phi = jnp.array(np.random.randn(n, B))
        y = jnp.array(np.random.randn(n))
        alpha = 0.1

        PhiTPhi, PhiTy, yTy, Phi_ret, y_ret = _precompute_gram(Phi, y, regularization=alpha)

        for indices in [[0, 1, 2], [3, 5, 7], [0, 1, 2, 3, 4]]:
            coeffs_gram, mse_gram = _solve_subset_gram(PhiTPhi, PhiTy, yTy, Phi_ret, y_ret, indices)
            coeffs_ridge, mse_ridge = fit_ridge(Phi[:, jnp.array(indices)], y, alpha)

            np.testing.assert_allclose(
                np.array(coeffs_gram), np.array(coeffs_ridge), rtol=1e-4, atol=1e-6
            )
            np.testing.assert_allclose(mse_gram, mse_ridge, rtol=1e-3, atol=1e-6)

    def test_fit_subset_gram_matches_fit_subset(self):
        """Verify full SelectionResult fields match between Gram and standard paths."""
        np.random.seed(789)
        n, B = 150, 6
        Phi = jnp.array(np.random.randn(n, B))
        y = Phi[:, 0] * 2.0 + Phi[:, 2] * -1.0 + jnp.array(np.random.randn(n) * 0.05)
        basis_names = [f"f{i}" for i in range(B)]
        complexities = jnp.ones(B)

        PhiTPhi, PhiTy, yTy, Phi_ret, y_ret = _precompute_gram(Phi, y)
        indices = [0, 2, 4]

        result_gram = _fit_subset_gram(
            PhiTPhi, PhiTy, yTy, Phi_ret, y_ret, indices, basis_names, complexities
        )
        result_std = fit_subset(Phi, y, indices, basis_names, complexities)

        # Coefficients differ at float32 level between solve (normal eqs) and lstsq (QR/SVD)
        np.testing.assert_allclose(
            np.array(result_gram.coefficients),
            np.array(result_std.coefficients),
            rtol=2e-2,
            atol=1e-4,
        )
        np.testing.assert_allclose(result_gram.mse, result_std.mse, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(result_gram.aic, result_std.aic, rtol=1e-2, atol=0.5)
        np.testing.assert_allclose(result_gram.bic, result_std.bic, rtol=1e-2, atol=0.5)
        np.testing.assert_allclose(result_gram.aicc, result_std.aicc, rtol=1e-2, atol=0.5)
        assert result_gram.selected_names == result_std.selected_names
        assert result_gram.n_samples == result_std.n_samples

    def test_forward_selection_unchanged_results(self):
        """Verify greedy forward still finds the correct terms with Gram acceleration."""
        library = (
            BasisLibrary(n_features=2, feature_names=["x", "y"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=3)
        )

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2.5 * X[:, 0] - 0.8 * X[:, 1] ** 2 + np.random.randn(100) * 0.01
        Phi = library.evaluate(jnp.array(X))

        path = greedy_forward_selection(
            Phi=Phi,
            y=jnp.array(y),
            basis_names=library.names,
            complexities=library.complexities,
            max_terms=5,
            information_criterion="bic",
        )

        best = path.best
        assert "x" in best.selected_names
        assert "y^2" in best.selected_names
        assert best.mse < 0.1

    def test_gram_singular_subset(self):
        """Collinear columns handled gracefully (no crash, finite MSE)."""
        np.random.seed(321)
        n = 50
        col = np.random.randn(n)
        # Create design matrix with two identical columns (singular Gram submatrix)
        Phi = jnp.array(np.column_stack([col, col, np.random.randn(n)]))
        y = jnp.array(np.random.randn(n))

        PhiTPhi, PhiTy, yTy, Phi_ret, y_ret = _precompute_gram(Phi, y)

        # Subset [0, 1] has a singular Gram block
        coeffs, mse = _solve_subset_gram(PhiTPhi, PhiTy, yTy, Phi_ret, y_ret, [0, 1])

        # Should not crash; MSE should be finite and non-negative
        assert np.isfinite(mse)
        assert mse >= 0.0

    def test_exhaustive_with_ridge(self):
        """Exhaustive search with ridge exercises the regularized Gram path."""
        library = BasisLibrary(n_features=2).add_constant().add_linear()

        np.random.seed(99)
        X = np.random.randn(60, 2)
        y = 1.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.randn(60) * 0.1
        Phi = library.evaluate(jnp.array(X))

        path = exhaustive_search(
            Phi=Phi,
            y=jnp.array(y),
            basis_names=library.names,
            complexities=library.complexities,
            max_terms=3,
            regularization=0.01,
        )

        best = path.best
        assert best.mse < 1.0
        assert len(path.results) > 0
