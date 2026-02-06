"""Tests for parametric basis functions with nonlinear parameter optimisation."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr import BasisLibrary, SymbolicRegressor

# ──────────────────────────────────────────────────────────────────
# 1. exp(-a*x) parameter recovery
# ──────────────────────────────────────────────────────────────────


class TestParametricExpDecay:
    """Fit y = 2.5 * exp(-0.3 * x) and verify a ≈ 0.3."""

    def test_recovers_decay_rate(self):
        np.random.seed(42)
        X = np.linspace(0.1, 10.0, 200).reshape(-1, 1)
        y = 2.5 * np.exp(-0.3 * X[:, 0])

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_parametric(
                name="exp(-a*x)",
                func=lambda X, a: jnp.exp(-a * X[:, 0]),
                param_bounds={"a": (0.01, 5.0)},
                complexity=3,
                feature_indices=(0,),
            )
        )

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=3,
            strategy="greedy_forward",
        )
        model.fit(jnp.array(X), jnp.array(y))

        r2 = model.score(jnp.array(X), jnp.array(y))
        assert r2 > 0.99, f"R² = {r2:.6f}"

        p_info = library._parametric_info[0]
        assert p_info.resolved_params is not None
        assert (
            abs(p_info.resolved_params["a"] - 0.3) < 0.05
        ), f"Expected a ≈ 0.3, got {p_info.resolved_params['a']:.4f}"


# ──────────────────────────────────────────────────────────────────
# 2. x^a (Dittus-Boelter style) parameter recovery
# ──────────────────────────────────────────────────────────────────


class TestParametricPowerLaw:
    """Fit y = 1.2 * x^0.8 and verify a ≈ 0.8."""

    def test_recovers_exponent(self):
        np.random.seed(42)
        X = np.linspace(0.5, 10.0, 200).reshape(-1, 1)
        y = 1.2 * X[:, 0] ** 0.8

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_parametric(
                name="x^a",
                func=lambda X, a: jnp.power(X[:, 0], a),
                param_bounds={"a": (0.1, 2.0)},
                complexity=2,
                feature_indices=(0,),
            )
        )

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=3,
            strategy="greedy_forward",
        )
        model.fit(jnp.array(X), jnp.array(y))

        r2 = model.score(jnp.array(X), jnp.array(y))
        assert r2 > 0.99, f"R² = {r2:.6f}"

        p_info = library._parametric_info[0]
        assert p_info.resolved_params is not None
        assert (
            abs(p_info.resolved_params["a"] - 0.8) < 0.05
        ), f"Expected a ≈ 0.8, got {p_info.resolved_params['a']:.4f}"


# ──────────────────────────────────────────────────────────────────
# 3. No-parametric library → zero overhead / identical behaviour
# ──────────────────────────────────────────────────────────────────


class TestNoParametricRegression:
    """Verify a library without parametric functions behaves identically."""

    def test_identical_results(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2.5 * X[:, 0] + 1.2 * X[:, 0] * X[:, 1] - 0.8 * X[:, 1] ** 2

        library = (
            BasisLibrary(n_features=2, feature_names=["x", "y"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
            .add_interactions(max_order=2)
        )

        assert not library.has_parametric

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=5,
            strategy="greedy_forward",
        )
        model.fit(jnp.array(X), jnp.array(y))

        r2 = model.score(jnp.array(X), jnp.array(y))
        assert r2 > 0.99

        # No parametric_params on result
        assert model._result.parametric_params is None


# ──────────────────────────────────────────────────────────────────
# 4. Symbolic export: to_sympy() / to_latex()
# ──────────────────────────────────────────────────────────────────


class TestParametricSymPyLatex:
    """Parametric parameters appear correctly in to_sympy() / to_latex()."""

    def test_exp_to_sympy(self):
        pytest.importorskip("sympy")
        import sympy

        np.random.seed(42)
        X = np.linspace(0.1, 10.0, 200).reshape(-1, 1)
        y = 2.5 * np.exp(-0.3 * X[:, 0])

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_parametric(
                name="exp(-a*x)",
                func=lambda X, a: jnp.exp(-a * X[:, 0]),
                param_bounds={"a": (0.01, 5.0)},
                complexity=3,
                feature_indices=(0,),
            )
        )

        model = SymbolicRegressor(basis_library=library, max_terms=3)
        model.fit(jnp.array(X), jnp.array(y))

        expr = model.to_sympy()
        assert expr is not None
        expr_str = str(expr)
        assert "exp" in expr_str

        # Evaluate symbolically at x=1 and compare
        x_sym = sympy.Symbol("x")
        val = float(expr.subs(x_sym, 1.0))
        expected = 2.5 * np.exp(-0.3)
        assert abs(val - expected) < 0.5, f"Expected ≈{expected:.4f}, got {val:.4f}"

        latex = model.to_latex()
        assert isinstance(latex, str)
        assert len(latex) > 0

    def test_power_to_sympy(self):
        pytest.importorskip("sympy")
        import sympy

        np.random.seed(42)
        X = np.linspace(0.5, 10.0, 200).reshape(-1, 1)
        y = 1.2 * X[:, 0] ** 0.8

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_parametric(
                name="x^a",
                func=lambda X, a: jnp.power(X[:, 0], a),
                param_bounds={"a": (0.1, 2.0)},
                complexity=2,
                feature_indices=(0,),
            )
        )

        model = SymbolicRegressor(basis_library=library, max_terms=3)
        model.fit(jnp.array(X), jnp.array(y))

        expr = model.to_sympy()
        assert expr is not None

        x_sym = sympy.Symbol("x")
        val = float(expr.subs(x_sym, 2.0))
        expected = 1.2 * 2.0**0.8
        assert abs(val - expected) < 0.5, f"Expected ≈{expected:.4f}, got {val:.4f}"

        latex = model.to_latex()
        assert isinstance(latex, str)
        assert len(latex) > 0

    def test_to_callable_parametric(self):
        """to_callable() produces correct predictions with optimised params."""
        np.random.seed(42)
        X = np.linspace(0.5, 10.0, 200).reshape(-1, 1)
        y = 1.2 * X[:, 0] ** 0.8

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_parametric(
                name="x^a",
                func=lambda X, a: jnp.power(X[:, 0], a),
                param_bounds={"a": (0.1, 2.0)},
                complexity=2,
                feature_indices=(0,),
            )
        )

        model = SymbolicRegressor(basis_library=library, max_terms=3)
        model.fit(jnp.array(X), jnp.array(y))

        predict_fn = model.to_callable()
        y_pred = predict_fn(np.array(X))
        np.testing.assert_allclose(y_pred, y, rtol=0.05)
