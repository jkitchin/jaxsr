"""
Basic Usage Example for JAXSR.

Demonstrates the core functionality of JAXSR for discovering
algebraic expressions from data.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jaxsr import BasisLibrary, SymbolicRegressor, fit_symbolic


def example_polynomial():
    """Discover a polynomial expression."""
    print("=" * 60)
    print("Example 1: Polynomial Expression Discovery")
    print("=" * 60)

    # Generate synthetic data: y = 2.5*x0 + 1.2*x0*x1 - 0.8*x1^2 + noise
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 2) * 2
    y = 2.5 * X[:, 0] + 1.2 * X[:, 0] * X[:, 1] - 0.8 * X[:, 1] ** 2
    y += np.random.randn(n_samples) * 0.1

    X_jax = jnp.array(X)
    y_jax = jnp.array(y)

    print(f"\nTrue model: y = 2.5*x0 + 1.2*x0*x1 - 0.8*x1^2")
    print(f"Data: {n_samples} samples, 2 features, noise std=0.1")

    # Build basis library
    library = (
        BasisLibrary(n_features=2, feature_names=["x0", "x1"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=3)
        .add_interactions(max_order=2)
    )

    print(f"\nBasis library: {len(library)} candidate functions")

    # Fit model
    model = SymbolicRegressor(
        basis_library=library,
        max_terms=5,
        strategy="greedy_forward",
        information_criterion="bic",
    )
    model.fit(X_jax, y_jax)

    # Results
    print(f"\nDiscovered expression:")
    print(f"  {model.expression_}")
    print(f"\nMetrics:")
    print(f"  R² score: {model.score(X_jax, y_jax):.6f}")
    print(f"  MSE: {model.metrics_['mse']:.6f}")
    print(f"  BIC: {model.metrics_['bic']:.2f}")
    print(f"  Complexity: {model.complexity_}")

    return model


def example_transcendental():
    """Discover an expression with transcendental functions."""
    print("\n" + "=" * 60)
    print("Example 2: Transcendental Expression Discovery")
    print("=" * 60)

    # Generate data: y = exp(-0.5*x) + log(x+1)
    np.random.seed(42)
    n_samples = 150
    X = np.random.uniform(0.1, 3.0, (n_samples, 1))
    y = np.exp(-0.5 * X[:, 0]) + np.log(X[:, 0] + 1)
    y += np.random.randn(n_samples) * 0.02

    X_jax = jnp.array(X)
    y_jax = jnp.array(y)

    print(f"\nTrue model: y = exp(-0.5*x) + log(x+1)")
    print(f"Data: {n_samples} samples, 1 feature")

    # Build library with transcendental functions
    library = (
        BasisLibrary(n_features=1, feature_names=["x"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=3)
        .add_transcendental(["log", "exp", "sqrt"])
    )

    print(f"\nBasis library: {len(library)} candidate functions")

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=6,
        strategy="greedy_forward",
    )
    model.fit(X_jax, y_jax)

    print(f"\nDiscovered expression:")
    print(f"  {model.expression_}")
    print(f"  R² score: {model.score(X_jax, y_jax):.6f}")

    return model


def example_convenience_function():
    """Use the fit_symbolic convenience function."""
    print("\n" + "=" * 60)
    print("Example 3: Quick Fitting with fit_symbolic")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 3.0 * X[:, 0] ** 2 - 2.0 * X[:, 1] + 1.0
    y += np.random.randn(100) * 0.05

    print(f"\nTrue model: y = 3*a^2 - 2*b + 1")

    # Quick fit
    model = fit_symbolic(
        jnp.array(X),
        jnp.array(y),
        feature_names=["a", "b"],
        max_terms=5,
        max_poly_degree=3,
    )

    print(f"\nDiscovered: {model.expression_}")
    print(f"R² = {model.score(jnp.array(X), jnp.array(y)):.4f}")

    return model


def example_pareto_front():
    """Explore the Pareto front of complexity vs accuracy."""
    print("\n" + "=" * 60)
    print("Example 4: Pareto Front Exploration")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    X = np.random.randn(150, 2)
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] ** 2 - 0.5 * X[:, 0] * X[:, 1]
    y += np.random.randn(150) * 0.1

    X_jax = jnp.array(X)
    y_jax = jnp.array(y)

    library = (
        BasisLibrary(n_features=2, feature_names=["x", "y"])
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
    model.fit(X_jax, y_jax)

    print("\nPareto Front (Complexity vs MSE):")
    print("-" * 50)
    for result in model.pareto_front_:
        print(f"  Complexity {result.complexity:2d} | MSE {result.mse:.6f}")
        print(f"    {result.expression()}")
        print()

    return model


def example_model_export():
    """Demonstrate model export capabilities."""
    print("\n" + "=" * 60)
    print("Example 5: Model Export")
    print("=" * 60)

    # Generate and fit
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2.0 * X[:, 0] + X[:, 1] ** 2

    model = fit_symbolic(
        jnp.array(X),
        jnp.array(y),
        feature_names=["a", "b"],
        max_terms=4,
    )

    # Human-readable expression
    print(f"\nExpression: {model.expression_}")

    # SymPy export
    try:
        sympy_expr = model.to_sympy()
        print(f"SymPy: {sympy_expr}")

        # LaTeX
        latex = model.to_latex()
        print(f"LaTeX: {latex}")
    except ImportError:
        print("(SymPy not available for symbolic export)")

    # Pure Python callable
    predict_fn = model.to_callable()
    X_test = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = predict_fn(X_test)
    print(f"\nPure NumPy predictions:")
    print(f"  X = {X_test.tolist()}")
    print(f"  y_pred = {y_pred.tolist()}")

    return model


def main():
    """Run all examples."""
    print("JAXSR: JAX-based Symbolic Regression")
    print("Basic Usage Examples")
    print("=" * 60)

    example_polynomial()
    example_transcendental()
    example_convenience_function()
    example_pareto_front()
    example_model_export()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
