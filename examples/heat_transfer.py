"""
Heat Transfer Example for JAXSR.

Demonstrates discovering heat transfer correlations from data, including:
- Nusselt number correlations
- Natural convection
- Forced convection
"""

import jax.numpy as jnp
import numpy as np

from jaxsr import BasisLibrary, SymbolicRegressor, Constraints


def example_forced_convection():
    """
    Discover Dittus-Boelter correlation for turbulent forced convection.

    True model: Nu = 0.023 * Re^0.8 * Pr^0.4
    """
    print("=" * 60)
    print("Example 1: Forced Convection (Dittus-Boelter)")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 100

    # Reynolds and Prandtl number ranges for turbulent flow
    Re = np.random.uniform(10000, 100000, n_samples)
    Pr = np.random.uniform(0.7, 100, n_samples)

    # Dittus-Boelter correlation
    Nu_true = 0.023 * Re ** 0.8 * Pr ** 0.4
    Nu = Nu_true * (1 + np.random.randn(n_samples) * 0.05)

    # Work in log space for power law discovery
    log_Re = np.log(Re)
    log_Pr = np.log(Pr)
    log_Nu = np.log(Nu)

    X = jnp.column_stack([log_Re, log_Pr])
    y = jnp.array(log_Nu)

    print(f"\nTrue model: Nu = 0.023 * Re^0.8 * Pr^0.4")
    print(f"Log form: ln(Nu) = {np.log(0.023):.3f} + 0.8*ln(Re) + 0.4*ln(Pr)")

    library = (
        BasisLibrary(n_features=2, feature_names=["ln_Re", "ln_Pr"])
        .add_constant()
        .add_linear()
        .add_interactions(max_order=2)
    )

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=4,
        strategy="greedy_forward",
    )
    model.fit(X, y)

    print(f"\nDiscovered expression (log space):")
    print(f"  {model.expression_}")
    print(f"  R² = {model.metrics_['r2']:.4f}")

    # Extract exponents
    if "ln_Re" in model.selected_features_:
        idx = model.selected_features_.index("ln_Re")
        re_exp = float(model.coefficients_[idx])
        print(f"\nRe exponent: {re_exp:.2f} (true: 0.80)")

    if "ln_Pr" in model.selected_features_:
        idx = model.selected_features_.index("ln_Pr")
        pr_exp = float(model.coefficients_[idx])
        print(f"Pr exponent: {pr_exp:.2f} (true: 0.40)")

    return model


def example_natural_convection():
    """
    Discover natural convection correlation.

    True model: Nu = C * Ra^n where Ra = Gr * Pr
    For vertical plate: Nu = 0.59 * Ra^0.25 (laminar)
    """
    print("\n" + "=" * 60)
    print("Example 2: Natural Convection (Vertical Plate)")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 80

    # Rayleigh number range for laminar flow
    Ra = np.random.uniform(1e4, 1e9, n_samples)

    # Churchill-Chu correlation (simplified for laminar)
    C = 0.59
    n = 0.25
    Nu_true = C * Ra ** n
    Nu = Nu_true * (1 + np.random.randn(n_samples) * 0.03)

    # Log transformation
    log_Ra = np.log10(Ra)
    log_Nu = np.log10(Nu)

    X = jnp.array(log_Ra).reshape(-1, 1)
    y = jnp.array(log_Nu)

    print(f"\nTrue model: Nu = 0.59 * Ra^0.25")
    print(f"Log form: log10(Nu) = {np.log10(C):.3f} + 0.25*log10(Ra)")

    library = (
        BasisLibrary(n_features=1, feature_names=["log_Ra"])
        .add_constant()
        .add_linear()
    )

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=2,
        strategy="exhaustive",
    )
    model.fit(X, y)

    print(f"\nDiscovered expression:")
    print(f"  {model.expression_}")
    print(f"  R² = {model.metrics_['r2']:.4f}")

    return model


def example_fin_efficiency():
    """
    Discover fin efficiency correlation.

    True model: eta = tanh(mL) / (mL)
    where mL = sqrt(hP/(kA)) * L
    """
    print("\n" + "=" * 60)
    print("Example 3: Fin Efficiency")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 100

    # mL parameter (dimensionless fin parameter)
    mL = np.random.uniform(0.1, 3.0, n_samples)

    # True fin efficiency
    eta_true = np.tanh(mL) / mL
    eta = eta_true + np.random.randn(n_samples) * 0.01

    X = jnp.array(mL).reshape(-1, 1)
    y = jnp.array(eta)

    print(f"\nTrue model: eta = tanh(mL) / mL")

    # Build library with hyperbolic functions
    library = (
        BasisLibrary(n_features=1, feature_names=["mL"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=4)
        .add_transcendental(["tanh", "inv"])
    )

    # Add the exact form as a custom function
    library.add_custom(
        name="tanh(mL)/mL",
        func=lambda X: jnp.tanh(X[:, 0]) / X[:, 0],
        complexity=3,
    )

    # Constraint: efficiency must be between 0 and 1
    constraints = Constraints().add_bounds("y", lower=0.0, upper=1.0)

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=3,
        strategy="greedy_forward",
        constraints=constraints,
    )
    model.fit(X, y)

    print(f"\nDiscovered expression:")
    print(f"  {model.expression_}")
    print(f"  R² = {model.metrics_['r2']:.4f}")

    return model


def example_heat_exchanger():
    """
    Discover heat exchanger effectiveness-NTU relationship.

    For parallel flow: eps = (1 - exp(-NTU*(1+C))) / (1+C)
    """
    print("\n" + "=" * 60)
    print("Example 4: Heat Exchanger Effectiveness")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 120

    # NTU and capacity ratio
    NTU = np.random.uniform(0.1, 5.0, n_samples)
    C = np.random.uniform(0.0, 1.0, n_samples)

    # Parallel flow effectiveness
    eps_true = (1 - np.exp(-NTU * (1 + C))) / (1 + C)
    eps = eps_true + np.random.randn(n_samples) * 0.01

    X = jnp.column_stack([NTU, C])
    y = jnp.array(eps)

    print(f"\nTrue model: eps = (1 - exp(-NTU*(1+C))) / (1+C)")

    library = (
        BasisLibrary(n_features=2, feature_names=["NTU", "C"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=2)
        .add_interactions(max_order=2)
        .add_transcendental(["exp"])
    )

    # Add specific forms
    library.add_custom(
        name="exp(-NTU)",
        func=lambda X: jnp.exp(-X[:, 0]),
        complexity=2,
    )
    library.add_custom(
        name="exp(-NTU*(1+C))",
        func=lambda X: jnp.exp(-X[:, 0] * (1 + X[:, 1])),
        complexity=3,
    )
    library.add_custom(
        name="1/(1+C)",
        func=lambda X: 1 / (1 + X[:, 1]),
        complexity=2,
    )

    # Effectiveness between 0 and 1
    constraints = Constraints().add_bounds("y", lower=0.0, upper=1.0)

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=5,
        strategy="greedy_forward",
        constraints=constraints,
    )
    model.fit(X, y)

    print(f"\nDiscovered expression:")
    print(f"  {model.expression_}")
    print(f"  R² = {model.metrics_['r2']:.4f}")

    return model


def main():
    """Run all heat transfer examples."""
    print("JAXSR: Heat Transfer Correlation Examples")
    print("Discovering Empirical Correlations from Data")
    print("=" * 60)

    example_forced_convection()
    example_natural_convection()
    example_fin_efficiency()
    example_heat_exchanger()

    print("\n" + "=" * 60)
    print("All heat transfer examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
