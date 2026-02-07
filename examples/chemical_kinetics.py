"""
Chemical Kinetics Example for JAXSR.

Demonstrates discovering rate laws from kinetic data, including:
- Langmuir-Hinshelwood kinetics
- Power law kinetics
- Arrhenius temperature dependence
"""

import jax.numpy as jnp
import numpy as np

from jaxsr import BasisLibrary, Constraints, SymbolicRegressor


def _diagnostics(model, X, y, prefix):
    """Print parameter significance table and save parity/residual plots."""
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    from jaxsr.plotting import plot_parity, plot_residuals

    intervals = model.coefficient_intervals(alpha=0.05)
    n, k = len(np.asarray(y)), len(model.selected_features_)
    df = n - k

    print(f"\n  Parameter significance ({prefix}):")
    print(
        f"  {'Term':>20s} {'Estimate':>10s} {'Std Err':>9s}" f" {'t':>8s} {'p-value':>10s} 95% CI"
    )
    print("  " + "-" * 80)
    for name, (est, lo, hi, se) in intervals.items():
        t_val = est / se if abs(se) > 1e-15 else float("inf")
        p_val = float(2 * (1 - sp_stats.t.cdf(abs(t_val), df))) if df > 0 else 0.0
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = ""
        print(
            f"  {name:>20s} {est:10.4f} {se:9.4f} {t_val:8.2f}"
            f" {p_val:10.2e} [{lo:.4f}, {hi:.4f}] {sig}"
        )
    print("  --- *** p<0.001, ** p<0.01, * p<0.05")

    X_arr = jnp.atleast_2d(jnp.asarray(X))
    y_arr = jnp.asarray(y)
    y_pred = model.predict(X_arr)
    tag = prefix.lower().replace(" ", "_").replace("-", "_")

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_parity(y_arr, y_pred, ax=ax, title=f"{prefix}: Parity")
    plt.savefig(f"{tag}_parity.png", dpi=150, bbox_inches="tight")
    plt.close()

    plot_residuals(model, X_arr, y_arr)
    plt.savefig(f"{tag}_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {tag}_parity.png, {tag}_residuals.png")


def example_langmuir_hinshelwood():
    """
    Discover Langmuir-Hinshelwood rate law.

    True model: r = k * C_A * C_B / (1 + K * C_A)
    """
    print("=" * 60)
    print("Example 1: Langmuir-Hinshelwood Kinetics")
    print("=" * 60)

    # Generate synthetic kinetic data
    np.random.seed(42)
    n_samples = 100

    # Concentration ranges typical for catalytic reactions
    C_A = np.random.uniform(0.1, 2.0, n_samples)
    C_B = np.random.uniform(0.1, 2.0, n_samples)

    # True kinetic parameters
    k = 2.5  # Rate constant
    K = 1.2  # Adsorption equilibrium constant

    # True rate law
    r_true = k * C_A * C_B / (1 + K * C_A)
    r = r_true + np.random.randn(n_samples) * 0.05

    X = jnp.column_stack([C_A, C_B])
    y = jnp.array(r)

    print("\nTrue model: r = 2.5*C_A*C_B / (1 + 1.2*C_A)")
    print(f"Data: {n_samples} samples")

    # Build basis library with appropriate functions for kinetics
    library = (
        BasisLibrary(n_features=2, feature_names=["C_A", "C_B"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=2)
        .add_interactions(max_order=2)
        .add_ratios()
        .add_transcendental(["inv"])
    )

    print(f"Basis library: {len(library)} candidate functions")

    # Add constraint: reaction rate must be non-negative
    constraints = Constraints().add_bounds("y", lower=0)

    # Fit model
    model = SymbolicRegressor(
        basis_library=library,
        max_terms=6,
        strategy="greedy_forward",
        information_criterion="bic",
        constraints=constraints,
    )
    model.fit(X, y)

    print("\nDiscovered expression:")
    print(f"  {model.expression_}")
    print("\nMetrics:")
    print(f"  R² = {model.metrics_['r2']:.4f}")
    print(f"  MSE = {model.metrics_['mse']:.6f}")
    _diagnostics(model, X, y, "Langmuir-Hinshelwood")

    return model


def example_power_law():
    """
    Discover power law kinetics.

    True model: r = k * C_A^a * C_B^b
    """
    print("\n" + "=" * 60)
    print("Example 2: Power Law Kinetics")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 100

    C_A = np.random.uniform(0.5, 3.0, n_samples)
    C_B = np.random.uniform(0.5, 3.0, n_samples)

    # True parameters
    k = 1.5
    a = 1.0  # First order in A
    b = 0.5  # Half order in B

    r_true = k * C_A**a * C_B**b
    r = r_true + np.random.randn(n_samples) * 0.02

    X = jnp.column_stack([C_A, C_B])
    y = jnp.array(r)

    print("\nTrue model: r = 1.5 * C_A^1.0 * C_B^0.5")

    # For power law, include sqrt for half-order
    library = (
        BasisLibrary(n_features=2, feature_names=["C_A", "C_B"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=2)
        .add_interactions(max_order=2)
        .add_transcendental(["sqrt"])
    )

    # Add custom basis function for C_A * sqrt(C_B)
    library.add_custom(
        name="C_A*sqrt(C_B)",
        func=lambda X: X[:, 0] * jnp.sqrt(X[:, 1]),
        complexity=2,
    )

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=5,
        strategy="greedy_forward",
    )
    model.fit(X, y)

    print("\nDiscovered expression:")
    print(f"  {model.expression_}")
    print(f"  R² = {model.metrics_['r2']:.4f}")
    _diagnostics(model, X, y, "Power Law")

    return model


def example_arrhenius():
    """
    Discover Arrhenius temperature dependence.

    True model: k = A * exp(-Ea/RT)
    """
    print("\n" + "=" * 60)
    print("Example 3: Arrhenius Temperature Dependence")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 50

    # Temperature range (K)
    T = np.random.uniform(300, 500, n_samples)

    # Arrhenius parameters
    A = 1e6  # Pre-exponential factor
    Ea = 50000  # Activation energy (J/mol)
    R = 8.314  # Gas constant (J/mol/K)

    # True rate constant
    k_true = A * np.exp(-Ea / (R * T))
    # Work in log space for better fitting
    log_k = np.log(k_true) + np.random.randn(n_samples) * 0.05

    # Use 1/T as the feature (linearized Arrhenius)
    X = jnp.array(1000 / T).reshape(-1, 1)  # 1000/T in 1/K
    y = jnp.array(log_k)

    print("\nTrue model: ln(k) = ln(A) - Ea/(R*T)")
    print(f"Or: ln(k) = {np.log(A):.2f} - {Ea/R/1000:.2f} * (1000/T)")

    # Simple linear library for linearized Arrhenius
    library = BasisLibrary(n_features=1, feature_names=["1000/T"]).add_constant().add_linear()

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=2,
        strategy="exhaustive",
    )
    model.fit(X, y)

    print("\nDiscovered expression:")
    print(f"  {model.expression_}")
    print(f"  R² = {model.metrics_['r2']:.4f}")

    # Extract parameters
    if "1" in model.selected_features_:
        idx_const = model.selected_features_.index("1")
        ln_A = float(model.coefficients_[idx_const])
        print("\nExtracted parameters:")
        print(f"  ln(A) = {ln_A:.2f} (true: {np.log(A):.2f})")

    if "1000/T" in model.selected_features_:
        idx_T = model.selected_features_.index("1000/T")
        slope = float(model.coefficients_[idx_T])
        Ea_fit = -slope * R * 1000
        print(f"  Ea = {Ea_fit:.0f} J/mol (true: {Ea} J/mol)")

    _diagnostics(model, X, y, "Arrhenius")

    return model


def example_competitive_adsorption():
    """
    Discover competitive adsorption kinetics.

    True model: r = k * C_A * C_B / (1 + K_A*C_A + K_B*C_B)
    """
    print("\n" + "=" * 60)
    print("Example 4: Competitive Adsorption")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 150

    C_A = np.random.uniform(0.1, 2.0, n_samples)
    C_B = np.random.uniform(0.1, 2.0, n_samples)

    # Kinetic parameters
    k = 3.0
    K_A = 0.8
    K_B = 1.5

    r_true = k * C_A * C_B / (1 + K_A * C_A + K_B * C_B)
    r = r_true + np.random.randn(n_samples) * 0.03

    X = jnp.column_stack([C_A, C_B])
    y = jnp.array(r)

    print("\nTrue model: r = 3.0*C_A*C_B / (1 + 0.8*C_A + 1.5*C_B)")

    # Build comprehensive library
    library = (
        BasisLibrary(n_features=2, feature_names=["C_A", "C_B"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=2)
        .add_interactions(max_order=2)
        .add_ratios()
    )

    # Add custom rational functions
    library.add_custom(
        name="C_A*C_B/(1+C_A)",
        func=lambda X: X[:, 0] * X[:, 1] / (1 + X[:, 0]),
        complexity=3,
    )
    library.add_custom(
        name="C_A*C_B/(1+C_B)",
        func=lambda X: X[:, 0] * X[:, 1] / (1 + X[:, 1]),
        complexity=3,
    )
    library.add_custom(
        name="C_A*C_B/(1+C_A+C_B)",
        func=lambda X: X[:, 0] * X[:, 1] / (1 + X[:, 0] + X[:, 1]),
        complexity=4,
    )

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=5,
        strategy="greedy_forward",
    )
    model.fit(X, y)

    print("\nDiscovered expression:")
    print(f"  {model.expression_}")
    print(f"  R² = {model.metrics_['r2']:.4f}")
    _diagnostics(model, X, y, "Competitive Adsorption")

    return model


def main():
    """Run all chemical kinetics examples."""
    print("JAXSR: Chemical Kinetics Examples")
    print("Discovering Rate Laws from Data")
    print("=" * 60)

    example_langmuir_hinshelwood()
    example_power_law()
    example_arrhenius()
    example_competitive_adsorption()

    print("\n" + "=" * 60)
    print("All chemical kinetics examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
