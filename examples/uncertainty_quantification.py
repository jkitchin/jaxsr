"""
Uncertainty Quantification Example for JAXSR.

Demonstrates the full range of UQ capabilities:
1. Classical OLS intervals (coefficient CIs, prediction/confidence bands)
2. Pareto front ensemble predictions
3. Bayesian Model Averaging
4. Conformal prediction (split and jackknife+)
5. Residual bootstrap
6. UQ visualization
"""

import jax.numpy as jnp
import numpy as np

from jaxsr import (
    BasisLibrary,
    BayesianModelAverage,
    SymbolicRegressor,
    bootstrap_coefficients,
    bootstrap_predict,
)


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


def example_classical_intervals():
    """
    Classical OLS prediction and confidence intervals.

    Since JAXSR models are linear-in-parameters (y = Phi @ beta),
    standard OLS inference applies directly:
      - Cov(beta) = s^2 * (Phi^T Phi)^{-1}
      - Prediction variance = s^2 * (1 + h(x_new))
    """
    print("=" * 60)
    print("Example 1: Classical OLS Intervals")
    print("=" * 60)

    # Generate known model: y = 2*x + 1 + noise
    np.random.seed(42)
    n = 100
    X = np.random.uniform(0, 5, (n, 1))
    y_true = 2.0 * X[:, 0] + 1.0
    y = y_true + np.random.randn(n) * 0.5

    X_jax = jnp.array(X)
    y_jax = jnp.array(y)

    # Fit model
    library = (
        BasisLibrary(n_features=1, feature_names=["x"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=3)
    )

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=3,
        strategy="greedy_forward",
    )
    model.fit(X_jax, y_jax)

    print("\nTrue model: y = 2*x + 1 (noise std = 0.5)")
    print(f"Discovered: {model.expression_}")

    # Noise estimate
    print(f"\nEstimated noise (sigma): {model.sigma_:.4f} (true: 0.5)")

    # Coefficient intervals
    print("\nCoefficient 95% confidence intervals:")
    intervals = model.coefficient_intervals(alpha=0.05)
    for name, (est, lo, hi, se) in intervals.items():
        print(f"  {name}: {est:.4f} [{lo:.4f}, {hi:.4f}]  (SE={se:.4f})")

    # Prediction intervals on new data
    X_new = jnp.linspace(0, 5, 5).reshape(-1, 1)
    y_pred, pred_lo, pred_hi = model.predict_interval(X_new, alpha=0.05)
    y_pred_c, conf_lo, conf_hi = model.confidence_band(X_new, alpha=0.05)

    print("\nPrediction and confidence intervals at selected points:")
    print(
        f"  {'x':>5}  {'y_pred':>8}  {'pred_lo':>8}  {'pred_hi':>8}  {'conf_lo':>8}  {'conf_hi':>8}"
    )
    for i in range(len(X_new)):
        print(
            f"  {float(X_new[i, 0]):5.1f}  "
            f"{float(y_pred[i]):8.3f}  "
            f"{float(pred_lo[i]):8.3f}  "
            f"{float(pred_hi[i]):8.3f}  "
            f"{float(conf_lo[i]):8.3f}  "
            f"{float(conf_hi[i]):8.3f}"
        )

    print("\nNote: Confidence band is narrower — it estimates E[y|x], not a new y.")

    # Covariance matrix
    cov = model.covariance_matrix_
    print("\nCoefficient covariance matrix:")
    print(f"  {np.array(cov)}")
    _diagnostics(model, X_jax, y_jax, "Classical OLS")

    return model


def example_ensemble_predictions():
    """
    Pareto front ensemble predictions.

    Measures structural/model uncertainty: how much do predictions
    vary across plausible model complexities?
    """
    print("\n" + "=" * 60)
    print("Example 2: Pareto Front Ensemble")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.uniform(-2, 3, (100, 1))
    y = 1.5 * X[:, 0] ** 2 - 0.5 * X[:, 0] + 2.0 + np.random.randn(100) * 0.5

    library = (
        BasisLibrary(n_features=1, feature_names=["x"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=4)
    )

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=5,
        strategy="greedy_forward",
    )
    model.fit(jnp.array(X), jnp.array(y))

    print(f"\nBest model: {model.expression_}")

    # Pareto front models
    print("\nPareto front models:")
    for r in model.pareto_front_:
        print(f"  Complexity {r.complexity}: {r.expression()}")

    # Ensemble predictions
    X_new = jnp.linspace(-2, 3, 5).reshape(-1, 1)
    result = model.predict_ensemble(X_new)

    print("\nEnsemble predictions at selected points:")
    print(f"  {'x':>5}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    for i in range(len(X_new)):
        print(
            f"  {float(X_new[i, 0]):5.1f}  "
            f"{float(result['y_mean'][i]):8.3f}  "
            f"{float(result['y_std'][i]):8.3f}  "
            f"{float(result['y_min'][i]):8.3f}  "
            f"{float(result['y_max'][i]):8.3f}"
        )

    _diagnostics(model, jnp.array(X), jnp.array(y), "Ensemble")

    return model


def example_bayesian_model_averaging():
    """
    Bayesian Model Averaging (BMA).

    Weights models by their BIC/AIC: w_k = exp(-0.5*IC_k) / Z.
    BMA variance includes both within-model and between-model components.
    """
    print("\n" + "=" * 60)
    print("Example 3: Bayesian Model Averaging")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.uniform(0, 5, (120, 1))
    y = 2.0 * X[:, 0] + 1.0 + np.random.randn(120) * 0.5

    library = (
        BasisLibrary(n_features=1, feature_names=["x"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=3)
    )

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=4,
        strategy="greedy_forward",
    )
    model.fit(jnp.array(X), jnp.array(y))

    # Create BMA
    bma = BayesianModelAverage(model, criterion="bic")

    print("\nBMA model weights (BIC-based):")
    for expr, weight in bma.weights.items():
        print(f"  {weight:.4f}  {expr}")

    # BMA predictions
    X_new = jnp.linspace(0, 5, 5).reshape(-1, 1)
    y_mean, y_std = bma.predict(X_new)

    print("\nBMA predictions:")
    print(f"  {'x':>5}  {'mean':>8}  {'std':>8}")
    for i in range(len(X_new)):
        print(f"  {float(X_new[i, 0]):5.1f}  {float(y_mean[i]):8.3f}  {float(y_std[i]):8.3f}")

    # Convenience method with intervals
    y_pred, lower, upper = model.predict_bma(X_new, criterion="bic", alpha=0.05)
    print("\n95% BMA prediction intervals:")
    print(f"  {'x':>5}  {'pred':>8}  {'lower':>8}  {'upper':>8}")
    for i in range(len(X_new)):
        print(
            f"  {float(X_new[i, 0]):5.1f}  "
            f"{float(y_pred[i]):8.3f}  "
            f"{float(lower[i]):8.3f}  "
            f"{float(upper[i]):8.3f}"
        )

    return model


def example_conformal_prediction():
    """
    Conformal prediction: distribution-free intervals.

    - Split conformal: uses held-out calibration set.
    - Jackknife+: uses LOO residuals from training data.

    Both provide finite-sample coverage guarantees.
    """
    print("\n" + "=" * 60)
    print("Example 4: Conformal Prediction")
    print("=" * 60)

    np.random.seed(42)
    n = 300
    X_all = np.random.uniform(0, 5, (n, 1))
    y_all = 2.0 * X_all[:, 0] + 1.0 + np.random.randn(n) * 0.5

    # Split into train / calibration / test
    X_train, y_train = jnp.array(X_all[:150]), jnp.array(y_all[:150])
    X_cal, y_cal = jnp.array(X_all[150:250]), jnp.array(y_all[150:250])
    X_test, y_test = jnp.array(X_all[250:]), jnp.array(y_all[250:])

    library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=2,
        strategy="greedy_forward",
    )
    model.fit(X_train, y_train)

    print(f"\nModel: {model.expression_}")

    # Split conformal
    y_pred, lower, upper = model.predict_conformal(
        X_test, alpha=0.10, method="split", X_cal=X_cal, y_cal=y_cal
    )
    covered = (y_test >= lower) & (y_test <= upper)
    coverage = float(jnp.mean(covered))
    print("\nSplit conformal (target 90% coverage):")
    print(f"  Actual coverage: {coverage:.1%}")
    print(f"  Avg interval width: {float(jnp.mean(upper - lower)):.3f}")

    # Jackknife+
    y_pred_j, lower_j, upper_j = model.predict_conformal(X_test, alpha=0.10, method="jackknife+")
    covered_j = (y_test >= lower_j) & (y_test <= upper_j)
    coverage_j = float(jnp.mean(covered_j))
    print("\nJackknife+ (target 90% coverage):")
    print(f"  Actual coverage: {coverage_j:.1%}")
    print(f"  Avg interval width: {float(jnp.mean(upper_j - lower_j)):.3f}")

    return model


def example_bootstrap():
    """
    Residual bootstrap: no Gaussian assumption needed.

    Resamples residuals to create y* = y_hat + e*, then refits.
    Vectorized with NumPy for efficiency.
    """
    print("\n" + "=" * 60)
    print("Example 5: Residual Bootstrap")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.uniform(0, 5, (100, 1))
    y = 2.0 * X[:, 0] + 1.0 + np.random.randn(100) * 0.5

    library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=2,
        strategy="greedy_forward",
    )
    model.fit(jnp.array(X), jnp.array(y))

    print(f"\nModel: {model.expression_}")

    # Bootstrap coefficient CIs
    result = bootstrap_coefficients(model, n_bootstrap=2000, alpha=0.05, seed=42)
    print("\nBootstrap 95% coefficient CIs (B=2000):")
    for i, name in enumerate(result["names"]):
        print(
            f"  {name}: {float(result['mean'][i]):.4f} "
            f"[{float(result['lower'][i]):.4f}, {float(result['upper'][i]):.4f}]  "
            f"(std={float(result['std'][i]):.4f})"
        )

    # Bootstrap prediction intervals
    X_new = jnp.linspace(0, 5, 5).reshape(-1, 1)
    pred_result = bootstrap_predict(model, X_new, n_bootstrap=2000, alpha=0.05, seed=42)

    print("\nBootstrap 95% prediction intervals:")
    print(f"  {'x':>5}  {'pred':>8}  {'lower':>8}  {'upper':>8}")
    for i in range(len(X_new)):
        print(
            f"  {float(X_new[i, 0]):5.1f}  "
            f"{float(pred_result['y_pred'][i]):8.3f}  "
            f"{float(pred_result['lower'][i]):8.3f}  "
            f"{float(pred_result['upper'][i]):8.3f}"
        )

    _diagnostics(model, jnp.array(X), jnp.array(y), "Bootstrap")

    return model


def example_visualization():
    """
    UQ visualization: fan charts, forest plots, BMA weights.
    """
    print("\n" + "=" * 60)
    print("Example 6: UQ Visualization")
    print("=" * 60)

    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        from jaxsr.plotting import (
            plot_bma_weights,
            plot_coefficient_intervals,
            plot_prediction_intervals,
        )
    except ImportError:
        print("matplotlib not available, skipping visualization example")
        return None

    np.random.seed(42)
    X = np.random.uniform(0, 5, (100, 1))
    y = 2.0 * X[:, 0] + 1.0 + np.random.randn(100) * 0.5

    library = (
        BasisLibrary(n_features=1, feature_names=["x"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=3)
    )

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=4,
        strategy="greedy_forward",
    )
    model.fit(jnp.array(X), jnp.array(y))

    X_plot = jnp.linspace(0, 5, 100).reshape(-1, 1)

    # Fan chart
    ax = plot_prediction_intervals(model, X_plot, y=jnp.array(y))
    ax.set_title("Prediction Intervals Fan Chart")
    plt.savefig("uq_prediction_intervals.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("\n  Saved: uq_prediction_intervals.png")

    # Coefficient forest plot
    ax = plot_coefficient_intervals(model)
    plt.savefig("uq_coefficient_intervals.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  Saved: uq_coefficient_intervals.png")

    # BMA weights
    ax = plot_bma_weights(model)
    plt.savefig("uq_bma_weights.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  Saved: uq_bma_weights.png")

    return model


def main():
    """Run all UQ examples."""
    print("JAXSR: Uncertainty Quantification Examples")
    print("=" * 60)

    example_classical_intervals()
    example_ensemble_predictions()
    example_bayesian_model_averaging()
    example_conformal_prediction()
    example_bootstrap()
    example_visualization()

    print("\n" + "=" * 60)
    print("All UQ examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
