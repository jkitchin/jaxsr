"""
Active Learning & Acquisition Functions Example for JAXSR.

Demonstrates how to use acquisition functions to intelligently select
the next experiments to run, balancing exploration and exploitation.

This example covers five scenarios:

1. Pure exploration: reduce uncertainty everywhere
2. Bayesian optimisation: find the minimum of an unknown function
3. Model discrimination: resolve which model structure is correct
4. Batch selection strategies: greedy vs penalized vs kriging believer
5. Full active learning loop: iteratively improve a model
"""

import jax.numpy as jnp
import numpy as np

from jaxsr import BasisLibrary, SymbolicRegressor
from jaxsr.acquisition import (
    LCB,
    ActiveLearner,
    AOptimal,
    BMAUncertainty,
    ConfidenceBandWidth,
    DOptimal,
    EnsembleDisagreement,
    ExpectedImprovement,
    ModelDiscrimination,
    ModelMin,
    PredictionVariance,
    ProbabilityOfImprovement,
    ThompsonSampling,
    suggest_points,
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


# =========================================================================
# Shared setup: fit a model we'll use across examples
# =========================================================================


def make_model():
    """Create a fitted model on synthetic data: y = x^2 - 3x + 2 + noise."""
    np.random.seed(42)
    X = np.random.uniform(0, 5, (40, 1))
    y = X[:, 0] ** 2 - 3.0 * X[:, 0] + 2.0 + np.random.randn(40) * 0.3

    library = (
        BasisLibrary(n_features=1, feature_names=["x"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=3)
    )
    model = SymbolicRegressor(basis_library=library, max_terms=4, strategy="greedy_forward")
    model.fit(jnp.array(X), jnp.array(y))
    _diagnostics(model, jnp.array(X), jnp.array(y), "Active Learning Base")
    return model


# =========================================================================
# Example 1: Pure Exploration - Reduce Uncertainty
# =========================================================================


def example_exploration():
    """
    Goal: improve model accuracy uniformly by sampling where
    prediction uncertainty is highest.

    Available acquisition functions for this goal:

    - PredictionVariance: the default.  Uses the OLS posterior to compute
      sigma^2(x).  Fast, exact for linear-in-parameter models.

    - ConfidenceBandWidth: similar, but reports the actual width of the
      confidence band at a specified significance level.

    - EnsembleDisagreement: uses the Pareto front of models with different
      complexities.  Good when you're unsure whether a simpler or more
      complex model is appropriate.

    - BMAUncertainty: Bayesian Model Averaging.  The most comprehensive
      measure -- combines noise uncertainty and model-selection uncertainty.

    - AOptimal: targets reduction in *parameter* uncertainty (trace of
      covariance matrix).  Use when you care about accurate coefficients.

    - DOptimal: maximises information gain (det of information matrix).
      Use when you want maximal information per experiment.
    """
    print("=" * 60)
    print("Example 1: Pure Exploration")
    print("=" * 60)

    model = make_model()
    bounds = [(0.0, 5.0)]

    print(f"Current model: {model.expression_}")
    print(f"Current R^2:   {model.score(model._X_train, model._y_train):.4f}")
    print(f"Training size: {len(model._y_train)}")
    print()

    # --- Try different exploration strategies ---
    strategies = [
        ("PredictionVariance", PredictionVariance()),
        ("ConfidenceBandWidth(95%)", ConfidenceBandWidth(alpha=0.05)),
        ("EnsembleDisagreement", EnsembleDisagreement()),
        ("BMAUncertainty", BMAUncertainty(criterion="bic")),
        ("AOptimal", AOptimal()),
        ("DOptimal", DOptimal()),
    ]

    for name, acq in strategies:
        result = suggest_points(model, bounds, acq, n_points=3, random_state=42)
        pts = np.array(result.points).ravel()
        print(f"  {name:30s} -> x = [{', '.join(f'{p:.2f}' for p in pts)}]")

    print()


# =========================================================================
# Example 2: Bayesian Optimisation - Find the Minimum
# =========================================================================


def example_optimisation():
    """
    Goal: find x that minimises y, using the fitted model as a surrogate.

    Available acquisition functions for this goal:

    - ModelMin / ModelMax: pure exploitation.  No exploration at all --
      just returns the predicted optimum.  Use only when you fully trust
      the model.

    - LCB (Lower Confidence Bound): y_hat - kappa*sigma.  The kappa
      parameter controls exploration vs exploitation:
        kappa=0  -> pure exploitation (ModelMin)
        kappa~2  -> balanced
        kappa>3  -> heavy exploration

    - UCB (Upper Confidence Bound): the mirror image for maximisation.

    - ExpectedImprovement (EI): the Bayesian optimisation gold standard.
      Naturally balances exploration and exploitation without a tuning
      parameter (just xi, which is usually small).  Recommended as the
      default for optimisation.

    - ProbabilityOfImprovement (PI): similar to EI but only cares about
      the *probability* of beating the current best, not the magnitude
      of improvement.  More exploitative than EI for the same xi.

    - ThompsonSampling: draws a random model from the posterior and
      optimises that.  Produces diverse batches naturally.
    """
    print("=" * 60)
    print("Example 2: Bayesian Optimisation (Minimise y)")
    print("=" * 60)

    model = make_model()
    bounds = [(0.0, 5.0)]

    print(f"Model: {model.expression_}")
    print("True minimum at x=1.5 (y = 2.25 - 4.5 + 2 = -0.25)")
    print()

    strategies = [
        ("ModelMin (exploit only)", ModelMin()),
        ("LCB kappa=0.5 (exploitative)", LCB(kappa=0.5)),
        ("LCB kappa=2 (balanced)", LCB(kappa=2.0)),
        ("LCB kappa=5 (exploratory)", LCB(kappa=5.0)),
        ("Expected Improvement", ExpectedImprovement(minimize=True)),
        ("Prob. of Improvement", ProbabilityOfImprovement(minimize=True)),
        ("Thompson Sampling", ThompsonSampling(minimize=True, seed=42)),
    ]

    for name, acq in strategies:
        result = suggest_points(model, bounds, acq, n_points=3, random_state=42)
        pts = np.array(result.points).ravel()
        print(f"  {name:35s} -> x = [{', '.join(f'{p:.2f}' for p in pts)}]")

    print()


# =========================================================================
# Example 3: Model Discrimination
# =========================================================================


def example_discrimination():
    """
    Goal: figure out which model form is correct.

    When the Pareto front contains models of different complexities that
    all fit the data similarly, you need data points that *discriminate*
    between them.

    - ModelDiscrimination: scores candidates by the maximum disagreement
      among Pareto-front models.

    - EnsembleDisagreement: standard deviation across Pareto models.
      Similar idea but uses std instead of max-min range.
    """
    print("=" * 60)
    print("Example 3: Model Discrimination")
    print("=" * 60)

    model = make_model()
    bounds = [(0.0, 5.0)]

    print(f"Best model: {model.expression_}")
    print(f"Pareto front has {len(model.pareto_front_)} models:")
    for r in model.pareto_front_:
        print(f"  complexity={r.complexity}, BIC={r.bic:.1f}: {r.expression()}")
    print()

    acqs = [
        ("ModelDiscrimination", ModelDiscrimination()),
        ("EnsembleDisagreement", EnsembleDisagreement()),
    ]

    for name, acq in acqs:
        result = suggest_points(model, bounds, acq, n_points=5, random_state=42)
        pts = np.array(result.points).ravel()
        print(f"  {name:25s} -> x = [{', '.join(f'{p:.2f}' for p in pts)}]")

    print()


# =========================================================================
# Example 4: Batch Selection Strategies
# =========================================================================


def example_batch_strategies():
    """
    Goal: select a *batch* of points that are collectively informative.

    When you select the top-k by acquisition score (greedy), the points
    can cluster in one region.  Batch strategies address this:

    - greedy: top-k by raw score.  Fast but may cluster.

    - penalized: after selecting the best candidate, nearby candidates
      are penalised before selecting the next.  Simple diversity.

    - kriging_believer: after selecting each point, the model is
      temporarily updated with a "fantasy" observation (y_hat) and
      re-scored.  More sophisticated -- later selections account for
      information gained by earlier ones.

    - d_optimal: selects the batch that maximises det(Phi^T Phi),
      ignoring the acquisition function entirely.  Best for pure
      space-filling / information maximisation.
    """
    print("=" * 60)
    print("Example 4: Batch Selection Strategies")
    print("=" * 60)

    model = make_model()
    bounds = [(0.0, 5.0)]

    learner = ActiveLearner(model, bounds, PredictionVariance(), random_state=42)

    for strategy in ["greedy", "penalized", "kriging_believer", "d_optimal"]:
        result = learner.suggest(n_points=5, batch_strategy=strategy)
        pts = sorted(np.array(result.points).ravel())
        spread = pts[-1] - pts[0]
        print(
            f"  {strategy:20s} -> x = [{', '.join(f'{p:.2f}' for p in pts)}]"
            f"  (spread={spread:.2f})"
        )

    print()


# =========================================================================
# Example 5: Full Active Learning Loop
# =========================================================================


def example_full_loop():
    """
    Goal: iteratively improve a model by running experiments.

    The workflow is:
    1. Fit an initial model on a small dataset.
    2. Use an acquisition function to suggest new points.
    3. "Run the experiment" (here: evaluate the true function + noise).
    4. Update the model with the new data.
    5. Repeat until converged or budget exhausted.
    """
    print("=" * 60)
    print("Example 5: Full Active Learning Loop")
    print("=" * 60)

    # True function (unknown to the model)
    def oracle(X):
        X = np.array(X)
        return X[:, 0] ** 2 - 3.0 * X[:, 0] + 2.0 + np.random.randn(len(X)) * 0.2

    # Start with very few points
    np.random.seed(0)
    X_init = np.random.uniform(0, 5, (15, 1))
    y_init = oracle(X_init)

    library = (
        BasisLibrary(n_features=1, feature_names=["x"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=3)
    )
    model = SymbolicRegressor(basis_library=library, max_terms=4, strategy="greedy_forward")
    model.fit(jnp.array(X_init), jnp.array(y_init))

    print(f"Initial model ({len(y_init)} points): {model.expression_}")
    print(f"  R^2 = {model.score(model._X_train, model._y_train):.4f}")
    print(f"  MSE = {model.metrics_['mse']:.4f}")

    # Active learning loop
    learner = ActiveLearner(
        model,
        bounds=[(0.0, 5.0)],
        acquisition=ExpectedImprovement(minimize=True),
        random_state=42,
    )

    n_iterations = 5
    points_per_iteration = 5

    for i in range(n_iterations):
        result = learner.suggest(
            n_points=points_per_iteration,
            batch_strategy="penalized",
        )

        y_new = oracle(np.array(result.points))
        learner.update(result.points, jnp.array(y_new))

        print(
            f"  Iteration {i + 1}: "
            f"n={learner.n_observations}, "
            f"R^2={model.score(model._X_train, model._y_train):.4f}, "
            f"MSE={model.metrics_['mse']:.4f}, "
            f"model={model.expression_}"
        )

    print(f"\nFinal model ({learner.n_observations} points): {model.expression_}")
    _diagnostics(model, model._X_train, model._y_train, "Active Learning Final")


# =========================================================================
# Example 6: Composite Acquisition Functions
# =========================================================================


def example_composite():
    """
    Goal: combine multiple objectives using weighted acquisition.

    You can weight and add acquisition functions together to balance
    different goals simultaneously.  Each component is min-max normalised
    before weighting so the weights are meaningful.

    Common recipes:
    - Balanced optimisation:  0.7 * EI + 0.3 * PredictionVariance
    - Exploration with model improvement:  0.5 * PredictionVariance + 0.5 * AOptimal
    - Multi-objective:  0.4 * ModelMin + 0.3 * PredictionVariance + 0.3 * DOptimal
    """
    print("=" * 60)
    print("Example 6: Composite Acquisition Functions")
    print("=" * 60)

    model = make_model()
    bounds = [(0.0, 5.0)]

    composites = [
        (
            "0.7*EI + 0.3*Variance",
            0.7 * ExpectedImprovement(minimize=True) + 0.3 * PredictionVariance(),
        ),
        (
            "0.5*LCB + 0.5*DOptimal",
            0.5 * LCB(kappa=2) + 0.5 * DOptimal(),
        ),
        (
            "Equal: EI + Var + AOptimal",
            ExpectedImprovement(minimize=True) + PredictionVariance() + AOptimal(),
        ),
    ]

    for name, acq in composites:
        result = suggest_points(model, bounds, acq, n_points=3, random_state=42)
        pts = np.array(result.points).ravel()
        print(f"  {name:30s} -> x = [{', '.join(f'{p:.2f}' for p in pts)}]")

    print()


# =========================================================================
# Decision Guide
# =========================================================================


def print_decision_guide():
    """Print a guide for choosing among acquisition functions."""
    print("=" * 60)
    print("Decision Guide: Choosing an Acquisition Function")
    print("=" * 60)

    guide = """
    WHAT IS YOUR GOAL?

    1. IMPROVE MODEL ACCURACY (explore everywhere)
       ├── Simple & fast?  -> PredictionVariance
       ├── Need coverage guarantee?  -> ConfidenceBandWidth(alpha=0.05)
       ├── Unsure about model form?  -> EnsembleDisagreement or BMAUncertainty
       ├── Tighten coefficient CIs?  -> AOptimal
       └── Max info per experiment?  -> DOptimal

    2. FIND THE OPTIMUM (minimise or maximise y)
       ├── Trust the model fully?  -> ModelMin / ModelMax
       ├── Want balanced exploration?  -> ExpectedImprovement (recommended)
       ├── Need probability of beating a threshold?  -> ProbabilityOfImprovement
       ├── Want explicit exploration knob?  -> LCB(kappa) / UCB(kappa)
       └── Want randomised exploration?  -> ThompsonSampling

    3. DECIDE WHICH MODEL IS CORRECT
       ├── Pareto models disagree?  -> ModelDiscrimination
       └── Quantify structural uncertainty?  -> EnsembleDisagreement

    4. MULTIPLE OBJECTIVES
       └── Combine with weights:  0.7 * EI + 0.3 * PredictionVariance

    BATCH STRATEGY SELECTION:
       ├── Fast, don't care about diversity?  -> greedy
       ├── Want spatial diversity?  -> penalized
       ├── Want information-aware batches?  -> kriging_believer
       └── Want maximum design efficiency?  -> d_optimal
    """
    print(guide)


# =========================================================================
# Main
# =========================================================================


def main():
    """Run all active learning examples."""
    print("JAXSR: Active Learning & Acquisition Functions")
    print("=" * 60)
    print()

    example_exploration()
    example_optimisation()
    example_discrimination()
    example_batch_strategies()
    example_full_loop()
    example_composite()
    print_decision_guide()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
