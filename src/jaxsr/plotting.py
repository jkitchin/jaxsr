"""
Visualization Tools for JAXSR.

Provides plotting functions for model analysis and results visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .regressor import SymbolicRegressor
    from .selection import SelectionResult
    from .uncertainty import BayesianModelAverage


def plot_pareto_front(
    pareto_front: List[SelectionResult],
    highlight_best: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_expressions: bool = True,
    max_label_length: int = 40,
) -> plt.Axes:
    """
    Plot the Pareto front of complexity vs accuracy.

    Parameters
    ----------
    pareto_front : list of SelectionResult
        Pareto-optimal models.
    highlight_best : bool
        Highlight the model with best BIC.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple
        Figure size if creating new figure.
    show_expressions : bool
        Show expression labels on points.
    max_label_length : int
        Maximum length for expression labels.

    Returns
    -------
    ax : plt.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if not pareto_front:
        ax.text(0.5, 0.5, "No Pareto front data", ha='center', va='center')
        return ax

    complexities = [r.complexity for r in pareto_front]
    mse_values = [r.mse for r in pareto_front]
    bic_values = [r.bic for r in pareto_front]

    # Find best by BIC
    best_idx = np.argmin(bic_values)

    # Plot all points
    ax.scatter(complexities, mse_values, s=100, c='steelblue', alpha=0.7, label='Models')

    # Connect points
    sorted_idx = np.argsort(complexities)
    ax.plot(
        [complexities[i] for i in sorted_idx],
        [mse_values[i] for i in sorted_idx],
        'b--', alpha=0.3
    )

    # Highlight best
    if highlight_best:
        ax.scatter(
            [complexities[best_idx]],
            [mse_values[best_idx]],
            s=200, c='red', marker='*', label='Best (BIC)', zorder=5
        )

    # Add labels
    if show_expressions:
        for i, result in enumerate(pareto_front):
            expr = result.expression()
            if len(expr) > max_label_length:
                expr = expr[:max_label_length-3] + "..."

            offset = (5, 5) if i % 2 == 0 else (5, -15)
            ax.annotate(
                expr,
                (complexities[i], mse_values[i]),
                xytext=offset,
                textcoords='offset points',
                fontsize=8,
                alpha=0.8,
            )

    ax.set_xlabel('Complexity')
    ax.set_ylabel('MSE')
    ax.set_title('Pareto Front: Complexity vs Accuracy')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    return ax


def plot_parity(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
    title: str = "Parity Plot",
    alpha: float = 0.6,
) -> plt.Axes:
    """
    Create a parity plot (predicted vs actual).

    Parameters
    ----------
    y_true : jnp.ndarray
        True values.
    y_pred : jnp.ndarray
        Predicted values.
    ax : plt.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    alpha : float
        Point transparency.

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Plot points
    ax.scatter(y_true, y_pred, alpha=alpha, c='steelblue', edgecolors='white', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    ax.plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        'k--', label='Perfect prediction'
    )

    # Add R² annotation
    ax.text(
        0.05, 0.95, f'R² = {r2:.4f}',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


def plot_residuals(
    model: SymbolicRegressor,
    X: jnp.ndarray,
    y: jnp.ndarray,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Create residual diagnostic plots.

    Creates three subplots:
    1. Residuals vs Predicted
    2. Residual histogram
    3. Q-Q plot

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model.
    X : jnp.ndarray
        Input data.
    y : jnp.ndarray
        True values.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
    """
    y_pred = model.predict(X)
    residuals = np.array(y - y_pred)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Residuals vs Predicted
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.6, c='steelblue')
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Predicted')
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1]
    ax.hist(residuals, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')

    # Overlay normal distribution
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    from scipy import stats
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')

    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[2]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_coefficient_path(
    model: SymbolicRegressor,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot coefficient values for selected terms.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    coefficients = np.array(model.coefficients_)
    names = model.selected_features_

    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(coefficients))[::-1]

    colors = ['steelblue' if c >= 0 else 'coral' for c in coefficients[sorted_idx]]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, coefficients[sorted_idx], color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in sorted_idx])
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Coefficient Magnitudes')
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_feature_importance(
    model: SymbolicRegressor,
    X: jnp.ndarray,
    y: jnp.ndarray,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot feature importance based on coefficient magnitudes and basis function values.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model.
    X : jnp.ndarray
        Input data.
    y : jnp.ndarray
        Target values.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get basis function evaluations
    Phi = model.basis_library.evaluate_subset(X, model.selected_indices_)
    coefficients = np.array(model.coefficients_)
    names = model.selected_features_

    # Contribution = |coefficient * std(basis function)|
    contributions = np.abs(coefficients) * np.std(np.array(Phi), axis=0)

    # Sort by contribution
    sorted_idx = np.argsort(contributions)[::-1]

    # Bar plot of contributions
    ax = axes[0]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, contributions[sorted_idx], color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in sorted_idx])
    ax.set_xlabel('Contribution (|coef| × std(basis))')
    ax.set_title('Term Importance')
    ax.grid(True, alpha=0.3, axis='x')

    # Cumulative contribution
    ax = axes[1]
    cumulative = np.cumsum(contributions[sorted_idx]) / np.sum(contributions)
    ax.plot(y_pos + 1, cumulative, 'o-', color='steelblue', linewidth=2)
    ax.axhline(y=0.95, color='r', linestyle='--', label='95%')
    ax.set_xlabel('Number of Terms')
    ax.set_ylabel('Cumulative Contribution')
    ax.set_title('Cumulative Importance')
    ax.set_xticks(y_pos + 1)
    ax.set_xticklabels([names[i] for i in sorted_idx], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_model_selection(
    selection_path,
    criterion: str = "bic",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot model selection criteria over the selection path.

    Parameters
    ----------
    selection_path : SelectionPath
        Selection path from model fitting.
    criterion : str
        Criterion to plot ("aic", "bic", "aicc", "mse").
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    results = selection_path.results
    n_terms = [r.n_terms for r in results]
    criterion_values = [getattr(r, criterion) for r in results]
    mse_values = [r.mse for r in results]

    # Criterion vs number of terms
    ax = axes[0]
    ax.plot(n_terms, criterion_values, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.axvline(
        x=n_terms[selection_path.best_index],
        color='r', linestyle='--',
        label=f'Selected ({n_terms[selection_path.best_index]} terms)'
    )
    ax.set_xlabel('Number of Terms')
    ax.set_ylabel(criterion.upper())
    ax.set_title(f'{criterion.upper()} vs Model Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MSE vs criterion (trade-off)
    ax = axes[1]
    scatter = ax.scatter(mse_values, criterion_values, c=n_terms, cmap='viridis', s=100)
    ax.scatter(
        [mse_values[selection_path.best_index]],
        [criterion_values[selection_path.best_index]],
        c='red', s=200, marker='*', label='Selected'
    )
    plt.colorbar(scatter, ax=ax, label='Number of Terms')
    ax.set_xlabel('MSE')
    ax.set_ylabel(criterion.upper())
    ax.set_title('MSE vs Information Criterion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_prediction_surface(
    model: SymbolicRegressor,
    bounds: List[Tuple[float, float]],
    fixed_values: Optional[Dict[str, float]] = None,
    n_points: int = 50,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Axes:
    """
    Plot 2D prediction surface for a model with 2 varying features.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model.
    bounds : list of tuple
        Bounds for the two varying features.
    fixed_values : dict, optional
        Fixed values for other features.
    n_points : int
        Number of points per dimension.
    ax : plt.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.

    Returns
    -------
    ax : plt.Axes
    """
    from mpl_toolkits.mplot3d import Axes3D

    if len(bounds) != 2:
        raise ValueError("Exactly 2 bounds must be specified for surface plot")

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    feature_names = model.basis_library.feature_names
    n_features = len(feature_names)

    # Create grid
    x1 = np.linspace(bounds[0][0], bounds[0][1], n_points)
    x2 = np.linspace(bounds[1][0], bounds[1][1], n_points)
    X1, X2 = np.meshgrid(x1, x2)

    # Create full feature matrix
    X_grid = np.zeros((n_points * n_points, n_features))

    # Fill in fixed values
    if fixed_values:
        for name, value in fixed_values.items():
            if name in feature_names:
                idx = feature_names.index(name)
                X_grid[:, idx] = value

    # Fill in varying features (first two by default)
    X_grid[:, 0] = X1.ravel()
    X_grid[:, 1] = X2.ravel()

    # Predict
    Y = model.predict(jnp.array(X_grid)).reshape(n_points, n_points)

    # Plot surface
    surf = ax.plot_surface(X1, X2, np.array(Y), cmap='viridis', alpha=0.8)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel('y')
    ax.set_title(f'Prediction Surface\n{model.expression_}')

    return ax


def plot_comparison(
    models: List[SymbolicRegressor],
    X: jnp.ndarray,
    y: jnp.ndarray,
    names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Compare multiple models visually.

    Parameters
    ----------
    models : list of SymbolicRegressor
        Models to compare.
    X : jnp.ndarray
        Test data.
    y : jnp.ndarray
        True values.
    names : list of str, optional
        Model names.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
    """
    if names is None:
        names = [f"Model {i+1}" for i in range(len(models))]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Compute metrics
    metrics = []
    for model in models:
        y_pred = model.predict(X)
        mse = float(jnp.mean((y - y_pred) ** 2))
        r2 = model.score(X, y)
        complexity = model.complexity_
        metrics.append({'mse': mse, 'r2': r2, 'complexity': complexity})

    # Bar plot of MSE
    ax = axes[0]
    x_pos = np.arange(len(models))
    ax.bar(x_pos, [m['mse'] for m in metrics], color='steelblue', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error')
    ax.grid(True, alpha=0.3, axis='y')

    # Bar plot of R²
    ax = axes[1]
    ax.bar(x_pos, [m['r2'] for m in metrics], color='seagreen', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('R²')
    ax.set_title('R² Score')
    ax.grid(True, alpha=0.3, axis='y')

    # Scatter: MSE vs Complexity
    ax = axes[2]
    for i, (model, name) in enumerate(zip(models, names)):
        ax.scatter(
            metrics[i]['complexity'],
            metrics[i]['mse'],
            s=150,
            label=name
        )
    ax.set_xlabel('Complexity')
    ax.set_ylabel('MSE')
    ax.set_title('Accuracy vs Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_learning_curve(
    model: SymbolicRegressor,
    X: jnp.ndarray,
    y: jnp.ndarray,
    train_sizes: Optional[List[float]] = None,
    cv: int = 5,
    random_state: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot learning curve showing model performance vs training set size.

    Parameters
    ----------
    model : SymbolicRegressor
        Model to evaluate.
    X : jnp.ndarray
        Full feature data.
    y : jnp.ndarray
        Full target data.
    train_sizes : list of float, optional
        Fractions of training data to use.
    cv : int
        Number of cross-validation folds.
    random_state : int, optional
        Random seed.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
    """
    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rng = np.random.RandomState(random_state)
    n_samples = len(y)

    train_scores_mean = []
    train_scores_std = []
    test_scores_mean = []
    test_scores_std = []

    for train_size in train_sizes:
        train_mse_list = []
        test_mse_list = []

        for _ in range(cv):
            # Split data
            indices = rng.permutation(n_samples)
            n_train = int(train_size * n_samples * 0.8)  # 80% for train in each fold
            n_test = int(n_samples * 0.2)

            train_idx = indices[:n_train]
            test_idx = indices[n_train:n_train + n_test]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Fit model
            model_clone = SymbolicRegressor(
                basis_library=model.basis_library,
                max_terms=model.max_terms,
                strategy=model.strategy,
                information_criterion=model.information_criterion,
            )
            model_clone.fit(X_train, y_train)

            # Compute MSE
            train_mse = float(jnp.mean((y_train - model_clone.predict(X_train)) ** 2))
            test_mse = float(jnp.mean((y_test - model_clone.predict(X_test)) ** 2))

            train_mse_list.append(train_mse)
            test_mse_list.append(test_mse)

        train_scores_mean.append(np.mean(train_mse_list))
        train_scores_std.append(np.std(train_mse_list))
        test_scores_mean.append(np.mean(test_mse_list))
        test_scores_std.append(np.std(test_mse_list))

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    train_sizes_abs = [int(t * n_samples * 0.8) for t in train_sizes]

    ax.fill_between(
        train_sizes_abs,
        np.array(train_scores_mean) - np.array(train_scores_std),
        np.array(train_scores_mean) + np.array(train_scores_std),
        alpha=0.2, color='blue'
    )
    ax.fill_between(
        train_sizes_abs,
        np.array(test_scores_mean) - np.array(test_scores_std),
        np.array(test_scores_mean) + np.array(test_scores_std),
        alpha=0.2, color='orange'
    )

    ax.plot(train_sizes_abs, train_scores_mean, 'o-', color='blue', label='Train MSE')
    ax.plot(train_sizes_abs, test_scores_mean, 'o-', color='orange', label='Test MSE')

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('MSE')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


# =============================================================================
# Uncertainty Quantification Plots
# =============================================================================


def plot_prediction_intervals(
    model: SymbolicRegressor,
    X: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    alpha: float = 0.05,
    sort_by: int = 0,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Axes:
    """
    Fan chart: inner band = confidence on E[y|x], outer band = prediction interval.

    For 1D data (single feature), plots against the feature. For multi-feature
    data, sorts by the specified feature index.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model.
    X : jnp.ndarray
        Input data for plotting.
    y : jnp.ndarray, optional
        Observed values to overlay.
    alpha : float
        Significance level (default 0.05 for 95% intervals).
    sort_by : int
        Feature index to sort/plot against on x-axis.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    ax : plt.Axes
    """
    from .uncertainty import prediction_interval

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    X = jnp.atleast_2d(jnp.asarray(X))
    x_vals = np.array(X[:, sort_by])
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]

    Phi_train = model._get_Phi_train()
    Phi_new = model.basis_library.evaluate_subset(X, model._result.selected_indices)

    result = prediction_interval(
        Phi_train, model._y_train, model._result.coefficients,
        Phi_new, alpha,
    )

    y_pred = np.array(result["y_pred"])[sort_idx]
    pred_lower = np.array(result["pred_lower"])[sort_idx]
    pred_upper = np.array(result["pred_upper"])[sort_idx]
    conf_lower = np.array(result["conf_lower"])[sort_idx]
    conf_upper = np.array(result["conf_upper"])[sort_idx]

    # Outer band: prediction interval
    ax.fill_between(
        x_sorted, pred_lower, pred_upper,
        alpha=0.15, color='steelblue', label=f'{int((1-alpha)*100)}% Prediction'
    )
    # Inner band: confidence band
    ax.fill_between(
        x_sorted, conf_lower, conf_upper,
        alpha=0.3, color='steelblue', label=f'{int((1-alpha)*100)}% Confidence'
    )
    # Mean prediction
    ax.plot(x_sorted, y_pred, '-', color='steelblue', linewidth=2, label='Prediction')

    # Observed data
    if y is not None:
        y_arr = np.array(y)
        ax.scatter(
            x_vals[sort_idx], y_arr[sort_idx],
            c='black', s=20, alpha=0.6, zorder=5, label='Observed'
        )

    feature_name = model.basis_library.feature_names[sort_by]
    ax.set_xlabel(feature_name)
    ax.set_ylabel('y')
    ax.set_title('Prediction Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_coefficient_intervals(
    model: SymbolicRegressor,
    alpha: float = 0.05,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Axes:
    """
    Forest plot: horizontal error bars for each coefficient CI, vertical line at 0.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model.
    alpha : float
        Significance level.
    ax : plt.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    intervals = model.coefficient_intervals(alpha)
    names = list(intervals.keys())
    estimates = [intervals[n][0] for n in names]
    lowers = [intervals[n][1] for n in names]
    uppers = [intervals[n][2] for n in names]

    y_pos = np.arange(len(names))
    errors = np.array([[est - lo, hi - est] for est, lo, hi in zip(estimates, lowers, uppers)]).T

    ax.errorbar(
        estimates, y_pos, xerr=errors,
        fmt='o', color='steelblue', ecolor='steelblue',
        elinewidth=2, capsize=4, markersize=6,
    )
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Coefficient Value')
    ax.set_title(f'{int((1-alpha)*100)}% Coefficient Confidence Intervals')
    ax.grid(True, alpha=0.3, axis='x')

    return ax


def plot_bma_weights(
    model: SymbolicRegressor,
    criterion: str = "bic",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 5),
    max_label_length: int = 50,
) -> plt.Axes:
    """
    Horizontal bar chart of BMA model weights with expression labels.

    Parameters
    ----------
    model : SymbolicRegressor
        Fitted model.
    criterion : str
        IC for computing weights.
    ax : plt.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.
    max_label_length : int
        Maximum expression label length.

    Returns
    -------
    ax : plt.Axes
    """
    from .uncertainty import BayesianModelAverage

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    bma = BayesianModelAverage(model, criterion=criterion)
    weights = bma.weights

    # Sort by weight
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    labels = []
    values = []
    for expr, w in sorted_items:
        label = expr if len(expr) <= max_label_length else expr[:max_label_length-3] + "..."
        labels.append(label)
        values.append(w)

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('BMA Weight')
    ax.set_title(f'Bayesian Model Average Weights ({criterion.upper()})')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    return ax
