"""
SINDy-style ODE Discovery for JAXSR.

Provides tools for discovering governing ordinary differential equations
from time-series data using sparse regression over nonlinear function libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .basis import BasisLibrary
from .metrics import (
    compute_aic,
    compute_aicc,
    compute_bic,
    compute_mse,
    compute_r2,
)
from .regressor import SymbolicRegressor


def estimate_derivatives(
    X: np.ndarray,
    t: np.ndarray,
    method: str = "finite_difference",
    **kwargs: Any,
) -> np.ndarray:
    """
    Estimate time derivatives dX/dt from time-series state data.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples,) or (n_samples, n_states)
        State variable measurements over time.
    t : np.ndarray of shape (n_samples,)
        Time values corresponding to each row of X.
    method : str
        Differentiation method: ``"finite_difference"``, ``"savgol"``, or
        ``"spline"``.
    **kwargs
        Extra keyword arguments forwarded to the chosen method:

        - ``"savgol"``: ``window_length`` (int, default 5),
          ``polyorder`` (int, default 3).
        - ``"spline"``: ``smooth`` (float, default 0.0).

    Returns
    -------
    dXdt : np.ndarray
        Estimated derivatives, same shape as *X*.

    Raises
    ------
    ValueError
        If *method* is unknown, shapes are inconsistent, *t* is not
        monotonically increasing, or savgol is used with non-uniform spacing.
    """
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got {X.ndim}-D")

    if t.shape[0] != X.shape[0]:
        raise ValueError(
            f"Length of t ({t.shape[0]}) must match number of rows in X ({X.shape[0]})"
        )

    if t.shape[0] < 2:
        raise ValueError("Need at least 2 time points")

    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("t must be strictly monotonically increasing")

    valid_methods = {"finite_difference", "savgol", "spline"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method {method!r}. Choose from {sorted(valid_methods)}")

    if method == "finite_difference":
        dXdt = np.gradient(X, t, axis=0)

    elif method == "savgol":
        # Savitzky–Golay requires uniform spacing
        dt_range = dt.max() - dt.min()
        if dt_range > 1e-10 * dt.mean():
            raise ValueError(
                "savgol method requires uniformly spaced time points. "
                f"Spacing varies by {dt_range:.2e}"
            )
        from scipy.signal import savgol_filter

        window_length = kwargs.get("window_length", 5)
        polyorder = kwargs.get("polyorder", 3)
        delta = float(dt[0])
        dXdt = savgol_filter(X, window_length, polyorder, deriv=1, delta=delta, axis=0)

    elif method == "spline":
        from scipy.interpolate import UnivariateSpline

        smooth = kwargs.get("smooth", 0.0)
        dXdt = np.empty_like(X)
        for j in range(X.shape[1]):
            spl = UnivariateSpline(t, X[:, j], s=smooth, k=4)
            dXdt[:, j] = spl.derivative()(t)

    return dXdt


@dataclass
class DynamicsResult:
    """
    Result container for ODE discovery.

    Attributes
    ----------
    models : dict[str, SymbolicRegressor]
        Fitted symbolic regressor for each state variable.
    equations : dict[str, str]
        Human-readable ODE for each state variable,
        e.g. ``{"x": "d(x)/dt = 1.0*x - 0.5*x*y"}``.
    derivatives : np.ndarray
        The estimated dX/dt array used for fitting.
    state_names : list[str]
        Names of the state variables.
    metrics : dict[str, dict[str, float]]
        Per-variable metrics (mse, r2, aic, bic, aicc).
    """

    models: dict[str, SymbolicRegressor]
    equations: dict[str, str]
    derivatives: np.ndarray
    state_names: list[str]
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        """
        Return a human-readable multi-line summary of the discovered ODEs.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = ["Discovered ODEs", "=" * 40]
        for name in self.state_names:
            lines.append(self.equations[name])
            if name in self.metrics:
                m = self.metrics[name]
                lines.append(
                    f"  R² = {m['r2']:.4f}, MSE = {m['mse']:.2e}, "
                    f"AIC = {m['aic']:.1f}, BIC = {m['bic']:.1f}"
                )
        return "\n".join(lines)


def discover_dynamics(
    X: np.ndarray,
    t: np.ndarray,
    state_names: list[str] | None = None,
    basis_library: BasisLibrary | None = None,
    max_terms: int = 5,
    strategy: str = "greedy_forward",
    information_criterion: str = "bic",
    derivative_method: str = "finite_difference",
    derivative_kw: dict[str, Any] | None = None,
) -> DynamicsResult:
    """
    Discover governing ODEs from time-series state data.

    Estimates derivatives, builds a basis library (if not provided), then
    fits one :class:`SymbolicRegressor` per state variable using the
    specified selection strategy.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples,) or (n_samples, n_states)
        State variable measurements over time.
    t : np.ndarray of shape (n_samples,)
        Time values corresponding to each row of X.
    state_names : list[str], optional
        Names for each state variable. Defaults to ``["x0", "x1", ...]``.
    basis_library : BasisLibrary, optional
        Candidate function library. If ``None``, a default library is built
        with constant, linear, polynomial (degree 2-3), and interaction terms.
    max_terms : int
        Maximum number of terms per ODE.
    strategy : str
        Selection strategy forwarded to :class:`SymbolicRegressor`.
    information_criterion : str
        Information criterion forwarded to :class:`SymbolicRegressor`.
    derivative_method : str
        Method for estimating derivatives (see :func:`estimate_derivatives`).
    derivative_kw : dict, optional
        Extra keyword arguments for :func:`estimate_derivatives`.

    Returns
    -------
    DynamicsResult
        Container with fitted models, equations, derivatives, and metrics.

    Raises
    ------
    ValueError
        If *basis_library* has the wrong number of features, or if inputs
        are inconsistent.
    """
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_states = X.shape[1]

    if state_names is None:
        state_names = [f"x{i}" for i in range(n_states)]

    if len(state_names) != n_states:
        raise ValueError(
            f"Length of state_names ({len(state_names)}) must match "
            f"number of state variables ({n_states})"
        )

    # Estimate derivatives
    dXdt = estimate_derivatives(X, t, method=derivative_method, **(derivative_kw or {}))

    # Build default basis library if not provided
    if basis_library is None:
        basis_library = (
            BasisLibrary(n_features=n_states, feature_names=state_names)
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=3)
            .add_interactions(max_order=2)
        )
    else:
        if basis_library.n_features != n_states:
            raise ValueError(
                f"basis_library.n_features ({basis_library.n_features}) must match "
                f"number of state variables ({n_states})"
            )

    # Fit one regressor per state variable
    models: dict[str, SymbolicRegressor] = {}
    equations: dict[str, str] = {}
    metrics: dict[str, dict[str, float]] = {}

    for i, name in enumerate(state_names):
        y_i = dXdt[:, i]

        reg = SymbolicRegressor(
            basis_library=basis_library,
            max_terms=max_terms,
            strategy=strategy,
            information_criterion=information_criterion,
        )
        reg.fit(X, y_i)
        models[name] = reg

        expr = reg.expression_
        equations[name] = f"d({name})/dt = {expr}"

        # Compute metrics
        y_pred = np.asarray(reg.predict(X))
        n = len(y_i)
        n_params = len(reg.selected_features_)
        mse = float(compute_mse(y_i, y_pred))
        metrics[name] = {
            "mse": mse,
            "r2": float(compute_r2(y_i, y_pred)),
            "aic": float(compute_aic(n, n_params, mse)),
            "bic": float(compute_bic(n, n_params, mse)),
            "aicc": float(compute_aicc(n, n_params, mse)),
        }

    return DynamicsResult(
        models=models,
        equations=equations,
        derivatives=dXdt,
        state_names=state_names,
        metrics=metrics,
    )
