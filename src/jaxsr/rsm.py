"""
Response Surface Methodology (RSM) for JAXSR.

Provides classical DOE design generators, coded-variable handling,
and surface geometry analysis (canonical form, stationary point).

The :class:`ResponseSurface` convenience class wraps
:class:`~jaxsr.regressor.SymbolicRegressor` and layers RSM-specific
functionality on top — including design generation, ANOVA, canonical
analysis, and contour plotting — while still allowing jaxsr to discover
non-polynomial models when the data support it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from .basis import BasisLibrary
from .regressor import SymbolicRegressor
from .uncertainty import anova

# =========================================================================
# Design Generators
# =========================================================================


def factorial_design(
    levels: int | list[int],
    n_factors: int | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Generate a full factorial design.

    Parameters
    ----------
    levels : int or list of int
        Number of levels per factor.  If a single int, all factors use
        the same number of levels.
    n_factors : int, optional
        Number of factors.  Required when *levels* is a scalar.
    bounds : list of (float, float), optional
        If provided, the design is scaled from coded to natural units.

    Returns
    -------
    X : np.ndarray
        Design matrix of shape ``(n_runs, n_factors)``.

    Examples
    --------
    >>> factorial_design(levels=3, n_factors=2)
    array([[-1., -1.],
           [-1.,  0.],
           ...])
    """
    if isinstance(levels, int):
        if n_factors is None:
            raise ValueError("n_factors is required when levels is an int")
        levels_per = [levels] * n_factors
    else:
        levels_per = list(levels)
        if n_factors is not None and len(levels_per) != n_factors:
            raise ValueError(f"Length of levels ({len(levels_per)}) != n_factors ({n_factors})")
        n_factors = len(levels_per)

    axes = [np.linspace(-1, 1, lv) for lv in levels_per]
    grid = np.array(list(itertools.product(*axes)))

    if bounds is not None:
        grid = _decode(grid, bounds)
    return grid


def fractional_factorial_design(
    n_factors: int,
    resolution: int = 3,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Generate a 2-level fractional factorial design.

    Uses a Hadamard-like construction.  For *k* factors at resolution III
    the design has 2^p runs where p is chosen so that 2^p >= k + 1.

    Parameters
    ----------
    n_factors : int
        Number of factors.
    resolution : int
        Minimum resolution (III, IV, or V).  Higher resolution avoids
        confounding main effects with low-order interactions.
    bounds : list of (float, float), optional
        If provided, the design is scaled from coded to natural units.

    Returns
    -------
    X : np.ndarray
        Design matrix of shape ``(n_runs, n_factors)`` with entries in
        {-1, +1} (or natural units if *bounds* given).
    """
    # Determine minimum number of base columns
    if resolution <= 3:
        # Resolution III: main effects not aliased with each other
        p = int(np.ceil(np.log2(n_factors + 1)))
    elif resolution <= 4:
        # Resolution IV: main effects not aliased with 2fi
        p = max(int(np.ceil(np.log2(n_factors + 1))), n_factors - 1)
        p = min(p, n_factors)  # can't exceed full factorial
    else:
        p = n_factors  # full factorial

    # Build the base 2^p full factorial in {-1, +1}
    base = np.array(list(itertools.product([-1, 1], repeat=p)))

    if p >= n_factors:
        X = base[:, :n_factors]
    else:
        # Extra columns = products of base columns
        X = base.copy()
        extra_needed = n_factors - p
        # Generate interactions of increasing order
        generated = []
        for order in range(2, p + 1):
            for combo in itertools.combinations(range(p), order):
                col = np.prod(base[:, list(combo)], axis=1, keepdims=True)
                generated.append(col)
                if len(generated) >= extra_needed:
                    break
            if len(generated) >= extra_needed:
                break
        X = np.hstack([base] + generated[:extra_needed])

    if bounds is not None:
        X = _decode(X, bounds)
    return X


def central_composite_design(
    n_factors: int,
    alpha: str | float = "rotatable",
    center_points: int = 1,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Generate a Central Composite Design (CCD).

    A CCD consists of:

    * A 2^k factorial cube (corners at +/-1).
    * 2k axial (star) points at +/-alpha along each axis.
    * Center point(s) at the origin.

    Parameters
    ----------
    n_factors : int
        Number of factors *k*.
    alpha : str or float
        Axial distance.  Special values:

        * ``"rotatable"`` — alpha = (2^k)^{1/4} for rotatability.
        * ``"face"`` — alpha = 1 (face-centered, all points within the cube).
        * A float for a custom distance.
    center_points : int
        Number of center-point replicates.
    bounds : list of (float, float), optional
        If provided, the design is scaled from coded to natural units.

    Returns
    -------
    X : np.ndarray
        Design matrix of shape ``(n_runs, n_factors)``.

    Examples
    --------
    >>> ccd = central_composite_design(3, alpha="face", center_points=3)
    >>> ccd.shape
    (17, 3)
    """
    k = n_factors

    if isinstance(alpha, str):
        if alpha == "rotatable":
            alpha_val = float((2**k) ** 0.25)
        elif alpha == "face":
            alpha_val = 1.0
        else:
            raise ValueError(f"Unknown alpha preset: {alpha!r}")
    else:
        alpha_val = float(alpha)

    # Factorial cube: 2^k points at {-1, +1}^k
    cube = np.array(list(itertools.product([-1, 1], repeat=k)), dtype=float)

    # Axial (star) points: +/-alpha along each axis
    axial = np.zeros((2 * k, k))
    for i in range(k):
        axial[2 * i, i] = alpha_val
        axial[2 * i + 1, i] = -alpha_val

    # Center points
    center = np.zeros((center_points, k))

    X = np.vstack([cube, axial, center])

    if bounds is not None:
        X = _decode(X, bounds)
    return X


def box_behnken_design(
    n_factors: int,
    center_points: int = 1,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Generate a Box-Behnken design.

    A Box-Behnken design avoids extreme corners (no point where all
    factors are simultaneously at their high or low levels), making it
    useful when those combinations are infeasible.

    Parameters
    ----------
    n_factors : int
        Number of factors (must be >= 3).
    center_points : int
        Number of center-point replicates.
    bounds : list of (float, float), optional
        If provided, the design is scaled from coded to natural units.

    Returns
    -------
    X : np.ndarray
        Design matrix of shape ``(n_runs, n_factors)``.

    Raises
    ------
    ValueError
        If ``n_factors < 3``.
    """
    if n_factors < 3:
        raise ValueError("Box-Behnken design requires at least 3 factors")

    rows = []
    # For each pair of factors: run a 2^2 factorial while others are at 0
    for i, j in itertools.combinations(range(n_factors), 2):
        for si in [-1, 1]:
            for sj in [-1, 1]:
                row = np.zeros(n_factors)
                row[i] = si
                row[j] = sj
                rows.append(row)

    # Center points
    for _ in range(center_points):
        rows.append(np.zeros(n_factors))

    X = np.array(rows)

    if bounds is not None:
        X = _decode(X, bounds)
    return X


# =========================================================================
# Variable Coding
# =========================================================================


def _decode(X_coded: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    """Map coded variables in [-1, +1] to natural units."""
    bounds_arr = np.asarray(bounds)
    centers = (bounds_arr[:, 1] + bounds_arr[:, 0]) / 2.0
    half_ranges = (bounds_arr[:, 1] - bounds_arr[:, 0]) / 2.0
    return X_coded * half_ranges + centers


def _encode(X_natural: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    """Map natural units to coded variables in [-1, +1]."""
    bounds_arr = np.asarray(bounds)
    centers = (bounds_arr[:, 1] + bounds_arr[:, 0]) / 2.0
    half_ranges = (bounds_arr[:, 1] - bounds_arr[:, 0]) / 2.0
    half_ranges = np.where(half_ranges == 0, 1.0, half_ranges)
    return (X_natural - centers) / half_ranges


def encode(
    X: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    """Convert natural-unit variables to coded [-1, +1] variables.

    Parameters
    ----------
    X : array-like of shape (n, k)
        Design points in natural units.
    bounds : list of (low, high)
        Natural-unit bounds for each factor.

    Returns
    -------
    X_coded : np.ndarray
        Coded design matrix.
    """
    return _encode(np.asarray(X), bounds)


def decode(
    X_coded: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    """Convert coded [-1, +1] variables to natural units.

    Parameters
    ----------
    X_coded : array-like of shape (n, k)
        Design points in coded units.
    bounds : list of (low, high)
        Natural-unit bounds for each factor.

    Returns
    -------
    X : np.ndarray
        Design matrix in natural units.
    """
    return _decode(np.asarray(X_coded), bounds)


# =========================================================================
# Canonical Analysis
# =========================================================================


@dataclass
class CanonicalAnalysis:
    """Result of canonical analysis on a fitted quadratic surface.

    Parameters
    ----------
    stationary_point : np.ndarray
        Location of the stationary point (coded or natural units).
    stationary_response : float
        Predicted response at the stationary point.
    eigenvalues : np.ndarray
        Eigenvalues of the B matrix (second-order coefficient matrix).
    eigenvectors : np.ndarray
        Corresponding eigenvectors (columns).
    nature : str
        Classification: ``"minimum"``, ``"maximum"``, or ``"saddle"``.
    b_vector : np.ndarray
        First-order coefficient vector.
    B_matrix : np.ndarray
        Second-order coefficient matrix (symmetric).
    warnings : list of str
        Diagnostic messages.
    """

    stationary_point: np.ndarray
    stationary_response: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    nature: str
    b_vector: np.ndarray
    B_matrix: np.ndarray
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # noqa: D105
        lines = [
            "Canonical Analysis",
            "-" * 40,
            f"Nature:              {self.nature}",
            f"Stationary point:    {np.array2string(self.stationary_point, precision=4)}",
            f"Predicted response:  {self.stationary_response:.4f}",
            f"Eigenvalues:         {np.array2string(self.eigenvalues, precision=4)}",
        ]
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def canonical_analysis(
    model: SymbolicRegressor,
    bounds: list[tuple[float, float]] | None = None,
) -> CanonicalAnalysis:
    """Perform canonical analysis on a fitted quadratic model.

    Extracts the first-order vector **b** and second-order matrix **B**
    from the model's expression, computes the stationary point
    ``x_s = -0.5 B^{-1} b``, and classifies the surface via the
    eigenvalues of **B**.

    The analysis works directly with whatever basis terms the model
    selected.  If the model is not purely quadratic (e.g. it includes
    ``log(x)`` terms), a warning is emitted and only the quadratic
    portion is analysed.

    Parameters
    ----------
    model : SymbolicRegressor
        A fitted model.
    bounds : list of (float, float), optional
        Factor bounds.  If provided, the stationary point is reported
        in natural units; otherwise in the model's native units.

    Returns
    -------
    CanonicalAnalysis
    """
    model._check_is_fitted()
    k = model.basis_library.n_features
    names = list(model._result.selected_names)
    coeffs = np.array(model._result.coefficients)
    feature_names = model.basis_library.feature_names
    warn: list[str] = []

    # --- Extract b (linear) and B (quadratic) ---
    b = np.zeros(k)
    B = np.zeros((k, k))
    intercept = 0.0
    unrecognised = []

    for name, coef in zip(names, coeffs, strict=False):
        coef = float(coef)
        if name == "1":
            intercept = coef
            continue

        # Linear term: "x0" or feature name
        if name in feature_names:
            idx = feature_names.index(name)
            b[idx] = coef
            continue

        # Quadratic term: "x0^2"
        matched_quad = False
        for i, fn in enumerate(feature_names):
            if name == f"{fn}^2":
                B[i, i] = coef
                matched_quad = True
                break
        if matched_quad:
            continue

        # Interaction: "x0*x1"
        matched_inter = False
        for i, fi in enumerate(feature_names):
            for j, fj in enumerate(feature_names):
                if i < j and name == f"{fi}*{fj}":
                    # Convention: B_ij = B_ji = coef / 2
                    B[i, j] = coef / 2.0
                    B[j, i] = coef / 2.0
                    matched_inter = True
                    break
            if matched_inter:
                break
        if matched_inter:
            continue

        # Anything else is not part of the quadratic model
        unrecognised.append(name)

    if unrecognised:
        warn.append(
            f"Non-quadratic terms ignored in canonical analysis: "
            f"{', '.join(unrecognised)}. Only the quadratic portion "
            f"is analysed."
        )

    # Check that we have at least some quadratic terms
    if np.allclose(B, 0):
        warn.append(
            "No quadratic or interaction terms found. "
            "Canonical analysis requires a second-order model."
        )
        return CanonicalAnalysis(
            stationary_point=np.full(k, np.nan),
            stationary_response=np.nan,
            eigenvalues=np.zeros(k),
            eigenvectors=np.eye(k),
            nature="linear (no curvature)",
            b_vector=b,
            B_matrix=B,
            warnings=warn,
        )

    # --- Stationary point: x_s = -0.5 * B^{-1} b ---
    try:
        x_s = -0.5 * np.linalg.solve(B, b)
    except np.linalg.LinAlgError:
        warn.append("B matrix is singular — stationary point does not exist.")
        x_s = np.full(k, np.nan)

    # Predicted response at stationary point
    y_s = intercept + float(b @ x_s + x_s @ B @ x_s) if np.all(np.isfinite(x_s)) else np.nan

    # --- Eigenanalysis ---
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Classify
    if np.all(eigenvalues > 0):
        nature = "minimum"
    elif np.all(eigenvalues < 0):
        nature = "maximum"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        nature = "saddle"
    else:
        nature = "stationary ridge"

    # Convert stationary point to natural units if bounds given
    if bounds is not None and np.all(np.isfinite(x_s)):
        x_s = _decode(x_s.reshape(1, -1), bounds).ravel()

    return CanonicalAnalysis(
        stationary_point=x_s,
        stationary_response=float(y_s),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        nature=nature,
        b_vector=b,
        B_matrix=B,
        warnings=warn,
    )


# =========================================================================
# ResponseSurface Convenience Class
# =========================================================================


class ResponseSurface:
    """Convenience class combining DOE design, fitting, and RSM analysis.

    Wraps :class:`~jaxsr.regressor.SymbolicRegressor` and adds:

    * Design generation (CCD, Box-Behnken, factorial).
    * Coded / natural variable bookkeeping.
    * ANOVA on the fitted model.
    * Canonical analysis (stationary point, eigenvalues, classification).
    * Contour plotting.

    The model is fitted using jaxsr's symbolic regression, so it may
    discover that a simpler or non-polynomial expression is more
    parsimonious.  Canonical analysis gracefully handles this by
    warning and analysing only the quadratic portion.

    Parameters
    ----------
    n_factors : int
        Number of input factors.
    bounds : list of (float, float)
        Natural-unit bounds for each factor.
    factor_names : list of str, optional
        Human-readable factor names.
    max_degree : int
        Maximum polynomial degree for the basis library (default 2
        for classical RSM; set to 3 if you want cubic terms too).
    include_interactions : bool
        Include interaction terms (default True).
    max_terms : int, optional
        Maximum number of terms the regressor may select.
    strategy : str
        Feature-selection strategy (default ``"greedy_forward"``).
    allow_transcendental : bool
        If True, also add ``["log", "exp", "sqrt", "inv"]`` to the
        basis library so jaxsr can discover non-polynomial models.

    Examples
    --------
    >>> rs = ResponseSurface(
    ...     n_factors=3,
    ...     bounds=[(300, 500), (1, 10), (0.01, 0.5)],
    ...     factor_names=["T", "P", "C"],
    ... )
    >>> X = rs.ccd(center_points=3)
    >>> y = run_experiments(X)
    >>> rs.fit(X, y)
    >>> print(rs.model.expression_)
    >>> print(rs.anova())
    >>> print(rs.canonical())
    """

    def __init__(
        self,
        n_factors: int,
        bounds: list[tuple[float, float]],
        factor_names: list[str] | None = None,
        max_degree: int = 2,
        include_interactions: bool = True,
        max_terms: int | None = None,
        strategy: str = "greedy_forward",
        allow_transcendental: bool = False,
        feature_types: list[str] | None = None,
        categories: dict[int, list] | None = None,
    ):
        self.n_factors = n_factors
        self.bounds = list(bounds)
        self.factor_names = factor_names or [f"x{i}" for i in range(n_factors)]
        self.max_degree = max_degree
        self.strategy = strategy
        self.feature_types = feature_types
        self.categories = categories

        if len(self.bounds) != n_factors:
            raise ValueError(
                f"Length of bounds ({len(self.bounds)}) must match " f"n_factors ({n_factors})"
            )

        # Build basis library
        library = (
            BasisLibrary(
                n_features=n_factors,
                feature_names=self.factor_names,
                feature_types=feature_types,
                categories=categories,
            )
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=max_degree)
        )
        if include_interactions:
            library.add_interactions(max_order=min(max_degree, n_factors))
        if allow_transcendental:
            library.add_transcendental(["log", "exp", "sqrt", "inv"])

        # Add categorical basis functions if any categorical features exist
        if library.categorical_indices:
            library.add_categorical_indicators()
            if library.continuous_indices:
                library.add_categorical_interactions()

        if max_terms is None:
            # Default: enough for a full quadratic + some headroom
            # Full quadratic = 1 + k + k*(k-1)/2 + k = 1 + 2k + C(k,2)
            max_terms = min(len(library), 1 + 2 * n_factors + n_factors * (n_factors - 1) // 2 + 2)

        self.model = SymbolicRegressor(
            basis_library=library,
            max_terms=max_terms,
            strategy=strategy,
        )

    # -- Design generators ------------------------------------------------

    def ccd(
        self,
        alpha: str | float = "rotatable",
        center_points: int = 1,
    ) -> np.ndarray:
        """Generate a Central Composite Design in natural units.

        Parameters
        ----------
        alpha : str or float
            ``"rotatable"``, ``"face"``, or a custom float.
        center_points : int
            Number of center-point replicates.

        Returns
        -------
        X : np.ndarray
            Design matrix in natural units.
        """
        return central_composite_design(
            self.n_factors,
            alpha=alpha,
            center_points=center_points,
            bounds=self.bounds,
        )

    def box_behnken(self, center_points: int = 1) -> np.ndarray:
        """Generate a Box-Behnken design in natural units.

        Parameters
        ----------
        center_points : int
            Number of center-point replicates.

        Returns
        -------
        X : np.ndarray
        """
        return box_behnken_design(
            self.n_factors,
            center_points=center_points,
            bounds=self.bounds,
        )

    def factorial(self, levels: int = 2) -> np.ndarray:
        """Generate a full factorial design in natural units.

        Parameters
        ----------
        levels : int
            Number of levels per factor.

        Returns
        -------
        X : np.ndarray
        """
        return factorial_design(
            levels=levels,
            n_factors=self.n_factors,
            bounds=self.bounds,
        )

    def fractional_factorial(self, resolution: int = 3) -> np.ndarray:
        """Generate a fractional factorial design in natural units.

        Parameters
        ----------
        resolution : int
            Minimum resolution (3, 4, or 5).

        Returns
        -------
        X : np.ndarray
        """
        return fractional_factorial_design(
            self.n_factors,
            resolution=resolution,
            bounds=self.bounds,
        )

    # -- Fitting ----------------------------------------------------------

    def fit(
        self,
        X: np.ndarray | jnp.ndarray,
        y: np.ndarray | jnp.ndarray,
    ) -> ResponseSurface:
        """Fit the response-surface model.

        Parameters
        ----------
        X : array-like of shape (n, n_factors)
            Input data in natural units.
        y : array-like of shape (n,)
            Observed responses.

        Returns
        -------
        self
        """
        self.model.fit(jnp.array(X), jnp.array(y))
        return self

    # -- Analysis ---------------------------------------------------------

    def anova(self, anova_type: str = "sequential"):
        """Run ANOVA on the fitted model.

        See :func:`jaxsr.uncertainty.anova` for details.
        """
        return anova(self.model, anova_type=anova_type)

    def canonical(self) -> CanonicalAnalysis:
        """Perform canonical analysis (stationary point, eigenvalues).

        The stationary point is returned in **natural units**.
        """
        return canonical_analysis(self.model, bounds=self.bounds)

    # -- Prediction helpers -----------------------------------------------

    def predict(self, X: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        """Predict response at new points (natural units)."""
        return self.model.predict(jnp.array(X))

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Convert natural units to coded [-1, +1] variables."""
        return _encode(np.asarray(X), self.bounds)

    def decode(self, X_coded: np.ndarray) -> np.ndarray:
        """Convert coded [-1, +1] variables to natural units."""
        return _decode(np.asarray(X_coded), self.bounds)

    # -- Plotting ---------------------------------------------------------

    def plot_contour(
        self,
        factors: tuple[int, int] = (0, 1),
        fixed: dict[int, float] | None = None,
        n_grid: int = 50,
        levels: int = 15,
        ax=None,
        figsize: tuple[int, int] = (8, 6),
        filled: bool = True,
        show_design: bool = True,
    ):
        """Plot 2D contour of the response surface.

        Parameters
        ----------
        factors : (int, int)
            Indices of the two factors to vary.
        fixed : dict, optional
            ``{factor_index: value}`` for held-constant factors (natural
            units).  Factors not in *factors* or *fixed* default to their
            midpoint.
        n_grid : int
            Grid resolution per axis.
        levels : int
            Number of contour levels.
        ax : matplotlib Axes, optional
            If ``None``, a new figure is created.
        figsize : tuple
            Figure size.
        filled : bool
            If True, use filled contours; otherwise contour lines only.
        show_design : bool
            If True and the model has training data, overlay the design
            points.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        self.model._check_is_fitted()

        i, j = factors
        k = self.n_factors

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Build grid in natural units
        xi = np.linspace(self.bounds[i][0], self.bounds[i][1], n_grid)
        xj = np.linspace(self.bounds[j][0], self.bounds[j][1], n_grid)
        Xi, Xj = np.meshgrid(xi, xj)

        # Determine fixed values
        midpoints = [(lo + hi) / 2 for lo, hi in self.bounds]
        fixed = fixed or {}
        fill_vals = list(midpoints)
        for idx, val in fixed.items():
            fill_vals[idx] = val

        # Construct full input matrix
        X_grid = np.tile(fill_vals, (n_grid * n_grid, 1))
        X_grid[:, i] = Xi.ravel()
        X_grid[:, j] = Xj.ravel()

        Y = np.array(self.model.predict(jnp.array(X_grid))).reshape(n_grid, n_grid)

        if filled:
            cs = ax.contourf(Xi, Xj, Y, levels=levels, cmap="viridis")
        else:
            cs = ax.contour(Xi, Xj, Y, levels=levels, cmap="viridis")
        plt.colorbar(cs, ax=ax, label="Response")

        # Overlay design points
        if show_design and self.model._X_train is not None:
            X_train = np.array(self.model._X_train)
            ax.scatter(
                X_train[:, i],
                X_train[:, j],
                c="red",
                edgecolors="white",
                s=50,
                zorder=5,
                label="Design points",
            )
            ax.legend()

        ax.set_xlabel(self.factor_names[i])
        ax.set_ylabel(self.factor_names[j])

        # Build subtitle with fixed factors
        fixed_strs = []
        for idx in range(k):
            if idx not in (i, j):
                fixed_strs.append(f"{self.factor_names[idx]}={fill_vals[idx]:.3g}")
        subtitle = f"  ({', '.join(fixed_strs)})" if fixed_strs else ""
        ax.set_title(f"Response Surface Contour{subtitle}")

        return ax

    def plot_surface(
        self,
        factors: tuple[int, int] = (0, 1),
        fixed: dict[int, float] | None = None,
        n_grid: int = 50,
        ax=None,
        figsize: tuple[int, int] = (10, 8),
    ):
        """Plot 3D surface of the response.

        Parameters are the same as :meth:`plot_contour` except there
        are no *levels* or *filled* options.

        Returns
        -------
        ax : matplotlib 3D Axes
        """
        import matplotlib.pyplot as plt

        self.model._check_is_fitted()

        i, j = factors

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

        xi = np.linspace(self.bounds[i][0], self.bounds[i][1], n_grid)
        xj = np.linspace(self.bounds[j][0], self.bounds[j][1], n_grid)
        Xi, Xj = np.meshgrid(xi, xj)

        midpoints = [(lo + hi) / 2 for lo, hi in self.bounds]
        fixed = fixed or {}
        fill_vals = list(midpoints)
        for idx, val in fixed.items():
            fill_vals[idx] = val

        X_grid = np.tile(fill_vals, (n_grid * n_grid, 1))
        X_grid[:, i] = Xi.ravel()
        X_grid[:, j] = Xj.ravel()

        Y = np.array(self.model.predict(jnp.array(X_grid))).reshape(n_grid, n_grid)

        ax.plot_surface(Xi, Xj, Y, cmap="viridis", alpha=0.8)
        ax.set_xlabel(self.factor_names[i])
        ax.set_ylabel(self.factor_names[j])
        ax.set_zlabel("Response")
        ax.set_title(f"Response Surface\n{self.model.expression_}")

        return ax

    def summary(self) -> str:
        """Return a text summary of the fitted response surface."""
        self.model._check_is_fitted()
        lines = [
            "Response Surface Summary",
            "=" * 50,
            f"Factors:    {self.factor_names}",
            f"Bounds:     {self.bounds}",
            f"Expression: {self.model.expression_}",
            f"R²:         {self.model.score(self.model._X_train, self.model._y_train):.6f}",
            f"MSE:        {self.model.metrics_['mse']:.6f}",
            f"BIC:        {self.model.metrics_['bic']:.2f}",
            f"N samples:  {len(self.model._y_train)}",
            "",
        ]

        # Canonical analysis
        ca = self.canonical()
        lines.append(str(ca))

        return "\n".join(lines)
