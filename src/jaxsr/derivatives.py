"""
Multivariate derivative estimation for surface and PDE-style discovery.

:func:`jaxsr.dynamics.estimate_derivatives` differentiates along a single axis, which
covers ODE discovery but not problems whose data is a *surface* and whose regression
needs more than one partial derivative -- PDE-style discovery
(``u_t = F(u, u_x, u_xx, ...)``) or transform discovery such as time-temperature
superposition, where ``y(x, T) = f(x + s(T))`` implies ``y_T = s'(T) * y_x`` and both
partials must come from one smoothed surface.

:class:`SurfaceDerivatives` fits a smoother to scattered or gridded N-D data and
returns *analytic* partial derivatives of that smoother, never finite differences of
noisy raw data. Three smoothers are available:

``"tensor_spline"``
    Penalized tensor-product B-splines (P-splines). Fast, works in any dimension for
    scattered or gridded data, smoothing chosen by GCV or from a known noise level.

``"local_poly"``
    Local polynomial (LOESS-style) regression. Robust to irregular sampling; the
    derivative of order *m* is read off the local polynomial coefficient.

``"gp"``
    Gaussian process with an anisotropic squared-exponential kernel. Gives derivative
    uncertainty directly and handles irregular sampling, at :math:`O(n^3)` cost.

The smoothing hyperparameter is always selected *without reference to any downstream
symbolic score* -- by GCV, by marginal likelihood, or from a supplied noise level.
Tuning a smoother against the regression that consumes it can manufacture whichever
law the regression prefers, and the failure is silent. The level actually used is
reported in :attr:`SurfaceDerivatives.smoothing_` and by
:meth:`SurfaceDerivatives.summary`, so smoothing-induced bias is visible rather than
inferred.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "SurfaceDerivatives",
    "estimate_partial_derivatives",
]

_VALID_METHODS = ("tensor_spline", "local_poly", "gp")
_JITTER = 1e-10


# ---------------------------------------------------------------------------
# Input normalization helpers
# ---------------------------------------------------------------------------


def _normalize_coords(
    coords: np.ndarray | Sequence[np.ndarray],
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize scattered or gridded inputs to flat ``(n, d)`` / ``(n,)`` arrays.

    Parameters
    ----------
    coords : np.ndarray of shape (n_points, n_dims), or sequence of 1-D arrays
        Sample locations. A sequence of ``d`` 1-D axis arrays is interpreted as a
        rectangular grid, in which case *values* must have shape
        ``(len(axis_0), ..., len(axis_{d-1}))``.
    values : np.ndarray
        Observed values, shape ``(n_points,)`` for scattered input or the grid shape
        for gridded input.

    Returns
    -------
    coords : np.ndarray of shape (n_points, n_dims)
        Flattened coordinates.
    values : np.ndarray of shape (n_points,)
        Flattened values.

    Raises
    ------
    ValueError
        If shapes are inconsistent or the inputs are not finite.
    """
    values = np.asarray(values, dtype=np.float64)

    # Gridded form: a sequence of d axis arrays paired with a d-dimensional values array.
    is_grid = (
        isinstance(coords, (list, tuple))
        and len(coords) > 0
        and values.ndim == len(coords)
        and all(np.ndim(axis) == 1 for axis in coords)
    )

    if is_grid:
        axes = [np.asarray(a, dtype=np.float64).ravel() for a in coords]
        shape = tuple(len(a) for a in axes)
        if values.shape != shape:
            raise ValueError(
                f"Gridded input: values shape {values.shape} does not match the grid "
                f"shape implied by the axes {shape}"
            )
        mesh = np.meshgrid(*axes, indexing="ij")
        coords_arr = np.column_stack([m.ravel() for m in mesh])
        values = values.ravel()
    else:
        coords_arr = np.asarray(coords, dtype=np.float64)
        if coords_arr.ndim == 1:
            coords_arr = coords_arr.reshape(-1, 1)
        if coords_arr.ndim != 2:
            raise ValueError(f"coords must be 1-D or 2-D, got {coords_arr.ndim}-D")
        values = values.ravel()

    if coords_arr.shape[0] != values.shape[0]:
        raise ValueError(
            f"Number of coordinate rows ({coords_arr.shape[0]}) must match the number "
            f"of values ({values.shape[0]})"
        )
    if not np.all(np.isfinite(coords_arr)):
        raise ValueError("coords contains non-finite values")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contains non-finite entries")

    return coords_arr, values


def _normalize_query(coords: np.ndarray, n_dims: int) -> np.ndarray:
    """
    Normalize query coordinates to a ``(n_query, n_dims)`` array.

    Parameters
    ----------
    coords : np.ndarray of shape (n_query, n_dims)
        Query locations. A 1-D array is read as a column of points when the surface is
        1-D, and as a single point otherwise.
    n_dims : int
        Expected number of dimensions.

    Returns
    -------
    np.ndarray of shape (n_query, n_dims)
        Query coordinates.

    Raises
    ------
    ValueError
        If the coordinate dimension does not match *n_dims* or entries are not finite.
    """
    query = np.asarray(coords, dtype=np.float64)
    if query.ndim == 1:
        if n_dims == 1:
            query = query.reshape(-1, 1)
        elif query.shape[0] == n_dims:
            query = query.reshape(1, n_dims)

    if query.ndim != 2 or query.shape[1] != n_dims:
        raise ValueError(
            f"Query coordinates must have shape (n_query, {n_dims}), got {query.shape}"
        )
    if not np.all(np.isfinite(query)):
        raise ValueError("Query coordinates contain non-finite values")
    return query


def _normalize_orders(order: Any, n_dims: int) -> np.ndarray:
    """
    Normalize a derivative-order specification to a ``(n_orders, n_dims)`` int array.

    Parameters
    ----------
    order : sequence
        Either a single order tuple such as ``(1, 0)`` or a sequence of them such as
        ``[(1, 0), (0, 1)]``. Each tuple gives the differentiation order per dimension.
    n_dims : int
        Number of coordinate dimensions.

    Returns
    -------
    np.ndarray of shape (n_orders, n_dims)
        Non-negative integer derivative orders.

    Raises
    ------
    ValueError
        If the specification is malformed, has the wrong length, or is negative.
    """
    if order is None:
        raise ValueError("order must be provided, e.g. order=[(1, 0), (0, 1)]")

    seq = list(order) if isinstance(order, (list, tuple, np.ndarray)) else [order]
    if len(seq) == 0:
        raise ValueError("order must contain at least one derivative order")

    if all(np.ndim(item) == 0 for item in seq):
        orders = [seq]
    else:
        orders = [list(item) for item in seq]

    arr = np.asarray(orders)
    if arr.ndim != 2 or arr.shape[1] != n_dims:
        raise ValueError(
            f"Each derivative order must have {n_dims} entries (one per dimension); "
            f"got {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.integer):
        if np.any(arr != np.round(arr)):
            raise ValueError("Derivative orders must be integers")
        arr = np.round(arr).astype(int)
    if np.any(arr < 0):
        raise ValueError("Derivative orders must be non-negative")
    return arr.astype(int)


def _normalize_sigma(sigma: float | np.ndarray | None, n_points: int) -> np.ndarray | None:
    """
    Normalize a noise specification to a per-point standard-deviation array.

    Parameters
    ----------
    sigma : float, np.ndarray of shape (n_points,), or None
        Measurement noise standard deviation, scalar or per point.
    n_points : int
        Number of data points.

    Returns
    -------
    np.ndarray of shape (n_points,) or None
        Per-point standard deviations, or ``None`` if *sigma* was ``None``.

    Raises
    ------
    ValueError
        If *sigma* is non-positive, non-finite, or the wrong length.
    """
    if sigma is None:
        return None
    arr = np.asarray(sigma, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n_points, float(arr))
    else:
        arr = arr.ravel()
    if arr.shape[0] != n_points:
        raise ValueError(f"sigma must be scalar or have {n_points} entries, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError("sigma must be finite and strictly positive")
    return arr


# ---------------------------------------------------------------------------
# B-spline helpers (tensor_spline)
# ---------------------------------------------------------------------------


def _knot_vector(lo: float, hi: float, n_basis: int, degree: int) -> np.ndarray:
    """
    Build a clamped, uniformly spaced knot vector.

    Parameters
    ----------
    lo, hi : float
        Lower and upper bounds of the data along this dimension.
    n_basis : int
        Number of B-spline basis functions.
    degree : int
        Spline degree.

    Returns
    -------
    np.ndarray
        Knot vector of length ``n_basis + degree + 1``.

    Raises
    ------
    ValueError
        If *n_basis* is smaller than ``degree + 1``.
    """
    if n_basis < degree + 1:
        raise ValueError(f"n_basis ({n_basis}) must be at least degree + 1 ({degree + 1})")
    if hi <= lo:
        hi = lo + 1.0
    n_interior = n_basis - degree - 1
    interior = np.linspace(lo, hi, n_interior + 2)[1:-1]
    return np.concatenate([np.full(degree + 1, lo), interior, np.full(degree + 1, hi)])


def _bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int, nu: int = 0) -> np.ndarray:
    """
    Evaluate all B-spline basis functions (or their derivatives) at *x*.

    Parameters
    ----------
    x : np.ndarray of shape (n_points,)
        Evaluation points.
    knots : np.ndarray
        Knot vector.
    degree : int
        Spline degree.
    nu : int
        Derivative order to evaluate.

    Returns
    -------
    np.ndarray of shape (n_points, n_basis)
        Basis (or basis-derivative) values.
    """
    from scipy.interpolate import BSpline

    n_basis = len(knots) - degree - 1
    out = np.zeros((x.shape[0], n_basis))
    if nu > degree:
        return out
    coef = np.zeros(n_basis)
    for j in range(n_basis):
        coef[:] = 0.0
        coef[j] = 1.0
        out[:, j] = BSpline(knots, coef, degree, extrapolate=True)(x, nu)
    return out


def _row_tensor(bases: list[np.ndarray]) -> np.ndarray:
    """
    Row-wise tensor (Khatri-Rao) product of per-dimension basis matrices.

    Parameters
    ----------
    bases : list of np.ndarray
        One ``(n_points, m_i)`` matrix per dimension.

    Returns
    -------
    np.ndarray of shape (n_points, prod(m_i))
        Tensor-product design matrix, with dimension 0 varying slowest.
    """
    out = bases[0]
    for basis in bases[1:]:
        out = (out[:, :, None] * basis[:, None, :]).reshape(basis.shape[0], -1)
    return out


def _difference_penalty(sizes: Sequence[int], penalty_order: int) -> np.ndarray:
    """
    Build the additive tensor-product difference penalty matrix.

    Parameters
    ----------
    sizes : sequence of int
        Number of basis functions per dimension.
    penalty_order : int
        Order of the difference penalty applied along each dimension.

    Returns
    -------
    np.ndarray of shape (prod(sizes), prod(sizes))
        Sum over dimensions of ``I (x) ... (x) D.T @ D (x) ... (x) I``.
    """
    total = int(np.prod(sizes))
    penalty = np.zeros((total, total))
    for axis, size in enumerate(sizes):
        order = min(penalty_order, size - 1)
        if order <= 0:
            block = np.eye(size)
        else:
            diff = np.diff(np.eye(size), n=order, axis=0)
            block = diff.T @ diff
        term = np.array([[1.0]])
        for other, other_size in enumerate(sizes):
            term = np.kron(term, block if other == axis else np.eye(other_size))
        penalty += term
    return penalty


# ---------------------------------------------------------------------------
# Gaussian process helpers
# ---------------------------------------------------------------------------


def _hermite_e(n: int, u: np.ndarray) -> np.ndarray:
    """
    Evaluate the probabilists' Hermite polynomial ``He_n``.

    Parameters
    ----------
    n : int
        Polynomial order.
    u : np.ndarray
        Evaluation points.

    Returns
    -------
    np.ndarray
        ``He_n(u)``, same shape as *u*.
    """
    coef = np.zeros(n + 1)
    coef[n] = 1.0
    return np.polynomial.hermite_e.hermeval(u, coef)


def _hermite_e_at_zero(n: int) -> float:
    """
    Evaluate ``He_n(0)``.

    Parameters
    ----------
    n : int
        Polynomial order.

    Returns
    -------
    float
        ``He_n(0)``: zero for odd *n*, ``(-1)^(n/2) * (n-1)!!`` otherwise.
    """
    if n % 2 == 1:
        return 0.0
    return float(_hermite_e(n, np.zeros(1))[0])


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


class SurfaceDerivatives:
    """
    Smoothed N-D surface with analytic partial derivatives.

    Fits a smoother to scattered or gridded data over ``n_dims`` coordinates and
    evaluates arbitrary mixed partial derivatives of that smoother analytically,
    together with their standard errors.

    Parameters
    ----------
    method : str
        Smoother to fit. One of ``"tensor_spline"`` (penalized tensor-product
        B-splines), ``"local_poly"`` (local polynomial regression), or ``"gp"``
        (Gaussian process with an anisotropic squared-exponential kernel).
    degree : int
        Spline degree for ``"tensor_spline"``, or the local polynomial degree for
        ``"local_poly"``. Must be at least the highest total derivative order
        requested. Ignored by ``"gp"``.
    n_basis : int or sequence of int, optional
        Number of B-spline basis functions per dimension (``"tensor_spline"`` only).
        Defaults to a value derived from the number of distinct coordinates per
        dimension, capped so the design matrix stays well determined.
    penalty_order : int
        Order of the difference penalty on the spline coefficients
        (``"tensor_spline"`` only). ``2`` penalizes curvature.
    smoothing : float or str
        How much to smooth, and how that level is chosen:

        - ``"auto"`` (default): GCV for ``"tensor_spline"`` and ``"local_poly"``,
          marginal likelihood for ``"gp"``.
        - ``"sigma"``: requires *sigma* at :meth:`fit`; chooses the smoothing level
          whose weighted residual sum of squares equals ``n_points`` (equivalently,
          unweighted residual sum of squares equal to ``n_points * sigma**2``).
        - float: use this value directly -- the ridge parameter ``lambda`` for
          ``"tensor_spline"``, the bandwidth for ``"local_poly"`` (in standardized
          coordinate units), or the noise variance for ``"gp"``.

        Never selected against a downstream regression score.
    smoothing_scale : float
        Multiplier applied to the selected smoothing level. Useful for deliberately
        over- or under-smoothing to expose the sensitivity of a discovered law to the
        derivative stage.
    length_scale : float or sequence of float, optional
        Fixed kernel length scales for ``"gp"`` in standardized coordinate units. If
        ``None`` they are learned by maximizing the log marginal likelihood.
    max_basis : int
        Guard on the total number of tensor-product spline basis functions.
    max_points : int
        Guard on the number of training points for ``"gp"``, whose cost is cubic.
    random_state : int, optional
        Seed for the subsampling used during ``"local_poly"`` bandwidth selection.

    Attributes
    ----------
    coords_ : np.ndarray of shape (n_points, n_dims)
        Training coordinates.
    values_ : np.ndarray of shape (n_points,)
        Training values.
    n_features_in_ : int
        Number of coordinate dimensions.
    smoothing_ : float
        Smoothing level actually used (``lambda``, bandwidth, or noise variance,
        depending on *method*).
    smoothing_source_ : str
        How :attr:`smoothing_` was chosen: ``"gcv"``, ``"marginal_likelihood"``,
        ``"sigma"``, or ``"fixed"``.
    effective_dof_ : float
        Effective degrees of freedom of the fitted smoother.
    residual_std_ : float
        Residual standard deviation of the fit at the training points.
    noise_std_ : float
        Noise level used for the reported uncertainties: the supplied *sigma* (its
        root-mean-square if per-point) when given, otherwise :attr:`residual_std_`.

    Raises
    ------
    ValueError
        If *method*, *degree*, *penalty_order*, *smoothing*, or *smoothing_scale* is
        invalid.

    See Also
    --------
    jaxsr.dynamics.estimate_derivatives : Derivatives along a single axis, for
        time-series / ODE discovery.

    Examples
    --------
    >>> import numpy as np
    >>> from jaxsr import SurfaceDerivatives
    >>> x = np.linspace(0, 1, 25)
    >>> T = np.linspace(0, 2, 15)
    >>> XX, TT = np.meshgrid(x, T, indexing="ij")
    >>> Z = np.sin(XX) * TT**2
    >>> est = SurfaceDerivatives(method="tensor_spline").fit([x, T], Z)
    >>> coords = np.column_stack([XX.ravel(), TT.ravel()])
    >>> z, dz = est.derivatives(coords, order=[(1, 0), (0, 1)])
    >>> dz.shape
    (375, 2)
    """

    def __init__(
        self,
        method: str = "tensor_spline",
        *,
        degree: int = 3,
        n_basis: int | Sequence[int] | None = None,
        penalty_order: int = 2,
        smoothing: float | str = "auto",
        smoothing_scale: float = 1.0,
        length_scale: float | Sequence[float] | None = None,
        max_basis: int = 512,
        max_points: int = 800,
        random_state: int | None = None,
    ) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(f"Unknown method {method!r}. Choose from {list(_VALID_METHODS)}")
        if not isinstance(degree, (int, np.integer)) or degree < 1:
            raise ValueError(f"degree must be a positive integer, got {degree!r}")
        if not isinstance(penalty_order, (int, np.integer)) or penalty_order < 0:
            raise ValueError(f"penalty_order must be a non-negative integer, got {penalty_order!r}")
        if isinstance(smoothing, str):
            if smoothing not in ("auto", "sigma"):
                raise ValueError(
                    f"smoothing must be 'auto', 'sigma', or a float, got {smoothing!r}"
                )
        else:
            smoothing = float(smoothing)
            if smoothing <= 0:
                raise ValueError("Numeric smoothing must be strictly positive")
        if smoothing_scale <= 0:
            raise ValueError("smoothing_scale must be strictly positive")
        if max_basis < 1:
            raise ValueError("max_basis must be a positive integer")
        if max_points < 1:
            raise ValueError("max_points must be a positive integer")

        self.method = method
        self.degree = int(degree)
        self.n_basis = n_basis
        self.penalty_order = int(penalty_order)
        self.smoothing = smoothing
        self.smoothing_scale = float(smoothing_scale)
        self.length_scale = length_scale
        self.max_basis = int(max_basis)
        self.max_points = int(max_points)
        self.random_state = random_state

        self._is_fitted = False

    # -- public API ---------------------------------------------------------

    def fit(
        self,
        coords: np.ndarray | Sequence[np.ndarray],
        values: np.ndarray,
        sigma: float | np.ndarray | None = None,
    ) -> SurfaceDerivatives:
        """
        Fit the smoother to surface data.

        Parameters
        ----------
        coords : np.ndarray of shape (n_points, n_dims), or sequence of 1-D arrays
            Sample locations. A sequence of ``n_dims`` 1-D axis arrays is treated as a
            rectangular grid, in which case *values* must have the grid shape.
        values : np.ndarray
            Observed values, shape ``(n_points,)`` for scattered coordinates or the
            grid shape for gridded coordinates.
        sigma : float or np.ndarray of shape (n_points,), optional
            Known measurement noise standard deviation, scalar or per point. Supplying
            it enables ``smoothing="sigma"`` and makes the reported derivative
            uncertainty reflect the measurement noise rather than the residual scatter.

        Returns
        -------
        SurfaceDerivatives
            The fitted estimator (``self``).

        Raises
        ------
        ValueError
            If the inputs are inconsistent, too small for the requested smoother, or
            if ``smoothing="sigma"`` was requested without *sigma*.
        """
        coords_arr, values_arr = _normalize_coords(coords, values)
        n_points, n_dims = coords_arr.shape

        if n_points < 4:
            raise ValueError(f"Need at least 4 data points, got {n_points}")

        sigma_arr = _normalize_sigma(sigma, n_points)
        if self.smoothing == "sigma" and sigma_arr is None:
            raise ValueError("smoothing='sigma' requires the sigma argument to fit()")

        self.coords_ = coords_arr
        self.values_ = values_arr
        self.n_features_in_ = n_dims
        self._sigma = sigma_arr
        self._weights = np.ones(n_points) if sigma_arr is None else 1.0 / sigma_arr**2

        if self.method == "tensor_spline":
            self._fit_tensor_spline()
        elif self.method == "local_poly":
            self._fit_local_poly()
        else:
            self._fit_gp()

        residuals = self.values_ - self._fitted_values
        dof = max(n_points - self.effective_dof_, 1.0)
        self.residual_std_ = float(np.sqrt(np.sum(residuals**2) / dof))
        if sigma_arr is None:
            self.noise_std_ = self.residual_std_
        else:
            self.noise_std_ = float(np.sqrt(np.mean(sigma_arr**2)))

        self._is_fitted = True
        return self

    def predict(
        self,
        coords: np.ndarray,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the smoothed surface.

        Parameters
        ----------
        coords : np.ndarray of shape (n_query, n_dims)
            Query locations.
        return_std : bool
            If ``True``, also return the standard error of the smoothed value.

        Returns
        -------
        values : np.ndarray of shape (n_query,)
            Smoothed values.
        std : np.ndarray of shape (n_query,)
            Standard errors. Only returned when *return_std* is ``True``.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("SurfaceDerivatives must be fitted before calling predict()")
        zero = tuple([0] * self.n_features_in_)
        out = self.derivatives(coords, order=[zero], return_std=return_std)
        if return_std:
            values, _, std = out
            return values, std[:, 0]
        return out[0]

    def derivatives(
        self,
        coords: np.ndarray,
        order: Any,
        return_std: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """
        Evaluate analytic partial derivatives of the fitted smoother.

        Parameters
        ----------
        coords : np.ndarray of shape (n_query, n_dims)
            Query locations. Use ``estimator.coords_`` to evaluate at the sample
            locations, including for data supplied in gridded form.
        order : tuple of int, or sequence of tuples
            Derivative orders per dimension. ``(1, 0)`` is a single first partial with
            respect to dimension 0; ``[(1, 0), (0, 1)]`` requests both first partials.
        return_std : bool
            If ``True``, also return the standard error of each partial derivative.

        Returns
        -------
        values : np.ndarray of shape (n_query,)
            The smoothed surface at the query points.
        partials : np.ndarray of shape (n_query, n_orders)
            One column per entry of *order*, in the given order. A single order tuple
            still yields a column vector of shape ``(n_query, 1)``.
        std : np.ndarray of shape (n_query, n_orders)
            Standard errors of *partials*. Only returned when *return_std* is ``True``.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        ValueError
            If *order* is malformed or exceeds what the smoother can differentiate.
        """
        if not self._is_fitted:
            raise RuntimeError("SurfaceDerivatives must be fitted before calling derivatives()")

        query = _normalize_query(coords, self.n_features_in_)
        orders = _normalize_orders(order, self.n_features_in_)
        self._validate_orders(orders)

        zero = np.zeros((1, self.n_features_in_), dtype=int)
        all_orders = np.vstack([zero, orders])

        if self.method == "tensor_spline":
            vals, stds = self._eval_tensor_spline(query, all_orders, return_std)
        elif self.method == "local_poly":
            vals, stds = self._eval_local_poly(query, all_orders, return_std)
        else:
            vals, stds = self._eval_gp(query, all_orders, return_std)

        values = vals[:, 0]
        partials = vals[:, 1:]
        if return_std:
            return values, partials, stds[:, 1:]
        return values, partials

    def summary(self) -> str:
        """
        Return a human-readable description of the fitted smoother.

        Reports the smoothing level actually used and how it was chosen, so that
        smoothing-induced bias in downstream results is visible rather than inferred.

        Returns
        -------
        str
            Multi-line summary.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("SurfaceDerivatives must be fitted before calling summary()")

        label = {
            "tensor_spline": "penalty lambda",
            "local_poly": "bandwidth",
            "gp": "noise variance",
        }[self.method]

        lines = [
            "SurfaceDerivatives",
            "=" * 40,
            f"method            : {self.method}",
            f"data              : {self.coords_.shape[0]} points, "
            f"{self.n_features_in_} dimensions",
            f"{label:<18}: {self.smoothing_:.6g} (chosen by {self.smoothing_source_})",
            f"effective dof     : {self.effective_dof_:.2f}",
            f"residual std      : {self.residual_std_:.6g}",
            f"noise std used    : {self.noise_std_:.6g}",
        ]
        if self.method == "gp":
            scales = ", ".join(f"{s:.4g}" for s in self._gp_length_scale)
            lines.append(f"length scales     : [{scales}] (standardized units)")
        elif self.method == "tensor_spline":
            lines.append(f"basis per dim     : {list(self._spline_sizes)} (degree {self.degree})")
        return "\n".join(lines)

    # -- validation ---------------------------------------------------------

    def _validate_orders(self, orders: np.ndarray) -> None:
        """
        Check that requested derivative orders are supported by the smoother.

        Parameters
        ----------
        orders : np.ndarray of shape (n_orders, n_dims)
            Requested derivative orders.

        Raises
        ------
        ValueError
            If an order exceeds what the fitted smoother can differentiate.
        """
        if self.method == "tensor_spline":
            worst = int(orders.max(initial=0))
            if worst > self.degree:
                raise ValueError(
                    f"Derivative order {worst} exceeds the spline degree {self.degree}; "
                    f"refit with degree >= {worst}"
                )
        elif self.method == "local_poly":
            worst = int(orders.sum(axis=1).max(initial=0))
            if worst > self.degree:
                raise ValueError(
                    f"Total derivative order {worst} exceeds the local polynomial degree "
                    f"{self.degree}; refit with degree >= {worst}"
                )

    # -- tensor-product P-splines ------------------------------------------

    def _default_basis_sizes(self) -> list[int]:
        """
        Choose a per-dimension basis size that stays well determined.

        Returns
        -------
        list of int
            Number of B-spline basis functions per dimension.

        Raises
        ------
        ValueError
            If an explicitly supplied *n_basis* is invalid or too large.
        """
        n_points, n_dims = self.coords_.shape
        floor = self.degree + 1

        if self.n_basis is not None:
            if np.ndim(self.n_basis) == 0:
                sizes = [int(self.n_basis)] * n_dims
            else:
                sizes = [int(v) for v in self.n_basis]  # type: ignore[union-attr]
            if len(sizes) != n_dims:
                raise ValueError(f"n_basis must be a scalar or have {n_dims} entries")
            if any(s < floor for s in sizes):
                raise ValueError(f"Each n_basis entry must be at least degree + 1 = {floor}")
            total = int(np.prod(sizes))
            if total > self.max_basis:
                raise ValueError(
                    f"n_basis={sizes} needs {total} tensor-product basis functions, "
                    f"above max_basis={self.max_basis}"
                )
            if total > n_points:
                raise ValueError(
                    f"n_basis={sizes} needs {total} basis functions but only {n_points} "
                    "data points are available"
                )
            return sizes

        # Cap the per-dimension resolution: GCV picks the penalty that is best for the
        # *fit*, which tends to under-smooth derivatives, so a moderate basis is a
        # useful second brake. Pass n_basis explicitly for finer structure.
        cap = max(floor, 12)
        sizes = []
        for dim in range(n_dims):
            n_unique = len(np.unique(self.coords_[:, dim]))
            sizes.append(int(np.clip(n_unique, floor, cap)))

        budget = min(self.max_basis, max(n_points // 2, floor**n_dims))
        while int(np.prod(sizes)) > budget and any(s > floor for s in sizes):
            largest = int(np.argmax(sizes))
            if sizes[largest] <= floor:
                break
            sizes[largest] -= 1
        if int(np.prod(sizes)) > self.max_basis:
            raise ValueError(
                f"Minimal tensor-product basis needs {int(np.prod(sizes))} functions, above "
                f"max_basis={self.max_basis}; reduce degree or raise max_basis"
            )
        return sizes

    def _spline_design(self, coords: np.ndarray, order: np.ndarray) -> np.ndarray:
        """
        Build the tensor-product design matrix for one derivative order.

        Parameters
        ----------
        coords : np.ndarray of shape (n_points, n_dims)
            Evaluation points.
        order : np.ndarray of shape (n_dims,)
            Derivative order per dimension.

        Returns
        -------
        np.ndarray of shape (n_points, n_basis_total)
            Design matrix.
        """
        bases = [
            _bspline_basis(coords[:, dim], self._spline_knots[dim], self.degree, int(order[dim]))
            for dim in range(self.n_features_in_)
        ]
        return _row_tensor(bases)

    def _fit_tensor_spline(self) -> None:
        """
        Fit penalized tensor-product B-splines and select the penalty weight.

        Raises
        ------
        ValueError
            If the basis specification is inconsistent with the data.
        """
        from scipy.linalg import cho_factor, cho_solve

        n_points = self.coords_.shape[0]
        sizes = self._default_basis_sizes()
        self._spline_sizes = sizes
        self._spline_knots = [
            _knot_vector(
                float(self.coords_[:, dim].min()),
                float(self.coords_[:, dim].max()),
                sizes[dim],
                self.degree,
            )
            for dim in range(self.n_features_in_)
        ]

        zero_order = np.zeros(self.n_features_in_, dtype=int)
        design = self._spline_design(self.coords_, zero_order)
        sqrt_w = np.sqrt(self._weights)
        design_w = design * sqrt_w[:, None]
        values_w = self.values_ * sqrt_w

        gram = design_w.T @ design_w
        rhs = design_w.T @ values_w
        penalty = _difference_penalty(sizes, self.penalty_order)

        # Scale the penalty so that lambda is dimensionless and O(1) grids work.
        scale = np.trace(gram) / max(np.trace(penalty), _JITTER)
        penalty = penalty * scale
        ridge = _JITTER * np.trace(gram) / gram.shape[0] * np.eye(gram.shape[0])

        def solve_for(lam: float) -> tuple[np.ndarray, float, float]:
            matrix = gram + lam * penalty + ridge
            factor = cho_factor(matrix, lower=True)
            coef = cho_solve(factor, rhs)
            resid = values_w - design_w @ coef
            rss = float(resid @ resid)
            edf = float(np.trace(cho_solve(factor, gram)))
            return coef, rss, edf

        lam_grid = np.geomspace(1e-8, 1e8, 21)
        results = [solve_for(float(lam)) for lam in lam_grid]
        rss_grid = np.array([r[1] for r in results])
        edf_grid = np.array([r[2] for r in results])

        if isinstance(self.smoothing, float):
            lam = self.smoothing
            source = "fixed"
        elif self.smoothing == "sigma":
            lam = float(_interp_log(lam_grid, rss_grid, float(n_points)))
            source = "sigma"
        else:
            denom = np.maximum(n_points - edf_grid, 1e-6)
            gcv = n_points * rss_grid / denom**2
            gcv[edf_grid >= n_points] = np.inf
            lam = float(lam_grid[int(np.argmin(gcv))])
            source = "gcv"

        lam *= self.smoothing_scale
        coef, _, edf = solve_for(lam)

        self.smoothing_ = lam
        self.smoothing_source_ = source
        self.effective_dof_ = edf
        self._spline_coef = coef
        self._fitted_values = design @ coef

        matrix = gram + lam * penalty + ridge
        factor = cho_factor(matrix, lower=True)
        self._spline_cov = cho_solve(factor, np.eye(matrix.shape[0]))

    def _eval_tensor_spline(
        self, query: np.ndarray, orders: np.ndarray, return_std: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate spline partials at the query points.

        Parameters
        ----------
        query : np.ndarray of shape (n_query, n_dims)
            Query points.
        orders : np.ndarray of shape (n_orders, n_dims)
            Derivative orders.
        return_std : bool
            Whether to compute standard errors.

        Returns
        -------
        values : np.ndarray of shape (n_query, n_orders)
            Evaluated partials.
        std : np.ndarray of shape (n_query, n_orders)
            Standard errors, left as zeros when *return_std* is ``False``.
        """
        scale = 1.0 if self._sigma is not None else self.residual_std_**2
        values = np.empty((query.shape[0], orders.shape[0]))
        stds = np.zeros_like(values)

        for k, order in enumerate(orders):
            design = self._spline_design(query, order)
            values[:, k] = design @ self._spline_coef
            if return_std:
                var = np.einsum("ij,jk,ik->i", design, self._spline_cov, design)
                stds[:, k] = np.sqrt(np.maximum(scale * var, 0.0))
        return values, stds

    # -- local polynomial regression ---------------------------------------

    def _poly_exponents(self) -> np.ndarray:
        """
        Enumerate monomial exponents up to the local polynomial degree.

        Returns
        -------
        np.ndarray of shape (n_terms, n_dims)
            Exponent vectors with total degree at most :attr:`degree`.
        """
        dims = self.n_features_in_
        combos = [
            exps
            for exps in itertools.product(range(self.degree + 1), repeat=dims)
            if sum(exps) <= self.degree
        ]
        combos.sort(key=lambda e: (sum(e), e))
        return np.asarray(combos, dtype=int)

    def _local_solve(
        self, point: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the weighted local polynomial around one query point.

        Parameters
        ----------
        point : np.ndarray of shape (n_dims,)
            Query point in standardized coordinates.

        Returns
        -------
        coef : np.ndarray of shape (n_terms,)
            Local polynomial coefficients in centered, standardized coordinates.
        var_diag : np.ndarray of shape (n_terms,)
            Sandwich variance of each coefficient, in units of the noise variance.
        value_weights : np.ndarray of shape (n_neighbors,)
            Linear smoother weights producing the fitted value at *point*.
        neighbors : np.ndarray of shape (n_neighbors,)
            Indices of the training points used, aligned with *value_weights*.
        """
        idx = np.asarray(self._tree.query_ball_point(point, self._bandwidth), dtype=int)
        if idx.size < self._min_points:
            _, idx = self._tree.query(point, k=self._min_points)
            idx = np.atleast_1d(np.asarray(idx, dtype=int))

        local = self._coords_z[idx] - point
        radius = float(np.max(np.linalg.norm(local, axis=1)))
        radius = max(radius, _JITTER)
        dist = np.linalg.norm(local, axis=1) / radius
        kernel = np.clip(1.0 - np.clip(dist, 0.0, 1.0) ** 3, 0.0, None) ** 3
        kernel = np.maximum(kernel, 1e-6)
        weights = kernel * self._weights[idx]

        design = np.prod(local[:, None, :] ** self._exponents[None, :, :], axis=2)
        gram = design.T @ (weights[:, None] * design)
        gram_pinv = np.linalg.pinv(gram)
        smoother = gram_pinv @ (design.T * weights)
        coef = smoother @ self.values_[idx]

        noise = 1.0 / self._weights[idx] if self._sigma is not None else np.ones(idx.size)
        var_diag = np.einsum("ij,j,ij->i", smoother, noise, smoother)
        return coef, var_diag, smoother[0], idx

    def _local_pass(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the local fit at a set of training points.

        Parameters
        ----------
        points : np.ndarray of shape (n_eval,)
            Indices into the training set.

        Returns
        -------
        fitted : np.ndarray of shape (n_eval,)
            Fitted values.
        hat : np.ndarray of shape (n_eval,)
            Diagonal of the smoother matrix at those points.
        """
        fitted = np.empty(points.shape[0])
        hat = np.empty(points.shape[0])
        for j, i in enumerate(points):
            coef, _, value_weights, neighbors = self._local_solve(self._coords_z[i])
            fitted[j] = coef[0]
            match = np.flatnonzero(neighbors == i)
            hat[j] = float(value_weights[match[0]]) if match.size else 0.0
        return fitted, hat

    def _fit_local_poly(self) -> None:
        """
        Fit local polynomial regression and select the bandwidth.

        Raises
        ------
        ValueError
            If there are too few points to support the requested local degree.
        """
        from scipy.spatial import cKDTree

        n_points = self.coords_.shape[0]
        self._center = self.coords_.mean(axis=0)
        spread = self.coords_.std(axis=0)
        spread[spread <= 0] = 1.0
        self._scale = spread
        self._coords_z = (self.coords_ - self._center) / self._scale

        self._exponents = self._poly_exponents()
        n_terms = self._exponents.shape[0]
        if n_points < n_terms + 1:
            raise ValueError(
                f"Local polynomial of degree {self.degree} in {self.n_features_in_} dimensions "
                f"needs more than {n_terms} points, got {n_points}"
            )
        self._min_points = min(2 * n_terms, n_points)
        self._tree = cKDTree(self._coords_z)

        knn_dist, _ = self._tree.query(self._coords_z, k=self._min_points)
        knn_dist = np.atleast_2d(knn_dist)[:, -1]
        h_min = float(np.percentile(knn_dist, 75))
        diameter = float(np.linalg.norm(self._coords_z.max(axis=0) - self._coords_z.min(axis=0)))
        h_max = max(diameter, 2 * h_min)
        h_min = max(h_min, _JITTER)

        rng = np.random.default_rng(self.random_state)
        n_probe = min(120, n_points)
        probe = rng.choice(n_points, size=n_probe, replace=False)

        if isinstance(self.smoothing, float):
            bandwidth = self.smoothing
            source = "fixed"
        else:
            grid = np.geomspace(h_min, h_max, 10)
            rss_grid = np.empty(grid.shape[0])
            score_grid = np.empty(grid.shape[0])
            for i, h in enumerate(grid):
                self._bandwidth = float(h)
                fitted, hat = self._local_pass(probe)
                resid = self.values_[probe] - fitted
                weighted = float(np.sum(self._weights[probe] * resid**2)) * n_points / n_probe
                rss_grid[i] = weighted
                denom = max(1.0 - float(np.mean(hat)), 1e-6)
                score_grid[i] = float(np.mean(resid**2)) / denom**2
            if self.smoothing == "sigma":
                bandwidth = float(_interp_log(grid, rss_grid, float(n_points)))
                source = "sigma"
            else:
                bandwidth = float(grid[int(np.argmin(score_grid))])
                source = "gcv"

        self._bandwidth = bandwidth * self.smoothing_scale
        self.smoothing_ = self._bandwidth
        self.smoothing_source_ = source

        fitted, hat = self._local_pass(np.arange(n_points))
        self._fitted_values = fitted
        self.effective_dof_ = float(np.sum(hat))

    def _eval_local_poly(
        self, query: np.ndarray, orders: np.ndarray, return_std: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate local polynomial partials at the query points.

        Parameters
        ----------
        query : np.ndarray of shape (n_query, n_dims)
            Query points.
        orders : np.ndarray of shape (n_orders, n_dims)
            Derivative orders.
        return_std : bool
            Whether to compute standard errors.

        Returns
        -------
        values : np.ndarray of shape (n_query, n_orders)
            Evaluated partials.
        std : np.ndarray of shape (n_query, n_orders)
            Standard errors, left as zeros when *return_std* is ``False``.
        """
        query_z = (query - self._center) / self._scale
        n_orders = orders.shape[0]

        # Map each requested order to its monomial column and derivative factor.
        columns = np.empty(n_orders, dtype=int)
        factors = np.empty(n_orders)
        for k, order in enumerate(orders):
            match = np.flatnonzero(np.all(self._exponents == order[None, :], axis=1))
            columns[k] = int(match[0])
            factors[k] = np.prod([math.factorial(int(o)) for o in order]) / np.prod(
                self._scale ** order.astype(float)
            )

        scale = 1.0 if self._sigma is not None else self.residual_std_**2
        values = np.empty((query.shape[0], n_orders))
        stds = np.zeros_like(values)

        for i in range(query.shape[0]):
            coef, var_diag, _, _ = self._local_solve(query_z[i])
            values[i] = coef[columns] * factors
            if return_std:
                stds[i] = np.sqrt(np.maximum(scale * var_diag[columns] * factors**2, 0.0))
        return values, stds

    # -- Gaussian process ---------------------------------------------------

    def _gp_kernel(self, a: np.ndarray, b: np.ndarray, sf2: float, ls: np.ndarray) -> np.ndarray:
        """
        Anisotropic squared-exponential kernel matrix.

        Parameters
        ----------
        a : np.ndarray of shape (n_a, n_dims)
            First set of standardized coordinates.
        b : np.ndarray of shape (n_b, n_dims)
            Second set of standardized coordinates.
        sf2 : float
            Signal variance.
        ls : np.ndarray of shape (n_dims,)
            Length scales.

        Returns
        -------
        np.ndarray of shape (n_a, n_b)
            Kernel matrix.
        """
        diff = (a[:, None, :] - b[None, :, :]) / ls[None, None, :]
        return sf2 * np.exp(-0.5 * np.sum(diff**2, axis=2))

    def _fit_gp(self) -> None:
        """
        Fit a Gaussian process and select its hyperparameters.

        Raises
        ------
        ValueError
            If the training set is larger than :attr:`max_points`.
        """
        from scipy.linalg import cho_factor, cho_solve
        from scipy.optimize import minimize

        n_points, n_dims = self.coords_.shape
        if n_points > self.max_points:
            raise ValueError(
                f"method='gp' has cubic cost and is capped at max_points={self.max_points}; "
                f"got {n_points} points. Subsample, raise max_points, or use "
                "method='tensor_spline'."
            )

        self._center = self.coords_.mean(axis=0)
        spread = self.coords_.std(axis=0)
        spread[spread <= 0] = 1.0
        self._scale = spread
        self._coords_z = (self.coords_ - self._center) / self._scale

        self._y_mean = float(self.values_.mean())
        y = self.values_ - self._y_mean
        y_var = max(float(np.var(y)), _JITTER)

        if self.length_scale is None:
            init_ls = np.ones(n_dims)
            fixed_ls = None
        else:
            fixed_ls = np.asarray(self.length_scale, dtype=np.float64).ravel()
            if fixed_ls.size == 1:
                fixed_ls = np.full(n_dims, float(fixed_ls[0]))
            if fixed_ls.size != n_dims:
                raise ValueError(f"length_scale must be a scalar or have {n_dims} entries")
            if np.any(fixed_ls <= 0):
                raise ValueError("length_scale entries must be strictly positive")
            init_ls = fixed_ls

        if isinstance(self.smoothing, float):
            noise_var = self.smoothing
            noise_source = "fixed"
        elif self._sigma is not None:
            # Covers both smoothing='sigma' and smoothing='auto' with a known noise level.
            noise_var = float(np.mean(self._sigma**2))
            noise_source = "sigma"
        else:
            noise_var = 0.01 * y_var
            noise_source = "marginal_likelihood"

        if self._sigma is None:
            noise_shape = np.ones(n_points)
        else:
            noise_shape = self._sigma**2 / float(np.mean(self._sigma**2))

        fit_noise = noise_source == "marginal_likelihood"

        def unpack(theta: np.ndarray) -> tuple[float, np.ndarray, float]:
            sf2 = math.exp(theta[0])
            ls = fixed_ls if fixed_ls is not None else np.exp(theta[1 : 1 + n_dims])
            nv = math.exp(theta[-1]) if fit_noise else noise_var
            return sf2, ls, nv

        def negative_log_marginal(theta: np.ndarray) -> float:
            sf2, ls, nv = unpack(theta)
            kernel = self._gp_kernel(self._coords_z, self._coords_z, sf2, ls)
            kernel[np.diag_indices(n_points)] += nv * noise_shape + _JITTER * sf2
            try:
                factor = cho_factor(kernel, lower=True)
            except np.linalg.LinAlgError:
                return 1e12
            alpha = cho_solve(factor, y)
            log_det = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
            return 0.5 * float(y @ alpha) + 0.5 * log_det

        theta0 = [math.log(y_var)]
        bounds = [(math.log(y_var) - 8, math.log(y_var) + 8)]
        if fixed_ls is None:
            theta0 += list(np.log(init_ls))
            bounds += [(math.log(0.02), math.log(20.0))] * n_dims
        if fit_noise:
            theta0.append(math.log(noise_var))
            bounds.append((math.log(1e-8 * y_var), math.log(y_var)))

        opt = minimize(
            negative_log_marginal,
            np.asarray(theta0),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200},
        )
        sf2, ls, noise_var = unpack(opt.x)
        noise_var *= self.smoothing_scale

        kernel = self._gp_kernel(self._coords_z, self._coords_z, sf2, ls)
        kernel_noisy = kernel.copy()
        kernel_noisy[np.diag_indices(n_points)] += noise_var * noise_shape + _JITTER * sf2
        factor = cho_factor(kernel_noisy, lower=True)
        self._gp_alpha = cho_solve(factor, y)
        self._gp_kinv = cho_solve(factor, np.eye(n_points))
        self._gp_signal_var = sf2
        self._gp_length_scale = ls
        self._gp_noise_var = noise_var

        self._fitted_values = kernel @ self._gp_alpha + self._y_mean
        self.effective_dof_ = float(np.trace(kernel @ self._gp_kinv))
        self.smoothing_ = float(noise_var)
        self.smoothing_source_ = noise_source

    def _gp_cross_covariance(self, query_z: np.ndarray, order: np.ndarray) -> np.ndarray:
        """
        Derivative cross-covariance between a partial at the query points and the data.

        Parameters
        ----------
        query_z : np.ndarray of shape (n_query, n_dims)
            Standardized query points.
        order : np.ndarray of shape (n_dims,)
            Derivative order per dimension.

        Returns
        -------
        np.ndarray of shape (n_query, n_points)
            The requested partial derivative of the kernel with respect to the query
            coordinates, in original (unstandardized) units.
        """
        ls = self._gp_length_scale
        diff = (query_z[:, None, :] - self._coords_z[None, :, :]) / ls[None, None, :]
        cov = self._gp_signal_var * np.exp(-0.5 * np.sum(diff**2, axis=2))
        for dim in range(self.n_features_in_):
            m = int(order[dim])
            if m == 0:
                continue
            step = ls[dim] * self._scale[dim]
            cov = cov * ((-1.0) ** m) * _hermite_e(m, diff[:, :, dim]) / step**m
        return cov

    def _gp_prior_variance(self, order: np.ndarray) -> float:
        """
        Prior variance of a partial derivative of the GP.

        Parameters
        ----------
        order : np.ndarray of shape (n_dims,)
            Derivative order per dimension.

        Returns
        -------
        float
            ``Var(d^order f(x))`` under the prior.
        """
        var = self._gp_signal_var * (-1.0) ** int(np.sum(order))
        for dim in range(self.n_features_in_):
            m = int(order[dim])
            if m == 0:
                continue
            step = self._gp_length_scale[dim] * self._scale[dim]
            var *= _hermite_e_at_zero(2 * m) / step ** (2 * m)
        return float(var)

    def _eval_gp(
        self, query: np.ndarray, orders: np.ndarray, return_std: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate GP posterior partials at the query points.

        Parameters
        ----------
        query : np.ndarray of shape (n_query, n_dims)
            Query points.
        orders : np.ndarray of shape (n_orders, n_dims)
            Derivative orders.
        return_std : bool
            Whether to compute posterior standard deviations.

        Returns
        -------
        values : np.ndarray of shape (n_query, n_orders)
            Posterior mean partials.
        std : np.ndarray of shape (n_query, n_orders)
            Posterior standard deviations, left as zeros when *return_std* is ``False``.
        """
        query_z = (query - self._center) / self._scale
        values = np.empty((query.shape[0], orders.shape[0]))
        stds = np.zeros_like(values)

        for k, order in enumerate(orders):
            cross = self._gp_cross_covariance(query_z, order)
            values[:, k] = cross @ self._gp_alpha
            if int(np.sum(order)) == 0:
                values[:, k] += self._y_mean
            if return_std:
                prior = self._gp_prior_variance(order)
                explained = np.einsum("ij,jk,ik->i", cross, self._gp_kinv, cross)
                stds[:, k] = np.sqrt(np.maximum(prior - explained, 0.0))
        return values, stds


def _interp_log(grid: np.ndarray, response: np.ndarray, target: float) -> float:
    """
    Invert a monotonically increasing response curve on a logarithmic grid.

    Parameters
    ----------
    grid : np.ndarray
        Strictly positive, increasing smoothing levels.
    response : np.ndarray
        Response (residual sum of squares) at each grid point; assumed increasing.
    target : float
        Desired response value.

    Returns
    -------
    float
        The grid value whose response matches *target*, clipped to the grid range.
    """
    monotone = np.maximum.accumulate(response)
    if target <= monotone[0]:
        return float(grid[0])
    if target >= monotone[-1]:
        return float(grid[-1])
    return float(np.exp(np.interp(target, monotone, np.log(grid))))


def estimate_partial_derivatives(
    coords: np.ndarray | Sequence[np.ndarray],
    values: np.ndarray,
    order: Any,
    method: str = "tensor_spline",
    sigma: float | np.ndarray | None = None,
    query: np.ndarray | Sequence[np.ndarray] | None = None,
    return_std: bool = False,
    **kwargs: Any,
) -> tuple[np.ndarray, ...]:
    """
    Estimate partial derivatives of a smoothed N-D surface in one call.

    Convenience wrapper around :class:`SurfaceDerivatives` for the common case of
    fitting a smoother and evaluating partials at the sample locations.

    Parameters
    ----------
    coords : np.ndarray of shape (n_points, n_dims), or sequence of 1-D arrays
        Sample locations, scattered or a rectangular grid (see
        :meth:`SurfaceDerivatives.fit`).
    values : np.ndarray
        Observed values, flat or grid-shaped to match *coords*.
    order : tuple of int, or sequence of tuples
        Derivative orders per dimension, e.g. ``[(1, 0), (0, 1)]``.
    method : str
        Smoother to use: ``"tensor_spline"``, ``"local_poly"``, or ``"gp"``.
    sigma : float or np.ndarray of shape (n_points,), optional
        Known measurement noise standard deviation.
    query : np.ndarray of shape (n_query, n_dims), optional
        Where to evaluate. Defaults to the (flattened) sample locations.
    return_std : bool
        If ``True``, also return the standard error of each partial derivative.
    **kwargs
        Extra keyword arguments forwarded to :class:`SurfaceDerivatives`.

    Returns
    -------
    values : np.ndarray of shape (n_query,)
        The smoothed surface at the query points.
    partials : np.ndarray of shape (n_query, n_orders)
        One column per requested derivative order.
    std : np.ndarray of shape (n_query, n_orders)
        Standard errors. Only returned when *return_std* is ``True``.

    Raises
    ------
    ValueError
        If the inputs or the derivative orders are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from jaxsr import estimate_partial_derivatives
    >>> x = np.linspace(0, 1, 20)
    >>> t = np.linspace(0, 1, 20)
    >>> XX, TT = np.meshgrid(x, t, indexing="ij")
    >>> Z = XX**2 + 3 * TT
    >>> _, dz = estimate_partial_derivatives([x, t], Z, order=[(1, 0), (0, 1)])
    >>> dz.shape
    (400, 2)
    """
    estimator = SurfaceDerivatives(method=method, **kwargs)
    estimator.fit(coords, values, sigma=sigma)
    target = estimator.coords_ if query is None else query
    return estimator.derivatives(target, order=order, return_std=return_std)
