r"""
Superposition: discover a symbolic transform that collapses a family of curves.

A large class of experiments produces *a set of curves indexed by a condition*, where
the scientific claim is that some transform of the axes collapses them onto a single
master curve. Rheology calls it time-temperature superposition; the same shape appears
in time-concentration and time-moisture superposition, finite-size scaling, Larson-Miller
creep master curves, and isoconversional kinetics.

The practitioner normally does two things by hand: shift each curve until the overlap
looks right, then *assert* a functional form (Arrhenius, WLF, a polynomial) for how the
shift depends on the condition. :class:`SuperpositionRegressor` learns that second step
symbolically, from a sparse library of candidate shift laws.

The model
---------
With the abscissa in log units and a dimensionless condition coordinate
:math:`q = (c - c_{ref}) / c_{ref}`,

.. math::

    y(x, q) = f\bigl(x + \sigma\, s(q)\bigr) + v(q),

where :math:`f` is the unknown master curve, :math:`s` the horizontal shift
(:math:`s = \log_{10} a_T`), :math:`v` an optional vertical shift, and :math:`\sigma`
a sign fixed by the domain (``+1`` for a frequency-like abscissa, ``-1`` for a
time-like one). Differentiating with respect to :math:`q` eliminates the unknown
master curve entirely:

.. math::

    y_q = \sigma\, s'(q)\, y_x + v'(q).

So a sparse regression of :math:`y_q` against the structured blocks
:math:`\Theta(q) \odot y_x \,|\, \Theta(q)` recovers :math:`s'` and :math:`v'` as
coefficient *functions*, which are integrated back with the anchor :math:`s(0) = 0`.
Both partials come from one smoothed surface per channel
(:class:`~jaxsr.derivatives.SurfaceDerivatives`), never from finite differences of
noisy data.

Two findings shape this API
---------------------------
**The transform is identifiable; the equation is not.** Over a realistic condition
window, :math:`1/(1+q)^2` (Arrhenius), :math:`1/(c_2+q)^2` (WLF) and low-order
polynomials span nearly the same function space. In a 90-fit synthetic study only 62%
of fits selected the true Arrhenius basis, yet all of the structural variants produced
the same transform to ~0.01 decades. :attr:`SuperpositionRegressor.shift_expression_`
is therefore *not* the headline result -- the transform is, together with a
structure-independent physical summary such as
:meth:`~SuperpositionRegressor.effective_activation_energy`.

**Expression stability is anti-correlated with validity.** In the same study a negative
control (two relaxation groups with different activation energies, for which no scalar
shift factor exists) produced a *perfectly stable* shift law -- 12/12 replicates agreed
on the structure -- while the genuine case disagreed 38% of the time. Only collapse on
*withheld conditions* separated them: at the noise floor for the real case, eight times
the noise floor for the control. :attr:`SuperpositionRegressor.validity_report_` is
consequently a graded verdict from leave-one-condition-out collapse, never a binary
claim read off a visual collapse.

Conventions
-----------
These are the classic source of silent error in superposition work, so they are
explicit and validated rather than assumed:

- the abscissa is *already in log units*, and the domain sets the sign
  (``"frequency"`` gives :math:`z = x + s`, ``"time"`` gives :math:`z = x - s`);
- temperatures must be in kelvin before reciprocal features are built
  (``condition_scale="kelvin"`` checks this and refuses a Celsius-looking column);
- the regression runs in the dimensionless :math:`q`, so design-matrix columns stay
  order-one instead of mixing :math:`T^{-2}` against polynomials;
- the master-curve smoother is fitted separately from the derivative surface -- reusing
  the surface that produced the shift law to also certify it would defeat the
  validation.

Examples
--------
>>> from jaxsr.superposition import SuperpositionRegressor
>>> model = SuperpositionRegressor(          # doctest: +SKIP
...     condition="temperature", abscissa="log_omega", response="log_Gp",
...     domain="frequency", candidate_families=("arrhenius", "wlf", "polynomial"),
... )
>>> model.fit(data)                          # doctest: +SKIP
>>> model.shift_factors([300.0, 320.0])      # doctest: +SKIP
>>> print(model.validity_report_.verdict)    # doctest: +SKIP
'supported'
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from .basis import BasisLibrary
from .derivatives import SurfaceDerivatives
from .regressor import SymbolicRegressor
from .uncertainty import summarize_selection_replicates

__all__ = [
    "MasterCurve",
    "ShiftTerm",
    "SuperpositionRegressor",
    "ValidityReport",
    "collapse_rmse",
]

#: Molar gas constant in J/(mol K), used by :meth:`SuperpositionRegressor.effective_activation_energy`.
GAS_CONSTANT = 8.314462618

_VALID_DOMAINS = ("frequency", "time")
_VALID_VERTICAL = ("none", "shared", "per_channel")
_VALID_VALIDATION = ("loco", "none")
_VALID_STABILITY = ("residual", "conditions")
_VALID_WEIGHTING = ("derivative_se", "none")

#: Smallest denominator tolerated by the reciprocal shift families before the column is
#: declared out of domain (signalled with NaN, which drops the whole column in ``fit``).
_MIN_DENOM = 1e-6


# ===========================================================================
# Candidate shift laws
# ===========================================================================


@dataclass(frozen=True)
class ShiftTerm:
    r"""
    One candidate term of a shift law, as a derivative/antiderivative pair.

    The regression works on :math:`s'(q)`, but the reported transform needs
    :math:`s(q)`. Carrying both halves analytically keeps the integration exact and
    lets :attr:`SuperpositionRegressor.shift_expression_` be a real symbolic
    expression rather than a quadrature table.

    Parameters
    ----------
    name : str
        Name of the *derivative* term as it appears in the design matrix, e.g.
        ``"1/(1+q)^2"``. Parametric terms name their free parameters, e.g.
        ``"1/(c2+q)^2"``.
    deriv : callable
        ``deriv(q, **params) -> np.ndarray``. Evaluates :math:`\theta_j(q)`, a
        candidate term of :math:`s'(q)`.
    antideriv : callable
        ``antideriv(q, **params) -> np.ndarray``. Evaluates
        :math:`\int_0^q \theta_j(u)\,du`, anchored so that the reference condition
        has zero shift.
    expression : str
        The antiderivative as a Python/sympy-parseable string in ``q``, with free
        parameters left as symbols.
    complexity : int
        Complexity score used for Pareto ranking.
    param_bounds : dict, optional
        ``{param_name: (lower, upper)}`` for a parametric term, or None.
    log_scale : bool
        Search the parameter in log space. Only meaningful with *param_bounds*.

    Examples
    --------
    >>> import numpy as np
    >>> term = ShiftTerm(
    ...     name="1/(1+q)^2",
    ...     deriv=lambda q: 1.0 / (1.0 + q) ** 2,
    ...     antideriv=lambda q: q / (1.0 + q),
    ...     expression="q/(1 + q)",
    ... )
    >>> float(term.antideriv(np.array([0.0]))[0])
    0.0
    """

    name: str
    deriv: Callable[..., np.ndarray]
    antideriv: Callable[..., np.ndarray]
    expression: str
    complexity: int = 2
    param_bounds: dict[str, tuple[float, float]] | None = None
    log_scale: bool = False

    @property
    def is_parametric(self) -> bool:
        """Whether the term carries free nonlinear parameters."""
        return bool(self.param_bounds)


def _constant_term() -> ShiftTerm:
    """Return the constant term of ``s'``, whose integral is a linear shift in ``q``."""
    return ShiftTerm(
        name="1",
        deriv=lambda q: np.ones_like(q),
        antideriv=lambda q: q,
        expression="q",
        complexity=1,
    )


def _polynomial_terms(degree: int) -> list[ShiftTerm]:
    """
    Build ``1, q, ..., q**degree`` as candidate terms of ``s'``.

    Parameters
    ----------
    degree : int
        Highest power of ``q`` in the *derivative*; the recovered shift law reaches
        one degree higher.

    Returns
    -------
    list of ShiftTerm
    """
    terms = [_constant_term()]
    for p in range(1, degree + 1):
        terms.append(
            ShiftTerm(
                name=f"q^{p}",
                deriv=lambda q, p=p: q**p,
                antideriv=lambda q, p=p: q ** (p + 1) / (p + 1),
                expression=f"q**{p + 1}/{p + 1}",
                complexity=1 + p,
            )
        )
    return terms


def _arrhenius_term() -> ShiftTerm:
    r"""
    Return the Arrhenius term.

    With :math:`\log_{10} a_T = A\,(1/T - 1/T_{ref})` and :math:`T = T_{ref}(1+q)`,
    the shift is :math:`s(q) = A\,(1/(1+q) - 1)/T_{ref}`, so
    :math:`s'(q) \propto 1/(1+q)^2`.

    Returns
    -------
    ShiftTerm
    """
    return ShiftTerm(
        name="1/(1+q)^2",
        deriv=lambda q: _safe_reciprocal_square(1.0 + q),
        antideriv=lambda q: q / (1.0 + q),
        expression="q/(1 + q)",
        complexity=2,
    )


def _wlf_term(bounds: tuple[float, float]) -> ShiftTerm:
    r"""
    Return the WLF term, whose denominator constant is fitted.

    With :math:`\log_{10} a_T = -C_1 (T - T_{ref}) / (C_2 + T - T_{ref})` and
    :math:`c_2 = C_2 / T_{ref}`, the shift is
    :math:`s(q) \propto q / (c_2 (c_2 + q))` and :math:`s'(q) \propto 1/(c_2+q)^2`.

    Parameters
    ----------
    bounds : tuple of float
        Search bounds ``(lower, upper)`` on the dimensionless ``c2 = C2 / c_ref``.

    Returns
    -------
    ShiftTerm
    """
    return ShiftTerm(
        name="1/(c2+q)^2",
        deriv=lambda q, c2: _safe_reciprocal_square(c2 + q),
        antideriv=lambda q, c2: q / (c2 * (c2 + q)),
        expression="q/(c2*(c2 + q))",
        complexity=3,
        param_bounds={"c2": (float(bounds[0]), float(bounds[1]))},
        log_scale=True,
    )


def _safe_reciprocal_square(d: Any) -> Any:
    """
    Return ``1 / d**2``, signalling an out-of-domain denominator with NaN.

    A NaN column is dropped wholesale by :meth:`SymbolicRegressor.fit`, which is the
    behaviour wanted here: a candidate whose pole sits inside the measured condition
    range should not be scored at all.

    Parameters
    ----------
    d : array-like
        Denominator values.

    Returns
    -------
    array-like
        ``1 / d**2`` where ``|d| > 1e-6``, NaN elsewhere. Works on both NumPy arrays
        and JAX arrays, so the same term serves the basis library and the
        integration path.
    """
    xp = jnp if isinstance(d, jnp.ndarray) else np
    safe = xp.where(xp.abs(d) > _MIN_DENOM, d, 1.0)
    return xp.where(xp.abs(d) > _MIN_DENOM, 1.0 / safe**2, xp.nan)


def _build_terms(
    families: Sequence[str],
    poly_degree: int,
    wlf_bounds: tuple[float, float],
    reciprocal_ok: bool,
) -> list[ShiftTerm]:
    """
    Assemble the candidate library from the requested families.

    Parameters
    ----------
    families : sequence of str
        Any of ``"polynomial"``, ``"arrhenius"``, ``"wlf"``.
    poly_degree : int
        Highest power of ``q`` for the polynomial family.
    wlf_bounds : tuple of float
        Bounds on the dimensionless WLF denominator constant.
    reciprocal_ok : bool
        Whether reciprocal-condition families are physically meaningful, i.e. the
        condition is a temperature in kelvin.

    Returns
    -------
    list of ShiftTerm
        Deduplicated by name, in a stable order.

    Raises
    ------
    ValueError
        If a family is unknown, if none are given, or if a reciprocal family is
        requested for a condition that is not an absolute temperature.
    """
    if not families:
        raise ValueError("candidate_families must name at least one family")

    known = {"polynomial", "arrhenius", "wlf"}
    unknown = [f for f in families if f not in known]
    if unknown:
        raise ValueError(f"Unknown candidate_families {unknown}. Choose from {sorted(known)}")

    reciprocal = [f for f in families if f in ("arrhenius", "wlf")]
    if reciprocal and not reciprocal_ok:
        raise ValueError(
            f"candidate_families {reciprocal} encode reciprocal-temperature physics and "
            "are only meaningful for an absolute temperature. Pass "
            "condition_scale='kelvin' (with the condition column in kelvin), or drop "
            "them and use candidate_families=('polynomial',)."
        )

    terms: list[ShiftTerm] = []
    if "polynomial" in families:
        terms.extend(_polynomial_terms(poly_degree))
    else:
        # Every law needs a constant available in s'; without it no pure offset can be
        # expressed and the fit is forced through a curved form it may not want.
        terms.append(_constant_term())
    if "arrhenius" in families:
        terms.append(_arrhenius_term())
    if "wlf" in families:
        terms.append(_wlf_term(wlf_bounds))

    seen: set[str] = set()
    unique: list[ShiftTerm] = []
    for term in terms:
        if term.name not in seen:
            seen.add(term.name)
            unique.append(term)
    return unique


# ===========================================================================
# Fitted laws
# ===========================================================================


@dataclass
class _Law:
    """
    A fitted shift law: a sparse combination of :class:`ShiftTerm` antiderivatives.

    Parameters
    ----------
    terms : list of tuple
        ``(coefficient, ShiftTerm, params)`` triples. *params* is the fitted value of
        each free nonlinear parameter, empty for non-parametric terms.
    """

    terms: list[tuple[float, ShiftTerm, dict[str, float]]] = field(default_factory=list)

    def value(self, q: np.ndarray) -> np.ndarray:
        """Evaluate the law itself, anchored at ``value(0) == 0``."""
        q = np.asarray(q, dtype=np.float64)
        out = np.zeros_like(q)
        for coef, term, params in self.terms:
            out = out + coef * np.asarray(term.antideriv(q, **params), dtype=np.float64)
        return out

    def derivative(self, q: np.ndarray) -> np.ndarray:
        """Evaluate the derivative of the law with respect to ``q``."""
        q = np.asarray(q, dtype=np.float64)
        out = np.zeros_like(q)
        for coef, term, params in self.terms:
            out = out + coef * np.asarray(term.deriv(q, **params), dtype=np.float64)
        return out

    @property
    def is_empty(self) -> bool:
        """Whether no term survived selection, i.e. the law is identically zero."""
        return not self.terms

    def expression(self) -> str:
        """Render the law as a human-readable expression in ``q``."""
        if not self.terms:
            return "0"
        parts = []
        for coef, term, params in self.terms:
            expr = term.expression
            for pname, value in params.items():
                expr = _substitute_symbol(expr, pname, value)
            parts.append(f"{coef:.6g}*({expr})")
        return " + ".join(parts)

    def term_names(self) -> list[str]:
        """Names of the selected derivative terms, in selection order."""
        return [term.name for _, term, _ in self.terms]


def _substitute_symbol(expression: str, symbol: str, value: float) -> str:
    """
    Replace whole-word occurrences of *symbol* in *expression* with *value*.

    Parameters
    ----------
    expression : str
        Expression text.
    symbol : str
        Parameter name to substitute.
    value : float
        Fitted value.

    Returns
    -------
    str
        Expression with the symbol replaced.
    """
    import re

    return re.sub(r"\b" + re.escape(symbol) + r"\b", f"{value:.6g}", expression)


# ===========================================================================
# Master curve
# ===========================================================================


@dataclass
class MasterCurve:
    """
    A reconstructed master curve for one channel, with an uncertainty band.

    Fitted by a smoother that is deliberately *separate* from the derivative surface
    used to discover the shift law -- reusing that surface to also certify the
    collapse would let the discovery grade its own homework.

    Parameters
    ----------
    channel : Any
        Channel label this curve belongs to.
    z : np.ndarray of shape (n_grid,)
        Reduced-coordinate grid spanning the observed range.
    y : np.ndarray of shape (n_grid,)
        Smoothed master curve on that grid.
    std : np.ndarray of shape (n_grid,)
        Standard error of the smoothed value.
    z_min : float
        Lowest reduced coordinate covered by data.
    z_max : float
        Highest reduced coordinate covered by data.
    n_points : int
        Number of observations that went into the curve.
    smoother : SurfaceDerivatives
        The fitted 1-D smoother, for evaluation at arbitrary points.
    """

    channel: Any
    z: np.ndarray
    y: np.ndarray
    std: np.ndarray
    z_min: float
    z_max: float
    n_points: int
    smoother: SurfaceDerivatives = field(repr=False)

    def predict(
        self,
        z: np.ndarray,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the master curve.

        Parameters
        ----------
        z : np.ndarray of shape (n_query,)
            Reduced coordinates. Values outside ``[z_min, z_max]`` are extrapolations
            of the smoother and are not flagged here; use :meth:`covers`.
        return_std : bool
            Also return the standard error of the smoothed value.

        Returns
        -------
        y : np.ndarray of shape (n_query,)
            Master-curve values.
        std : np.ndarray of shape (n_query,)
            Standard errors. Only returned when *return_std* is True.
        """
        z = np.asarray(z, dtype=np.float64).ravel()
        return self.smoother.predict(z.reshape(-1, 1), return_std=return_std)

    def covers(self, z: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask of query points inside the curve's observed range.

        Parameters
        ----------
        z : np.ndarray of shape (n_query,)
            Reduced coordinates.

        Returns
        -------
        np.ndarray of bool, shape (n_query,)
        """
        z = np.asarray(z, dtype=np.float64).ravel()
        return (z >= self.z_min) & (z <= self.z_max)


def collapse_rmse(
    z: np.ndarray,
    y: np.ndarray,
    channel: np.ndarray | None = None,
    smoothing: float | str = "auto",
    degree: int = 3,
    sigma: float | None = None,
) -> float:
    """
    Root-mean-square scatter of points about a smooth curve through them.

    This is the quantity that decides whether a proposed collapse is any good: fit a
    single smooth curve per channel to the reduced coordinates and measure how far the
    data sit from it. It is public because it is also the right way to score a
    collapse produced by hand or by another tool.

    Parameters
    ----------
    z : np.ndarray of shape (n_samples,)
        Reduced abscissa (shifted).
    y : np.ndarray of shape (n_samples,)
        Reduced response (vertically shifted, if applicable).
    channel : np.ndarray of shape (n_samples,), optional
        Channel label per row. Each channel gets its own master curve; the returned
        value pools their residuals.
    smoothing : float or str
        Passed to :class:`~jaxsr.derivatives.SurfaceDerivatives`. ``"auto"`` selects
        the smoothing level by GCV.
    degree : int
        Spline degree of the master-curve smoother.
    sigma : float, optional
        Known measurement noise standard deviation. Required by
        ``smoothing="sigma"``, and otherwise only sharpens the reported uncertainty.

    Returns
    -------
    float
        Pooled root-mean-square residual.

    Raises
    ------
    ValueError
        If the arrays disagree in length, contain non-finite entries, or a channel has
        too few distinct abscissa values to smooth.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.linspace(-2, 2, 60)
    >>> y = np.tanh(z)
    >>> bool(collapse_rmse(z, y) < 1e-3)
    True
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if z.shape != y.shape:
        raise ValueError(f"z has {z.size} values but y has {y.size}")
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(y))):
        raise ValueError("z and y must be finite")

    labels = np.zeros(z.size, dtype=int) if channel is None else np.asarray(channel).ravel()
    if labels.shape != z.shape:
        raise ValueError(f"channel has {labels.size} labels but z has {z.size} values")

    total = 0.0
    count = 0
    for label in _unique_preserving_order(labels):
        mask = labels == label
        curve = _fit_master_curve(
            label, z[mask], y[mask], smoothing=smoothing, degree=degree, sigma=sigma
        )
        resid = y[mask] - curve.predict(z[mask])
        total += float(np.sum(resid**2))
        count += int(mask.sum())
    return math.sqrt(total / count)


def _fit_master_curve(
    channel: Any,
    z: np.ndarray,
    y: np.ndarray,
    smoothing: float | str = "auto",
    degree: int = 3,
    n_grid: int = 200,
    sigma: float | None = None,
) -> MasterCurve:
    """
    Fit a 1-D smoother to reduced coordinates and package it as a MasterCurve.

    Parameters
    ----------
    channel : Any
        Channel label.
    z, y : np.ndarray of shape (n_samples,)
        Reduced coordinates and responses.
    smoothing : float or str
        Smoothing level for the smoother.
    degree : int
        Spline degree.
    n_grid : int
        Number of grid points on which to report the curve and its band.
    sigma : float, optional
        Known measurement noise standard deviation.

    Returns
    -------
    MasterCurve

    Raises
    ------
    ValueError
        If too few distinct abscissa values are present to fit the smoother.
    """
    order = np.argsort(z)
    z_sorted = z[order]
    y_sorted = y[order]

    n_distinct = len(np.unique(z_sorted))
    effective_degree = int(min(degree, max(1, n_distinct - 2)))
    if n_distinct < effective_degree + 2:
        raise ValueError(
            f"Channel {channel!r} has only {n_distinct} distinct reduced abscissa "
            "values; at least 3 are needed to fit a master curve."
        )

    if smoothing == "sigma" and sigma is None:
        raise ValueError("smoothing='sigma' needs a sigma; pass one or use 'auto'")
    smoother = SurfaceDerivatives(
        method="tensor_spline",
        degree=effective_degree,
        smoothing=smoothing,
    ).fit(z_sorted.reshape(-1, 1), y_sorted, sigma=sigma)

    z_lo, z_hi = float(z_sorted[0]), float(z_sorted[-1])
    grid = np.linspace(z_lo, z_hi, n_grid)
    values, std = smoother.predict(grid.reshape(-1, 1), return_std=True)

    return MasterCurve(
        channel=channel,
        z=grid,
        y=np.asarray(values),
        std=np.asarray(std),
        z_min=z_lo,
        z_max=z_hi,
        n_points=int(z.size),
        smoother=smoother,
    )


# ===========================================================================
# Validity report
# ===========================================================================


@dataclass
class ValidityReport:
    """
    Graded verdict on whether the discovered collapse actually holds.

    The verdict comes from collapse on *withheld* conditions, not from the in-sample
    collapse and not from the stability of the selected expression. A negative control
    with no valid shift factor can produce a beautiful in-sample collapse and a
    perfectly reproducible shift law; only a condition that took no part in the
    smoothing or the discovery separates the cases.

    Parameters
    ----------
    verdict : str
        ``"supported"``, ``"weakly_supported"``, ``"not_supported"``, or
        ``"not_evaluated"`` when validation was disabled or impossible.
    noise_floor : float
        Estimated measurement noise standard deviation, in response units. The
        collapse cannot be better than this.
    noise_floor_source : str
        How the floor was estimated: ``"replicates"``, ``"curve_smoother"``, or
        ``"surface"`` (in decreasing order of directness).
    in_sample_collapse : float
        Pooled RMSE of the collapse on the conditions used for fitting. Reported for
        context only -- it is not the verdict.
    holdout : list of dict
        One entry per withheld condition, with keys ``"condition"``,
        ``"collapse_rmse"``, ``"ratio"`` (to the noise floor),
        ``"shift_predicted"``, ``"shift_aligned"``, ``"shift_error"``, and
        ``"coverage"`` (fraction of the withheld curve landing inside the master
        curve's range).
    holdout_collapse_median : float or None
        Median of the held-out collapse RMSEs.
    holdout_ratio_median : float or None
        Median held-out collapse RMSE as a multiple of the noise floor. This is the
        number the verdict is read from.
    shift_error_median : float or None
        Median absolute difference between the predicted shift for a withheld
        condition and the shift that would have aligned it best, in the units of the
        abscissa (decades, for a base-10 log abscissa).
    thresholds : tuple of float
        The ``(supported, weakly_supported)`` multiples of the noise floor used.
    flags : list of str
        Machine-readable warnings raised during fitting or validation.

    Examples
    --------
    >>> report = ValidityReport(                                  # doctest: +SKIP
    ...     verdict="supported", noise_floor=0.030, ...
    ... )
    >>> report.verdict                                            # doctest: +SKIP
    'supported'
    """

    verdict: str
    noise_floor: float
    noise_floor_source: str
    in_sample_collapse: float
    holdout: list[dict[str, Any]] = field(default_factory=list)
    holdout_collapse_median: float | None = None
    holdout_ratio_median: float | None = None
    shift_error_median: float | None = None
    thresholds: tuple[float, float] = (2.0, 4.0)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the report to a plain dictionary.

        Returns
        -------
        dict
            All fields, with the per-condition list copied.
        """
        return {
            "verdict": self.verdict,
            "noise_floor": self.noise_floor,
            "noise_floor_source": self.noise_floor_source,
            "in_sample_collapse": self.in_sample_collapse,
            "holdout": [dict(entry) for entry in self.holdout],
            "holdout_collapse_median": self.holdout_collapse_median,
            "holdout_ratio_median": self.holdout_ratio_median,
            "shift_error_median": self.shift_error_median,
            "thresholds": tuple(self.thresholds),
            "flags": list(self.flags),
        }

    def summary(self) -> str:
        """
        Return a human-readable summary of the verdict and the numbers behind it.

        Returns
        -------
        str
            Multi-line text.
        """
        lines = [
            "Superposition validity",
            "=" * 46,
            f"verdict              : {self.verdict}",
            f"noise floor          : {self.noise_floor:.4g} (from {self.noise_floor_source})",
            f"in-sample collapse   : {self.in_sample_collapse:.4g}",
        ]
        if self.holdout_collapse_median is not None:
            lines.append(
                f"held-out collapse    : {self.holdout_collapse_median:.4g} "
                f"({self.holdout_ratio_median:.2f} x noise floor, "
                f"{len(self.holdout)} conditions)"
            )
        if self.shift_error_median is not None:
            lines.append(f"held-out shift error : {self.shift_error_median:.4g}")
        lines.append(
            f"thresholds           : supported <= {self.thresholds[0]:g}x, "
            f"weakly <= {self.thresholds[1]:g}x noise floor"
        )
        if self.flags:
            lines.append(f"flags                : {', '.join(self.flags)}")
        return "\n".join(lines)


# ===========================================================================
# Internal data container
# ===========================================================================


@dataclass
class _Dataset:
    """Normalized tidy table plus the derived condition coordinate."""

    condition: np.ndarray
    x: np.ndarray
    y: np.ndarray
    channel: np.ndarray
    q: np.ndarray
    reference: float
    channels: list[Any]
    conditions: list[float]

    def subset(self, mask: np.ndarray) -> _Dataset:
        """Return a new dataset holding only the rows selected by *mask*."""
        return _Dataset(
            condition=self.condition[mask],
            x=self.x[mask],
            y=self.y[mask],
            channel=self.channel[mask],
            q=self.q[mask],
            reference=self.reference,
            channels=[c for c in self.channels if np.any(self.channel[mask] == c)],
            conditions=sorted(float(v) for v in np.unique(self.condition[mask])),
        )


def _unique_preserving_order(values: np.ndarray) -> list[Any]:
    """
    Return the distinct entries of *values* in first-appearance order.

    Parameters
    ----------
    values : np.ndarray
        Any 1-D array, including string or object dtype.

    Returns
    -------
    list
        Distinct values.
    """
    seen: list[Any] = []
    known: set[Any] = set()
    for value in values.tolist():
        if value not in known:
            known.add(value)
            seen.append(value)
    return seen


# ===========================================================================
# The estimator
# ===========================================================================


class SuperpositionRegressor:
    r"""
    Discover a symbolic transform that collapses a family of curves.

    Given a tidy table of ``(condition, abscissa, response[, channel])``, learn a
    sparse symbolic horizontal shift :math:`s(c)` -- and optionally a vertical shift
    :math:`v(c)` -- reconstruct the master curve, and *test whether the collapse
    actually holds* on conditions withheld from the fit.

    The abscissa must already be in log units. The shift is reported in those same
    units, so with a base-10 log abscissa the shift factors are
    :math:`\log_{10} a_T` in decades.

    Parameters
    ----------
    condition : str
        Name of the condition column (temperature, concentration, pressure, ...).
    abscissa : str
        Name of the log-abscissa column.
    response : str
        Name of the response column.
    channel : str, optional
        Name of the channel column. Channels (e.g. ``G'`` and ``G''``) share one
        horizontal shift but get their own master curve, and their own vertical shift
        when ``vertical_shift="per_channel"``. If None the whole table is one channel,
        labelled with the response column name.
    domain : str
        ``"frequency"`` for a frequency-like abscissa (:math:`z = x + s`) or
        ``"time"`` for a time-like one (:math:`z = x - s`).

        This *names* the transform rather than changing it: the collapse is identical
        either way, because the data alone fixes the reduced coordinate. What the
        domain decides is whether the offset is reported as :math:`+\log a_T` or
        :math:`-\log a_T` -- so getting it wrong flips the sign of every shift factor
        and of :meth:`effective_activation_energy`, while every plot still looks
        perfect. Hence no guessing from the data.
    condition_scale : str, optional
        ``"kelvin"`` declares the condition to be an absolute temperature, which
        enables the reciprocal-condition families and
        :meth:`effective_activation_energy`, and rejects a Celsius-looking column.
        None treats the condition as a generic scalar; only the polynomial family is
        then available.
    reference : float, optional
        Reference condition, anchoring :math:`s(c_{ref}) = 0`. Defaults to the median
        of the measured conditions, which keeps :math:`q` small at both ends.
    vertical_shift : str
        ``"none"``, ``"shared"`` (one :math:`v(c)` for all channels), or
        ``"per_channel"``.
    candidate_families : sequence of str
        Any of ``"arrhenius"``, ``"wlf"``, ``"polynomial"``.
    poly_degree : int
        Highest power of :math:`q` in the polynomial family's contribution to
        :math:`s'`; the recovered shift law reaches one degree higher.
    wlf_c2_bounds : tuple of float
        Search bounds on the dimensionless WLF denominator constant
        :math:`c_2 = C_2 / c_{ref}`.
    max_terms : int
        Maximum number of terms across both shift laws.
    selection : str
        Information criterion for selection: ``"aic"``, ``"aicc"``, or ``"bic"``.
    strategy : str
        Selection strategy, passed to :class:`~jaxsr.regressor.SymbolicRegressor`.
    smoother : str
        Derivative-surface smoother: ``"tensor_spline"``, ``"local_poly"``, or
        ``"gp"``.
    smoothing : float or str
        Smoothing level for the derivative surface: ``"auto"`` (GCV or marginal
        likelihood), ``"sigma"`` to match the noise level measured from replicates or
        from single-condition curves, or a number. Never chosen against the symbolic
        score -- tuning a smoother against the regression that consumes it can
        manufacture whichever law the regression prefers, and the failure is silent.
    surface_degree : int
        Spline / local-polynomial degree of the derivative surface.
    weighting : str
        How to weight rows of the :math:`y_q` regression. ``"derivative_se"`` (the
        default) weights by :math:`1/\sigma^2` of the smoother's estimate of that
        target, so rows near the edges of the condition range -- where derivative
        standard errors blow up, and where a shift law is most tempted to bend -- stop
        dominating the selection. ``"none"`` fits unweighted.
    log_base : float
        Base of the log abscissa. Only used to convert a shift slope into an
        activation energy.
    validation : str
        ``"loco"`` for leave-one-condition-out validation (the default and the
        module's verdict) or ``"none"`` to skip it.
    max_holdout_conditions : int, optional
        Cap on how many conditions are withheld, for large tables. Conditions are
        sampled evenly across the measured range.
    collapse_thresholds : tuple of float
        ``(supported, weakly_supported)`` multiples of the noise floor.
    n_stability : int
        Size of the stability ensemble. 0 disables it.
    stability_resampling : str
        ``"residual"`` re-draws measurement noise and re-runs the whole pipeline;
        ``"conditions"`` resamples whole curves with replacement. Both perturb the
        smoothing stage, which a row-wise bootstrap cannot.
    random_state : int, optional
        Seed for the stability ensemble.

    Attributes
    ----------
    shift_expression_ : str
        The selected symbolic shift law in ``q``. **Not the headline result** --
        structurally different laws routinely produce the same transform to within
        0.01 decades. Report the transform and a physical summary instead.
    vertical_expressions_ : dict
        Channel label (or ``"shared"``) mapped to the selected vertical shift law.
    master_curve_ : dict
        Channel label mapped to a :class:`MasterCurve`.
    validity_report_ : ValidityReport
        The verdict, from held-out collapse.
    stability_ : dict or None
        Selection frequencies, parameter distributions and the spread of the physical
        summary over the ensemble; None when ``n_stability == 0``.
    channels_ : list
        Channel labels, in first-appearance order.
    conditions_ : list of float
        Distinct conditions, sorted.
    reference_ : float
        Reference condition actually used.
    surfaces_ : dict
        Channel label mapped to the fitted :class:`~jaxsr.derivatives.SurfaceDerivatives`
        used for the derivative stage. Kept so that the smoothing level is inspectable.
    noise_floor_ : float
        Measurement noise standard deviation, in response units. The collapse cannot
        be better than this, so it is the yardstick the verdict is read against.
    noise_floor_source_ : str
        How the floor was measured: ``"replicates"``, ``"curve_smoother"``, or
        ``"surface"`` (in decreasing order of directness).
    selection_model_ : SymbolicRegressor
        The fitted sparse regression over the structured blocks, exposed for the usual
        diagnostics (``selection_path_``, ``pareto_front_``, ``metrics_``). Its
        expression is written in the *working* features (``q``, ``y_x``, channel
        indicators) and its terms are block functions, so it is for inspection rather
        than export -- do not hand it to ``to_sympy()`` or ``to_callable()``. The
        result you want is :meth:`shift_factors`.

    Raises
    ------
    ValueError
        If any constructor argument is invalid.

    Examples
    --------
    >>> from jaxsr.superposition import SuperpositionRegressor
    >>> model = SuperpositionRegressor(                 # doctest: +SKIP
    ...     condition="temperature", abscissa="log_omega", response="log_Gp",
    ...     domain="frequency", condition_scale="kelvin",
    ... )
    >>> model.fit({"temperature": T, "log_omega": x, "log_Gp": y})   # doctest: +SKIP
    >>> model.shift_factors([280.0, 320.0])             # doctest: +SKIP
    array([ 0.83, -0.72])
    >>> model.validity_report_.verdict                  # doctest: +SKIP
    'supported'

    See Also
    --------
    jaxsr.derivatives.SurfaceDerivatives : The derivative stage this builds on.
    jaxsr.basis.BasisLibrary.add_block : The structured blocks the regression uses.
    """

    def __init__(
        self,
        condition: str = "condition",
        abscissa: str = "x",
        response: str = "y",
        channel: str | None = None,
        *,
        domain: str = "frequency",
        condition_scale: str | None = "kelvin",
        reference: float | None = None,
        vertical_shift: str = "none",
        candidate_families: Sequence[str] = ("arrhenius", "wlf", "polynomial"),
        poly_degree: int = 2,
        wlf_c2_bounds: tuple[float, float] = (0.05, 5.0),
        max_terms: int = 3,
        selection: str = "bic",
        strategy: str = "exhaustive",
        smoother: str = "tensor_spline",
        smoothing: float | str = "auto",
        surface_degree: int = 3,
        weighting: str = "derivative_se",
        log_base: float = 10.0,
        validation: str = "loco",
        max_holdout_conditions: int | None = None,
        collapse_thresholds: tuple[float, float] = (2.0, 4.0),
        n_stability: int = 0,
        stability_resampling: str = "residual",
        random_state: int | None = None,
    ) -> None:
        if domain not in _VALID_DOMAINS:
            raise ValueError(f"domain must be one of {list(_VALID_DOMAINS)}, got {domain!r}")
        if condition_scale not in (None, "kelvin"):
            raise ValueError(f"condition_scale must be 'kelvin' or None, got {condition_scale!r}")
        if vertical_shift not in _VALID_VERTICAL:
            raise ValueError(
                f"vertical_shift must be one of {list(_VALID_VERTICAL)}, got {vertical_shift!r}"
            )
        if vertical_shift == "per_channel" and channel is None:
            raise ValueError("vertical_shift='per_channel' requires a channel column")
        if validation not in _VALID_VALIDATION:
            raise ValueError(
                f"validation must be one of {list(_VALID_VALIDATION)}, got {validation!r}"
            )
        if weighting not in _VALID_WEIGHTING:
            raise ValueError(
                f"weighting must be one of {list(_VALID_WEIGHTING)}, got {weighting!r}"
            )
        if stability_resampling not in _VALID_STABILITY:
            raise ValueError(
                f"stability_resampling must be one of {list(_VALID_STABILITY)}, "
                f"got {stability_resampling!r}"
            )
        if poly_degree < 0:
            raise ValueError(f"poly_degree must be non-negative, got {poly_degree}")
        if max_terms < 1:
            raise ValueError(f"max_terms must be at least 1, got {max_terms}")
        if n_stability < 0:
            raise ValueError(f"n_stability must be non-negative, got {n_stability}")
        if log_base <= 1.0:
            raise ValueError(f"log_base must be greater than 1, got {log_base}")
        lo, hi = wlf_c2_bounds
        if not 0 < lo < hi:
            raise ValueError(f"wlf_c2_bounds must satisfy 0 < lower < upper, got {wlf_c2_bounds}")
        first, second = collapse_thresholds
        if not 0 < first <= second:
            raise ValueError(
                f"collapse_thresholds must satisfy 0 < supported <= weakly_supported, "
                f"got {collapse_thresholds}"
            )
        if max_holdout_conditions is not None and max_holdout_conditions < 2:
            raise ValueError("max_holdout_conditions must be at least 2 when given")

        self.condition = condition
        self.abscissa = abscissa
        self.response = response
        self.channel = channel
        self.domain = domain
        self.condition_scale = condition_scale
        self.reference = reference
        self.vertical_shift = vertical_shift
        self.candidate_families = tuple(candidate_families)
        self.poly_degree = int(poly_degree)
        self.wlf_c2_bounds = (float(lo), float(hi))
        self.max_terms = int(max_terms)
        self.selection = selection
        self.strategy = strategy
        self.smoother = smoother
        self.smoothing = smoothing
        self.surface_degree = int(surface_degree)
        self.weighting = weighting
        self.log_base = float(log_base)
        self.validation = validation
        self.max_holdout_conditions = max_holdout_conditions
        self.collapse_thresholds = (float(first), float(second))
        self.n_stability = int(n_stability)
        self.stability_resampling = stability_resampling
        self.random_state = random_state

        self._is_fitted = False
        self._flags: list[str] = []

    # -- sign convention ----------------------------------------------------

    @property
    def _sign(self) -> float:
        """+1 when the reduced coordinate adds the shift, -1 when it subtracts it."""
        return 1.0 if self.domain == "frequency" else -1.0

    # -- public API ---------------------------------------------------------

    def fit(self, data: Any) -> SuperpositionRegressor:
        """
        Discover the transform, reconstruct the master curve, and test the collapse.

        Parameters
        ----------
        data : mapping or pandas.DataFrame
            Tidy table. Any object supporting ``data[column_name]`` works, so a dict
            of 1-D arrays and a DataFrame are both accepted. The columns named by
            *condition*, *abscissa*, *response* and (if given) *channel* must be
            present and equal in length.

        Returns
        -------
        self : SuperpositionRegressor
            The fitted estimator.

        Raises
        ------
        KeyError
            If a required column is missing.
        ValueError
            If the table is malformed, holds non-finite values, has fewer than four
            distinct conditions, or declares kelvin for a column that cannot be one.
        """
        self._flags = []
        dataset = self._prepare(data)

        self.channels_ = list(dataset.channels)
        self.conditions_ = list(dataset.conditions)
        self.reference_ = dataset.reference
        self._dataset = dataset

        # Estimated before the surface fit, and from the raw data alone, so it can be
        # handed to the smoother as a known sigma. Nothing here looks at the symbolic
        # score -- tuning a smoother against the regression that consumes it can
        # manufacture whichever law the regression prefers.
        self.noise_floor_, self.noise_floor_source_ = self._measured_noise(dataset)

        self.surfaces_, partials = self._fit_surfaces(dataset, self.noise_floor_)
        self.selection_model_, blocks, terms = self._select(dataset, partials)
        self._shift_law, self._vertical_laws = self._extract_laws(
            self.selection_model_, blocks, terms
        )

        if self._shift_law.is_empty:
            self._flags.append("no_shift_terms_selected")
            warnings.warn(
                "No horizontal-shift term survived selection: the fitted transform is "
                "the identity. Either the curves already overlap, or no scalar shift "
                "explains them. See validity_report_.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.master_curve_ = self._fit_master_curves(dataset, self._shift_law, self._vertical_laws)
        self.validity_report_ = self._validate(dataset, partials)
        self.stability_ = self._run_stability(dataset) if self.n_stability else None

        self._is_fitted = True
        return self

    def shift_factors(self, condition: np.ndarray | None = None) -> np.ndarray:
        r"""
        Evaluate the horizontal shift :math:`s(c) = \log_{b} a_T`.

        This -- not :attr:`shift_expression_` -- is the identified quantity. Laws with
        different structure routinely agree here to within 0.01 decades.

        Parameters
        ----------
        condition : array-like, optional
            Conditions at which to evaluate, in the same units as the fitted column.
            Defaults to the distinct measured conditions.

        Returns
        -------
        np.ndarray of shape (n_conditions,)
            Shift in abscissa units, zero at the reference condition.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        Examples
        --------
        >>> model.shift_factors([300.0, 320.0])     # doctest: +SKIP
        array([ 0.  , -0.71])
        """
        self._check_is_fitted()
        values = np.asarray(self.conditions_ if condition is None else condition, dtype=np.float64)
        return self._shift_law.value(self._to_q(values.ravel()))

    def vertical_shifts(
        self,
        condition: np.ndarray | None = None,
        channel: Any = None,
    ) -> np.ndarray:
        """
        Evaluate the vertical shift :math:`v(c)`.

        Parameters
        ----------
        condition : array-like, optional
            Conditions at which to evaluate. Defaults to the measured conditions.
        channel : Any, optional
            Channel label, required when ``vertical_shift="per_channel"``.

        Returns
        -------
        np.ndarray of shape (n_conditions,)
            Vertical shift in response units, zero at the reference condition and
            identically zero when ``vertical_shift="none"``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If *channel* is missing or unknown for a per-channel vertical shift.
        """
        self._check_is_fitted()
        values = np.asarray(self.conditions_ if condition is None else condition, dtype=np.float64)
        return self._vertical_law_for(channel).value(self._to_q(values.ravel()))

    def transform(self, data: Any = None) -> dict[str, np.ndarray]:
        """
        Apply the discovered transform, returning the reduced coordinates.

        Parameters
        ----------
        data : mapping or pandas.DataFrame, optional
            Table with the same columns as :meth:`fit`. Defaults to the training data.

        Returns
        -------
        dict
            The input columns plus ``"z"`` (reduced abscissa) and ``"w"`` (reduced
            response). Returned as a plain dict of arrays so the caller can build
            whatever table type they use.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        KeyError
            If a required column is missing from *data*.
        """
        self._check_is_fitted()
        dataset = self._dataset if data is None else self._prepare(data, allow_new=True)
        z, w = self._reduce(dataset, self._shift_law, self._vertical_laws)
        out = {
            self.condition: dataset.condition,
            self.abscissa: dataset.x,
            self.response: dataset.y,
            "z": z,
            "w": w,
        }
        if self.channel is not None:
            out[self.channel] = dataset.channel
        return out

    def predict(
        self,
        condition: np.ndarray,
        abscissa: np.ndarray,
        channel: Any = None,
    ) -> np.ndarray:
        """
        Predict the response at new conditions by shifting the master curve.

        Parameters
        ----------
        condition : array-like of shape (n_samples,)
            Conditions, including ones never measured -- extrapolating to those is the
            point of learning the shift law symbolically.
        abscissa : array-like of shape (n_samples,)
            Log abscissa values.
        channel : Any, optional
            Channel label. Required when the model was fitted with several channels.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted response.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If the arrays disagree in length, or the channel is missing or unknown.
        """
        self._check_is_fitted()
        condition = np.asarray(condition, dtype=np.float64).ravel()
        abscissa = np.asarray(abscissa, dtype=np.float64).ravel()
        if condition.shape != abscissa.shape:
            raise ValueError(
                f"condition has {condition.size} values but abscissa has {abscissa.size}"
            )

        label = self._resolve_channel(channel)
        curve = self.master_curve_[label]
        q = self._to_q(condition)
        z = abscissa + self._sign * self._shift_law.value(q)
        return np.asarray(curve.predict(z)) + self._vertical_law_for(label).value(q)

    def effective_activation_energy(self, gas_constant: float = GAS_CONSTANT) -> float:
        r"""
        Report the activation energy implied by the shift law at the reference.

        Defined as :math:`E_{eff} = -\ln(b)\, R\, T_{ref}\, s'(0)` with :math:`b` the
        log base of the abscissa. This is the summary to compare across fits: in a
        90-fit study spanning nine distinct selected structures it came back at
        54.2 +/- 1.3 kJ/mol against a true 55.9, while the *expressions* disagreed 38%
        of the time.

        Parameters
        ----------
        gas_constant : float
            Molar gas constant, in the energy units wanted for the result. Defaults to
            8.314462618 J/(mol K), so the result is in J/mol.

        Returns
        -------
        float
            Effective activation energy at the reference condition.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If the condition is not an absolute temperature
            (``condition_scale="kelvin"``), in which case the quantity is meaningless.

        Examples
        --------
        >>> model.effective_activation_energy() / 1000.0    # kJ/mol   # doctest: +SKIP
        54.2
        """
        self._check_is_fitted()
        if self.condition_scale != "kelvin":
            raise ValueError(
                "effective_activation_energy() needs an absolute temperature; fit with "
                "condition_scale='kelvin' and the condition column in kelvin."
            )
        slope = float(self._shift_law.derivative(np.zeros(1))[0])
        return -math.log(self.log_base) * gas_constant * self.reference_ * slope

    @property
    def shift_expression_(self) -> str:
        """
        The selected shift law as a string in ``q``.

        Returns
        -------
        str
            The law, with any fitted nonlinear parameter substituted. ``"0"`` when no
            horizontal term survived selection.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        Notes
        -----
        Deliberately *not* the headline result. Over a realistic condition window,
        ``1/(1+q)^2``, ``1/(c2+q)^2`` and low-order polynomials span nearly the same
        function space, so this expression is far less reproducible than the transform
        it encodes. Read :meth:`shift_factors`, and
        :meth:`effective_activation_energy` for a structure-independent summary.
        """
        self._check_is_fitted()
        return self._shift_law.expression()

    @property
    def vertical_expressions_(self) -> dict[Any, str]:
        """
        Selected vertical-shift laws, keyed by channel or by ``"shared"``.

        Returns
        -------
        dict
            Key mapped to the law as a string in ``q``. Empty when
            ``vertical_shift="none"``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_is_fitted()
        return {key: law.expression() for key, law in self._vertical_laws.items()}

    def summary(self) -> str:
        """
        Return a human-readable report, leading with the transform and the verdict.

        Returns
        -------
        str
            Multi-line text.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_is_fitted()
        shifts = self.shift_factors()
        lines = [
            "SuperpositionRegressor",
            "=" * 46,
            f"domain            : {self.domain} (z = x {'+' if self._sign > 0 else '-'} s(c))",
            f"reference         : {self.reference_:.6g}",
            f"channels          : {self.channels_}",
            f"conditions        : {len(self.conditions_)}",
            "",
            "Transform (the identified quantity)",
            "-" * 46,
        ]
        for cond, shift in zip(self.conditions_, shifts, strict=False):
            lines.append(f"  {cond:>12.6g}  ->  {shift:+.4f}")
        if self.condition_scale == "kelvin" and not self._shift_law.is_empty:
            energy = self.effective_activation_energy() / 1000.0
            lines.append(f"  effective activation energy : {energy:.3g} kJ/mol")

        lines += [
            "",
            "Selected law (not the headline -- see shift_expression_ notes)",
            "-" * 46,
            f"  s(q) = {self._shift_law.expression()}",
        ]
        for key, expr in self.vertical_expressions_.items():
            lines.append(f"  v_{key}(q) = {expr}")

        lines += ["", self.validity_report_.summary()]
        if self.stability_ is not None:
            lines += [
                "",
                "Stability over the ensemble",
                "-" * 46,
                f"  replicates            : {self.stability_['n_replicates']}",
                f"  distinct structures   : {self.stability_['n_distinct_structures']}",
                f"  stability score       : {self.stability_['stability_score']:.2f}",
            ]
            energy = self.stability_.get("effective_activation_energy")
            if energy:
                lines.append(
                    f"  E_eff (kJ/mol)        : {energy['mean'] / 1000:.3g} "
                    f"+/- {energy['sd'] / 1000:.2g}"
                )
            lines.append(
                "  NOTE: a stable expression is not evidence of a valid collapse -- "
                "read the verdict above."
            )
        return "\n".join(lines)

    # -- input handling -----------------------------------------------------

    def _prepare(self, data: Any, allow_new: bool = False) -> _Dataset:
        """
        Normalize a tidy table into arrays and derive the condition coordinate.

        Parameters
        ----------
        data : mapping or pandas.DataFrame
            Input table.
        allow_new : bool
            Skip the "enough conditions to fit" checks, for tables that are being
            transformed rather than fitted.

        Returns
        -------
        _Dataset

        Raises
        ------
        KeyError
            If a required column is missing.
        ValueError
            If the table is malformed or violates the declared conventions.
        """
        columns = [self.condition, self.abscissa, self.response]
        if self.channel is not None:
            columns.append(self.channel)

        raw: dict[str, np.ndarray] = {}
        for name in columns:
            try:
                raw[name] = np.asarray(data[name])
            except (KeyError, IndexError) as exc:
                raise KeyError(
                    f"Column {name!r} is missing from the data (looked for "
                    f"{columns}). Set the column names on the constructor."
                ) from exc

        condition = np.asarray(raw[self.condition], dtype=np.float64).ravel()
        x = np.asarray(raw[self.abscissa], dtype=np.float64).ravel()
        y = np.asarray(raw[self.response], dtype=np.float64).ravel()
        if self.channel is None:
            channel = np.array([self.response] * condition.size, dtype=object)
        else:
            channel = np.asarray(raw[self.channel]).ravel()

        sizes = {condition.size, x.size, y.size, channel.size}
        if len(sizes) != 1:
            raise ValueError(
                f"Columns have unequal lengths: "
                f"{ {name: np.asarray(raw[name]).ravel().size for name in columns} }"
            )
        if condition.size == 0:
            raise ValueError("The data table is empty")
        for name, values in ((self.condition, condition), (self.abscissa, x), (self.response, y)):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Column {name!r} contains non-finite values")

        reference = self._resolve_reference(condition)
        if self.condition_scale == "kelvin":
            if np.any(condition <= 0):
                raise ValueError(
                    f"condition_scale='kelvin' but column {self.condition!r} has "
                    "non-positive values. Convert to kelvin before fitting."
                )
            if np.max(condition) < 150.0:
                warnings.warn(
                    f"condition_scale='kelvin' but column {self.condition!r} tops out at "
                    f"{np.max(condition):.4g}, which looks like Celsius. Reciprocal-condition "
                    "families will be nonsense if it is. Convert to kelvin.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            scale = reference
        else:
            span = float(np.max(condition) - np.min(condition))
            scale = span / 2.0 if span > 0 else max(abs(reference), 1.0)

        if allow_new:
            # Transforming new rows must reuse the coordinate the model was fitted in;
            # re-deriving it from a different table would silently move the anchor.
            reference, scale = self._q_reference, self._q_scale
        else:
            if len(np.unique(condition)) < 4:
                raise ValueError(
                    f"Superposition needs at least 4 distinct conditions to separate a shift "
                    f"law from the master curve; got {len(np.unique(condition))}."
                )
            self._q_scale = scale
            self._q_reference = reference

        q = (condition - reference) / scale
        conditions = sorted(float(v) for v in np.unique(condition))

        return _Dataset(
            condition=condition,
            x=x,
            y=y,
            channel=channel,
            q=q,
            reference=reference,
            channels=_unique_preserving_order(channel),
            conditions=conditions,
        )

    def _resolve_reference(self, condition: np.ndarray) -> float:
        """
        Pick the reference condition, defaulting to the median of those measured.

        Parameters
        ----------
        condition : np.ndarray
            Condition values.

        Returns
        -------
        float
            Reference condition.

        Raises
        ------
        ValueError
            If an explicit reference is zero while the condition is a generic scalar
            scaled by its own magnitude, or if it is non-positive under kelvin.
        """
        if self.reference is not None:
            reference = float(self.reference)
            if self.condition_scale == "kelvin" and reference <= 0:
                raise ValueError(f"reference must be positive in kelvin, got {reference}")
            return reference
        distinct = np.unique(condition)
        return float(np.median(distinct))

    def _to_q(self, condition: np.ndarray) -> np.ndarray:
        """Map condition values onto the dimensionless coordinate used for fitting."""
        return (np.asarray(condition, dtype=np.float64) - self._q_reference) / self._q_scale

    def _resolve_channel(self, channel: Any) -> Any:
        """
        Resolve a channel argument to a fitted channel label.

        Parameters
        ----------
        channel : Any
            Requested label, or None to use the only channel.

        Returns
        -------
        Any
            A label present in :attr:`channels_`.

        Raises
        ------
        ValueError
            If the label is unknown, or omitted when several channels exist.
        """
        if channel is None:
            if len(self.channels_) != 1:
                raise ValueError(
                    f"This model has {len(self.channels_)} channels {self.channels_}; "
                    "pass channel= to say which one."
                )
            return self.channels_[0]
        if channel not in self.channels_:
            raise ValueError(f"Unknown channel {channel!r}. Fitted channels: {self.channels_}")
        return channel

    def _vertical_law_for(self, channel: Any) -> _Law:
        """Return the vertical law that applies to *channel*."""
        if self.vertical_shift == "none":
            return _Law()
        if self.vertical_shift == "shared":
            return self._vertical_laws["shared"]
        return self._vertical_laws[self._resolve_channel(channel)]

    def _check_is_fitted(self) -> None:
        """Raise if the estimator has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError("SuperpositionRegressor must be fitted before calling this method")

    # -- derivative stage ---------------------------------------------------

    def _fit_surfaces(
        self,
        dataset: _Dataset,
        sigma: float | None = None,
    ) -> tuple[dict[Any, SurfaceDerivatives], dict[str, np.ndarray]]:
        """
        Fit one smoothed surface per channel and read off both first partials.

        Parameters
        ----------
        dataset : _Dataset
            Normalized data.
        sigma : float, optional
            Measurement noise standard deviation, measured from the raw data. Passing
            it makes the reported derivative uncertainties reflect the measurement
            noise, and is required by ``smoothing="sigma"``.

        Returns
        -------
        surfaces : dict
            Channel label mapped to the fitted smoother.
        partials : dict
            ``"y_x"``, ``"y_q"``, ``"y_q_se"`` and ``"fitted"`` arrays, aligned with the
            dataset rows.

        Raises
        ------
        ValueError
            If a channel has too few points to support the surface, or
            ``smoothing="sigma"`` was asked for but no noise estimate is available.
        """
        if self.smoothing == "sigma" and sigma is None:
            raise ValueError(
                "smoothing='sigma' needs a measurable noise level, but the data has "
                "neither replicate points nor curves dense enough to smooth "
                "individually. Use smoothing='auto' or a numeric value."
            )
        y_x = np.zeros(dataset.x.size)
        y_q = np.zeros(dataset.x.size)
        y_q_se = np.zeros(dataset.x.size)
        fitted = np.zeros(dataset.x.size)
        surfaces: dict[Any, SurfaceDerivatives] = {}

        for label in dataset.channels:
            mask = dataset.channel == label
            coords = np.column_stack([dataset.x[mask], dataset.q[mask]])
            if coords.shape[0] < (self.surface_degree + 1) ** 2:
                raise ValueError(
                    f"Channel {label!r} has {coords.shape[0]} points, too few for a "
                    f"degree-{self.surface_degree} surface. Lower surface_degree or "
                    "measure more points."
                )
            surface = SurfaceDerivatives(
                method=self.smoother,
                degree=self.surface_degree,
                smoothing=self.smoothing,
            ).fit(coords, dataset.y[mask], sigma=sigma)
            values, partials, errors = surface.derivatives(
                coords, order=[(1, 0), (0, 1)], return_std=True
            )
            surfaces[label] = surface
            fitted[mask] = np.asarray(values)
            y_x[mask] = np.asarray(partials[:, 0])
            y_q[mask] = np.asarray(partials[:, 1])
            y_q_se[mask] = np.asarray(errors[:, 1])

        return surfaces, {"y_x": y_x, "y_q": y_q, "y_q_se": y_q_se, "fitted": fitted}

    # -- structured regression ----------------------------------------------

    def _terms(self) -> list[ShiftTerm]:
        """Build the candidate term library for the configured families."""
        return _build_terms(
            self.candidate_families,
            self.poly_degree,
            self.wlf_c2_bounds,
            reciprocal_ok=self.condition_scale == "kelvin",
        )

    def _theta_library(self, terms: Sequence[ShiftTerm]) -> BasisLibrary:
        """
        Wrap the candidate terms as a one-feature basis library over ``q``.

        Parameters
        ----------
        terms : sequence of ShiftTerm
            Candidate terms of the shift derivative.

        Returns
        -------
        BasisLibrary
            Library over the single feature ``"q"``, ready to be multiplied into a
            structured block.
        """
        theta = BasisLibrary(n_features=1, feature_names=["q"])
        for term in terms:
            if term.is_parametric:
                theta.add_parametric(
                    name=term.name,
                    func=_make_parametric_column(term),
                    param_bounds=dict(term.param_bounds or {}),
                    complexity=term.complexity,
                    feature_indices=(0,),
                    log_scale=term.log_scale,
                )
            else:
                theta.add_custom(
                    name=term.name,
                    func=_make_column(term),
                    complexity=term.complexity,
                    feature_indices=(0,),
                )
        return theta

    def _build_design(
        self,
        dataset: _Dataset,
        partials: dict[str, np.ndarray],
        terms: Sequence[ShiftTerm],
    ) -> tuple[np.ndarray, np.ndarray, BasisLibrary, dict[str, tuple[int, int]]]:
        r"""
        Build the structured design :math:`\Theta(q) \odot y_x \,|\, \Theta(q)`.

        Parameters
        ----------
        dataset : _Dataset
            Normalized data.
        partials : dict
            Output of :meth:`_fit_surfaces`.
        terms : sequence of ShiftTerm
            Candidate terms.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_working_features)
            Working feature matrix: ``q``, ``y_x``, and one indicator per channel when
            the vertical shift is per-channel.
        y : np.ndarray of shape (n_samples,)
            The regression target :math:`y_q`.
        library : BasisLibrary
            Library holding the structured blocks.
        blocks : dict
            Block label mapped to the ``(start, stop)`` slice of library indices, so
            that a selected index can be traced back to a block and a term.
        """
        theta = self._theta_library(terms)

        feature_names = ["q", "y_x"]
        columns = [dataset.q, partials["y_x"]]
        per_channel = self.vertical_shift == "per_channel"
        if per_channel:
            for label in dataset.channels:
                feature_names.append(f"ind[{label}]")
                columns.append((dataset.channel == label).astype(np.float64))

        library = BasisLibrary(n_features=len(feature_names), feature_names=feature_names)
        blocks: dict[str, tuple[int, int]] = {}

        start = len(library.basis_functions)
        library.add_block(theta, multiply_by="y_x", block_name="horizontal")
        blocks["horizontal"] = (start, len(library.basis_functions))

        if self.vertical_shift == "shared":
            start = len(library.basis_functions)
            library.add_block(self._theta_library(terms), block_name="vertical")
            blocks["vertical"] = (start, len(library.basis_functions))
        elif per_channel:
            for label in dataset.channels:
                start = len(library.basis_functions)
                name = f"vertical[{label}]"
                library.add_block(
                    self._theta_library(terms),
                    multiply_by=f"ind[{label}]",
                    block_name=name,
                )
                blocks[name] = (start, len(library.basis_functions))

        return np.column_stack(columns), partials["y_q"], library, blocks

    def _select(
        self,
        dataset: _Dataset,
        partials: dict[str, np.ndarray],
    ) -> tuple[SymbolicRegressor, dict[str, tuple[int, int]], list[ShiftTerm]]:
        """
        Run the sparse regression that recovers the coefficient functions.

        Parameters
        ----------
        dataset : _Dataset
            Normalized data.
        partials : dict
            Output of :meth:`_fit_surfaces`.

        Returns
        -------
        model : SymbolicRegressor
            The fitted selection model.
        blocks : dict
            Block index ranges, as returned by :meth:`_build_design`.
        terms : list of ShiftTerm
            The candidate terms, in the order the blocks hold them.
        """
        terms = self._terms()
        X, y, library, blocks = self._build_design(dataset, partials, terms)

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=self.max_terms,
            strategy=self.strategy,
            information_criterion=self.selection,
        )
        model.fit(jnp.asarray(X), jnp.asarray(y), sample_weight=self._row_weights(partials))
        return model, blocks, terms

    def _row_weights(self, partials: dict[str, np.ndarray]) -> np.ndarray | None:
        r"""
        Weight each row of the :math:`y_q` regression by the precision of that target.

        The regression target is an *estimate*, and the smoother's uncertainty in it is
        far from uniform: derivative standard errors blow up near the edges of the
        condition range, which is exactly where a shift law is most tempted to bend.
        Weighting by :math:`1/\sigma^2` stops those rows from dominating the selection.

        Parameters
        ----------
        partials : dict
            Output of :meth:`_fit_surfaces`, whose ``"y_q_se"`` entry holds the
            smoother's standard error for each target value.

        Returns
        -------
        np.ndarray of shape (n_samples,) or None
            Weights, or None when weighting is disabled or the standard errors are
            unusable (non-finite, or all equal, in which case weighting is a no-op).
        """
        if self.weighting == "none":
            return None
        errors = np.asarray(partials["y_q_se"], dtype=np.float64)
        if not np.all(np.isfinite(errors)) or np.all(errors <= 0):
            return None
        # A zero standard error would be infinite precision; floor it at a small
        # fraction of the typical error rather than letting one row take over.
        floor = 0.05 * float(np.median(errors[errors > 0]))
        weights = 1.0 / np.maximum(errors, floor) ** 2
        if np.ptp(weights) == 0.0:
            return None
        return weights / weights.mean()

    def _extract_laws(
        self,
        model: SymbolicRegressor,
        blocks: dict[str, tuple[int, int]],
        terms: Sequence[ShiftTerm],
    ) -> tuple[_Law, dict[Any, _Law]]:
        """
        Split the selected coefficients back into a shift law and vertical laws.

        Parameters
        ----------
        model : SymbolicRegressor
            The fitted selection model.
        blocks : dict
            Block index ranges.
        terms : sequence of ShiftTerm
            Candidate terms, in block order.

        Returns
        -------
        shift_law : _Law
            The horizontal shift, with the domain sign already folded in so that
            ``shift_law.value(q)`` is ``s(q)`` directly.
        vertical_laws : dict
            Channel label (or ``"shared"``) mapped to its vertical law. Empty laws are
            still present, so callers never have to test for a missing key.
        """
        params = model._result.parametric_params or {}
        coefficients = np.asarray(model.coefficients_, dtype=np.float64)
        indices = [int(i) for i in np.asarray(model.selected_indices_)]

        shift_law = _Law()
        vertical_laws: dict[Any, _Law] = {}
        if self.vertical_shift == "shared":
            vertical_laws["shared"] = _Law()
        elif self.vertical_shift == "per_channel":
            vertical_laws = {label: _Law() for label in self.channels_}

        for coef, index in zip(coefficients, indices, strict=False):
            label, offset = _locate_block(index, blocks)
            if label is None:
                continue
            term = terms[offset]
            fitted_params = dict(params.get(index, {})) if term.is_parametric else {}
            if term.is_parametric and not fitted_params:
                # Profile optimisation did not report a value; fall back to the
                # midpoint the library was registered with rather than guessing.
                fitted_params = {
                    name: float(np.sqrt(lo * hi)) if term.log_scale else float((lo + hi) / 2)
                    for name, (lo, hi) in (term.param_bounds or {}).items()
                }
            if label == "horizontal":
                shift_law.terms.append((self._sign * float(coef), term, fitted_params))
            elif label == "vertical":
                vertical_laws["shared"].terms.append((float(coef), term, fitted_params))
            else:
                channel = label[len("vertical[") : -1]
                key = _match_channel(channel, self.channels_)
                vertical_laws[key].terms.append((float(coef), term, fitted_params))

        return shift_law, vertical_laws

    # -- collapse -----------------------------------------------------------

    def _reduce(
        self,
        dataset: _Dataset,
        shift_law: _Law,
        vertical_laws: dict[Any, _Law],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map raw rows onto reduced coordinates under a given pair of laws.

        Parameters
        ----------
        dataset : _Dataset
            Rows to transform.
        shift_law : _Law
            Horizontal shift law (already sign-corrected).
        vertical_laws : dict
            Vertical laws by channel or ``"shared"``.

        Returns
        -------
        z : np.ndarray of shape (n_samples,)
            Reduced abscissa.
        w : np.ndarray of shape (n_samples,)
            Reduced response.
        """
        z = dataset.x + self._sign * shift_law.value(dataset.q)
        w = np.array(dataset.y, dtype=np.float64)
        if self.vertical_shift == "shared":
            w = w - vertical_laws["shared"].value(dataset.q)
        elif self.vertical_shift == "per_channel":
            for label, law in vertical_laws.items():
                mask = dataset.channel == label
                if np.any(mask):
                    w[mask] = w[mask] - law.value(dataset.q[mask])
        return z, w

    def _fit_master_curves(
        self,
        dataset: _Dataset,
        shift_law: _Law,
        vertical_laws: dict[Any, _Law],
    ) -> dict[Any, MasterCurve]:
        """
        Fit one master curve per channel from the reduced coordinates.

        Parameters
        ----------
        dataset : _Dataset
            Rows to collapse.
        shift_law, vertical_laws
            The discovered laws.

        Returns
        -------
        dict
            Channel label mapped to a :class:`MasterCurve`.
        """
        z, w = self._reduce(dataset, shift_law, vertical_laws)
        curves: dict[Any, MasterCurve] = {}
        for label in dataset.channels:
            mask = dataset.channel == label
            curves[label] = _fit_master_curve(
                label,
                z[mask],
                w[mask],
                smoothing=self.smoothing,
                sigma=getattr(self, "noise_floor_", None),
            )
        return curves

    # -- validation ---------------------------------------------------------

    def _validate(self, dataset: _Dataset, partials: dict[str, np.ndarray]) -> ValidityReport:
        """
        Grade the collapse, primarily from conditions withheld from the whole pipeline.

        Parameters
        ----------
        dataset : _Dataset
            Normalized data.
        partials : dict
            Output of :meth:`_fit_surfaces`, used only for the noise-floor fallback.

        Returns
        -------
        ValidityReport
        """
        noise_floor, source = self.noise_floor_, self.noise_floor_source_
        if noise_floor is None:
            # Last resort: the derivative surface smooths across conditions, so a
            # collapse failure can hide inside its residuals. Grading against it is
            # conservative in the wrong direction, hence the flag.
            noise_floor = float(np.std(dataset.y - partials["fitted"]))
            source = "surface"
            self.noise_floor_, self.noise_floor_source_ = noise_floor, source
        if source != "replicates":
            self._flags.append(f"noise_floor_from_{source}")

        z, w = self._reduce(dataset, self._shift_law, self._vertical_laws)
        residual_sq = 0.0
        for label in dataset.channels:
            mask = dataset.channel == label
            resid = w[mask] - np.asarray(self.master_curve_[label].predict(z[mask]))
            residual_sq += float(np.sum(resid**2))
        in_sample = math.sqrt(residual_sq / z.size)

        report = ValidityReport(
            verdict="not_evaluated",
            noise_floor=noise_floor,
            noise_floor_source=source,
            in_sample_collapse=in_sample,
            thresholds=self.collapse_thresholds,
            flags=self._flags,
        )

        if self.validation == "none":
            report.flags.append("validation_disabled")
            return report
        if self._shift_law.is_empty:
            report.verdict = "not_supported"
            return report
        if len(dataset.conditions) < 5:
            report.flags.append("too_few_conditions_for_holdout")
            return report

        floor = max(noise_floor, 1e-12)
        report.holdout = self._leave_one_condition_out(dataset)
        if not report.holdout:
            report.flags.append("holdout_failed")
            return report
        for entry in report.holdout:
            entry["ratio"] = entry["collapse_rmse"] / floor

        collapses = np.array([entry["collapse_rmse"] for entry in report.holdout])
        errors = np.array([entry["shift_error"] for entry in report.holdout])
        errors = errors[np.isfinite(errors)]
        report.holdout_collapse_median = float(np.median(collapses))
        report.shift_error_median = float(np.median(errors)) if errors.size else None

        report.holdout_ratio_median = report.holdout_collapse_median / floor
        first, second = self.collapse_thresholds
        if report.holdout_ratio_median <= first:
            report.verdict = "supported"
        elif report.holdout_ratio_median <= second:
            report.verdict = "weakly_supported"
        else:
            report.verdict = "not_supported"
        return report

    def _holdout_conditions(self, dataset: _Dataset) -> list[float]:
        """
        Choose which conditions to withhold, spreading them across the range.

        Parameters
        ----------
        dataset : _Dataset
            Normalized data.

        Returns
        -------
        list of float
            Conditions to hold out, one at a time.
        """
        conditions = list(dataset.conditions)
        cap = self.max_holdout_conditions
        if cap is None or cap >= len(conditions):
            return conditions
        picks = np.linspace(0, len(conditions) - 1, cap)
        return [conditions[int(round(i))] for i in picks]

    def _leave_one_condition_out(self, dataset: _Dataset) -> list[dict[str, Any]]:
        """
        Refit the whole pipeline without each condition and score the withheld curve.

        The withheld condition takes no part in the smoothing or in the discovery; it
        is shifted by prediction alone. That is the only test that separated a genuine
        superposition from a negative control in the study behind this module.

        Parameters
        ----------
        dataset : _Dataset
            Normalized data.

        Returns
        -------
        list of dict
            One entry per successfully evaluated condition.
        """
        entries: list[dict[str, Any]] = []

        for condition in self._holdout_conditions(dataset):
            held = dataset.condition == condition
            kept = ~held
            if len(np.unique(dataset.condition[kept])) < 4:
                continue

            train = dataset.subset(kept)
            try:
                _, partials = self._fit_surfaces(train, self.noise_floor_)
                model, blocks, terms = self._select(train, partials)
                shift_law, vertical_laws = self._extract_laws(model, blocks, terms)
                curves = self._fit_master_curves(train, shift_law, vertical_laws)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
            if shift_law.is_empty:
                continue

            entry = self._score_holdout(dataset, held, shift_law, vertical_laws, curves)
            if entry is not None:
                entry["condition"] = float(condition)
                entries.append(entry)

        return entries

    def _score_holdout(
        self,
        dataset: _Dataset,
        held: np.ndarray,
        shift_law: _Law,
        vertical_laws: dict[Any, _Law],
        curves: dict[Any, MasterCurve],
    ) -> dict[str, Any] | None:
        """
        Score one withheld curve against a master curve built without it.

        Parameters
        ----------
        dataset : _Dataset
            Full data.
        held : np.ndarray of bool
            Mask of the withheld rows.
        shift_law, vertical_laws : _Law, dict
            Laws discovered without the withheld condition.
        curves : dict
            Master curves built without the withheld condition.

        Returns
        -------
        dict or None
            Scores for this condition, or None when too little of the shifted curve
            lands inside the master curve's range to say anything.
        """
        holdout = dataset.subset(held)
        q = holdout.q
        predicted = float(shift_law.value(q[:1])[0])

        residual_sq = 0.0
        n_used = 0
        n_total = 0
        for label in holdout.channels:
            if label not in curves:
                continue
            mask = holdout.channel == label
            z = holdout.x[mask] + self._sign * predicted
            w = holdout.y[mask] - self._vertical_from(vertical_laws, label, q[mask])
            curve = curves[label]
            inside = curve.covers(z)
            n_total += int(mask.sum())
            if not np.any(inside):
                continue
            resid = w[inside] - np.asarray(curve.predict(z[inside]))
            residual_sq += float(np.sum(resid**2))
            n_used += int(inside.sum())

        if n_used == 0 or n_used < 0.25 * max(n_total, 1):
            self._flags.append(f"low_holdout_coverage_at_{holdout.conditions[0]:g}")
            return None

        aligned = self._best_alignment_shift(holdout, vertical_laws, curves, predicted)
        return {
            "collapse_rmse": math.sqrt(residual_sq / n_used),
            "ratio": float("nan"),  # filled in by _validate, which knows the noise floor
            "shift_predicted": predicted,
            "shift_aligned": aligned,
            "shift_error": abs(predicted - aligned) if aligned is not None else float("nan"),
            "coverage": n_used / max(n_total, 1),
        }

    def _vertical_from(self, vertical_laws: dict[Any, _Law], label: Any, q: np.ndarray) -> Any:
        """Evaluate the vertical law applying to *label*, or zero when there is none."""
        if self.vertical_shift == "none":
            return 0.0
        if self.vertical_shift == "shared":
            return vertical_laws["shared"].value(q)
        law = vertical_laws.get(label)
        return 0.0 if law is None else law.value(q)

    def _best_alignment_shift(
        self,
        holdout: _Dataset,
        vertical_laws: dict[Any, _Law],
        curves: dict[Any, MasterCurve],
        centre: float,
        half_width: float = 1.5,
    ) -> float | None:
        """
        Find the shift that would have aligned the withheld curve best.

        Comparing this against the *predicted* shift separates two failure modes: a
        curve that collapses badly because the law extrapolated wrongly, and one that
        does not collapse under any shift at all.

        Parameters
        ----------
        holdout : _Dataset
            The withheld rows.
        vertical_laws : dict
            Vertical laws to apply before aligning.
        curves : dict
            Master curves to align against.
        centre : float
            Shift to search around, normally the predicted one.
        half_width : float
            Half-width of the search window, in abscissa units.

        Returns
        -------
        float or None
            Best-aligning shift, or None when no candidate had usable overlap.
        """
        q = holdout.q
        prepared = []
        for label in holdout.channels:
            if label not in curves:
                continue
            mask = holdout.channel == label
            w = holdout.y[mask] - self._vertical_from(vertical_laws, label, q[mask])
            prepared.append((holdout.x[mask], w, curves[label]))
        if not prepared:
            return None

        def score(shift: float) -> float:
            """Mean squared collapse residual at a trial shift, inf if too little overlap."""
            total = 0.0
            used = 0
            for x, w, curve in prepared:
                z = x + self._sign * shift
                inside = curve.covers(z)
                if not np.any(inside):
                    continue
                resid = w[inside] - np.asarray(curve.predict(z[inside]))
                total += float(np.sum(resid**2))
                used += int(inside.sum())
            if used < 3:
                return float("inf")
            return total / used

        best = None
        for width, n_points in ((half_width, 61), (half_width / 30.0, 41)):
            grid = np.linspace(
                (centre if best is None else best) - width,
                (centre if best is None else best) + width,
                n_points,
            )
            scores = [score(float(s)) for s in grid]
            if not np.any(np.isfinite(scores)):
                return None
            best = float(grid[int(np.argmin(scores))])
        return best

    def _measured_noise(self, dataset: _Dataset) -> tuple[float | None, str]:
        """
        Estimate measurement noise from the raw data, before any surface is fitted.

        Two estimators, in decreasing order of directness: the scatter of true
        replicates, and the residual scatter of a smooth curve fitted to each single
        condition separately. Neither looks across conditions, so neither can absorb a
        failure of the collapse -- which is what makes the result usable both as the
        smoother's sigma and as the yardstick the collapse is graded against.

        Parameters
        ----------
        dataset : _Dataset
            Normalized data.

        Returns
        -------
        floor : float or None
            Noise standard deviation in response units, or None when the data
            supports neither estimator.
        source : str
            ``"replicates"``, ``"curve_smoother"``, or ``"unavailable"``.
        """
        keys = np.column_stack(
            [
                np.array([dataset.channels.index(c) for c in dataset.channel], dtype=np.float64),
                dataset.condition,
                dataset.x,
            ]
        )
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inverse = np.asarray(inverse).ravel()
        if np.any(counts > 1):
            numerator = 0.0
            dof = 0
            for group in np.flatnonzero(counts > 1):
                values = dataset.y[inverse == group]
                numerator += float(np.sum((values - values.mean()) ** 2))
                dof += values.size - 1
            if dof >= 4:
                return math.sqrt(numerator / dof), "replicates"

        numerator = 0.0
        weight = 0
        for label in dataset.channels:
            for condition in dataset.conditions:
                mask = (dataset.channel == label) & (dataset.condition == condition)
                x = dataset.x[mask]
                if len(np.unique(x)) < 8:
                    continue
                try:
                    curve = SurfaceDerivatives(
                        method="tensor_spline",
                        degree=min(3, self.surface_degree),
                        smoothing="auto",
                    ).fit(x.reshape(-1, 1), dataset.y[mask])
                except (ValueError, np.linalg.LinAlgError):
                    continue
                numerator += float(curve.residual_std_**2) * int(mask.sum())
                weight += int(mask.sum())
        if weight > 0:
            return math.sqrt(numerator / weight), "curve_smoother"

        return None, "unavailable"

    # -- stability ----------------------------------------------------------

    def _run_stability(self, dataset: _Dataset) -> dict[str, Any]:
        """
        Re-run the whole pipeline over an ensemble and summarize what moves.

        Resampling the *rows* fed to the sparse regression would perturb nothing about
        the smoother that produced them, so each replicate regenerates the data and
        re-runs the derivative stage.

        Parameters
        ----------
        dataset : _Dataset
            Normalized data.

        Returns
        -------
        dict
            The keys of :func:`~jaxsr.uncertainty.summarize_selection_replicates`, plus
            ``"n_failed"``, ``"shift_factor_quantiles"`` and
            ``"effective_activation_energy"``.
        """
        rng = np.random.RandomState(self.random_state)
        models: list[SymbolicRegressor] = []
        shift_samples: list[np.ndarray] = []
        energy_samples: list[float] = []
        n_failed = 0

        conditions = np.asarray(dataset.conditions, dtype=np.float64)
        for _ in range(self.n_stability):
            replicate = self._resample(dataset, rng)
            try:
                _, partials = self._fit_surfaces(replicate, self.noise_floor_)
                model, blocks, terms = self._select(replicate, partials)
                shift_law, _ = self._extract_laws(model, blocks, terms)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                n_failed += 1
                continue
            models.append(model)
            shift_samples.append(shift_law.value(self._to_q(conditions)))
            if self.condition_scale == "kelvin" and not shift_law.is_empty:
                slope = float(shift_law.derivative(np.zeros(1))[0])
                energy_samples.append(
                    -math.log(self.log_base) * GAS_CONSTANT * self.reference_ * slope
                )

        if not models:
            return {
                "n_replicates": 0,
                "n_failed": n_failed,
                "n_distinct_structures": 0,
                "stability_score": float("nan"),
                "feature_frequencies": {},
                "shift_factor_quantiles": {},
                "effective_activation_energy": None,
                "resampling": self.stability_resampling,
            }

        summary = summarize_selection_replicates(
            models,
            reference=self.selection_model_,
            resampling=f"pipeline:{self.stability_resampling}",
        )
        summary["n_failed"] = n_failed

        stacked = np.vstack(shift_samples)
        summary["shift_factor_quantiles"] = {
            float(cond): (
                float(np.quantile(stacked[:, i], 0.05)),
                float(np.quantile(stacked[:, i], 0.5)),
                float(np.quantile(stacked[:, i], 0.95)),
            )
            for i, cond in enumerate(conditions)
        }
        summary["effective_activation_energy"] = (
            {
                "mean": float(np.mean(energy_samples)),
                "sd": float(np.std(energy_samples, ddof=1)) if len(energy_samples) > 1 else 0.0,
                "q05": float(np.quantile(energy_samples, 0.05)),
                "q95": float(np.quantile(energy_samples, 0.95)),
                "n": len(energy_samples),
            }
            if energy_samples
            else None
        )
        return summary

    def _resample(self, dataset: _Dataset, rng: np.random.RandomState) -> _Dataset:
        """
        Draw one ensemble replicate of the raw data.

        Parameters
        ----------
        dataset : _Dataset
            Normalized data.
        rng : numpy.random.RandomState
            Random state, so the ensemble is reproducible under ``random_state``.

        Returns
        -------
        _Dataset
            A replicate with the same shape as the original (residual resampling) or
            with whole curves drawn with replacement (condition resampling).
        """
        if self.stability_resampling == "conditions":
            picks = rng.choice(len(dataset.conditions), size=len(dataset.conditions), replace=True)
            rows = np.concatenate(
                [np.flatnonzero(dataset.condition == dataset.conditions[p]) for p in picks]
            )
            return dataset.subset(rows)

        y = np.array(dataset.y, dtype=np.float64)
        for label in dataset.channels:
            mask = dataset.channel == label
            surface = self.surfaces_[label]
            coords = np.column_stack([dataset.x[mask], dataset.q[mask]])
            fitted = np.asarray(surface.predict(coords))
            residual = dataset.y[mask] - fitted
            residual = residual - residual.mean()
            y[mask] = fitted + residual[rng.randint(0, residual.size, size=residual.size)]

        return _Dataset(
            condition=dataset.condition,
            x=dataset.x,
            y=y,
            channel=dataset.channel,
            q=dataset.q,
            reference=dataset.reference,
            channels=list(dataset.channels),
            conditions=list(dataset.conditions),
        )


# ===========================================================================
# Helpers
# ===========================================================================


def _make_column(term: ShiftTerm) -> Callable[[Any], Any]:
    """Wrap a term's derivative as a basis-library column over feature 0 (``q``)."""

    def column(X: Any) -> Any:
        """Evaluate the term on the working feature matrix's ``q`` column."""
        return term.deriv(X[:, 0])

    return column


def _make_parametric_column(term: ShiftTerm) -> Callable[..., Any]:
    """Wrap a parametric term's derivative as a basis-library column."""

    def column(X: Any, **params: float) -> Any:
        """Evaluate the parametric term on the ``q`` column at the given parameters."""
        return term.deriv(X[:, 0], **params)

    return column


def _locate_block(index: int, blocks: dict[str, tuple[int, int]]) -> tuple[str | None, int]:
    """
    Map a library index back to its block label and its offset within that block.

    Parameters
    ----------
    index : int
        Index into the working basis library.
    blocks : dict
        Block label mapped to ``(start, stop)``.

    Returns
    -------
    label : str or None
        Block label, or None when the index belongs to no block.
    offset : int
        Position of the term inside the block.
    """
    for label, (start, stop) in blocks.items():
        if start <= index < stop:
            return label, index - start
    return None, -1


def _match_channel(text: str, channels: Sequence[Any]) -> Any:
    """
    Recover a channel label from its string rendering in a block name.

    Parameters
    ----------
    text : str
        The label as it was interpolated into the block name.
    channels : sequence
        Fitted channel labels.

    Returns
    -------
    Any
        The matching label.

    Raises
    ------
    ValueError
        If no channel renders to *text*.
    """
    for label in channels:
        if str(label) == text:
            return label
    raise ValueError(f"Block names a channel {text!r} that is not among {list(channels)}")
