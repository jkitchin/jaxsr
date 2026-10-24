"""
Embed fitted JAXSR models in mixed-integer optimization models.

A JAXSR model is a weighted sum of basis terms, so it can be written into an
optimization model term by term. When every selected term is constant, linear,
a hinge ``max(0, x - k)``, ``abs(x)``, a categorical indicator, or an
indicator times a continuous feature, the model is piecewise linear and the
embedding is an exact mixed-integer linear program (MILP):

==============================  ===============================================
Term                            Formulation
==============================  ===============================================
``1``, ``x``                    written directly
``max(0, x - k)``               aux ``h >= 0``, ``h >= x - k``, plus a binary
                                and two big-M rows forcing equality
``abs(x)``                      aux ``t >= x``, ``t >= -x``, plus a binary and
                                two big-M rows forcing equality
``I(c=v)``                      one-hot binary for category ``v`` of ``c``
``I(c=v)*x``                    exact McCormick product of binary and ``x``
==============================  ===============================================

The big-M constants come from the input bounds, which are therefore required
and are enforced by the embedding (the surrogate is only valid inside them).
Polynomial, interaction, transcendental, power-law and ratio terms are not
MILP-representable; pass ``allow_nonlinear=True`` to write them as nonlinear
expressions (giving an MIQCP or MINLP).

Two backends are supported: Pyomo (:func:`to_pyomo`) and discopt
(:func:`to_discopt`). Both share one formulation, so they produce the same
constraints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .regressor import SymbolicRegressor

#: Term types that have an exact MILP formulation.
MILP_TERM_TYPES = frozenset({"constant", "linear", "hinge", "indicator", "categorical_interaction"})

#: Term types that can be written as nonlinear expressions (``allow_nonlinear=True``).
NONLINEAR_TERM_TYPES = frozenset(
    {"polynomial", "interaction", "polynomial_interaction", "power", "ratio", "transcendental"}
)

_NONLINEAR_FUNCS = ("log", "exp", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh")

_SENSES = (None, "min", "max")


def _term_kind(bf) -> str:
    """Classify a basis function as ``"milp"``, ``"nonlinear"`` or ``"unsupported"``."""
    if bf.func_type in MILP_TERM_TYPES:
        return "milp"
    if bf.func_type == "transcendental":
        name = bf.func_config.get("func_name")
        if name == "abs":
            return "milp"
        if name in _NONLINEAR_FUNCS or name in ("inv", "square"):
            return "nonlinear"
        return "unsupported"
    if bf.func_type in NONLINEAR_TERM_TYPES:
        return "nonlinear"
    return "unsupported"


def _selected_terms(model: SymbolicRegressor) -> list[tuple[float, Any]]:
    """Return ``(coefficient, BasisFunction)`` for each selected term of a fitted model."""
    from .regressor import SymbolicRegressor

    if not isinstance(model, SymbolicRegressor):
        raise TypeError(f"Expected a SymbolicRegressor, got {type(model).__name__}")
    if not getattr(model, "_is_fitted", False):
        raise ValueError("Model is not fitted. Call fit() first.")
    if model.basis_library.has_parametric:
        raise ValueError("Models with parametric basis functions cannot be embedded.")

    lib = model.basis_library
    coefs = np.asarray(model.coefficients_, dtype=float)
    indices = np.asarray(model.selected_indices_, dtype=int)
    return [
        (float(c), lib.basis_functions[int(i)])
        for c, i in zip(coefs, indices, strict=True)
        if c != 0.0
    ]


def milp_incompatible_terms(model: SymbolicRegressor) -> list[str]:
    """
    List the selected terms of a fitted model that are not MILP-representable.

    Parameters
    ----------
    model : SymbolicRegressor
        A fitted model.

    Returns
    -------
    names : list of str
        Names of selected terms with no exact MILP formulation. An empty list
        means :func:`to_pyomo` and :func:`to_discopt` produce a pure MILP.

    Raises
    ------
    TypeError
        If ``model`` is not a SymbolicRegressor.
    ValueError
        If the model is not fitted or uses parametric basis functions.
    """
    return [bf.name for _, bf in _selected_terms(model) if _term_kind(bf) != "milp"]


def _resolve_bounds(
    model: SymbolicRegressor,
    bounds: dict[str, tuple[float, float]] | list[tuple[float, float]] | None,
) -> dict[int, tuple[float, float]]:
    """
    Bounds for each continuous feature.

    Priority: explicit ``bounds``, then the library's ``feature_bounds``, then the
    range of the training data.
    """
    lib = model.basis_library
    resolved: dict[int, tuple[float, float]] = {}
    if isinstance(bounds, dict):
        for key, (lo, hi) in bounds.items():
            if key not in lib.feature_names:
                raise ValueError(f"Unknown feature name {key!r} in bounds")
            resolved[lib.feature_names.index(key)] = (float(lo), float(hi))
    elif bounds is not None:
        if len(bounds) != lib.n_features:
            raise ValueError(f"bounds must have {lib.n_features} entries, got {len(bounds)}")
        resolved = {i: (float(lo), float(hi)) for i, (lo, hi) in enumerate(bounds)}

    X_train = getattr(model, "_X_train", None)
    for i in lib.continuous_indices:
        if i in resolved:
            continue
        if lib.feature_bounds is not None:
            lo, hi = lib.feature_bounds[i]
            resolved[i] = (float(lo), float(hi))
        elif X_train is not None:
            col = np.asarray(X_train, dtype=float)[:, i]
            resolved[i] = (float(np.min(col)), float(np.max(col)))
        else:
            raise ValueError(
                f"No bounds for feature '{lib.feature_names[i]}'. Pass bounds=... "
                "or set feature_bounds on the basis library."
            )

    for i in lib.continuous_indices:
        lo, hi = resolved[i]
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
            raise ValueError(
                f"Invalid bounds ({lo}, {hi}) for feature '{lib.feature_names[i]}'; "
                "finite bounds with lower <= upper are required for big-M constants."
            )
    return resolved


class _Backend:
    """Minimal modeling-layer interface the shared formulation is written against."""

    def continuous(self, key: str, lb: float, ub: float) -> Any:
        raise NotImplementedError

    def binary(self, key: str) -> Any:
        raise NotImplementedError

    def integer(self, key: str, lb: int, ub: int) -> Any:
        raise NotImplementedError

    def constrain(self, expr: Any) -> None:
        raise NotImplementedError

    def func(self, name: str) -> Callable[[Any], Any]:
        raise NotImplementedError


@dataclass
class _Formulation:
    """What the shared formulation built, in backend objects."""

    inputs: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    onehot: dict[tuple[str, Any], Any] = field(default_factory=dict)
    binaries: list[Any] = field(default_factory=list)


def _formulate(
    model: SymbolicRegressor,
    backend: _Backend,
    inputs: dict[str, Any] | None,
    output: Any,
    bounds: dict[str, tuple[float, float]] | list[tuple[float, float]] | None,
    output_sense: str | None,
    allow_nonlinear: bool,
) -> _Formulation:
    """Write ``model`` into ``backend`` and return the created variables."""
    if output_sense not in _SENSES:
        raise ValueError(f"output_sense must be None, 'min' or 'max', got {output_sense!r}")

    terms = _selected_terms(model)
    lib = model.basis_library
    names = lib.feature_names

    unsupported = [bf.name for _, bf in terms if _term_kind(bf) == "unsupported"]
    if unsupported:
        raise ValueError(f"These terms cannot be embedded in an optimization model: {unsupported}")
    nonlinear = [bf.name for _, bf in terms if _term_kind(bf) == "nonlinear"]
    if nonlinear and not allow_nonlinear:
        raise ValueError(
            f"These terms are not MILP-representable: {nonlinear}. Refit with a "
            "piecewise-linear library (add_linear, add_hinges, abs, categorical terms) "
            "or pass allow_nonlinear=True to embed them as nonlinear expressions."
        )

    if inputs is not None:
        unknown = set(inputs) - set(names)
        if unknown:
            raise ValueError(f"Unknown feature names in inputs: {sorted(unknown)}")

    feature_bounds = _resolve_bounds(model, bounds)
    form = _Formulation()

    # Input variables, bounded so the big-M constants below are valid.
    x: dict[int, Any] = {}
    for i in lib.continuous_indices:
        lo, hi = feature_bounds[i]
        x[i] = backend.continuous(f"in_{names[i]}", lo, hi)
        form.inputs[names[i]] = x[i]

    # Categorical inputs: one-hot binaries plus an integer-coded input that
    # matches the 0..K-1 encoding JAXSR uses for categorical columns.
    for i in lib.categorical_indices:
        cats = lib.categories[i]
        x[i] = backend.integer(f"in_{names[i]}", 0, len(cats) - 1)
        form.inputs[names[i]] = x[i]
        b = []
        for k, cat in enumerate(cats):
            b_k = backend.binary(f"onehot_{names[i]}_{k}")
            form.onehot[(names[i], cat)] = b_k
            form.binaries.append(b_k)
            b.append(b_k)
        backend.constrain(sum(b) == 1)
        backend.constrain(sum(k * b_k for k, b_k in enumerate(b)) == x[i])

    for name, var in (inputs or {}).items():
        backend.constrain(form.inputs[name] == var)

    def needs_exact(coef: float) -> bool:
        # A convex piece only needs its lower envelope when the output is pushed
        # in the direction that makes the envelope tight.
        if output_sense == "min":
            return coef < 0
        if output_sense == "max":
            return coef > 0
        return True

    def max0(j: int, coef: float, expr: Any, lo: float, hi: float) -> Any:
        """Variable equal to ``max(0, expr)`` with ``expr`` in ``[lo, hi]``."""
        if hi <= 0:
            return 0.0
        if lo >= 0:
            return expr
        h = backend.continuous(f"hinge_{j}", 0.0, hi)
        backend.constrain(h >= expr)
        if needs_exact(coef):
            z = backend.binary(f"hinge_on_{j}")
            form.binaries.append(z)
            backend.constrain(h <= expr - lo * (1 - z))
            backend.constrain(h <= hi * z)
        return h

    def abs_term(j: int, coef: float, expr: Any, lo: float, hi: float) -> Any:
        """Variable equal to ``|expr|`` with ``expr`` in ``[lo, hi]``."""
        if lo >= 0:
            return expr
        if hi <= 0:
            return -expr
        t = backend.continuous(f"abs_{j}", 0.0, max(-lo, hi))
        backend.constrain(t >= expr)
        backend.constrain(t >= -expr)
        if needs_exact(coef):
            z = backend.binary(f"abs_pos_{j}")
            form.binaries.append(z)
            backend.constrain(t <= expr - 2 * lo * (1 - z))
            backend.constrain(t <= -expr + 2 * hi * z)
        return t

    def onehot_of(fc: dict[str, Any], key: str) -> Any:
        i = fc[key]
        return form.onehot[(names[i], lib.categories[i][fc["category_index"]])]

    total = 0.0
    # Interval bounds on the output, accumulated from each term's range over
    # the input box; nonlinear terms leave the output unbounded.
    out_lo, out_hi = 0.0, 0.0
    for j, (coef, bf) in enumerate(terms):
        fc = bf.func_config
        ft = bf.func_type
        term_range: tuple[float, float] = (-np.inf, np.inf)
        if ft == "constant":
            term = 1.0
            term_range = (1.0, 1.0)
        elif ft == "linear":
            term = x[fc["feature_index"]]
            term_range = feature_bounds[fc["feature_index"]]
        elif ft == "hinge":
            i, knot, d = fc["feature_index"], fc["knot"], fc["direction"]
            lo, hi = feature_bounds[i]
            ends = (d * (lo - knot), d * (hi - knot))
            term = max0(j, coef, d * (x[i] - knot), min(ends), max(ends))
            term_range = (max(0.0, min(ends)), max(0.0, max(ends)))
        elif ft == "transcendental" and fc["func_name"] == "abs":
            i = fc["feature_index"]
            lo, hi = feature_bounds[i]
            term = abs_term(j, coef, x[i], lo, hi)
            term_range = (0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)), max(abs(lo), abs(hi)))
        elif ft == "indicator":
            term = onehot_of(fc, "feature_index")
            term_range = (0.0, 1.0)
        elif ft == "categorical_interaction":
            b = onehot_of(fc, "categorical_index")
            xc = x[fc["continuous_index"]]
            lo, hi = feature_bounds[fc["continuous_index"]]
            term_range = (min(0.0, lo), max(0.0, hi))
            term = backend.continuous(f"catprod_{j}", *term_range)
            backend.constrain(term <= hi * b)
            backend.constrain(term >= lo * b)
            backend.constrain(term <= xc - lo * (1 - b))
            backend.constrain(term >= xc - hi * (1 - b))
        elif ft == "polynomial":
            term = x[fc["feature_index"]] ** fc["degree"]
        elif ft == "interaction":
            term = 1.0
            for i in fc["feature_indices"]:
                term = term * x[i]
        elif ft == "polynomial_interaction":
            term = 1.0
            for i, e in enumerate(fc["exponents"]):
                if e == 1:
                    term = term * x[i]
                elif e > 1:
                    term = term * x[i] ** e
        elif ft == "power":
            term = x[fc["feature_index"]] ** fc["exponent"]
        elif ft == "ratio":
            term = x[fc["numerator_index"]] / x[fc["denominator_index"]]
        else:  # nonlinear transcendental
            xi = x[fc["feature_index"]]
            fname = fc["func_name"]
            if fname == "inv":
                term = 1.0 / xi
            elif fname == "square":
                term = xi**2
            else:
                term = backend.func(fname)(xi)
        total = total + coef * term
        scaled = (coef * term_range[0], coef * term_range[1])
        out_lo += min(scaled)
        out_hi += max(scaled)

    form.output = backend.continuous("out", out_lo, out_hi)
    backend.constrain(form.output == total)
    if output is not None:
        backend.constrain(output == form.output)
    return form


# ----------------------------------------------------------------------
# Pyomo
# ----------------------------------------------------------------------


class _PyomoBackend(_Backend):
    """Writes into a Pyomo block using a few sparse indexed components."""

    def __init__(self, block: Any):
        import pyomo.environ as pyo

        self.pyo = pyo
        self.block = block
        block.aux = pyo.Var(pyo.Any, dense=False)
        block.binaries = pyo.Var(pyo.Any, dense=False, within=pyo.Binary)
        block.integers = pyo.Var(pyo.Any, dense=False, within=pyo.Integers)
        block.cons = pyo.ConstraintList()

    def continuous(self, key: str, lb: float, ub: float) -> Any:
        v = self.block.aux[key]
        v.setlb(None if np.isinf(lb) else lb)
        v.setub(None if np.isinf(ub) else ub)
        return v

    def binary(self, key: str) -> Any:
        return self.block.binaries[key]

    def integer(self, key: str, lb: int, ub: int) -> Any:
        v = self.block.integers[key]
        v.setlb(lb)
        v.setub(ub)
        return v

    def constrain(self, expr: Any) -> None:
        if isinstance(expr, bool):
            if not expr:
                raise ValueError("Formulation produced an infeasible constant constraint")
            return
        self.block.cons.add(expr)

    def func(self, name: str) -> Callable[[Any], Any]:
        return getattr(self.pyo, name)


def to_pyomo(
    model: SymbolicRegressor,
    block: Any = None,
    inputs: dict[str, Any] | None = None,
    output: Any = None,
    bounds: dict[str, tuple[float, float]] | list[tuple[float, float]] | None = None,
    output_sense: str | None = None,
    allow_nonlinear: bool = False,
) -> Any:
    """
    Write a fitted model into a Pyomo block.

    Parameters
    ----------
    model : SymbolicRegressor
        A fitted model.
    block : pyomo Block, optional
        Block to populate, e.g. ``m.surrogate = pyo.Block()``. If None, a new
        ``ConcreteModel`` is created and returned.
    inputs : dict, optional
        Maps feature names to your own Pyomo variables or expressions; each
        is linked to the block's input with an equality constraint.
        Categorical inputs are integer-coded ``0..K-1``.
    output : pyomo Var or expression, optional
        Linked to the block's output with an equality constraint.
    bounds : dict or list, optional
        Input bounds, as ``{name: (lo, hi)}`` or a list in feature order.
        Defaults to the library's ``feature_bounds``, then the training data
        range. The block enforces them.
    output_sense : {None, "min", "max"}
        If the output is only ever minimized (or only maximized) in your
        problem, hinge and ``abs`` terms whose coefficient makes them convex
        in that direction are written without binaries. Leave None (the
        default) for an exact formulation that is valid in any problem.
    allow_nonlinear : bool
        If True, write non-MILP terms as nonlinear Pyomo expressions.

    Returns
    -------
    block : pyomo Block
        The populated block, with components ``aux`` (inputs ``aux["in_<name>"]``,
        output ``aux["out"]`` and auxiliary variables), ``binaries``, ``integers``
        and ``cons``. ``block.jaxsr_inputs`` and ``block.jaxsr_output`` point to
        the input and output variables.

    Raises
    ------
    ImportError
        If Pyomo is not installed.
    ValueError
        If the model has terms that cannot be embedded, has non-MILP terms
        and ``allow_nonlinear`` is False, or bounds are missing or invalid.

    Examples
    --------
    >>> import pyomo.environ as pyo
    >>> m = pyo.ConcreteModel()
    >>> m.x = pyo.Var(bounds=(0, 1))
    >>> m.sr = pyo.Block()
    >>> to_pyomo(model, m.sr, inputs={"x": m.x})
    >>> m.obj = pyo.Objective(expr=m.sr.jaxsr_output)
    """
    try:
        import pyomo.environ as pyo
    except ImportError as e:
        raise ImportError("to_pyomo requires Pyomo: pip install pyomo") from e

    if block is None:
        block = pyo.ConcreteModel()
    form = _formulate(
        model, _PyomoBackend(block), inputs, output, bounds, output_sense, allow_nonlinear
    )
    # Plain Python attributes, not Pyomo components: convenient handles.
    object.__setattr__(block, "jaxsr_inputs", form.inputs)
    object.__setattr__(block, "jaxsr_output", form.output)
    object.__setattr__(block, "jaxsr_onehot", form.onehot)
    return block


# ----------------------------------------------------------------------
# discopt
# ----------------------------------------------------------------------


@dataclass
class DiscoptEmbedding:
    """
    Variables created by :func:`to_discopt`.

    Attributes
    ----------
    inputs : dict
        Feature name to input variable (integer-coded for categorical features).
    output : discopt Variable
        Variable equal to the model prediction.
    onehot : dict
        ``(feature name, category)`` to the one-hot binary for that category.
    binaries : list
        All binary variables the embedding created.
    """

    inputs: dict[str, Any]
    output: Any
    onehot: dict[tuple[str, Any], Any]
    binaries: list[Any]


class _DiscoptBackend(_Backend):
    """Writes into a discopt Model, prefixing every variable name."""

    def __init__(self, dm_model: Any, prefix: str):
        import discopt.modeling as dm

        self.dm = dm
        self.model = dm_model
        self.prefix = prefix

    def _name(self, key: str) -> str:
        return f"{self.prefix}_{key}"

    def continuous(self, key: str, lb: float, ub: float) -> Any:
        kwargs = {}
        if np.isfinite(lb):
            kwargs["lb"] = lb
        if np.isfinite(ub):
            kwargs["ub"] = ub
        return self.model.continuous(self._name(key), **kwargs)

    def binary(self, key: str) -> Any:
        return self.model.binary(self._name(key))

    def integer(self, key: str, lb: int, ub: int) -> Any:
        return self.model.integer(self._name(key), lb=lb, ub=ub)

    def constrain(self, expr: Any) -> None:
        if isinstance(expr, bool):
            if not expr:
                raise ValueError("Formulation produced an infeasible constant constraint")
            return
        self.model.subject_to(expr)

    def func(self, name: str) -> Callable[[Any], Any]:
        return getattr(self.dm, name)


def to_discopt(
    model: SymbolicRegressor,
    dm_model: Any,
    inputs: dict[str, Any] | None = None,
    output: Any = None,
    bounds: dict[str, tuple[float, float]] | list[tuple[float, float]] | None = None,
    output_sense: str | None = None,
    allow_nonlinear: bool = False,
    prefix: str = "jaxsr",
) -> DiscoptEmbedding:
    """
    Write a fitted model into a discopt model.

    Parameters
    ----------
    model : SymbolicRegressor
        A fitted model.
    dm_model : discopt.Model
        Model to add variables and constraints to.
    inputs : dict, optional
        Maps feature names to your own discopt variables or expressions; each
        is linked to the embedding's input with an equality constraint.
        Categorical inputs are integer-coded ``0..K-1``.
    output : discopt Variable or expression, optional
        Linked to the embedding's output with an equality constraint.
    bounds : dict or list, optional
        Input bounds, as ``{name: (lo, hi)}`` or a list in feature order.
        Defaults to the library's ``feature_bounds``, then the training data
        range. The embedding enforces them.
    output_sense : {None, "min", "max"}
        See :func:`to_pyomo`.
    allow_nonlinear : bool
        If True, write non-MILP terms as nonlinear discopt expressions.
    prefix : str
        Prefix for every variable name, so several models can be embedded in
        one discopt model (default ``"jaxsr"``).

    Returns
    -------
    embedding : DiscoptEmbedding
        The input, output and binary variables created.

    Raises
    ------
    ImportError
        If discopt is not installed.
    ValueError
        If the model has terms that cannot be embedded, has non-MILP terms
        and ``allow_nonlinear`` is False, or bounds are missing or invalid.

    Examples
    --------
    >>> import discopt
    >>> m = discopt.Model("design")
    >>> emb = to_discopt(model, m)
    >>> m.minimize(emb.output)
    >>> result = m.solve()
    """
    try:
        import discopt.modeling  # noqa: F401  (import check only)
    except ImportError as e:
        raise ImportError("to_discopt requires discopt: pip install discopt") from e

    form = _formulate(
        model,
        _DiscoptBackend(dm_model, prefix),
        inputs,
        output,
        bounds,
        output_sense,
        allow_nonlinear,
    )
    return DiscoptEmbedding(
        inputs=form.inputs, output=form.output, onehot=form.onehot, binaries=form.binaries
    )
