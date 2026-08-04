"""
A NumPy-backed stand-in for the slice of JAX that JAXSR actually uses.

``jaxlib`` is a compiled XLA wheel with no Emscripten build, so JAX cannot run
under Pyodide.  JAXSR, however, uses JAX almost entirely as "NumPy with a
different import name": across ``src/jaxsr`` there are roughly 1300 ``jnp.*``
references, all of which name functions that NumPy also provides, plus a very
small tail of genuinely JAX-specific API:

===========================  ===============================================
JAX surface                  Replacement provided here
===========================  ===============================================
``jax.numpy``                NumPy, with ``array``/``asarray`` returning an
                             ``ndarray`` subclass that supports ``.at[]``
``jax.jit``                  identity decorator
``jax.grad``                 central finite differences
``jax.lax.erf``              ``scipy.special.erf``
``jax.random.PRNGKey``       ``numpy.random.Generator``
``jax.random.split``         independent child generators
===========================  ===============================================

Call :func:`install` *before* importing ``jaxsr``.  It registers the stand-in
modules in :data:`sys.modules`, so ``import jax.numpy as jnp`` inside JAXSR
resolves here without any change to the library itself.

Notes
-----
NumPy defaults to float64 where JAX defaults to float32.  That is a deliberate
and welcome difference: ``selection.py`` takes its closed-form O(k) MSE path
only for float64 input, and the finite-difference steps in ``constraints.py``
were tuned around float32 cancellation, so they become more accurate rather
than less.  Expect small numerical differences from a real JAX run.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
from scipy.special import erf as _scipy_erf

__all__ = ["install", "is_installed"]


# =============================================================================
# Functional index updates: JAX's ``x.at[idx].set(v)`` syntax
# =============================================================================


class _IndexUpdateRef:
    """
    A pending functional update to ``array`` at ``index``.

    Every method returns a *new* array, matching JAX's immutable-update
    semantics.  The original array is never modified.
    """

    __slots__ = ("_array", "_index")

    def __init__(self, array: np.ndarray, index: Any) -> None:
        self._array = array
        self._index = index

    def _updated(self, op: str, values: Any) -> np.ndarray:
        out = np.array(self._array, copy=True)
        if op == "set":
            out[self._index] = values
        elif op == "add":
            out[self._index] += values
        elif op == "subtract":
            out[self._index] -= values
        elif op == "multiply":
            out[self._index] *= values
        elif op == "divide":
            out[self._index] /= values
        elif op == "power":
            out[self._index] **= values
        elif op == "min":
            out[self._index] = np.minimum(out[self._index], values)
        elif op == "max":
            out[self._index] = np.maximum(out[self._index], values)
        else:  # pragma: no cover - guarded by the public methods below
            raise ValueError(f"Unknown index update operation: {op}")
        return out.view(_ShimArray)

    def get(self) -> np.ndarray:
        """Return ``array[index]``."""
        return np.asarray(self._array[self._index]).view(_ShimArray)

    def set(self, values: Any) -> np.ndarray:
        """Return a copy with ``array[index] = values``."""
        return self._updated("set", values)

    def add(self, values: Any) -> np.ndarray:
        """Return a copy with ``array[index] += values``."""
        return self._updated("add", values)

    def subtract(self, values: Any) -> np.ndarray:
        """Return a copy with ``array[index] -= values``."""
        return self._updated("subtract", values)

    def multiply(self, values: Any) -> np.ndarray:
        """Return a copy with ``array[index] *= values``."""
        return self._updated("multiply", values)

    # JAX spells this both ways.
    mul = multiply

    def divide(self, values: Any) -> np.ndarray:
        """Return a copy with ``array[index] /= values``."""
        return self._updated("divide", values)

    def power(self, values: Any) -> np.ndarray:
        """Return a copy with ``array[index] **= values``."""
        return self._updated("power", values)

    def min(self, values: Any) -> np.ndarray:
        """Return a copy with ``array[index] = minimum(array[index], values)``."""
        return self._updated("min", values)

    def max(self, values: Any) -> np.ndarray:
        """Return a copy with ``array[index] = maximum(array[index], values)``."""
        return self._updated("max", values)


class _IndexUpdateHelper:
    """The object returned by ``array.at``; indexing it yields an update ref."""

    __slots__ = ("_array",)

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def __getitem__(self, index: Any) -> _IndexUpdateRef:
        return _IndexUpdateRef(self._array, index)


class _ShimArray(np.ndarray):
    """
    A plain ``ndarray`` that additionally understands JAX's ``.at[]`` syntax.

    JAXSR calls ``.at[]`` at roughly 30 sites across ``sampling.py``,
    ``selection.py`` and ``constraints.py``, on arrays originating from many
    different NumPy entry points.  ``_wrap_result`` therefore converts results
    at the ``jnp`` module boundary, and NumPy's own subclass propagation
    through ufuncs, slicing and arithmetic carries the accessor from there.
    """

    @property
    def at(self) -> _IndexUpdateHelper:
        """Functional index-update accessor (see :class:`_IndexUpdateRef`)."""
        return _IndexUpdateHelper(self)

    def block_until_ready(self) -> _ShimArray:
        """No-op; present so JAX benchmarking idioms do not break."""
        return self


def _array(obj: Any, dtype: Any = None, **kwargs: Any) -> np.ndarray:
    """``jnp.array`` equivalent: always copies, returns a ``.at``-capable array."""
    kwargs.setdefault("copy", True)
    return np.array(obj, dtype=dtype, **kwargs).view(_ShimArray)


def _asarray(obj: Any, dtype: Any = None, **kwargs: Any) -> np.ndarray:
    """``jnp.asarray`` equivalent: avoids copying, returns a ``.at``-capable array."""
    return np.asarray(obj, dtype=dtype, **kwargs).view(_ShimArray)


# =============================================================================
# jax.jit / jax.grad
# =============================================================================


def _jit(fun: Any = None, **_kwargs: Any) -> Any:
    """
    Identity stand-in for :func:`jax.jit`.

    Supports both ``@jit`` and ``@jit(static_argnums=...)`` spellings.  JAXSR
    uses ``@jit`` on four small helpers in ``utils.py``; without XLA there is
    nothing to compile, and NumPy evaluates them eagerly.
    """
    if fun is None:
        return lambda f: f
    return fun


# Relative step for central differences.  For float64 the error-optimal step is
# around eps**(1/3) ~= 6e-6, so 1e-6 sits in the right regime.
#
# This must NOT be shrunk toward scipy's own default finite-difference step
# (~1.5e-8).  The penalty functions in constraints.py are themselves built from
# finite differences, and at 1.5e-8 they look constant to the optimizer -- that
# is precisely the bug jax.grad was introduced to fix.
_GRAD_STEP = 1e-6


def _grad(fun: Any, argnums: int = 0, **_kwargs: Any) -> Any:
    """
    Central finite-difference stand-in for :func:`jax.grad`.

    Parameters
    ----------
    fun : callable
        Scalar-valued function to differentiate.
    argnums : int
        Index of the argument to differentiate with respect to.

    Returns
    -------
    grad_fun : callable
        Function returning the gradient of ``fun`` with the same shape as the
        differentiated argument.
    """

    def grad_fun(*args: Any, **kwargs: Any) -> np.ndarray:
        args = list(args)
        x0 = np.asarray(args[argnums], dtype=np.float64)
        flat = x0.ravel()
        out = np.zeros_like(flat)
        for i in range(flat.size):
            step = _GRAD_STEP * max(1.0, abs(float(flat[i])))
            plus = flat.copy()
            plus[i] += step
            minus = flat.copy()
            minus[i] -= step

            args[argnums] = _array(plus.reshape(x0.shape))
            f_plus = float(fun(*args, **kwargs))
            args[argnums] = _array(minus.reshape(x0.shape))
            f_minus = float(fun(*args, **kwargs))

            out[i] = (f_plus - f_minus) / (2.0 * step)
        return _array(out.reshape(x0.shape))

    return grad_fun


def _value_and_grad(fun: Any, argnums: int = 0, **kwargs: Any) -> Any:
    """Stand-in for :func:`jax.value_and_grad`."""
    grad_fun = _grad(fun, argnums=argnums, **kwargs)

    def value_and_grad_fun(*args: Any, **kw: Any) -> tuple[float, np.ndarray]:
        return float(fun(*args, **kw)), grad_fun(*args, **kw)

    return value_and_grad_fun


# =============================================================================
# Module construction
# =============================================================================


def _wrap_result(fn: Any) -> Any:
    """
    Wrap a NumPy function so a plain ``ndarray`` result gains ``.at``.

    ``.at[]`` is used at ~30 sites across ``sampling.py``, ``selection.py`` and
    ``constraints.py``, on arrays produced by a wide range of NumPy entry
    points (``zeros``, ``where``, ``column_stack``, ...).  Wrapping every
    non-ufunc function is more robust than enumerating the constructors that
    happen to be used today.  Ufuncs are deliberately left alone: they already
    preserve ndarray subclasses, and wrapping them would hide ``reduce`` and
    ``outer``.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        if type(out) is np.ndarray:
            return out.view(_ShimArray)
        return out

    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    wrapper.__wrapped__ = fn
    return wrapper


def _build_linalg_module() -> types.ModuleType:
    """
    Build a ``jax.numpy.linalg`` stand-in that never raises.

    This is the one place where NumPy and JAX genuinely disagree rather than
    merely differing in spelling.  NumPy raises :class:`~numpy.linalg.LinAlgError`
    on a singular or non-converging problem; JAX returns an array of NaN.  JAXSR
    is written against the JAX contract -- ``selection.py:290-295`` solves the
    Gram system, tests the result with ``isfinite``, and falls back to ``lstsq``
    -- so the shim must return NaN rather than raise, or that fallback is never
    reached.
    """
    mod = types.ModuleType("jax.numpy.linalg")
    mod.__doc__ = "Non-raising stand-in for jax.numpy.linalg (see jax_shim)."
    for name in dir(np.linalg):
        if not name.startswith("_"):
            setattr(mod, name, getattr(np.linalg, name))

    nan = np.nan

    def solve(a: Any, b: Any) -> np.ndarray:
        """Solve ``a @ x == b``, returning NaN instead of raising when singular."""
        try:
            return np.linalg.solve(a, b).view(_ShimArray)
        except np.linalg.LinAlgError:
            return np.full(np.shape(b), nan).view(_ShimArray)

    def lstsq(a: Any, b: Any, rcond: Any = None) -> tuple:
        """Least squares, returning NaN coefficients instead of raising."""
        try:
            return np.linalg.lstsq(a, b, rcond=rcond)
        except np.linalg.LinAlgError:
            a = np.asarray(a)
            b = np.asarray(b)
            x = np.full((a.shape[1], *b.shape[1:]), nan)
            s = np.full((min(a.shape),), nan)
            return x.view(_ShimArray), np.array([]), 0, s

    def svd(a: Any, full_matrices: bool = True, compute_uv: bool = True, **kw: Any) -> Any:
        """SVD, returning NaN factors instead of raising when it does not converge."""
        try:
            return np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, **kw)
        except np.linalg.LinAlgError:
            a = np.asarray(a)
            m, n = a.shape[-2], a.shape[-1]
            k = min(m, n)
            s = np.full((*a.shape[:-2], k), nan).view(_ShimArray)
            if not compute_uv:
                return s
            u_cols, vh_rows = (m, n) if full_matrices else (k, k)
            u = np.full((*a.shape[:-2], m, u_cols), nan).view(_ShimArray)
            vh = np.full((*a.shape[:-2], vh_rows, n), nan).view(_ShimArray)
            return u, s, vh

    def pinv(a: Any, *args: Any, **kw: Any) -> np.ndarray:
        """Pseudo-inverse, returning NaN instead of raising."""
        try:
            return np.linalg.pinv(a, *args, **kw).view(_ShimArray)
        except np.linalg.LinAlgError:
            a = np.asarray(a)
            return np.full((*a.shape[:-2], a.shape[-1], a.shape[-2]), nan).view(_ShimArray)

    def inv(a: Any) -> np.ndarray:
        """Matrix inverse, returning NaN instead of raising when singular."""
        try:
            return np.linalg.inv(a).view(_ShimArray)
        except np.linalg.LinAlgError:
            return np.full(np.shape(a), nan).view(_ShimArray)

    def cholesky(a: Any, **kw: Any) -> np.ndarray:
        """Cholesky factor, returning NaN instead of raising when not PD."""
        try:
            return np.linalg.cholesky(a, **kw).view(_ShimArray)
        except np.linalg.LinAlgError:
            return np.full(np.shape(a), nan).view(_ShimArray)

    def eigh(a: Any, **kw: Any) -> tuple:
        """Symmetric eigendecomposition, returning NaN instead of raising."""
        try:
            return np.linalg.eigh(a, **kw)
        except np.linalg.LinAlgError:
            a = np.asarray(a)
            w = np.full(a.shape[:-1], nan).view(_ShimArray)
            return w, np.full(a.shape, nan).view(_ShimArray)

    mod.solve = solve
    mod.lstsq = lstsq
    mod.svd = svd
    mod.pinv = pinv
    mod.inv = inv
    mod.cholesky = cholesky
    mod.eigh = eigh
    return mod


def _build_numpy_module() -> types.ModuleType:
    """Build a ``jax.numpy`` stand-in from the public NumPy namespace."""
    mod = types.ModuleType("jax.numpy")
    mod.__doc__ = "NumPy-backed stand-in for jax.numpy (see jax_shim)."
    for name in dir(np):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(np, name)
        except AttributeError:
            # NumPy raises for a handful of removed legacy aliases.
            continue
        # Wrap plain functions so their results carry ``.at``; leave classes,
        # modules, ufuncs and constants exactly as NumPy defines them.
        if (
            callable(obj)
            and not isinstance(obj, (type, np.ufunc))
            and not isinstance(obj, types.ModuleType)
        ):
            obj = _wrap_result(obj)
        setattr(mod, name, obj)

    mod.array = _array
    mod.asarray = _asarray
    mod.ndarray = np.ndarray
    mod.DeviceArray = np.ndarray
    mod.linalg = _build_linalg_module()
    return mod


def _build_lax_module() -> types.ModuleType:
    """Build a ``jax.lax`` stand-in (JAXSR only uses ``lax.erf``)."""
    mod = types.ModuleType("jax.lax")
    mod.__doc__ = "Minimal stand-in for jax.lax (see jax_shim)."
    mod.erf = _scipy_erf
    mod.stop_gradient = lambda x: x
    return mod


def _build_random_module() -> types.ModuleType:
    """Build a ``jax.random`` stand-in backed by ``numpy.random.Generator``."""
    mod = types.ModuleType("jax.random")
    mod.__doc__ = "Minimal stand-in for jax.random (see jax_shim)."

    def PRNGKey(seed: int) -> np.random.Generator:  # noqa: N802 - JAX's name
        """Return a NumPy generator standing in for a JAX PRNG key."""
        return np.random.default_rng(int(seed))

    def split(key: np.random.Generator, num: int = 2) -> list[np.random.Generator]:
        """Split a key into ``num`` independent generators."""
        seeds = key.integers(0, 2**31 - 1, size=num)
        return [np.random.default_rng(int(s)) for s in seeds]

    mod.PRNGKey = PRNGKey
    mod.key = PRNGKey
    mod.split = split
    mod.normal = lambda key, shape=(), dtype=float: _array(key.standard_normal(shape), dtype)
    mod.uniform = lambda key, shape=(), dtype=float, minval=0.0, maxval=1.0: _array(
        key.uniform(minval, maxval, shape), dtype
    )
    return mod


def is_installed() -> bool:
    """
    Report whether the shim currently occupies ``sys.modules["jax"]``.

    Returns
    -------
    bool
        True if the stand-in modules are installed.
    """
    jax_mod = sys.modules.get("jax")
    return getattr(jax_mod, "__jaxsr_shim__", False) is True


def install(force: bool = False) -> types.ModuleType:
    """
    Register the NumPy-backed stand-in modules in :data:`sys.modules`.

    Must be called before ``import jaxsr``.

    Parameters
    ----------
    force : bool
        Replace a genuine JAX installation if one is already imported.  By
        default an already-imported real JAX is left alone and reused.

    Returns
    -------
    module
        The ``jax`` stand-in module.

    Raises
    ------
    RuntimeError
        If ``jaxsr`` has already been imported, in which case it has already
        bound the real ``jnp`` and installing the shim would have no effect.
    """
    if is_installed():
        return sys.modules["jax"]

    if "jaxsr" in sys.modules:
        raise RuntimeError("jaxsr is already imported; install the shim before importing jaxsr.")

    if "jax" in sys.modules and not force:
        return sys.modules["jax"]

    jnp = _build_numpy_module()
    lax = _build_lax_module()
    random = _build_random_module()

    jax = types.ModuleType("jax")
    jax.__doc__ = "NumPy-backed stand-in for JAX (see jax_shim)."
    jax.__jaxsr_shim__ = True
    jax.__version__ = "0.0.0+jaxsr-shim"
    jax.numpy = jnp
    jax.lax = lax
    jax.random = random
    jax.jit = _jit
    jax.grad = _grad
    jax.value_and_grad = _value_and_grad
    jax.device_put = lambda x, *a, **kw: _asarray(x)
    jax.block_until_ready = lambda x: x
    jax.Array = np.ndarray

    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = jnp
    sys.modules["jax.numpy.linalg"] = jnp.linalg
    sys.modules["jax.lax"] = lax
    sys.modules["jax.random"] = random
    return jax
