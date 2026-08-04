"""
Tests for the NumPy-backed JAX shim used by the browser build.

``webapp/py/jax_shim.py`` lets JAXSR run under Pyodide, where ``jaxlib`` is
unavailable.  These tests check that the shim reproduces the parts of the JAX
contract that JAXSR actually relies on, and that a fit through the shim agrees
with the same fit on real JAX.

The shim has to be installed before ``jaxsr`` is imported, so the shim-side
runs happen in a subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIM_DIR = REPO_ROOT / "webapp" / "py"

pytestmark = pytest.mark.skipif(
    not (SHIM_DIR / "jax_shim.py").exists(),
    reason="webapp/py/jax_shim.py not present",
)

_PREAMBLE = f"""
import sys, json
sys.path.insert(0, {str(SHIM_DIR)!r})
import jax_shim
jax_shim.install(force=True)
import numpy as np
np.seterr(all="ignore")
"""


def run_under_shim(body: str) -> dict:
    """
    Execute ``body`` in a subprocess with the shim installed.

    Parameters
    ----------
    body : str
        Python source. It must ``print(json.dumps(...))`` its result.

    Returns
    -------
    dict
        The parsed JSON the subprocess printed on its last line.

    Raises
    ------
    AssertionError
        If the subprocess exits non-zero.
    """
    script = _PREAMBLE + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


# =============================================================================
# The JAX contract the shim has to reproduce
# =============================================================================


class TestJaxContract:
    """Behaviours JAXSR depends on that plain NumPy does not provide."""

    def test_at_index_update_is_functional(self):
        """``x.at[i].set(v)`` returns a new array and leaves the original alone."""
        result = run_under_shim("""
            import jax.numpy as jnp
            x = jnp.zeros(4)
            y = x.at[1].set(5.0).at[2].add(3.0)
            print(json.dumps({"x": np.asarray(x).tolist(), "y": np.asarray(y).tolist()}))
            """)
        assert result["x"] == [0.0, 0.0, 0.0, 0.0]
        assert result["y"] == [0.0, 5.0, 3.0, 0.0]

    def test_at_available_on_derived_arrays(self):
        """``.at`` survives the array-producing calls JAXSR chains together."""
        result = run_under_shim("""
            import jax.numpy as jnp
            checks = {
                "zeros": hasattr(jnp.zeros(3), "at"),
                "where": hasattr(jnp.where(jnp.zeros(3) > 0, 0.0, 1.0), "at"),
                "column_stack": hasattr(jnp.column_stack([jnp.zeros(3)] * 2), "at"),
                "arithmetic": hasattr(jnp.zeros(3) * 2 + 1, "at"),
                "slice": hasattr(jnp.zeros((3, 3))[:, 0], "at"),
                "asarray": hasattr(jnp.asarray([[1.0, 2.0]]), "at"),
            }
            print(json.dumps(checks))
            """)
        assert all(result.values()), result

    def test_singular_solve_returns_nan_instead_of_raising(self):
        """JAX linalg never raises; selection.py's lstsq fallback depends on that."""
        result = run_under_shim("""
            import jax.numpy as jnp
            singular = jnp.array([[1.0, 2.0], [2.0, 4.0]])
            out = jnp.linalg.solve(singular, jnp.array([1.0, 2.0]))
            print(json.dumps({"finite": bool(jnp.all(jnp.isfinite(out)))}))
            """)
        assert result["finite"] is False

    def test_grad_matches_analytic_gradient(self):
        """The finite-difference stand-in is accurate enough to drive L-BFGS-B."""
        result = run_under_shim("""
            import jax, jax.numpy as jnp
            f = lambda w: jnp.sum(w ** 3) + 2.0 * jnp.sum(w ** 2)
            w = jnp.array([0.5, -1.5, 3.0])
            got = np.asarray(jax.grad(f)(w))
            want = 3 * np.asarray(w) ** 2 + 4 * np.asarray(w)
            print(json.dumps({"got": got.tolist(), "want": want.tolist()}))
            """)
        assert np.allclose(result["got"], result["want"], rtol=1e-5, atol=1e-6)

    def test_jit_is_transparent(self):
        """``@jit`` decorated helpers still work, with and without arguments."""
        result = run_under_shim("""
            from jax import jit
            import jax.numpy as jnp

            @jit
            def double(x):
                return x * 2

            @jit(static_argnums=(1,))
            def scale(x, k):
                return x * k

            print(json.dumps({
                "double": np.asarray(double(jnp.array([1.0, 2.0]))).tolist(),
                "scale": np.asarray(scale(jnp.array([1.0, 2.0]), 3)).tolist(),
            }))
            """)
        assert result["double"] == [2.0, 4.0]
        assert result["scale"] == [3.0, 6.0]


# =============================================================================
# End-to-end agreement with real JAX
# =============================================================================

_FIT_BODY = """
from jaxsr import BasisLibrary, SymbolicRegressor

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 2)) * 2
y = 2.5 * X[:, 0] + 1.2 * X[:, 0] * X[:, 1] - 0.8 * X[:, 1] ** 2

library = (
    BasisLibrary(n_features=2, feature_names=["x0", "x1"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=3)
    .add_interactions(max_order=2)
)
model = SymbolicRegressor(
    basis_library=library, max_terms=5, strategy="greedy_forward",
    information_criterion="bic",
).fit(X, y)
print(json.dumps({
    "terms": list(model.selected_features_),
    "coefficients": [float(c) for c in model.coefficients_],
    "r2": float(model.metrics_["r2"]),
}))
"""


@pytest.fixture(scope="module")
def shim_fit() -> dict:
    """Fit the reference problem in a subprocess running on the shim."""
    return run_under_shim(_FIT_BODY)


class TestFitAgreement:
    """A fit through the shim must match the same fit on real JAX."""

    def test_recovers_the_generating_terms(self, shim_fit):
        """The shim recovers the three terms the data was generated from."""
        assert set(shim_fit["terms"]) >= {"x0", "x0*x1", "x1^2"}
        assert shim_fit["r2"] > 0.99

    def test_matches_real_jax(self, shim_fit):
        """Selected terms and coefficients agree with the real-JAX fit."""
        from jaxsr import BasisLibrary, SymbolicRegressor

        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 2)) * 2
        y = 2.5 * X[:, 0] + 1.2 * X[:, 0] * X[:, 1] - 0.8 * X[:, 1] ** 2
        library = (
            BasisLibrary(n_features=2, feature_names=["x0", "x1"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=3)
            .add_interactions(max_order=2)
        )
        model = SymbolicRegressor(
            basis_library=library,
            max_terms=5,
            strategy="greedy_forward",
            information_criterion="bic",
        ).fit(X, y)

        assert list(shim_fit["terms"]) == list(model.selected_features_)
        # JAX defaults to float32, the shim runs float64, so this is a
        # precision comparison rather than a bit-for-bit one.
        assert np.allclose(
            shim_fit["coefficients"],
            [float(c) for c in model.coefficients_],
            rtol=1e-3,
            atol=1e-4,
        )
