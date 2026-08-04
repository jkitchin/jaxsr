#!/usr/bin/env python
"""
Run the JAXSR test suite against the NumPy-backed JAX shim.

The browser build of JAXSR runs on ``webapp/py/jax_shim.py`` instead of real
JAX, because ``jaxlib`` has no Emscripten wheel.  That substitution is only
trustworthy if the library behaves the same way on top of it, so this script
installs the shim and then runs the ordinary test suite through it.

Usage
-----
    python scripts/test_under_numpy.py [pytest args...]

Examples
--------
    python scripts/test_under_numpy.py
    python scripts/test_under_numpy.py tests/test_regressor.py -x
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIM_DIR = REPO_ROOT / "webapp" / "py"


def main(argv: list[str]) -> int:
    """
    Install the shim, then hand off to pytest.

    Parameters
    ----------
    argv : list of str
        Arguments forwarded to pytest.  Defaults to ``["tests/"]``.

    Returns
    -------
    int
        The pytest exit code.
    """
    if "jaxsr" in sys.modules:  # pragma: no cover - defensive
        raise RuntimeError("jaxsr imported too early; run this script directly.")

    sys.path.insert(0, str(SHIM_DIR))
    import jax_shim

    jax_shim.install(force=True)

    import numpy as np

    # JAX silently produces inf/nan where NumPy warns; JAXSR already filters
    # non-finite basis columns in regressor.py, so match JAX's quiet behaviour.
    np.seterr(all="ignore")

    import jaxsr

    print(f"jaxsr {jaxsr.__version__} running on {sys.modules['jax'].__version__}")

    import pytest

    args = argv or ["tests/"]
    return pytest.main([*args, "-p", "no:cacheprovider"])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
