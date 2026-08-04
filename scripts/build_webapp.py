#!/usr/bin/env python
"""
Build the assets the JAXSR browser app needs.

The app installs JAXSR into Pyodide from a wheel served alongside the page, so
the wheel has to exist before the page can boot.  This script builds it and
writes ``webapp/manifest.json`` describing what the page should load.

Usage
-----
    python scripts/build_webapp.py [--skip-build]

``--skip-build`` reuses an existing wheel in ``dist/``, which is handy when
iterating on the front end.

The wheel and manifest are generated artifacts and are gitignored; CI rebuilds
them before publishing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
WEBAPP = REPO_ROOT / "webapp"
WHEELS = WEBAPP / "wheels"
DIST = REPO_ROOT / "dist"

# Pinned so the page always boots against a known NumPy/SciPy.  Bumping this is
# a deliberate act: check that numpy, scipy and sympy are still available in the
# new release before changing it.
#
# 0.29.3 ships CPython 3.13 / numpy 2.2 / scipy 1.14 / sympy 1.13, which is the
# closest match to the environment the NumPy shim is validated against by
# scripts/test_under_numpy.py.  Newer Pyodide releases run CPython 3.14, which
# is outside the Python versions jaxsr's CI covers.
PYODIDE_VERSION = "0.29.3"
PYODIDE_CDN = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"

# Loaded before jaxsr.  sympy is deliberately absent: it costs several MB and is
# only needed for LaTeX output, so the page fetches it on demand.
CORE_PACKAGES = ["numpy", "scipy", "micropip"]


def build_wheel() -> Path:
    """
    Build the jaxsr wheel into ``dist/``.

    Returns
    -------
    Path
        The freshly built wheel.

    Raises
    ------
    SystemExit
        If the build fails or produces no wheel.
    """
    print("Building jaxsr wheel...")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(DIST)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("Wheel build failed. Is `build` installed? pip install build")
    return newest_wheel()


def newest_wheel() -> Path:
    """
    Return the most recently modified jaxsr wheel in ``dist/``.

    Returns
    -------
    Path
        Path to the wheel.

    Raises
    ------
    SystemExit
        If no wheel is present.
    """
    wheels = sorted(DIST.glob("jaxsr-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise SystemExit(f"No jaxsr wheel found in {DIST}. Run without --skip-build.")
    return wheels[-1]


def main() -> int:
    """
    Stage the wheel and write the manifest.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-build", action="store_true", help="reuse an existing wheel in dist/"
    )
    args = parser.parse_args()

    wheel = newest_wheel() if args.skip_build else build_wheel()

    import make_example_workbook

    make_example_workbook.main([])

    WHEELS.mkdir(parents=True, exist_ok=True)
    for stale in WHEELS.glob("jaxsr-*.whl"):
        stale.unlink()
    staged = WHEELS / wheel.name
    shutil.copy2(wheel, staged)

    version = wheel.name.split("-")[1]
    manifest = {
        "jaxsrVersion": version,
        "wheel": f"wheels/{wheel.name}",
        "pyodideVersion": PYODIDE_VERSION,
        "pyodideIndexURL": PYODIDE_CDN,
        "corePackages": CORE_PACKAGES,
        "pythonModules": ["py/jax_shim.py", "py/kernel.py"],
    }
    manifest_path = WEBAPP / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    size_mb = staged.stat().st_size / 1e6
    print(f"  wheel     {staged.relative_to(REPO_ROOT)} ({size_mb:.2f} MB)")
    print(f"  manifest  {manifest_path.relative_to(REPO_ROOT)}")
    print(f"  jaxsr     {version} on pyodide {PYODIDE_VERSION}")
    print(f"\nServe locally with:\n  python -m http.server -d {WEBAPP.relative_to(REPO_ROOT)} 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
