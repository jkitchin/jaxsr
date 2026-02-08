"""
JAXSR Streamlit App — interactive DOE workflow.

Launch with ``jaxsr app`` or ``python -m jaxsr.app``.
"""

from __future__ import annotations

import os

# Force JAX to use CPU backend — avoids "default_memory_space" errors
# in Streamlit environments. Must be set before JAX is imported.
os.environ["JAX_PLATFORMS"] = "cpu"


def launch_app(port: int = 8501, study_path: str | None = None) -> None:
    """
    Launch the JAXSR Streamlit DOE application.

    Parameters
    ----------
    port : int
        Port to run Streamlit on.
    study_path : str or None
        Optional path to a ``.jaxsr`` study file to pre-load.

    Raises
    ------
    ImportError
        If streamlit is not installed.
    """
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        raise ImportError(
            "Streamlit is required for the JAXSR app. " "Install it with: pip install jaxsr[app]"
        ) from None

    import sys
    from pathlib import Path

    main_script = str(Path(__file__).parent / "main.py")

    sys.argv = [
        "streamlit",
        "run",
        main_script,
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]

    if study_path is not None:
        sys.argv += ["--", "--study", study_path]

    stcli.main()
