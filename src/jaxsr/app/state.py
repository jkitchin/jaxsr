"""Session state helpers, auto-save, and prerequisite guards."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Session-state key constants
# ---------------------------------------------------------------------------
_KEY_STUDY = "jaxsr_study"
_KEY_PATH = "jaxsr_study_path"
_KEY_LAST_SAVE = "jaxsr_last_save"


# ---------------------------------------------------------------------------
# Read / Write helpers
# ---------------------------------------------------------------------------


def get_study():
    """Return the current ``DOEStudy`` or *None*."""
    return st.session_state.get(_KEY_STUDY)


def get_study_path() -> str | None:
    """Return the file-path associated with the loaded study."""
    return st.session_state.get(_KEY_PATH)


def set_study(study, path: str | None = None) -> None:
    """
    Store *study* in session state.

    Parameters
    ----------
    study : DOEStudy
        The study object to store.
    path : str or None
        On-disk path associated with the study.
    """
    st.session_state[_KEY_STUDY] = study
    if path is not None:
        st.session_state[_KEY_PATH] = path


def auto_save(study=None, path: str | None = None) -> None:
    """
    Persist the study to disk and update the session timestamp.

    Parameters
    ----------
    study : DOEStudy or None
        Study to save.  Defaults to ``get_study()``.
    path : str or None
        File path.  Defaults to ``get_study_path()``.
    """
    study = study or get_study()
    path = path or get_study_path()
    if study is None or path is None:
        return
    try:
        study.save(path)
    except (OSError, PermissionError) as e:
        st.warning(f"Auto-save failed: {e}")
        return
    st.session_state[_KEY_LAST_SAVE] = time.time()


def last_save_time() -> float | None:
    """Return the UNIX timestamp of the last auto-save, or *None*."""
    return st.session_state.get(_KEY_LAST_SAVE)


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


def invalidate_model_caches() -> None:
    """Clear cached ANOVA / canonical results after a model change."""
    for key in list(st.session_state.keys()):
        if key.startswith("jaxsr_cache_"):
            del st.session_state[key]


# ---------------------------------------------------------------------------
# Prerequisite guards — call at the top of each page
# ---------------------------------------------------------------------------


def require_study():
    """Stop the page with a warning if no study is loaded."""
    if get_study() is None:
        st.warning("No study loaded. Go to **Setup** to create or load a study.")
        st.stop()
    return get_study()


def require_design(study):
    """Stop the page if no design has been generated."""
    if study.design_points is None:
        st.warning("No design generated yet. Go to **Design** to create one.")
        st.stop()


def require_observations(study):
    """Stop the page if the study has no observations."""
    if study.n_observations == 0:
        st.warning("No observations yet. Go to **Data** to import results.")
        st.stop()


def require_fitted(study):
    """Stop the page if the study has no fitted model."""
    if not study.is_fitted:
        st.warning("No model fitted yet. Go to **Fit** to fit a model.")
        st.stop()


def study_file_path(study_name: str) -> str:
    """Return a default ``.jaxsr`` file path for the given study name."""
    return str(Path.cwd() / f"{study_name}.jaxsr")
