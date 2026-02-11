"""Reusable Streamlit widgets for the JAXSR app."""

from __future__ import annotations

import time

import streamlit as st

# ---------------------------------------------------------------------------
# Factor editor
# ---------------------------------------------------------------------------


def factor_editor(key: str = "factor_editor") -> list[dict]:
    """
    Render a dynamic form for adding/editing factors.

    Parameters
    ----------
    key : str
        Streamlit widget key prefix for uniqueness.

    Returns
    -------
    factors : list[dict]
        Each dict has keys: ``name``, ``type`` (``"continuous"`` or
        ``"categorical"``), and either ``bounds`` (tuple) or ``levels``
        (list of str).
    """
    if f"{key}_factors" not in st.session_state:
        st.session_state[f"{key}_factors"] = []

    factors: list[dict] = st.session_state[f"{key}_factors"]

    # --- Add-factor form ---------------------------------------------------
    with st.expander("Add a factor", expanded=len(factors) == 0):
        col1, col2 = st.columns([2, 1])
        with col1:
            name = st.text_input("Factor name", key=f"{key}_name")
        with col2:
            ftype = st.selectbox("Type", ["continuous", "categorical"], key=f"{key}_type")

        if ftype == "continuous":
            c1, c2, c3 = st.columns(3)
            with c1:
                low = st.number_input("Low bound", value=0.0, key=f"{key}_low")
            with c2:
                high = st.number_input("High bound", value=1.0, key=f"{key}_high")
            with c3:
                units = st.text_input("Units (optional)", key=f"{key}_units")
        else:
            levels_str = st.text_input("Levels (comma-separated)", key=f"{key}_levels")
            units = ""

        if st.button("Add factor", key=f"{key}_add"):
            if not name.strip():
                st.error("Factor name cannot be empty.")
            elif any(f["name"] == name.strip() for f in factors):
                st.error(f"Factor '{name.strip()}' already exists.")
            else:
                entry: dict = {"name": name.strip(), "type": ftype}
                if ftype == "continuous":
                    if low >= high:
                        st.error("Low bound must be less than high bound.")
                    else:
                        entry["bounds"] = (low, high)
                        if units.strip():
                            entry["units"] = units.strip()
                        factors.append(entry)
                        st.session_state[f"{key}_factors"] = factors
                        st.rerun()
                else:
                    levels = [lv.strip() for lv in levels_str.split(",") if lv.strip()]
                    if len(levels) < 2:
                        st.error("Categorical factors need at least 2 levels.")
                    else:
                        entry["levels"] = levels
                        factors.append(entry)
                        st.session_state[f"{key}_factors"] = factors
                        st.rerun()

    # --- Current factors table ---------------------------------------------
    if factors:
        st.markdown("**Defined factors:**")
        for i, f in enumerate(factors):
            cols = st.columns([3, 2, 2, 1])
            with cols[0]:
                st.text(f["name"])
            with cols[1]:
                st.text(f["type"])
            with cols[2]:
                if f["type"] == "continuous":
                    st.text(f'{f["bounds"][0]} – {f["bounds"][1]}')
                else:
                    st.text(", ".join(f["levels"]))
            with cols[3]:
                if st.button("Remove", key=f"{key}_rm_{i}"):
                    factors.pop(i)
                    st.session_state[f"{key}_factors"] = factors
                    st.rerun()

    return factors


# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------


def metric_cards(metrics: dict[str, float]) -> None:
    """
    Render a row of ``st.metric`` widgets for model metrics.

    Parameters
    ----------
    metrics : dict[str, float]
        Mapping of metric name to value (e.g. ``{"R²": 0.98, "MSE": 0.01}``).
    """
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items(), strict=False):
        with col:
            if isinstance(value, float):
                st.metric(name, f"{value:.4g}")
            else:
                st.metric(name, str(value))


# ---------------------------------------------------------------------------
# Sidebar status
# ---------------------------------------------------------------------------


def study_status_sidebar(study) -> None:
    """
    Render study status in the sidebar.

    Parameters
    ----------
    study : DOEStudy or None
        The current study, or None if no study is loaded.
    """
    from jaxsr.app.state import last_save_time

    with st.sidebar:
        st.markdown("### Study Status")
        if study is None:
            st.info("No study loaded.")
            return

        st.markdown(f"**Name:** {study.name}")
        st.markdown(f"**Factors:** {study.n_factors}")
        st.markdown(f"**Observations:** {study.n_observations}")
        st.markdown(f"**Fitted:** {'Yes' if study.is_fitted else 'No'}")

        if study.design_points is not None:
            n_design = len(study.design_points)
            n_pending = len(study.pending_points) if study.pending_points is not None else 0
            st.markdown(f"**Design points:** {n_design} ({n_pending} pending)")

        ts = last_save_time()
        if ts is not None:
            elapsed = time.time() - ts
            if elapsed < 60:
                st.caption(f"Last saved {int(elapsed)}s ago")
            else:
                st.caption(f"Last saved {int(elapsed / 60)}m ago")
