"""Page 1 — Create or load a DOE study and define factors."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import streamlit as st

from jaxsr.app.components import factor_editor, study_status_sidebar
from jaxsr.app.state import auto_save, get_study, set_study, study_file_path

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
study_status_sidebar(get_study())

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.header("Study Setup")

tab_new, tab_load = st.tabs(["Create New Study", "Load Existing Study"])

# ========================== Create New Study ===============================
with tab_new:
    st.subheader("Create a new DOE study")

    col1, col2 = st.columns(2)
    with col1:
        study_name = st.text_input("Study name", value="my_study")
    with col2:
        response_name = st.text_input("Response variable name", value="Response")

    description = st.text_area("Description (optional)", height=80)

    st.markdown("---")
    st.subheader("Define Factors")
    factors = factor_editor(key="setup")

    st.markdown("---")
    if st.button("Create Study", type="primary", disabled=len(factors) == 0):
        from jaxsr.study import DOEStudy

        factor_names = [f["name"] for f in factors]
        bounds = []
        feature_types = []
        categories = {}

        for i, f in enumerate(factors):
            feature_types.append(f["type"])
            if f["type"] == "continuous":
                bounds.append(f["bounds"])
            else:
                levels = f["levels"]
                bounds.append((0, len(levels) - 1))
                categories[i] = levels

        study = DOEStudy(
            name=study_name.strip() or "my_study",
            factor_names=factor_names,
            bounds=bounds,
            feature_types=(
                feature_types if any(ft == "categorical" for ft in feature_types) else None
            ),
            categories=categories if categories else None,
            description=description,
        )

        path = study_file_path(study.name)
        set_study(study, path)
        auto_save(study, path)
        st.success(f"Study **{study.name}** created with {study.n_factors} factors.")
        st.info(f"Saved to `{path}`")

# ========================== Load Existing Study ============================
with tab_load:
    st.subheader("Load a .jaxsr study file")

    upload_method = st.radio(
        "How to load?",
        ["Upload file", "Enter file path"],
        horizontal=True,
        key="load_method",
    )

    if upload_method == "Upload file":
        uploaded = st.file_uploader(
            "Upload .jaxsr file",
            type=["jaxsr"],
            key="study_upload",
        )
        if uploaded is not None:
            # Write to temp file, then load
            with tempfile.NamedTemporaryFile(suffix=".jaxsr", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            from jaxsr.study import DOEStudy

            study = DOEStudy.load(tmp_path)
            # Save to a local path so auto-save works
            local_path = study_file_path(study.name)
            set_study(study, local_path)
            auto_save(study, local_path)
            st.success(f"Loaded study **{study.name}** ({study.n_observations} observations).")
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    else:
        file_path = st.text_input("Path to .jaxsr file", key="study_path_input")
        if st.button("Load", key="load_path_btn") and file_path:
            p = Path(file_path)
            if not p.exists():
                st.error(f"File not found: {file_path}")
            else:
                from jaxsr.study import DOEStudy

                study = DOEStudy.load(str(p))
                set_study(study, str(p))
                st.success(f"Loaded study **{study.name}** ({study.n_observations} observations).")
