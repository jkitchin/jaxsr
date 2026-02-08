"""Page 2 — Generate experimental designs and download templates."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pandas as pd
import streamlit as st

from jaxsr.app.components import study_status_sidebar
from jaxsr.app.state import auto_save, get_study, require_study

# ---------------------------------------------------------------------------
study_status_sidebar(get_study())
study = require_study()

# ---------------------------------------------------------------------------
st.header("Generate Design")

# Design method selection
METHODS = [
    "latin_hypercube",
    "sobol",
    "halton",
    "factorial",
    "ccd",
    "box_behnken",
    "fractional_factorial",
]

col1, col2 = st.columns(2)
with col1:
    method = st.selectbox("Design method", METHODS, index=0)
with col2:
    n_points = st.number_input(
        "Number of points",
        min_value=3,
        max_value=1000,
        value=20,
        step=1,
        help="For factorial / CCD / Box-Behnken this is ignored (determined by factors).",
    )

# Method-specific options
kwargs: dict = {}
if method == "ccd":
    c1, c2 = st.columns(2)
    with c1:
        alpha = st.selectbox("Alpha", ["rotatable", "face", "spherical"], index=0)
        kwargs["alpha"] = alpha
    with c2:
        center_pts = st.number_input("Center points", min_value=1, value=1, step=1)
        kwargs["center_points"] = center_pts
elif method == "box_behnken":
    center_pts = st.number_input("Center points", min_value=1, value=1, step=1)
    kwargs["center_points"] = center_pts
elif method == "fractional_factorial":
    resolution = st.number_input("Resolution", min_value=3, max_value=5, value=3, step=1)
    kwargs["resolution"] = resolution
elif method == "factorial":
    levels = st.number_input("Levels", min_value=2, max_value=5, value=2, step=1)
    kwargs["levels"] = levels

seed = st.number_input("Random seed (0 = random)", min_value=0, value=42, step=1)
random_state = seed if seed > 0 else None

st.markdown("---")

# Generate button
if st.button("Generate Design", type="primary"):
    try:
        X = study.create_design(
            method=method, n_points=n_points, random_state=random_state, **kwargs
        )
        auto_save()
        st.success(f"Generated {len(X)} design points using **{method}**.")
    except Exception as e:
        st.error(f"Error generating design: {e}")

# ---------------------------------------------------------------------------
# Preview design
# ---------------------------------------------------------------------------
if study.design_points is not None:
    st.subheader("Design Preview")
    X = study.design_points
    df = pd.DataFrame(X, columns=study.factor_names)
    df.index = np.arange(1, len(df) + 1)
    df.index.name = "Run"
    st.dataframe(df, use_container_width=True)

    st.markdown(f"**{len(X)} design points**")

    # Download buttons
    st.subheader("Download")
    col_csv, col_xlsx = st.columns(2)

    with col_csv:
        csv_buf = df.to_csv()
        st.download_button(
            "Download CSV",
            data=csv_buf,
            file_name=f"{study.name}_design.csv",
            mime="text/csv",
        )

    with col_xlsx:
        try:
            import tempfile
            from pathlib import Path

            from jaxsr.excel import generate_template

            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                generate_template(study, tmp.name)
                tmp_path = tmp.name

            with open(tmp_path, "rb") as f:
                xlsx_data = f.read()

            st.download_button(
                "Download Excel Template",
                data=xlsx_data,
                file_name=f"{study.name}_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            Path(tmp_path).unlink(missing_ok=True)
        except ImportError:
            st.info("Install `openpyxl` and `xlsxwriter` for Excel export.")
