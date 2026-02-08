"""Page 7 — Canonical analysis and suggest next experiments."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pandas as pd
import streamlit as st

from jaxsr.app.components import metric_cards, study_status_sidebar
from jaxsr.app.state import auto_save, get_study, require_fitted, require_study

# ---------------------------------------------------------------------------
study_status_sidebar(get_study())
study = require_study()
require_fitted(study)

model = study.model

# ---------------------------------------------------------------------------
st.header("Optimization")

# ---------------------------------------------------------------------------
# Canonical analysis
# ---------------------------------------------------------------------------
st.subheader("Canonical Analysis")

try:
    from jaxsr.rsm import canonical_analysis

    ca = canonical_analysis(model, bounds=study.bounds)

    # Nature badge
    nature_colors = {"maximum": "🟢", "minimum": "🔵", "saddle": "🟡"}
    badge = nature_colors.get(ca.nature, "⚪")
    st.markdown(f"**Nature:** {badge} {ca.nature}")

    # Stationary point
    sp_data = {study.factor_names[i]: f"{float(v):.4g}" for i, v in enumerate(ca.stationary_point)}
    st.markdown("**Stationary point:**")
    st.json(sp_data)

    # Predicted response at stationary point
    metric_cards({"Predicted Response": ca.stationary_response})

    # Eigenvalues
    st.markdown("**Eigenvalues:**")
    eig_df = pd.DataFrame(
        {"Eigenvalue": [f"{float(v):.4g}" for v in ca.eigenvalues]},
        index=[f"λ{i + 1}" for i in range(len(ca.eigenvalues))],
    )
    st.dataframe(eig_df, use_container_width=True)

    # Warnings
    if ca.warnings:
        for w in ca.warnings:
            st.warning(w)

except Exception as e:
    st.info(f"Canonical analysis not available: {e}")
    st.caption("Canonical analysis requires a quadratic (second-order) model.")

# ---------------------------------------------------------------------------
# Suggest next experiments
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Suggest Next Experiments")

col1, col2 = st.columns(2)
with col1:
    n_suggest = st.number_input("Number of points", min_value=1, max_value=100, value=5, step=1)
with col2:
    suggest_strategy = st.selectbox(
        "Strategy",
        ["space_filling", "uncertainty", "error", "leverage", "gradient", "random"],
        index=0,
    )

if st.button("Suggest", type="primary"):
    with st.spinner("Computing suggestions..."):
        try:
            next_pts = study.suggest_next(n_points=n_suggest, strategy=suggest_strategy)
            auto_save()

            st.success(f"Suggested {len(next_pts)} new experiments.")

            df_next = pd.DataFrame(next_pts, columns=study.factor_names)
            df_next.index = np.arange(1, len(df_next) + 1)
            df_next.index.name = "Run"
            st.dataframe(df_next, use_container_width=True)

            # Download
            col_csv, col_xlsx = st.columns(2)
            with col_csv:
                csv_buf = df_next.to_csv()
                st.download_button(
                    "Download CSV",
                    data=csv_buf,
                    file_name=f"{study.name}_next_experiments.csv",
                    mime="text/csv",
                )
            with col_xlsx:
                try:
                    import tempfile
                    from pathlib import Path

                    from jaxsr.excel import generate_template

                    # Create a temporary study with the suggested points as design
                    from jaxsr.study import DOEStudy

                    tmp_study = DOEStudy(
                        name=f"{study.name}_next",
                        factor_names=study.factor_names,
                        bounds=study.bounds,
                        feature_types=study.feature_types,
                        categories=study.categories,
                    )
                    tmp_study.create_design(method="latin_hypercube", n_points=len(next_pts))
                    # Override design points with our suggestions
                    tmp_study._design_points = next_pts

                    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                        generate_template(tmp_study, tmp.name)
                        tmp_path = tmp.name

                    with open(tmp_path, "rb") as f:
                        xlsx_data = f.read()

                    st.download_button(
                        "Download Excel Template",
                        data=xlsx_data,
                        file_name=f"{study.name}_next_template.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                    Path(tmp_path).unlink(missing_ok=True)
                except ImportError:
                    st.info("Install `openpyxl` and `xlsxwriter` for Excel export.")
        except Exception as e:
            st.error(f"Error suggesting experiments: {e}")
