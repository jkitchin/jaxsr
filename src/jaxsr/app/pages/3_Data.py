"""Page 3 — Import experimental results from Excel or CSV."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pandas as pd
import streamlit as st

from jaxsr.app.components import study_status_sidebar
from jaxsr.app.state import (
    auto_save,
    get_study,
    invalidate_model_caches,
    require_study,
)

# ---------------------------------------------------------------------------
study_status_sidebar(get_study())
study = require_study()

# ---------------------------------------------------------------------------
st.header("Import Data")

# ---------------------------------------------------------------------------
# Design completion status
# ---------------------------------------------------------------------------
if study.design_points is not None:
    total = len(study.design_points)
    completed = study.n_observations
    progress = min(completed / total, 1.0) if total > 0 else 0.0
    st.progress(progress, text=f"{completed} / {total} design points completed")

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload completed data file",
    type=["xlsx", "xls", "csv"],
    key="data_upload",
)

if uploaded is not None:
    file_name = uploaded.name

    if file_name.endswith((".xlsx", ".xls")):
        # Try the JAXSR template reader first
        try:
            import tempfile
            from pathlib import Path

            from jaxsr.excel import read_completed_template

            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            X, y = read_completed_template(study, tmp_path)
            Path(tmp_path).unlink(missing_ok=True)

            st.success(f"Read {len(y)} observations from Excel template.")
            df_preview = pd.DataFrame(X, columns=study.factor_names)
            df_preview["Response"] = y
            st.dataframe(df_preview, use_container_width=True)

            if st.button("Import these observations", type="primary", key="import_xlsx"):
                study.add_observations(X, y)
                invalidate_model_caches()
                auto_save()
                st.success(f"Added {len(y)} observations. Total: {study.n_observations}")
                st.rerun()

        except ImportError:
            st.error("Install `openpyxl` for Excel import: `pip install jaxsr[excel]`")
        except (ValueError, KeyError) as e:
            st.error(f"Error reading Excel file: {e}")

    elif file_name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded)
            st.dataframe(df, use_container_width=True)

            # Expect columns: factor_names + at least one response column
            expected_cols = study.factor_names
            if all(col in df.columns for col in expected_cols):
                # Find the response column (first column not in factor_names)
                response_cols = [c for c in df.columns if c not in expected_cols]
                if response_cols:
                    response_col = st.selectbox("Response column", response_cols)
                else:
                    st.error("No response column found (all columns match factor names).")
                    st.stop()

                X = df[expected_cols].values.astype(float)
                y = df[response_col].values.astype(float)

                # Drop rows with NaN response
                mask = ~np.isnan(y)
                X, y = X[mask], y[mask]

                if len(y) == 0:
                    st.warning("No valid (non-NaN) response values found.")
                else:
                    st.info(f"Found {len(y)} observations with response column '{response_col}'.")
                    if st.button("Import these observations", type="primary", key="import_csv"):
                        study.add_observations(X, y)
                        invalidate_model_caches()
                        auto_save()
                        st.success(f"Added {len(y)} observations. Total: {study.n_observations}")
                        st.rerun()
            else:
                missing = [c for c in expected_cols if c not in df.columns]
                st.error(f"Missing expected factor columns: {missing}")
        except (ValueError, KeyError) as e:
            st.error(f"Error reading CSV: {e}")

# ---------------------------------------------------------------------------
# Manual entry (expander)
# ---------------------------------------------------------------------------
with st.expander("Manual observation entry"):
    st.markdown("Enter a single observation manually.")
    manual_cols = st.columns(study.n_factors + 1)
    manual_x = []
    for i, col in enumerate(manual_cols[:-1]):
        with col:
            val = st.number_input(study.factor_names[i], key=f"manual_x_{i}")
            manual_x.append(val)
    with manual_cols[-1]:
        manual_y = st.number_input("Response", key="manual_y")

    if st.button("Add observation", key="manual_add"):
        X_new = np.array([manual_x])
        y_new = np.array([manual_y])
        study.add_observations(X_new, y_new)
        invalidate_model_caches()
        auto_save()
        st.success(f"Added 1 observation. Total: {study.n_observations}")
        st.rerun()

# ---------------------------------------------------------------------------
# Current data preview
# ---------------------------------------------------------------------------
if study.n_observations > 0:
    st.markdown("---")
    st.subheader("Current Data")
    df_all = pd.DataFrame(study.X, columns=study.factor_names)
    df_all["Response"] = study.y
    df_all.index = np.arange(1, len(df_all) + 1)
    df_all.index.name = "Obs"
    st.dataframe(df_all, use_container_width=True)
    st.markdown(f"**{study.n_observations} total observations**")
