"""Page 4 — Fit a symbolic regression model."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import pandas as pd
import streamlit as st

from jaxsr.app.components import metric_cards, study_status_sidebar
from jaxsr.app.state import (
    auto_save,
    get_study,
    invalidate_model_caches,
    require_observations,
    require_study,
)

# ---------------------------------------------------------------------------
study_status_sidebar(get_study())
study = require_study()
require_observations(study)

# ---------------------------------------------------------------------------
st.header("Fit Model")

# Fitting controls
col1, col2, col3 = st.columns(3)
with col1:
    max_terms = st.number_input("Max terms", min_value=1, max_value=50, value=10, step=1)
with col2:
    strategy = st.selectbox(
        "Strategy",
        ["greedy_forward", "greedy_backward", "exhaustive", "lasso_path"],
        index=0,
    )
with col3:
    criterion = st.selectbox(
        "Information criterion",
        ["aicc", "aic", "bic"],
        index=0,
    )

# Basis configuration
with st.expander("Basis configuration"):
    max_poly = st.number_input("Max polynomial degree", min_value=1, max_value=6, value=2, step=1)
    include_transcendental = st.checkbox("Include transcendental functions", value=False)
    include_ratios = st.checkbox("Include ratio terms", value=False)

    basis_config: dict = {
        "max_poly_degree": max_poly,
        "include_transcendental": include_transcendental,
        "include_ratios": include_ratios,
    }

st.markdown("---")

# Fit button
if st.button("Fit Model", type="primary"):
    with st.spinner("Fitting model..."):
        try:
            invalidate_model_caches()
            model = study.fit(
                max_terms=max_terms,
                strategy=strategy,
                information_criterion=criterion,
                basis_config=basis_config,
            )
            auto_save()
            st.success("Model fitted successfully!")
        except (ValueError, RuntimeError) as e:
            st.error(f"Fitting error: {e}")

# ---------------------------------------------------------------------------
# Display results if fitted
# ---------------------------------------------------------------------------
if study.is_fitted:
    model = study.model
    st.markdown("---")
    st.subheader("Fitted Model")

    # LaTeX equation
    try:
        latex_str = model.to_latex()
        st.latex(latex_str)
    except (ImportError, ValueError):
        st.code(model.expression_, language=None)

    # Metrics
    metrics = model.metrics_
    display_metrics = {}
    for key in ["r2", "mse", "aic", "aicc", "bic"]:
        if key in metrics:
            label = key.upper() if key != "r2" else "R²"
            display_metrics[label] = metrics[key]
    if display_metrics:
        metric_cards(display_metrics)

    # Coefficient table
    st.subheader("Coefficients")
    names = model.selected_features_
    coefs = model.coefficients_
    coef_df = pd.DataFrame({"Term": names, "Coefficient": [float(c) for c in coefs]})
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

    # ANOVA table
    st.subheader("ANOVA")
    try:
        from jaxsr.uncertainty import anova

        result = anova(model)

        # Separate term rows from summary rows for % contribution
        summary_sources = {"Model", "Residual", "Total"}
        term_rows = [r for r in result.rows if r.source not in summary_sources]
        summary_rows = [r for r in result.rows if r.source in summary_sources]

        # Total model SS for percentage calculation
        total_row = next((r for r in summary_rows if r.source == "Total"), None)
        residual_row = next((r for r in summary_rows if r.source == "Residual"), None)
        total_ss = total_row.sum_sq if total_row and total_row.sum_sq > 0 else 1.0

        # Warn if residual MS is near machine precision (F-tests unreliable)
        if residual_row and residual_row.mean_sq < 1e-10:
            st.warning(
                "Residual variance is near machine precision "
                f"(MS = {residual_row.mean_sq:.2e}). "
                "F-test p-values may be unreliable — use **% Contribution** "
                "to judge term importance instead."
            )

        anova_data = []
        for row in term_rows:
            pct = 100.0 * row.sum_sq / total_ss if total_ss > 0 else 0.0
            anova_data.append(
                {
                    "Source": row.source,
                    "DF": row.df,
                    "Sum Sq": f"{row.sum_sq:.4g}",
                    "% Contribution": f"{pct:.2f}%",
                    "Mean Sq": f"{row.mean_sq:.4g}",
                    "F": f"{row.f_value:.4g}" if row.f_value is not None else "—",
                    "p-value": f"{row.p_value:.4g}" if row.p_value is not None else "—",
                }
            )
        # Add summary rows without % contribution
        for row in summary_rows:
            anova_data.append(
                {
                    "Source": row.source,
                    "DF": row.df,
                    "Sum Sq": f"{row.sum_sq:.4g}",
                    "% Contribution": "",
                    "Mean Sq": f"{row.mean_sq:.4g}",
                    "F": f"{row.f_value:.4g}" if row.f_value is not None else "—",
                    "p-value": f"{row.p_value:.4g}" if row.p_value is not None else "—",
                }
            )
        st.dataframe(pd.DataFrame(anova_data), use_container_width=True, hide_index=True)
    except (ValueError, RuntimeError, ImportError) as e:
        st.warning(f"Could not compute ANOVA: {e}")

    # Cross-validation (optional)
    with st.expander("Cross-Validation"):
        cv_folds = st.number_input("CV folds", min_value=2, max_value=20, value=5, step=1)
        if st.button("Run Cross-Validation", key="run_cv"):
            try:
                import jax.numpy as jnp

                from jaxsr.metrics import cross_validate

                cv_result = cross_validate(
                    model,
                    jnp.array(study.X),
                    jnp.array(study.y),
                    cv=cv_folds,
                )
                metric_cards(
                    {
                        "Mean Test Score": cv_result["mean_test_score"],
                        "Std Test Score": cv_result["std_test_score"],
                        "Mean Train Score": cv_result["mean_train_score"],
                    }
                )
            except (ValueError, RuntimeError) as e:
                st.error(f"Cross-validation error: {e}")
