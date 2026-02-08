"""Page 5 — Diagnostic plots: parity, residuals, Q-Q, coefficient intervals."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from jaxsr.app.components import study_status_sidebar
from jaxsr.app.state import get_study, require_fitted, require_study

# ---------------------------------------------------------------------------
study_status_sidebar(get_study())
study = require_study()
require_fitted(study)

model = study.model
X = jnp.array(study.X)
y = jnp.array(study.y)
y_pred = model.predict(X)

# ---------------------------------------------------------------------------
st.header("Diagnostics")

# ---------------------------------------------------------------------------
# Parity plot
# ---------------------------------------------------------------------------
st.subheader("Parity Plot")
try:
    from jaxsr.plotting import plot_parity

    fig_ax = plot_parity(y, y_pred)
    # plot_parity returns an Axes; get its figure
    st.pyplot(fig_ax.figure)
except Exception as e:
    st.warning(f"Could not create parity plot: {e}")

# ---------------------------------------------------------------------------
# Residual plots
# ---------------------------------------------------------------------------
st.subheader("Residual Plots")
try:
    from jaxsr.plotting import plot_residuals

    fig = plot_residuals(model, X, y)
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not create residual plots: {e}")

# ---------------------------------------------------------------------------
# Residuals vs each factor
# ---------------------------------------------------------------------------
st.subheader("Residuals vs Factors")
residuals = np.array(y - y_pred)

n_factors = study.n_factors
cols_per_row = min(n_factors, 3)

for start in range(0, n_factors, cols_per_row):
    fig, axes = plt.subplots(
        1,
        min(cols_per_row, n_factors - start),
        figsize=(5 * min(cols_per_row, n_factors - start), 4),
    )
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for idx, ax in enumerate(axes):
        factor_idx = start + idx
        ax.scatter(np.array(X[:, factor_idx]), residuals, alpha=0.6)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(study.factor_names[factor_idx])
        ax.set_ylabel("Residual")
        ax.set_title(f"Residuals vs {study.factor_names[factor_idx]}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Coefficient intervals
# ---------------------------------------------------------------------------
st.subheader("Coefficient Intervals")
alpha = st.slider("Confidence level (1 − α)", 0.80, 0.99, 0.95, 0.01, key="ci_alpha")
ci_alpha = 1.0 - alpha

try:
    intervals = model.coefficient_intervals(alpha=ci_alpha)
    # intervals: dict[str, (est, lo, hi, se)]
    ci_data = []
    for name, (est, lo, hi, se) in intervals.items():
        ci_data.append(
            {
                "Term": name,
                "Estimate": f"{est:.4g}",
                "Lower": f"{lo:.4g}",
                "Upper": f"{hi:.4g}",
                "SE": f"{se:.4g}",
            }
        )
    st.dataframe(pd.DataFrame(ci_data), use_container_width=True, hide_index=True)

    # Forest plot
    try:
        from jaxsr.plotting import plot_coefficient_intervals

        fig_ax = plot_coefficient_intervals(model, alpha=ci_alpha)
        st.pyplot(fig_ax.figure)
    except Exception:
        # Manual forest plot fallback
        fig, ax = plt.subplots(figsize=(8, max(3, len(intervals) * 0.5)))
        names = list(intervals.keys())
        ests = [intervals[n][0] for n in names]
        los = [intervals[n][1] for n in names]
        his = [intervals[n][2] for n in names]
        y_pos = range(len(names))
        ax.barh(
            y_pos,
            [h - lo for h, lo in zip(his, los, strict=False)],
            left=los,
            height=0.4,
            alpha=0.3,
        )
        ax.scatter(ests, y_pos, color="black", zorder=5)
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names)
        ax.set_xlabel("Coefficient Value")
        ax.set_title(f"{int(alpha * 100)}% Confidence Intervals")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
except Exception as e:
    st.warning(f"Could not compute coefficient intervals: {e}")

# ---------------------------------------------------------------------------
# Prediction intervals
# ---------------------------------------------------------------------------
st.subheader("Prediction Intervals")
try:
    from jaxsr.plotting import plot_prediction_intervals

    fig_ax = plot_prediction_intervals(model, X, y, alpha=ci_alpha)
    st.pyplot(fig_ax.figure)
except Exception as e:
    st.warning(f"Could not create prediction interval plot: {e}")
