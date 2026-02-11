"""Page 6 — Interactive contour and 3D surface plots."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
import streamlit as st

from jaxsr.app.components import study_status_sidebar
from jaxsr.app.state import get_study, require_fitted, require_study

# ---------------------------------------------------------------------------
study_status_sidebar(get_study())
study = require_study()
require_fitted(study)

model = study.model
n_factors = study.n_factors

if n_factors < 2:
    st.warning("Surface plots require at least 2 continuous factors.")
    st.stop()

# ---------------------------------------------------------------------------
st.header("Response Surface")

# Factor selection
col1, col2 = st.columns(2)
with col1:
    x_factor = st.selectbox("X axis", study.factor_names, index=0)
with col2:
    remaining = [f for f in study.factor_names if f != x_factor]
    y_factor = st.selectbox("Y axis", remaining, index=0)

x_idx = study.factor_names.index(x_factor)
y_idx = study.factor_names.index(y_factor)

# Sliders for held-fixed factors
held_fixed = {}
other_indices = [i for i in range(n_factors) if i not in (x_idx, y_idx)]

if other_indices:
    st.subheader("Held-fixed factor values")
    cols = st.columns(min(len(other_indices), 3))
    for ci, idx in enumerate(other_indices):
        with cols[ci % len(cols)]:
            lo, hi = study.bounds[idx]
            mid = (lo + hi) / 2.0
            val = st.slider(
                study.factor_names[idx],
                float(lo),
                float(hi),
                float(mid),
                key=f"fixed_{idx}",
            )
            held_fixed[idx] = val

# Grid resolution
n_grid = st.slider("Grid resolution", 20, 100, 50, 5)

# ---------------------------------------------------------------------------
# Build meshgrid
# ---------------------------------------------------------------------------
x_lo, x_hi = study.bounds[x_idx]
y_lo, y_hi = study.bounds[y_idx]

x_vals = np.linspace(x_lo, x_hi, n_grid)
y_vals = np.linspace(y_lo, y_hi, n_grid)
xx, yy = np.meshgrid(x_vals, y_vals)

# Construct full input matrix
X_grid = np.zeros((n_grid * n_grid, n_factors))
X_grid[:, x_idx] = xx.ravel()
X_grid[:, y_idx] = yy.ravel()
for idx, val in held_fixed.items():
    X_grid[:, idx] = val

# Predict
z_pred = np.array(model.predict(jnp.array(X_grid)))
zz = z_pred.reshape(n_grid, n_grid)

# ---------------------------------------------------------------------------
# Plotly surface + contour
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go

    tab_contour, tab_3d = st.tabs(["Contour", "3D Surface"])

    with tab_contour:
        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                x=x_vals,
                y=y_vals,
                z=zz,
                colorscale="Viridis",
                contours={"showlabels": True},
            )
        )

        # Overlay design points if available
        if study.X is not None and study.n_observations > 0:
            fig.add_trace(
                go.Scatter(
                    x=study.X[:, x_idx],
                    y=study.X[:, y_idx],
                    mode="markers",
                    marker={"size": 8, "color": "red", "symbol": "x"},
                    name="Design points",
                )
            )

        # Mark stationary point if available
        try:
            from jaxsr.rsm import canonical_analysis

            ca = canonical_analysis(model, bounds=study.bounds)
            if ca.stationary_point is not None:
                sp = ca.stationary_point
                fig.add_trace(
                    go.Scatter(
                        x=[float(sp[x_idx])],
                        y=[float(sp[y_idx])],
                        mode="markers",
                        marker={"size": 14, "color": "gold", "symbol": "star"},
                        name=f"Stationary ({ca.nature})",
                    )
                )
        except (ValueError, RuntimeError, ImportError):
            pass

        fig.update_layout(
            xaxis_title=x_factor,
            yaxis_title=y_factor,
            title=f"Response Surface: {y_factor} vs {x_factor}",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_3d:
        fig3d = go.Figure(
            data=[
                go.Surface(
                    x=x_vals,
                    y=y_vals,
                    z=zz,
                    colorscale="Viridis",
                )
            ]
        )
        fig3d.update_layout(
            scene={
                "xaxis_title": x_factor,
                "yaxis_title": y_factor,
                "zaxis_title": "Response",
            },
            title=f"3D Surface: Response vs {x_factor}, {y_factor}",
            height=700,
        )
        st.plotly_chart(fig3d, use_container_width=True)

except ImportError:
    # Matplotlib fallback
    st.info("Install `plotly` for interactive plots. Falling back to matplotlib.")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Contour
    cs = ax1.contourf(xx, yy, zz, levels=20, cmap="viridis")
    plt.colorbar(cs, ax=ax1)
    if study.X is not None:
        ax1.scatter(study.X[:, x_idx], study.X[:, y_idx], c="red", marker="x", s=50)
    ax1.set_xlabel(x_factor)
    ax1.set_ylabel(y_factor)
    ax1.set_title("Contour Plot")

    # 3D
    ax2.remove()
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(xx, yy, zz, cmap="viridis", alpha=0.8)
    ax2.set_xlabel(x_factor)
    ax2.set_ylabel(y_factor)
    ax2.set_zlabel("Response")
    ax2.set_title("3D Surface")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
