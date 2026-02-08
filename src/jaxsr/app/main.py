"""
JAXSR DOE App — main entry point.

Run via ``jaxsr app`` or ``streamlit run src/jaxsr/app/main.py``.
"""

from __future__ import annotations

import os

# Force JAX to use CPU backend — avoids "default_memory_space" errors
# on macOS Metal / Apple Silicon. Must be set before JAX is imported.
os.environ["JAX_PLATFORMS"] = "cpu"

import streamlit as st

from jaxsr.app.components import study_status_sidebar
from jaxsr.app.state import get_study

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="JAXSR DOE",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
study = get_study()
study_status_sidebar(study)

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
st.title("JAXSR — Design of Experiments")

st.markdown("""
Welcome to the **JAXSR DOE App**, an interactive workflow for designing
experiments, fitting symbolic regression models, and optimizing responses.

### Workflow

1. **Setup** — create or load a study, define factors
2. **Design** — generate an experimental design, download an Excel template
3. **Data** — import experimental results
4. **Fit** — fit a symbolic regression model
5. **Diagnostics** — parity plots, residuals, Q-Q, coefficient intervals
6. **Surface** — interactive contour and 3D surface plots
7. **Optimize** — canonical analysis, suggest next experiments
8. **Export** — LaTeX, JSON, Word/Excel reports

Use the **sidebar** to navigate between pages.
""")

if study is None:
    st.info("Get started by navigating to **Setup** in the sidebar.")
else:
    st.success(f"Study **{study.name}** is loaded with {study.n_observations} observations.")
    if study.is_fitted:
        st.code(study.model.expression_, language=None)
