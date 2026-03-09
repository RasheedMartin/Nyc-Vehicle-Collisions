"""pages/6_pipeline.py — Automated MLOps pipeline overview."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import polars as pl
from datetime import datetime
from utils import load_data, load_train_meta

train_meta = load_train_meta()
_year_list = load_data()["year"].drop_nulls().cast(pl.Int32).unique().sort().to_list()
year_label = (
    f"{_year_list[0]} – {_year_list[-1]}"
    if len(_year_list) > 1
    else str(_year_list[0])
    if _year_list
    else ""
)

try:
    trained_at = datetime.fromisoformat(train_meta["trained_at"]).strftime(
        "%b %d, %Y  %I:%M %p"
    )
except Exception:
    trained_at = train_meta.get("trained_at", "—")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
<div style="padding:2rem 0 1rem 0;">
  <div style="font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.2em;
              text-transform:uppercase;color:#6b6880;">06 · Data Pipeline</div>
  <h1 style="font-size:clamp(2rem,5vw,3.5rem);margin:0.3rem 0 0 0;color:#f0ecff;">
    Automated MLOps Pipeline
  </h1>
  <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;
              text-transform:uppercase;color:#6b6880;margin-top:0.4rem;">
    {year_label}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="font-size:1rem;color:#e8e3f0;line-height:1.75;margin-bottom:1.5rem;">
  Every Monday at 6am, an Apache Airflow DAG automatically pulls new crash records from NYC Open Data,
  retrains the XGBoost model, uploads the artifacts to Cloudflare R2, and triggers a Railway redeploy
  via GitHub Actions — keeping the dashboard current without any manual intervention.
</div>
""",
    unsafe_allow_html=True,
)

# ── Pipeline flow ─────────────────────────────────────────────────────────────

STAGES = [
    ("NYC Open Data", "Socrata API", "#4f46e5"),
    ("Fetch", "Incremental pull", "#6d28d9"),
    ("Clean", "Merge + engineer", "#6d28d9"),
    ("Features", "Encode + fit", "#6d28d9"),
    ("Train", "XGBoost + SMOTE", "#6d28d9"),
    ("R2 Upload", "Cloudflare R2", "#7c3aed"),
    ("API Reload", "Hot-swap model", "#8b5cf6"),
    ("GitHub Actions", "workflow_dispatch", "#9333ea"),
    ("Railway", "Live redeploy", "#a855f7"),
]

parts = []
for i, (title, subtitle, color) in enumerate(STAGES):
    parts.append(
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'min-width:80px;flex-shrink:0;">'
        '<div style="background:#0d0720;border:1px solid '
        + color
        + ";border-radius:6px;"
        'padding:0.6rem 0.75rem;text-align:center;width:100%;">'
        "<div style=\"font-family:'DM Mono',monospace;font-size:0.72rem;font-weight:600;"
        "color:" + color + ';letter-spacing:0.05em;">' + title + "</div>"
        "<div style=\"font-family:'DM Mono',monospace;font-size:0.58rem;"
        'color:#6b6880;margin-top:0.25rem;">' + subtitle + "</div>"
        "</div></div>"
    )
    if i < len(STAGES) - 1:
        parts.append(
            '<div style="color:#4c1d95;font-size:1rem;padding:0 0.25rem;'
            'align-self:center;flex-shrink:0;">&#9654;</div>'
        )

boxes = "".join(parts)

st.markdown(
    '<div style="background:#080414;border:1px solid #1f1535;border-radius:8px;'
    'padding:1.5rem 1.5rem 1rem 1.5rem;margin:0.5rem 0 1.5rem 0;">'
    "<div style=\"font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.15em;"
    'text-transform:uppercase;color:#7c3aed;margin-bottom:1rem;">Pipeline</div>'
    '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:0.4rem;overflow-x:auto;'
    'padding-bottom:0.5rem;">' + boxes + "</div>"
    "<div style=\"font-family:'DM Mono',monospace;font-size:0.6rem;color:#4c1d95;margin-top:1rem;\">"
    "&#8635; &nbsp;Scheduled every Monday at 6 am &middot; Orchestrated by Apache Airflow (Docker)"
    "</div></div>",
    unsafe_allow_html=True,
)

st.divider()

# ── Last run + infrastructure ─────────────────────────────────────────────────

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown(
        """
<div style="font-family:'DM Mono',monospace;font-size:0.63rem;letter-spacing:0.15em;
            text-transform:uppercase;color:#7c3aed;margin-bottom:0.75rem;">Last Model Run</div>
""",
        unsafe_allow_html=True,
    )

    st.metric("Retrained", trained_at)
    st.metric("Training rows", f"{train_meta['n_train']:,}")
    st.metric("Test rows", f"{train_meta['n_test']:,}")
    st.metric(
        "CV F1 (macro)",
        f"{train_meta['cv_f1_mean']:.3f} ± {train_meta['cv_f1_std']:.3f}",
    )
    st.metric("Test F1 (macro)", f"{train_meta['test_metrics']['f1']:.3f}")

with col_r:
    st.markdown(
        """
<div style="font-family:'DM Mono',monospace;font-size:0.63rem;letter-spacing:0.15em;
            text-transform:uppercase;color:#7c3aed;margin-bottom:0.75rem;">Infrastructure</div>
""",
        unsafe_allow_html=True,
    )

    rows = [
        ("Orchestration", "Apache Airflow (Docker)"),
        ("Schedule", "Every Monday, 6 am"),
        ("Artifact store", "Cloudflare R2"),
        ("CI/CD", "GitHub Actions"),
        ("Hosting", "Railway"),
        ("Data source", "NYC Open Data (Socrata)"),
        ("Model", "XGBoost multiclass"),
        ("Resampling", "SMOTE (train split only)"),
        ("Features", f"{train_meta['n_features']} encoded columns"),
    ]
    table_rows = "".join(
        f"<tr>"
        f'<td style="color:#6b6880;padding:0.28rem 0.75rem 0.28rem 0;'
        f"font-family:'DM Mono',monospace;font-size:0.78rem;white-space:nowrap;\">{k}</td>"
        f'<td style="color:#e8e3f0;padding:0.28rem 0;'
        f"font-family:'DM Mono',monospace;font-size:0.78rem;\">{v}</td>"
        f"</tr>"
        for k, v in rows
    )
    st.markdown(
        f"""
<div style="background:#0d0720;border:1px solid #1f1535;border-radius:6px;padding:1rem 1.2rem;">
  <table style="width:100%;border-collapse:collapse;">{table_rows}</table>
</div>
""",
        unsafe_allow_html=True,
    )

st.divider()

# ── XGBoost params ────────────────────────────────────────────────────────────

with st.expander("XGBoost Hyperparameters"):
    params = train_meta.get("xgboost_params", {})
    param_rows = "".join(
        f"<tr>"
        f'<td style="color:#6b6880;padding:0.25rem 1rem 0.25rem 0;'
        f"font-family:'DM Mono',monospace;font-size:0.78rem;\">{k}</td>"
        f'<td style="color:#a855f7;padding:0.25rem 0;'
        f"font-family:'DM Mono',monospace;font-size:0.78rem;\">{v}</td>"
        f"</tr>"
        for k, v in params.items()
    )
    st.markdown(
        f"""
<table style="border-collapse:collapse;">{param_rows}</table>
""",
        unsafe_allow_html=True,
    )
