"""
app.py — Queens Vehicle Collision Dashboard
Run with: streamlit run app/app.py
"""

import json
import joblib
import streamlit as st
import polars as pl
from pathlib import Path

st.set_page_config(
    page_title="Queens Collision Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed" / "QUEENS"
MODELS_DIR    = ROOT / "models" / "QUEENS"

DATA_PATH         = PROCESSED_DIR / "collisions_queens.parquet"
FEATURE_META_PATH = PROCESSED_DIR / "feature_meta.json"
TRAIN_META_PATH   = MODELS_DIR    / "train_meta.json"
MODEL_PATH        = MODELS_DIR    / "severity_model.joblib"
PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"

# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pl.DataFrame:
    if not DATA_PATH.exists():
        full_path = ROOT / "data" / "processed" / "collisions.parquet"
        return pl.read_parquet(full_path).filter(pl.col("borough") == "QUEENS")
    return pl.read_parquet(DATA_PATH)

@st.cache_data
def load_feature_meta() -> dict:
    with open(FEATURE_META_PATH) as f:
        return json.load(f)

@st.cache_data
def load_train_meta() -> dict:
    with open(TRAIN_META_PATH) as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)



# ── Landing ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding:3rem 0 2rem 0;">
  <div style="font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:0.2em;
              text-transform:uppercase; color:#6b6880; margin-bottom:0.8rem;">
    NYC · Vehicle Collision Intelligence
  </div>
  <h1 style="font-family:'Bebas Neue',sans-serif; font-size:clamp(3rem,8vw,5.5rem);
             line-height:0.95; margin:0; color:#f0ecff; letter-spacing:0.02em;">
    Queens<br/>
    <span style="color:#ef4444;">Crash</span><br/>
    Report
  </h1>
  <div style="width:60px; height:3px; background:#ef4444; margin:1.5rem 0;"></div>
  <p style="font-family:'DM Sans',sans-serif; font-size:1rem; color:#9d99ae;
            max-width:520px; line-height:1.75; font-weight:300;">
    An evidence-based analysis of motor vehicle collisions in Queens, New York —
    examining where crashes cluster, what causes them, and what policy changes
    could save lives.
  </p>
</div>
""", unsafe_allow_html=True)

cards = [
    ("01", "Overview",           "Borough-wide statistics, severity breakdown, and year-over-year trends."),
    ("02 — 03", "Maps & Factors","WebGL hotspot maps and contributing factor analysis for policy insight."),
    ("04", "Severity Predictor", "XGBoost model trained on Queens data. Simulate conditions and predict crash severity."),
]

cols = st.columns(3)
for col, (num, title, desc) in zip(cols, cards):
    with col:
        with st.container():
            st.markdown(f"""
            <div style="background:#13131f; border:1px solid #1f1f30; padding:1.5rem;
                        border-radius:6px; height:100%;">
            <div style="font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:0.15em;
                 ß       text-transform:uppercase; color:#6b6880; margin-bottom:0.7rem;">{num}</div>
            <div style="font-family:'Bebas Neue',sans-serif; font-size:1.25rem;
                        color:#f0ecff; margin-bottom:0.5rem; letter-spacing:0.04em;">{title}</div>
            <div style="font-size:0.84rem; color:#9d99ae; line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:3rem; font-family:'DM Mono',monospace; font-size:0.62rem;
            letter-spacing:0.1em; color:#2e2e42;">
  DATA SOURCE: NYC OPEN DATA · MOTOR VEHICLE COLLISIONS · NYPD
</div>
""",  unsafe_allow_html=True)