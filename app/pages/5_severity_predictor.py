"""pages/4_severity_predictor.py — Interactive XGBoost severity predictor."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import polars as pl
from main import load_data, load_model, load_preprocessor, load_feature_meta, load_train_meta
from theme import SEVERITY_COLORS, SEVERITY_ORDER, CHART_BG, GRID_COLOR, TEXT, SUBTEXT, ACCENT

model        = load_model()
preprocessor = load_preprocessor()
feature_meta = load_feature_meta()
train_meta   = load_train_meta()

_year_list = load_data()["year"].drop_nulls().cast(pl.Int32).unique().sort().to_list()
year_label = f"{_year_list[0]} – {_year_list[-1]}" if len(_year_list) > 1 else str(_year_list[0]) if _year_list else ""

catalogue = feature_meta["category_catalogue"]
nom_feats = feature_meta["nominal_features"]
ord_feats = feature_meta["ordered_features"]
num_feats = feature_meta["numeric_features"]

LAYOUT = dict(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
              font=dict(family="DM Sans", color=TEXT, size=11))
XAXIS  = dict(gridcolor=GRID_COLOR, tickfont=dict(family="DM Mono", size=10), color=SUBTEXT)
YAXIS  = dict(gridcolor=GRID_COLOR, tickfont=dict(family="DM Mono", size=10), color=SUBTEXT)

# HTML: Bebas Neue header only
st.markdown(f"""
<div style="padding:2rem 0 1rem 0;">
  <div style="font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.2em;
              text-transform:uppercase;color:#6b6880;">05 · Severity Predictor</div>
  <h1 style="font-size:clamp(2rem,5vw,3.5rem);margin:0.3rem 0 0 0;color:#f0ecff;">
    Location Risk Predictor
  </h1>
  <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;
              text-transform:uppercase;color:#6b6880;margin-top:0.4rem;">
    {year_label}
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:1rem;color:#e8e3f0;line-height:1.75;margin-bottom:0.5rem;">
  Given a street and time in Queens — say Jamaica Ave on a winter weekday at noon —
  how severe are crashes likely to be there? Select a location and time below to get
  a severity probability distribution based on historical Queens crash patterns.
</div>
""", unsafe_allow_html=True)
st.caption("XGBoost model trained on Queens collision data · one row per crash · location + time features only")

# ── Model summary — native expander + metrics ─────────────────────────────────

with st.expander("Model Performance Summary"):
    m = train_meta["test_metrics"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy",  f"{m['accuracy']:.1%}")
    mc2.metric("Precision", f"{m['precision']:.1%}")
    mc3.metric("Recall",    f"{m['recall']:.1%}")
    mc4.metric("F1 Score",  f"{m['f1']:.1%}")
    st.caption(
        f"Trained on {train_meta['n_train']:,} rows · "
        f"Tested on {train_meta['n_test']:,} rows · "
        f"CV F1: {train_meta['cv_f1_mean']:.3f} ± {train_meta['cv_f1_std']:.3f} · "
        f"Borough: {train_meta['borough']}"
    )

cv_f1   = train_meta["cv_f1_mean"]
test_f1 = train_meta["test_metrics"]["f1"]
gap     = cv_f1 - test_f1

st.markdown(f"""
<div style="border-left:3px solid #7c3aed;padding:1.2rem 1.5rem;
            background:#0f0a1a;border-radius:0 6px 6px 0;margin:1rem 0;">
  <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
              text-transform:uppercase;color:#7c3aed;margin-bottom:0.8rem;">
    What the Numbers Tell Us
  </div>
  <div style="font-size:0.95rem;color:#e8e3f0;line-height:1.8;">
  <strong style="color:#a855f7;">Class imbalance dominates the target.</strong>
  Fatal crashes are rare — they represent less than 0.2% of Queens collisions.
  No-Injury outcomes account for the vast majority of records even after deduplication
  to crash level. A model that predicts "No Injury" every time achieves ~{m['accuracy']:.0%} accuracy,
  making raw accuracy a meaningless metric. SMOTE oversamples the minority classes during training,
  but the test set is unbalanced real data — which is why test F1 ({test_f1:.2f}) is the number
  that matters.
  <br/><br/>
  <strong style="color:#a855f7;">Cross-validation overstates performance.</strong>
  CV F1 is <strong>{cv_f1:.2f}</strong> but test F1 drops to <strong>{test_f1:.2f}</strong>
  — a {gap:.2f}-point gap. SMOTE was applied inside the CV loop, so each fold was trained
  and scored on synthetic minority samples. Real-world crash data is far noisier than
  interpolated synthetic points.
  <br/><br/>
  <strong style="color:#a855f7;">Location and time predict <em>where</em> crashes cluster — not <em>how bad</em> they are.</strong>
  Street name and time of day are good signals for crash frequency and hotspot identification
  (as the Hotspot Map shows). They are weaker predictors of severity because the same
  corridor at the same hour can produce a fender-bender or a fatality depending on
  variables not in the dataset: impact speed, vehicle type, and angle of collision.
  <br/><br/>
  <strong style="color:#a855f7;">What would improve this model.</strong>
  NYC Open Data's companion
  <a href="https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Vehicles/bm4k-52h4"
     style="color:#a855f7;" target="_blank">Motor Vehicle Collisions – Vehicles</a>
  table (joinable via <code style="font-family:'DM Mono',monospace;font-size:0.85em;">collision_id</code>)
  contains <strong>vehicle type</strong> (motorcycle vs. sedan vs. truck — the single strongest
  available proxy for severity), <strong>point of impact</strong> (frontal vs. side),
  and <strong>pre-crash action</strong> (maneuver at time of crash). Intersection-level
  granularity — pairing <code style="font-family:'DM Mono',monospace;font-size:0.85em;">on_street_name</code>
  with <code style="font-family:'DM Mono',monospace;font-size:0.85em;">cross_street_name</code> —
  would also sharpen the location signal beyond street-corridor level.
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()
st.subheader("Try It Yourself")
st.markdown("#### Crash Conditions")

col_l, col_r = st.columns(2, gap="large")
input_values = {}

with col_l:
    st.write("**Location & Time**")
    if "on_street_name" in catalogue:
        default_street = "JAMAICA AVENUE"
        street_idx = catalogue["on_street_name"].index(default_street) if default_street in catalogue["on_street_name"] else 0
        input_values["on_street_name"] = st.selectbox("Street", catalogue["on_street_name"], index=street_idx)
    if "time_of_day" in catalogue:
        input_values["time_of_day"] = st.selectbox("Time of Day", catalogue["time_of_day"], index=2)
    if "season" in catalogue:
        input_values["season"] = st.selectbox("Season", catalogue["season"], index=0)

with col_r:
    st.write("**Day Details**")
    input_values["is_weekend"]   = int(st.checkbox("Weekend"))
    input_values["is_rush_hour"] = int(st.checkbox("Rush hour (7–10am or 4–8pm)"))
    if "month" in num_feats:
        input_values["month"] = st.slider("Month", 1, 12, 6)
    if "day_of_week" in num_feats:
        input_values["day_of_week"] = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

st.divider()

if st.button("Predict Severity →"):
    all_cols = nom_feats + ord_feats + num_feats
    row = {c: [input_values.get(c, 0 if c in num_feats else "Unknown")] for c in all_cols}
    X = preprocessor.transform(pd.DataFrame(row)).astype(np.float32)
    proba = model.predict_proba(X)[0]

    predicted_idx   = int(np.argmax(proba))
    predicted_label = SEVERITY_ORDER[predicted_idx]
    predicted_color = SEVERITY_COLORS[predicted_label]
    tint = {"No Injury":"#0a1a0f","Minor Injury":"#1a1500",
            "Major Injury":"#1a0e00","Fatal":"#1a0808"}

    # HTML: coloured border + Bebas Neue large result — can't replicate with st.metric
    st.markdown(f"""
    <div style="margin:1rem 0 1.5rem 0;padding:2rem;background:{tint[predicted_label]};
                border:1px solid {predicted_color};border-radius:6px;
                border-left:4px solid {predicted_color};">
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                  text-transform:uppercase;color:{predicted_color};margin-bottom:0.4rem;">
        Predicted Outcome
      </div>
      <div style="font-family:'Bebas Neue',sans-serif;font-size:3.2rem;
                  color:{predicted_color};line-height:1;letter-spacing:0.03em;">
        {predicted_label}
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.8rem;
                  color:#6b6880;margin-top:0.5rem;">
        Confidence: {proba[predicted_idx]:.1%}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Probability by Severity Class")
    fig_p = go.Figure(go.Bar(
        x=SEVERITY_ORDER, y=[p*100 for p in proba],
        marker_color=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER],
        text=[f"{p:.1%}" for p in proba],
        textposition="outside",
        textfont=dict(family="DM Mono", size=11, color=TEXT),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
    ))
    fig_p.update_layout(LAYOUT, height=280, margin=dict(t=20,b=20,l=40,r=20),
                        xaxis=XAXIS, yaxis=dict(**YAXIS, title="Probability (%)", range=[0,108]))
    st.plotly_chart(fig_p, width="stretch")

    st.subheader("What Drives This Prediction")
    top_imp = sorted(train_meta["feature_importances"].items(),
                     key=lambda x: x[1], reverse=True)[:10]
    fig_i = go.Figure(go.Bar(
        x=[v for _,v in top_imp],
        y=[f.replace("_"," ") for f,_ in top_imp],
        orientation="h",
        marker=dict(color=[v for _,v in top_imp],
                    colorscale=[[0,"#1f1f30"],[1,"#7c3aed"]], showscale=False),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig_i.update_layout(LAYOUT, height=300, margin=dict(t=10,b=20,l=230,r=20),
                        xaxis=XAXIS, yaxis=dict(**YAXIS, autorange="reversed"))
    st.plotly_chart(fig_i, width="stretch")

    st.caption(
        "MODEL DISCLAIMER · Predictions are probabilistic estimates based on historical Queens "
        "collision patterns. Fatal class performance is limited by low sample count (~48 training cases). "
        "Intended for policy analysis, not individual risk assessment."
    )