"""pages/4_severity_predictor.py — Interactive XGBoost severity predictor."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from main import load_data, load_model, load_preprocessor, load_feature_meta, load_train_meta
from theme import SEVERITY_COLORS, SEVERITY_ORDER, CHART_BG, GRID_COLOR, TEXT, SUBTEXT, ACCENT

model        = load_model()
preprocessor = load_preprocessor()
feature_meta = load_feature_meta()
train_meta   = load_train_meta()

catalogue = feature_meta["category_catalogue"]
nom_feats = feature_meta["nominal_features"]
ord_feats = feature_meta["ordered_features"]
num_feats = feature_meta["numeric_features"]

LAYOUT = dict(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
              font=dict(family="DM Sans", color=TEXT, size=11))
XAXIS  = dict(gridcolor=GRID_COLOR, tickfont=dict(family="DM Mono", size=10), color=SUBTEXT)
YAXIS  = dict(gridcolor=GRID_COLOR, tickfont=dict(family="DM Mono", size=10), color=SUBTEXT)

# HTML: Bebas Neue header only
st.markdown("""
<div style="padding:2rem 0 1rem 0;">
  <div style="font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.2em;
              text-transform:uppercase;color:#6b6880;">04 · Severity Predictor</div>
  <h1 style="font-size:clamp(2rem,5vw,3.5rem);margin:0.3rem 0 0 0;color:#f0ecff;">
    Predict Crash Severity
  </h1>
</div>
""", unsafe_allow_html=True)

st.caption("Simulate crash conditions and see what the XGBoost model predicts — "
           "trained exclusively on Queens collision data.")

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

st.divider()
st.subheader("Crash Conditions")

col_l, col_r = st.columns(2, gap="large")
input_values = {}

with col_l:
    st.write("**Incident Details**")
    for feat in ["contributing_factor_vehicle_1", "contributing_factor_vehicle_2"]:
        if feat in catalogue:
            label = "Primary Factor" if "1" in feat else "Secondary Factor"
            default = "Driver Inattention/Distraction"
            idx = catalogue[feat].index(default) if default in catalogue[feat] else 0
            input_values[feat] = st.selectbox(label, catalogue[feat], index=idx)
    if "time_of_day" in catalogue:
        input_values["time_of_day"] = st.selectbox("Time of Day", catalogue["time_of_day"], index=2)
    if "season" in catalogue:
        input_values["season"] = st.selectbox("Season", catalogue["season"], index=0)
    input_values["is_weekend"]   = int(st.checkbox("Weekend crash"))
    input_values["is_rush_hour"] = int(st.checkbox("Rush hour (7–10am or 4–8pm)"))

with col_r:
    st.write("**Person Details**")
    for feat, default in [("person_type","Occupant"), ("person_sex",None),
                          ("ejection","Not Ejected"), ("bodily_injury",None)]:
        if feat in catalogue:
            label = feat.replace("_"," ").title()
            idx = catalogue[feat].index(default) if default and default in catalogue[feat] else 0
            input_values[feat] = st.selectbox(label, catalogue[feat], index=idx)
    if "person_age" in num_feats:
        input_values["person_age"] = st.slider("Person Age", 16, 90, 35)
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