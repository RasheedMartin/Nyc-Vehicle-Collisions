"""pages/5_severity_predictor.py — Interactive XGBoost severity predictor.

Calls the FastAPI inference service instead of loading the model locally.
Set INFERENCE_API_URL in your Streamlit environment variables.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os
import requests as http_requests
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from utils import load_data
from theme import SEVERITY_COLORS, CHART_BG, GRID_COLOR, TEXT, SUBTEXT
from dotenv import load_dotenv


load_dotenv()
# ── API config ────────────────────────────────────────────────────────────────

API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

TIMEOUT = 10  # seconds

LAYOUT = dict(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_BG,
    font=dict(family="DM Sans", color=TEXT, size=11),
)
XAXIS = dict(
    gridcolor=GRID_COLOR, tickfont=dict(family="DM Mono", size=10), color=SUBTEXT
)
YAXIS = dict(
    gridcolor=GRID_COLOR, tickfont=dict(family="DM Mono", size=10), color=SUBTEXT
)


# ── Cached API calls ──────────────────────────────────────────────────────────


@st.cache_data(ttl=3600)  # refresh metadata once per hour
def fetch_meta() -> dict | None:
    """Pull feature catalogue + train metrics from the API."""
    try:
        resp = http_requests.get(f"{API_URL}/meta", headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Could not reach inference API at `{API_URL}`: {e}", icon="🚨")
        return None


def call_predict(payload: dict) -> dict | None:
    """POST to /predict and return the response dict, or None on error."""
    try:
        resp = http_requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except http_requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        st.error(f"API error: {detail}", icon="🚨")
    except Exception as e:
        st.error(f"Request failed: {e}", icon="🚨")
    return None


# ── Page ──────────────────────────────────────────────────────────────────────

_year_list = load_data()["year"].drop_nulls().cast(pl.Int32).unique().sort().to_list()
year_label = (
    f"{_year_list[0]} – {_year_list[-1]}"
    if len(_year_list) > 1
    else str(_year_list[0])
    if _year_list
    else ""
)

st.markdown(
    f"""
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
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="font-size:1rem;color:#e8e3f0;line-height:1.75;margin-bottom:0.5rem;">
  Given a street and time in Queens — say Jamaica Ave on a winter weekday at noon —
  how severe are crashes likely to be there? Select a location and time below to get
  a severity probability distribution based on historical Queens crash patterns.
</div>
""",
    unsafe_allow_html=True,
)
st.caption(
    "XGBoost model trained on Queens collision data · one row per crash · location + time features only"
)

# ── Load metadata from API ────────────────────────────────────────────────────

api_meta = fetch_meta()
if api_meta is None:
    st.stop()

train_meta = {
    "test_metrics": api_meta["test_metrics"],
    "cv_f1_mean": api_meta.get("cv_f1_mean"),
    "cv_f1_std": api_meta.get("cv_f1_std"),
    "cv_severe_auc_mean": api_meta.get("cv_severe_auc_mean"),
    "cv_severe_auc_std": api_meta.get("cv_severe_auc_std"),
    "base_rates": api_meta.get("base_rates") or {},
    "model_version": api_meta.get("model_version", 1),
    "n_train": api_meta["n_train"],
    "n_test": api_meta["n_test"],
    "trained_at": api_meta["trained_at"],
    "feature_importances": api_meta["feature_importances"],
    "borough": "QUEENS",
}
catalogue = api_meta["category_catalogue"]
severity_order = api_meta["severity_order"]

# ── Model summary ─────────────────────────────────────────────────────────────

IS_V2 = train_meta["model_version"] >= 2
m = train_meta["test_metrics"]

if IS_V2:
    base = train_meta["base_rates"]
    base_severe = base.get("Major Injury", 0) + base.get("Fatal", 0)
    with st.expander("Model Performance Summary"):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Severe-crash AUC", f"{m['severe_auc']:.3f}")
        mc2.metric("Fatal AUC", f"{m['fatal_auc']:.3f}" if m.get("fatal_auc") else "—")
        mc3.metric("Top-10% risk lift", f"{m['top_decile_lift']:.2f}×")
        mc4.metric("Log-loss skill vs base rate", f"{m['log_loss_skill']:+.1%}")
        st.caption(
            f"Trained on {train_meta['n_train']:,} crashes · tested on {train_meta['n_test']:,} "
            f"held-out crashes with the real class mix · CV severe AUC "
            f"{train_meta['cv_severe_auc_mean']:.3f} ± {train_meta['cv_severe_auc_std']:.3f}"
        )
        cal = m.get("calibration", {})
        if cal:
            st.caption(
                "Calibration (mean predicted vs observed): "
                + " · ".join(
                    f"{k} {v['predicted']:.2%} / {v['observed']:.2%}"
                    for k, v in cal.items()
                )
            )

    st.markdown(
        f"""
<div style="border-left:3px solid #7c3aed;padding:1.2rem 1.5rem;
            background:#0f0a1a;border-radius:0 6px 6px 0;margin:1rem 0;">
  <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
              text-transform:uppercase;color:#7c3aed;margin-bottom:0.8rem;">
    How to Read This Model
  </div>
  <div style="font-size:0.95rem;color:#e8e3f0;line-height:1.8;">
  <strong style="color:#a855f7;">Probabilities are calibrated.</strong>
  Across held-out crashes the model's average predicted probabilities match observed
  rates: about {base.get("Major Injury", 0):.1%} of Queens crashes are major-injury and
  {base.get("Fatal", 0):.2%} are fatal. A prediction of "4% severe" means roughly 4 in 100
  crashes at that street and time are major-injury or fatal.
  <br/><br/>
  <strong style="color:#a855f7;">Read it as relative risk, not a forecast.</strong>
  Most crashes anywhere are no-injury, so the single most likely outcome is almost always
  "No Injury". The useful signal is how a street and time compare with the Queens average.
  The 10% of street/time combinations the model rates riskiest have
  <strong>{m["top_decile_lift"]:.1f}×</strong> the borough's severe-crash rate
  ({m["top_decile_severe_rate"]:.1%} vs {m["overall_severe_rate"]:.1%}).
  <br/><br/>
  <strong style="color:#a855f7;">Location and time are weak severity signals.</strong>
  A severe-crash AUC of {m["severe_auc"]:.2f} means the model ranks risk better than chance but
  far from perfectly. The same corridor at the same hour can produce a fender-bender or a
  fatality depending on speed, vehicle type and angle of impact, and none of those are features here.
  <br/><br/>
  <strong style="color:#a855f7;">What would improve this model.</strong>
  NYC Open Data's companion
  <a href="https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Vehicles/bm4k-52h4"
     style="color:#a855f7;" target="_blank">Motor Vehicle Collisions – Vehicles</a>
  table has <strong>vehicle type</strong>, <strong>point of impact</strong> and
  <strong>pre-crash action</strong>, the strongest available proxies for severity.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

if not IS_V2:
    with st.expander("Model Performance Summary"):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy", f"{m['accuracy']:.1%}")
        mc2.metric("Precision", f"{m['precision']:.1%}")
        mc3.metric("Recall", f"{m['recall']:.1%}")
        mc4.metric("F1 Score", f"{m['f1']:.1%}")
        st.caption(
            f"Trained on {train_meta['n_train']:,} rows · "
            f"Tested on {train_meta['n_test']:,} rows · "
            f"CV F1: {train_meta['cv_f1_mean']:.3f} ± {train_meta['cv_f1_std']:.3f} · "
            f"Borough: {train_meta['borough']}"
        )

    cv_f1 = train_meta["cv_f1_mean"]
    test_f1 = train_meta["test_metrics"]["f1"]
    gap = cv_f1 - test_f1

    st.markdown(
        f"""
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
      to crash level. A model that predicts "No Injury" every time achieves ~{m["accuracy"]:.0%} accuracy,
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
      table contains <strong>vehicle type</strong>, <strong>point of impact</strong>,
      and <strong>pre-crash action</strong> — the strongest available proxies for severity.
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.divider()
st.subheader("Try It Yourself")
st.markdown("#### Crash Conditions")

col_l, col_r = st.columns(2, gap="large")
input_values = {}

with col_l:
    st.write("**Location & Time**")
    if "on_street_name" in catalogue:
        default_street = "JAMAICA AVENUE"
        streets = catalogue["on_street_name"]
        street_idx = streets.index(default_street) if default_street in streets else 0
        input_values["on_street_name"] = st.selectbox(
            "Street", streets, index=street_idx
        )
    if "time_of_day" in catalogue:
        input_values["time_of_day"] = st.selectbox(
            "Time of Day", catalogue["time_of_day"], index=2
        )
    if "season" in catalogue:
        input_values["season"] = st.selectbox("Season", catalogue["season"], index=0)

with col_r:
    st.write("**Day Details**")
    input_values["is_weekend"] = int(st.checkbox("Weekend"))
    input_values["is_rush_hour"] = int(st.checkbox("Rush hour (7–10am or 4–8pm)"))
    input_values["month"] = st.slider("Month", 1, 12, 6)
    input_values["day_of_week"] = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

st.divider()

if st.button("Predict Severity →"):
    with st.spinner("Calling inference API…"):
        result = call_predict(input_values)

    if result:
        predicted_label = result["predicted_label"]
        confidence = result["confidence"]
        proba = [p["probability"] for p in result["probabilities"]]
        labels = [p["label"] for p in result["probabilities"]]

        predicted_color = SEVERITY_COLORS.get(predicted_label, "#ffffff")
        tint = {
            "No Injury": "#0a1a0f",
            "Minor Injury": "#1a1500",
            "Major Injury": "#1a0e00",
            "Fatal": "#1a0808",
        }

        rr = result.get("severe_relative_risk")
        if rr is not None:
            sp = result["severe_probability"]
            if rr >= 1.25:
                rr_color, rr_word = "#ef4444", "above"
            elif rr <= 0.8:
                rr_color, rr_word = "#4ade80", "below"
            else:
                rr_color, rr_word = "#facc15", "close to"
            st.markdown(
                f"""
        <div style="margin:1rem 0 1.5rem 0;padding:2rem;background:#13131f;
                    border:1px solid {rr_color};border-left:4px solid {rr_color};border-radius:6px;">
          <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                      text-transform:uppercase;color:{rr_color};margin-bottom:0.4rem;">
            Severe-crash risk (major injury or fatal)
          </div>
          <div style="font-family:'Bebas Neue',sans-serif;font-size:3.2rem;
                      color:{rr_color};line-height:1;letter-spacing:0.03em;">
            {rr:.2f}× Queens average
          </div>
          <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#6b6880;margin-top:0.5rem;">
            {sp:.1%} of crashes here are expected to be severe, vs {base_severe:.1%} borough-wide
            · {rr_word} average
          </div>
        </div>
        """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div style="margin:1rem 0 1.5rem 0;padding:2rem;background:{tint.get(predicted_label, "#13131f")};
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
                Confidence: {confidence:.1%}
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.subheader("Probability by Severity Class")
        fig_p = go.Figure(
            go.Bar(
                x=labels,
                y=[p * 100 for p in proba],
                marker_color=[SEVERITY_COLORS.get(s, "#555") for s in labels],
                text=[f"{p:.1%}" for p in proba],
                textposition="outside",
                textfont=dict(family="DM Mono", size=11, color=TEXT),
                hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
            )
        )
        if train_meta["base_rates"]:
            fig_p.add_trace(
                go.Scatter(
                    x=labels,
                    y=[train_meta["base_rates"].get(lb, 0) * 100 for lb in labels],
                    mode="markers",
                    marker=dict(
                        symbol="line-ew-open", size=40, color=TEXT, line=dict(width=2)
                    ),
                    name="Queens average",
                    hovertemplate="<b>%{x}</b><br>Queens average %{y:.2f}%<extra></extra>",
                )
            )
            fig_p.update_layout(
                showlegend=True, legend=dict(orientation="h", y=1.12, x=0)
            )
        fig_p.update_layout(
            LAYOUT,
            height=280,
            margin=dict(t=20, b=20, l=40, r=20),
            xaxis=XAXIS,
            yaxis=dict(**YAXIS, title="Probability (%)", range=[0, 108]),
        )
        st.plotly_chart(fig_p, width="stretch")

        st.subheader("What Drives This Prediction")
        top_imp = sorted(
            train_meta["feature_importances"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        fig_i = go.Figure(
            go.Bar(
                x=[v for _, v in top_imp],
                y=[f.replace("_", " ") for f, _ in top_imp],
                orientation="h",
                marker=dict(
                    color=[v for _, v in top_imp],
                    colorscale=[[0, "#1f1f30"], [1, "#7c3aed"]],
                    showscale=False,
                ),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            )
        )
        fig_i.update_layout(
            LAYOUT,
            height=300,
            margin=dict(t=10, b=20, l=230, r=20),
            xaxis=XAXIS,
            yaxis=dict(**YAXIS, autorange="reversed"),
        )
        st.plotly_chart(fig_i, width="stretch")

        st.caption(
            "MODEL DISCLAIMER · Predictions are probabilistic estimates based on historical Queens "
            "collision patterns. Fatal crashes are ~0.1% of the data, so fatal estimates are the least certain. "
            "Intended for policy analysis, not individual risk assessment."
        )
