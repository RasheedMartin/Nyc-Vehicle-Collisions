"""pages/3_contributing_factors.py — Why crashes happen."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import polars as pl
import plotly.graph_objects as go
from main import (
    load_data
)

from theme import  SEVERITY_COLORS, SEVERITY_ORDER, CHART_BG, GRID_COLOR, TEXT, SUBTEXT, ACCENT

df = load_data()

_year_list = df["year"].drop_nulls().cast(pl.Int32).unique().sort().to_list()
year_label = f"{_year_list[0]} – {_year_list[-1]}" if len(_year_list) > 1 else str(_year_list[0]) if _year_list else ""

LAYOUT = dict(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
              font=dict(family="DM Sans", color=TEXT, size=11))
XAXIS  = dict(gridcolor=GRID_COLOR, color=SUBTEXT)
YAXIS  = dict(gridcolor=GRID_COLOR, tickfont=dict(family="DM Mono", size=10), color=SUBTEXT)
LEGEND = dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)")

st.markdown(f"""
<div style="padding:2rem 0 1.5rem 0;">
  <div style="font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.2em;
              text-transform:uppercase;color:#6b6880;">03 · Contributing Factors</div>
  <h1 style="font-size:clamp(2rem,5vw,3.5rem);margin:0.3rem 0 0 0;color:#f0ecff;">
    Why Crashes Happen
  </h1>
  <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;
              text-transform:uppercase;color:#6b6880;margin-top:0.4rem;">
    {year_label}
  </div>
</div>
""", unsafe_allow_html=True)

selected_severity = st.multiselect("Filter by Severity", SEVERITY_ORDER, default=SEVERITY_ORDER)
filtered = df.filter(pl.col("accident_severity").is_in(selected_severity))
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Top 15 factors ────────────────────────────────────────────────────────────

st.markdown("#### Top Contributing Factors")
if "contributing_factor_vehicle_1" in df.columns:
    factors = (
        filtered
        .filter(pl.col("contributing_factor_vehicle_1").is_not_null() &
                (pl.col("contributing_factor_vehicle_1") != "Unspecified"))
        .group_by("contributing_factor_vehicle_1").len()
        .sort("len", descending=True).head(15)
    ).to_pandas()

    fig = go.Figure(go.Bar(
        x=factors["len"], y=factors["contributing_factor_vehicle_1"],
        orientation="h",
        marker=dict(color=factors["len"],
                    colorscale=[[0,"#1f1f30"],[0.5,"#7c3aed"],[1,"#ef4444"]],
                    showscale=False),
        text=factors["len"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        textfont=dict(family="DM Mono", size=10, color=SUBTEXT),
        hovertemplate="<b>%{y}</b><br>%{x:,} crashes<extra></extra>",
    ))
    xaxis = dict(**XAXIS, tickfont=dict(family="DM Mono", size=10))
    fig.update_layout(LAYOUT, height=420, margin=dict(t=10,b=20,l=260,r=80),
                      xaxis=xaxis, yaxis=dict(**YAXIS, autorange="reversed"))
    st.plotly_chart(fig, width='stretch')

st.markdown("<hr style='border-color:#1f1f30;margin:1.5rem 0'>", unsafe_allow_html=True)

# ── Fatal factors + severity split ────────────────────────────────────────────

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Top Factors in Fatal Crashes")
    if "contributing_factor_vehicle_1" in df.columns:
        ff = (
            df.filter(pl.col("accident_severity") == "Fatal")
              .filter(pl.col("contributing_factor_vehicle_1").is_not_null() &
                      (pl.col("contributing_factor_vehicle_1") != "Unspecified"))
              .group_by("contributing_factor_vehicle_1").len()
              .sort("len", descending=True).head(8)
        ).to_pandas()
        fig2 = go.Figure(go.Bar(
            x=ff["len"], y=ff["contributing_factor_vehicle_1"],
            orientation="h", marker_color=ACCENT,
            hovertemplate="<b>%{y}</b><br>%{x} fatal crashes<extra></extra>",
        ))
        xaxis = dict(**XAXIS, tickfont=dict(family="DM Mono", size=10))
        fig2.update_layout(LAYOUT, height=320, margin=dict(t=10,b=20,l=220,r=20),
                           xaxis=xaxis, yaxis=dict(**YAXIS, autorange="reversed"))
        st.plotly_chart(fig2, width='stretch')

with col_b:
    st.markdown("#### Severity Split by Top Factor")
    if "contributing_factor_vehicle_1" in df.columns:
        top8 = (
            df.filter(pl.col("contributing_factor_vehicle_1").is_not_null() &
                      (pl.col("contributing_factor_vehicle_1") != "Unspecified"))
              .group_by("contributing_factor_vehicle_1").len()
              .sort("len", descending=True).head(8)["contributing_factor_vehicle_1"].to_list()
        )
        split = (
            df.filter(pl.col("contributing_factor_vehicle_1").is_in(top8))
              .group_by(["contributing_factor_vehicle_1","accident_severity"]).len()
        ).to_pandas()
        fig3 = go.Figure()
        for s in SEVERITY_ORDER:
            sub = split[split["accident_severity"] == s]
            fig3.add_trace(go.Bar(
                x=sub["contributing_factor_vehicle_1"], y=sub["len"],
                name=s, marker_color=SEVERITY_COLORS[s],
                hovertemplate=f"<b>{s}</b><br>%{{x}}<br>%{{y:,}} crashes<extra></extra>",
            ))
        xaxis = dict(**XAXIS, tickangle=-35, tickfont=dict(family="DM Sans",size=9))
    
        fig3.update_layout(LAYOUT, barmode="stack", height=320,
                           margin=dict(t=10,b=100,l=40,r=20),
                           xaxis=xaxis,
                           yaxis=YAXIS, legend=LEGEND)
        st.plotly_chart(fig3, width='stretch')

st.markdown("<hr style='border-color:#1f1f30;margin:1.5rem 0'>", unsafe_allow_html=True)

# ── Person type ───────────────────────────────────────────────────────────────

st.markdown("#### Who Gets Hurt — Victim Type by Severity")
if "person_type" in df.columns:
    pt = (filtered.filter(pl.col("person_type").is_not_null() &
                          (pl.col("person_type") != "Unknown"))
                  .group_by(["person_type","accident_severity"]).len()).to_pandas()
    fig4 = go.Figure()
    for s in SEVERITY_ORDER:
        sub = pt[pt["accident_severity"] == s]
        fig4.add_trace(go.Bar(
            x=sub["person_type"], y=sub["len"], name=s,
            marker_color=SEVERITY_COLORS[s],
            hovertemplate=f"<b>{s}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ))
    xaxis = dict(**XAXIS, tickfont=dict(family="DM Mono", size=10))
    fig4.update_layout(LAYOUT, barmode="group", height=280,
                       margin=dict(t=10,b=40,l=40,r=20),
                       xaxis=xaxis, yaxis=YAXIS, legend=LEGEND)
    st.plotly_chart(fig4, width='stretch')

# ── Policy callouts ───────────────────────────────────────────────────────────

if "contributing_factor_vehicle_1" in df.columns:
    top_factor = (
        df.filter(pl.col("contributing_factor_vehicle_1").is_not_null() &
                  (pl.col("contributing_factor_vehicle_1") != "Unspecified"))
          .group_by("contributing_factor_vehicle_1").len()
          .sort("len", descending=True).head(1)["contributing_factor_vehicle_1"][0]
    )
    fatal_count = len(df.filter(
        (pl.col("contributing_factor_vehicle_1") == top_factor) &
        (pl.col("accident_severity") == "Fatal")
    ))
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"""
        <div style="border-left:3px solid #ef4444;padding:1.2rem 1.5rem;
                    background:#1a0f0f;border-radius:0 6px 6px 0;height:100%;">
          <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                      text-transform:uppercase;color:#ef4444;margin-bottom:0.5rem;">Top Factor</div>
          <div style="font-size:1rem;color:#e8e3f0;line-height:1.75;">
            <strong>{top_factor}</strong> is the leading contributing factor in Queens,
            linked to <strong>{fatal_count}</strong> fatal collisions.
            Automated enforcement directly targets this behaviour.
          </div>
        </div>""", unsafe_allow_html=True)
    with p2:
        # Compute pedestrian/cyclist share of fatal crashes from actual data
        if "person_type" in df.columns:
            vuln_fatal = len(df.filter(
                (pl.col("accident_severity") == "Fatal") &
                (pl.col("person_type").is_in(["Pedestrian", "Bicyclist"]))
            ))
            total_fatal = len(df.filter(pl.col("accident_severity") == "Fatal"))
            vuln_pct = vuln_fatal / total_fatal * 100 if total_fatal > 0 else 0
        else:
            vuln_fatal, total_fatal, vuln_pct = 0, 0, 0

        st.markdown(f"""
        <div style="border-left:3px solid #f97316;padding:1.2rem 1.5rem;
                    background:#1a1000;border-radius:0 6px 6px 0;height:100%;">
          <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                      text-transform:uppercase;color:#f97316;margin-bottom:0.5rem;">Vulnerable Road Users</div>
          <div style="font-size:1rem;color:#e8e3f0;line-height:1.75;">
            In this dataset, <strong>{vuln_pct:.0f}%</strong> of fatal crashes
            ({vuln_fatal} of {total_fatal}) involved a pedestrian or cyclist.
            NYC DOT Vision Zero identifies protected infrastructure —
            pedestrian islands, protected bike lanes, leading pedestrian intervals —
            as the highest-impact intervention for this group.
            <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#6b6880;
                         display:block;margin-top:0.6rem;">
              Source: NYC DOT Vision Zero Year 9 Report (2022) ·
              <a href="https://www.nyc.gov/html/dot/html/pedestrians/visionzero.shtml"
                 style="color:#6b6880;" target="_blank">nyc.gov/dot/visionzero</a>
            </span>
          </div>
        </div>""", unsafe_allow_html=True)