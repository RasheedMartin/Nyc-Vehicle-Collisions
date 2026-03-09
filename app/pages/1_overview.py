"""pages/1_overview.py — Queens at a glance."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import polars as pl
import plotly.graph_objects as go
from utils import load_data
from theme import SEVERITY_COLORS, SEVERITY_ORDER, CHART_BG, GRID_COLOR, TEXT, SUBTEXT

df = load_data()

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
LEGEND = dict(
    font=dict(family="DM Mono", size=10),
    bgcolor="rgba(0,0,0,0)",
    bordercolor="rgba(0,0,0,0)",
)

_year_list = df["year"].drop_nulls().cast(pl.Int32).unique().sort().to_list()
year_label = (
    f"{_year_list[0]} – {_year_list[-1]}"
    if len(_year_list) > 1
    else str(_year_list[0])
    if _year_list
    else ""
)

# HTML: Bebas Neue font + clamp sizing + monospace label — can't do this with st.header
st.markdown(
    f"""
<div style="padding:2rem 0 1rem 0;">
  <div style="font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.2em;
              text-transform:uppercase;color:#6b6880;">01 · Overview</div>
  <h1 style="font-size:clamp(2rem,5vw,3.5rem);margin:0.3rem 0 0 0;color:#f0ecff;">
    Queens at a Glance
  </h1>
  <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;
              text-transform:uppercase;color:#6b6880;margin-top:0.4rem;">
    {year_label}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── KPIs — st.metric styled by global CSS ────────────────────────────────────

total_crashes = df["collision_id"].n_unique()
total_injured = int(df["number_of_persons_injured"].sum())
total_killed = int(df["number_of_persons_killed"].sum())
fatal_crashes = df.filter(pl.col("accident_severity") == "Fatal")[
    "collision_id"
].n_unique()
pct_fatal = (fatal_crashes / total_crashes) * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Crashes", f"{total_crashes:,}")
c2.metric("People Injured", f"{total_injured:,}")
c3.metric("People Killed", f"{total_killed:,}")
c4.metric("Fatal Crashes", f"{fatal_crashes:,}")
c5.metric("Fatal Rate", f"{pct_fatal:.2f}%")

st.divider()

# ── Severity donut + Hourly bar ───────────────────────────────────────────────

col_a, col_b = st.columns([1, 2])

with col_a:
    st.subheader("Severity Breakdown")
    sev = (
        df.group_by("accident_severity").len().sort("len", descending=True).to_pandas()
    )
    fig = go.Figure(
        go.Pie(
            labels=sev["accident_severity"],
            values=sev["len"],
            hole=0.62,
            marker_colors=[
                SEVERITY_COLORS.get(s, "#555") for s in sev["accident_severity"]
            ],
            textinfo="percent",
            textfont=dict(family="DM Mono", size=11, color="#e8e3f0"),
            hovertemplate="<b>%{label}</b><br>%{value:,} crashes (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        LAYOUT,
        height=300,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=True,
        legend=LEGEND,
    )
    st.plotly_chart(fig, width="stretch")

with col_b:
    st.subheader("Crashes by Hour of Day")
    if "hour" in df.columns:

        def to_ampm(h: int) -> str:
            if h == 0:
                return "12 AM"
            if h < 12:
                return f"{h} AM"
            if h == 12:
                return "12 PM"
            return f"{h - 12} PM"

        hourly = (
            df.filter(pl.col("hour").is_not_null())
            .group_by(["hour", "accident_severity"])
            .len()
            .sort("hour")
        ).to_pandas()

        hour_labels = [to_ampm(h) for h in range(24)]

        fig2 = go.Figure()
        for s in SEVERITY_ORDER:
            sub = hourly[hourly["accident_severity"] == s]
            fig2.add_trace(
                go.Bar(
                    x=sub["hour"],
                    y=sub["len"],
                    name=s,
                    marker_color=SEVERITY_COLORS[s],
                    hovertemplate=f"<b>{s}</b><br>%{{text}} · %{{y:,}} crashes<extra></extra>",
                    text=sub["hour"].map(lambda h: to_ampm(h)),
                    textposition="none",
                )
            )
        fig2.update_layout(
            LAYOUT,
            barmode="stack",
            height=300,
            margin=dict(t=20, b=40, l=40, r=20),
            xaxis=dict(
                **XAXIS,
                title="Hour of Day",
                tickmode="array",
                tickvals=list(range(24)),
                ticktext=hour_labels,
                tickangle=-45,
            ),
            yaxis=dict(**YAXIS, title="Crashes"),
            legend=LEGEND,
        )
        st.plotly_chart(fig2, width="stretch")

st.divider()

# ── Annual trend + Day×Season heatmap ────────────────────────────────────────

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Annual Crash Trend")
    if "year" in df.columns:
        yearly = (
            df.filter(pl.col("year").is_not_null() & (pl.col("year") >= 2013))
            .group_by(["year", "accident_severity"])
            .len()
            .sort("year")
        ).to_pandas()
        fig3 = go.Figure()
        for s in SEVERITY_ORDER:
            sub = yearly[yearly["accident_severity"] == s]
            fig3.add_trace(
                go.Scatter(
                    x=sub["year"],
                    y=sub["len"],
                    name=s,
                    mode="lines+markers",
                    line=dict(color=SEVERITY_COLORS[s], width=2),
                    marker=dict(size=5),
                    hovertemplate=f"<b>{s}</b><br>%{{x}}: %{{y:,}} crashes<extra></extra>",
                )
            )
        fig3.update_layout(
            LAYOUT,
            height=300,
            margin=dict(t=20, b=40, l=40, r=20),
            xaxis=XAXIS,
            yaxis=YAXIS,
            legend=LEGEND,
        )
        st.plotly_chart(fig3, width="stretch")

with col_d:
    st.subheader("Crashes by Day & Season")
    if "day_name" in df.columns and "season" in df.columns:
        DAY_ORDER = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
        hm = (
            df.filter(pl.col("day_name").is_not_null() & pl.col("season").is_not_null())
            .group_by(["day_name", "season"])
            .len()
        ).to_pandas()
        pivot = (
            hm.pivot(index="day_name", columns="season", values="len")
            .reindex(index=DAY_ORDER, columns=SEASON_ORDER)
            .fillna(0)
        )
        fig4 = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[[0, "#13131f"], [0.5, "#7c3aed"], [1, "#ef4444"]],
                hovertemplate="<b>%{y} · %{x}</b><br>%{z:,} crashes<extra></extra>",
                showscale=True,
                colorbar=dict(
                    tickfont=dict(family="DM Mono", size=9, color=SUBTEXT),
                    outlinewidth=0,
                ),
            )
        )
        fig4.update_layout(
            LAYOUT,
            height=300,
            margin=dict(t=20, b=40, l=90, r=20),
            xaxis=dict(tickfont=dict(family="DM Mono", size=10), color=SUBTEXT),
            yaxis=dict(tickfont=dict(family="DM Mono", size=10), color=SUBTEXT),
        )
        st.plotly_chart(fig4, width="stretch")

# ── Policy callout — HTML: red left-border + background tint ─────────────────

if "hour" in df.columns:
    peak_hour = (
        df.filter(pl.col("accident_severity").is_in(["Fatal", "Major Injury"]))
        .filter(pl.col("hour").is_not_null())
        .group_by("hour")
        .len()
        .sort("len", descending=True)
        .head(1)["hour"][0]
    )

    def to_ampm(h: int) -> str:
        if h == 0:
            return "12:00 AM"
        if h < 12:
            return f"{h}:00 AM"
        if h == 12:
            return "12:00 PM"
        return f"{h - 12}:00 PM"

    st.markdown(
        f"""
    <div style="border-left:3px solid #ef4444;padding:1.2rem 1.5rem;
                background:#1a0f0f;border-radius:0 6px 6px 0;margin-top:0.5rem;">
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                  text-transform:uppercase;color:#ef4444;margin-bottom:0.5rem;">Policy Insight</div>
      <div style="font-size:1rem;color:#e8e3f0;line-height:1.75;">
        The highest concentration of severe and fatal crashes in Queens occurs at
        <strong>{to_ampm(peak_hour)}</strong>. Targeted enforcement during this window —
        speed cameras, red-light cameras, increased patrol presence — could
        meaningfully reduce the injury toll.
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
