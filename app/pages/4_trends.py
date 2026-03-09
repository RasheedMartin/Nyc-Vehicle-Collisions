"""pages/5_trends.py — Severity trends over time."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import polars as pl
import plotly.graph_objects as go
from main import load_data
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

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

_year_list = df["year"].drop_nulls().cast(pl.Int32).unique().sort().to_list()
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
              text-transform:uppercase;color:#6b6880;">04 · Trends</div>
  <h1 style="font-size:clamp(2rem,5vw,3.5rem);margin:0.3rem 0 0 0;color:#f0ecff;">
    Severity Over Time
  </h1>
  <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;
              text-transform:uppercase;color:#6b6880;margin-top:0.4rem;">
    {year_label}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Build monthly series ───────────────────────────────────────────────────────

monthly = (
    df.filter(
        pl.col("crash_date").is_not_null()
        & pl.col("year").is_not_null()
        & (pl.col("year") >= 2013)
    )
    .with_columns(
        [
            pl.col("crash_date").dt.year().alias("yr"),
            pl.col("crash_date").dt.month().alias("mo"),
        ]
    )
    .group_by(["yr", "mo", "accident_severity"])
    .len()
    .sort(["yr", "mo"])
)

# Totals per month (all severities) for fatal rate
monthly_total = (
    monthly.group_by(["yr", "mo"])
    .agg(pl.col("len").sum().alias("total"))
    .sort(["yr", "mo"])
)
monthly_fatal = monthly.filter(pl.col("accident_severity") == "Fatal").select(
    ["yr", "mo", pl.col("len").alias("fatal")]
)
fatal_rate = (
    monthly_total.join(monthly_fatal, on=["yr", "mo"], how="left")
    .with_columns(pl.col("fatal").fill_null(0))
    .with_columns((pl.col("fatal") / pl.col("total") * 100).alias("fatal_pct"))
    .sort(["yr", "mo"])
    .with_columns(
        (
            pl.col("yr").cast(pl.Utf8) + "-" + pl.col("mo").cast(pl.Utf8).str.zfill(2)
        ).alias("label")
    )
)

# ── Monthly crash trend by severity ───────────────────────────────────────────

st.subheader("Monthly Crash Volume by Severity")

monthly_pd = monthly.with_columns(
    (pl.col("yr").cast(pl.Utf8) + "-" + pl.col("mo").cast(pl.Utf8).str.zfill(2)).alias(
        "label"
    )
).to_pandas()

fig1 = go.Figure()
for s in SEVERITY_ORDER:
    sub = monthly_pd[monthly_pd["accident_severity"] == s]
    fig1.add_trace(
        go.Scatter(
            x=sub["label"],
            y=sub["len"],
            name=s,
            mode="lines",
            line=dict(color=SEVERITY_COLORS[s], width=1.5),
            hovertemplate=f"<b>{s}</b><br>%{{x}}: %{{y:,}} crashes<extra></extra>",
        )
    )
fig1.update_layout(
    LAYOUT,
    height=320,
    margin=dict(t=20, b=40, l=50, r=20),
    xaxis=dict(**XAXIS, tickangle=-45, nticks=20),
    yaxis=dict(**YAXIS, title="Crashes / month"),
    legend=LEGEND,
)
st.plotly_chart(fig1, width="stretch")

st.divider()

# ── Fatal rate + rolling average ──────────────────────────────────────────────

st.subheader("Fatal Crash Rate Over Time")
st.caption("Monthly % of crashes resulting in a fatality · 12-month rolling average")

fatal_pd = fatal_rate.to_pandas()

# Rolling 12-month average
fatal_pd = fatal_pd.sort_values("label")
fatal_pd["rolling_12"] = fatal_pd["fatal_pct"].rolling(12, min_periods=6).mean()

covid_years = fatal_pd[fatal_pd["label"].str.startswith(("2020", "2021"))]
pre_covid = fatal_pd[fatal_pd["label"].str.startswith(("2018", "2019"))]

pre_avg = pre_covid["rolling_12"].mean() if len(pre_covid) > 0 else None
covid_avg = covid_years["rolling_12"].mean() if len(covid_years) > 0 else None
covid_peak = covid_years["fatal_pct"].max() if len(covid_years) > 0 else None
covid_peak_label = (
    covid_years.loc[covid_years["fatal_pct"].idxmax(), "label"]
    if len(covid_years) > 0
    else None
)

fig2 = go.Figure()

# COVID shaded band
fig2.add_vrect(
    x0="2020-01",
    x1="2021-12",
    fillcolor="#7c3aed",
    opacity=0.08,
    layer="below",
    line_width=0,
)
fig2.add_vline(x="2020-03", line=dict(color="#7c3aed", width=1, dash="dot"))

# Pre-COVID baseline reference line
if pre_avg is not None:
    fig2.add_hline(
        y=pre_avg,
        line=dict(color="#6b6880", width=1, dash="dash"),
        annotation_text=f"Pre-COVID avg {pre_avg:.3f}%",
        annotation_font=dict(family="DM Mono", size=9, color="#6b6880"),
        annotation_position="top left",
    )

# Monthly noise
fig2.add_trace(
    go.Scatter(
        x=fatal_pd["label"],
        y=fatal_pd["fatal_pct"],
        name="Monthly rate",
        mode="lines",
        line=dict(color="#6b6880", width=1),
        opacity=0.4,
        hovertemplate="<b>%{x}</b><br>Fatal rate: %{y:.3f}%<extra></extra>",
    )
)

# Rolling average
fig2.add_trace(
    go.Scatter(
        x=fatal_pd["label"],
        y=fatal_pd["rolling_12"],
        name="12-month avg",
        mode="lines",
        line=dict(color="#ef4444", width=2.5),
        hovertemplate="<b>%{x}</b><br>12-mo avg: %{y:.3f}%<extra></extra>",
    )
)

# Spike annotation
if covid_peak is not None and covid_peak_label is not None:
    fig2.add_annotation(
        x=covid_peak_label,
        y=covid_peak,
        text=f"Peak {covid_peak:.3f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ef4444",
        font=dict(family="DM Mono", size=9, color="#ef4444"),
        bgcolor="#1a0808",
        bordercolor="#ef4444",
        borderwidth=1,
        ax=40,
        ay=-30,
    )

# COVID label inside the band
fig2.add_annotation(
    x="2020-09",
    text="COVID-19",
    showarrow=False,
    font=dict(family="DM Mono", size=9, color="#7c3aed"),
    yref="paper",
    yanchor="bottom",
    y=0.02,
)

fig2.update_layout(
    LAYOUT,
    height=320,
    margin=dict(t=30, b=40, l=60, r=20),
    xaxis=dict(**XAXIS, tickangle=-45, nticks=20),
    yaxis=dict(**YAXIS, title="Fatal crashes (%)"),
    legend=LEGEND,
)
st.plotly_chart(fig2, width="stretch")

# ── Policy callout ────────────────────────────────────────────────────────────

if len(covid_years) > 0 and len(pre_covid) > 0:
    covid_rate = covid_years["fatal_pct"].mean()
    pre_rate = pre_covid["fatal_pct"].mean()
    delta_pct = ((covid_rate - pre_rate) / pre_rate) * 100

    st.markdown(
        f"""
    <div style="border-left:3px solid #7c3aed;padding:1.2rem 1.5rem;
                background:#0f0a1a;border-radius:0 6px 6px 0;margin-top:0.5rem;">
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                  text-transform:uppercase;color:#7c3aed;margin-bottom:0.5rem;">COVID-19 Effect</div>
      <div style="font-size:1rem;color:#e8e3f0;line-height:1.75;">
        During 2020–2021, total crash volume in Queens dropped sharply —
        but the fatal crash rate <strong>{"rose" if delta_pct > 0 else "fell"} {abs(delta_pct):.0f}%</strong>
        compared to 2018–2019 ({pre_rate:.3f}% → {covid_rate:.3f}%).
        Fewer cars, higher speeds. Vision Zero data shows the same pattern citywide:
        emptier streets led to riskier driving behaviour, not safer outcomes.
        <br/><br/>
        The downward trend visible before 2020 reflects NYC's
        <strong>Vision Zero</strong> initiative, launched in January 2014.
        The programme reduced the citywide speed limit to 25 mph, expanded
        automated speed cameras near schools, and redesigned dozens of dangerous
        intersections. Queens saw measurable improvement in fatal crash rates from
        2014 through 2019 — gains that COVID-era speeding behaviour partially reversed.
        <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#6b6880;
                     display:block;margin-top:0.6rem;">
          Source: NYC DOT Vision Zero Year 9 Report (2022) ·
          <a href="https://www.nyc.gov/html/dot/html/pedestrians/visionzero.shtml"
             style="color:#6b6880;" target="_blank">nyc.gov/dot/visionzero</a>
        </span>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
st.divider()

# ── Year-over-year comparison ──────────────────────────────────────────────────

st.subheader("Year-over-Year: Total Monthly Crashes")
st.caption("Each line is one year — shows whether Queens is improving")

yoy_pd = (
    monthly.group_by(["yr", "mo"])
    .agg(pl.col("len").sum().alias("total"))
    .sort(["yr", "mo"])
    .to_pandas()
)

years = sorted(yoy_pd["yr"].unique())
palette = [
    "#6b6880",
    "#6b6880",
    "#6b6880",
    "#6b6880",
    "#6b6880",
    "#6b6880",
    "#6b6880",
    "#7c3aed",
    "#a855f7",
    "#ef4444",
]
# Older years grey, recent years highlighted
year_colors = {yr: palette[min(i, len(palette) - 1)] for i, yr in enumerate(years)}
year_colors[max(years)] = "#ef4444"
if len(years) >= 2:
    year_colors[sorted(years)[-2]] = "#a855f7"
if len(years) >= 3:
    year_colors[sorted(years)[-3]] = "#7c3aed"

fig3 = go.Figure()
for yr in years:
    sub = yoy_pd[yoy_pd["yr"] == yr].sort_values("mo")
    is_recent = yr >= sorted(years)[-3]
    fig3.add_trace(
        go.Scatter(
            x=sub["mo"],
            y=sub["total"],
            name=str(yr),
            mode="lines",
            line=dict(color=year_colors[yr], width=2.5 if is_recent else 1),
            opacity=1.0 if is_recent else 0.4,
            hovertemplate=f"<b>{yr}</b><br>%{{x}}: %{{y:,}} crashes<extra></extra>",
        )
    )
fig3.update_layout(
    LAYOUT,
    height=320,
    margin=dict(t=20, b=40, l=50, r=20),
    xaxis=dict(
        **XAXIS, tickvals=list(range(1, 13)), ticktext=MONTH_NAMES, title="Month"
    ),
    yaxis=dict(**YAXIS, title="Total crashes"),
    legend=LEGEND,
)
st.plotly_chart(fig3, width="stretch")


