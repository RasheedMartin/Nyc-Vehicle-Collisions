"""
pages/2_hotspot_map.py

Back on pydeck (deck.gl) — pure client-side WebGL.
Folium's st_folium triggers a Python rerun on every zoom/pan because it's a
two-way bridge designed for receiving map events. For a display-only map
pydeck is strictly better: no roundtrips, handles 150k+ points smoothly.

Fix for the old vars() __dict__ error: never put a Python list inside a
pandas cell. Instead, explode RGB into separate r/g/b integer columns and
reference them in get_fill_color=["r","g","b",200].
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import polars as pl
import pydeck as pdk
from utils import load_data
from theme import SEVERITY_ORDER, HEATMAP_COLOR_RANGE

df = load_data()

_year_list = df["year"].drop_nulls().cast(pl.Int32).unique().sort().to_list()
year_label = (
    f"{_year_list[0]} – {_year_list[-1]}"
    if len(_year_list) > 1
    else str(_year_list[0])
    if _year_list
    else ""
)

# ── Header (HTML: Bebas Neue + custom sizing) ─────────────────────────────────

st.markdown(
    f"""
<div style="padding:2rem 0 1rem 0;">
  <div style="font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.2em;
              text-transform:uppercase;color:#6b6880;">02 · Hotspot Map</div>
  <h1 style="font-size:clamp(2rem,5vw,3.5rem);margin:0.3rem 0 0 0;color:#f0ecff;">
    Where Crashes Happen
  </h1>
  <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;
              text-transform:uppercase;color:#6b6880;margin-top:0.4rem;">
    {year_label}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Filters — native components ───────────────────────────────────────────────

col_f1, col_f2, col_f3 = st.columns([2, 2, 1])

with col_f1:
    selected_severity = st.multiselect(
        "Severity", SEVERITY_ORDER, default=["Fatal", "Major Injury"]
    )
with col_f2:
    map_style = st.selectbox("Map Style", ["Heatmap", "Scatter Points", "Density Grid"])
with col_f3:
    year_range = None
    if "year" in df.columns:
        years = sorted(df["year"].drop_nulls().unique().to_list())
        if len(years) == 1:
            st.caption(f"Year: {years[0]}")
            year_range = (years[0], years[0])
        elif len(years) > 1:
            year_range = st.select_slider(
                "Year", options=years, value=(years[0], years[-1])
            )

# ── Filter ────────────────────────────────────────────────────────────────────

map_df = df.filter(
    pl.col("accident_severity").is_in(selected_severity)
    & pl.col("latitude").is_not_null()
    & pl.col("longitude").is_not_null()
)
if year_range and "year" in df.columns:
    map_df = map_df.filter(pl.col("year").is_between(year_range[0], year_range[1]))
if len(map_df) > 150_000:
    map_df = map_df.sample(n=150_000, seed=42)

st.caption(f"Showing {len(map_df):,} crashes")

# ── Build pandas payload ──────────────────────────────────────────────────────

SEVERITY_RGB = {
    "No Injury": (74, 222, 128),
    "Minor Injury": (250, 204, 21),
    "Major Injury": (249, 115, 22),
    "Fatal": (239, 68, 68),
}

map_pd = map_df.select(
    [
        "latitude",
        "longitude",
        "accident_severity",
        pl.col("crash_date").cast(pl.Utf8).alias("date"),
        pl.col("contributing_factor_vehicle_1").alias("factor"),
    ]
).to_pandas()

# Separate r/g/b columns — pydeck's JSON serialiser cannot handle Python
# lists stored inside pandas cells (causes vars() __dict__ TypeError)
map_pd["r"] = map_pd["accident_severity"].map(
    lambda s: SEVERITY_RGB.get(s, (150, 150, 150))[0]
)
map_pd["g"] = map_pd["accident_severity"].map(
    lambda s: SEVERITY_RGB.get(s, (150, 150, 150))[1]
)
map_pd["b"] = map_pd["accident_severity"].map(
    lambda s: SEVERITY_RGB.get(s, (150, 150, 150))[2]
)

# ── Layer ─────────────────────────────────────────────────────────────────────

view_state = pdk.ViewState(latitude=40.7282, longitude=-73.7949, zoom=11, pitch=0)

if map_style == "Heatmap":
    layer = pdk.Layer(
        "HeatmapLayer",
        data=map_pd,
        get_position=["longitude", "latitude"],
        get_weight=1,
        radius_pixels=45,
        intensity=1.0,
        threshold=0.05,
        pickable=True,
        color_range=HEATMAP_COLOR_RANGE,
    )

    tooltip = {
        "text": "Crash density hotspot",
        "style": {
            "backgroundColor": "#13131f",
            "color": "#e8e3f0",
            "fontFamily": "DM Mono",
            "fontSize": "11px",
            "border": "1px solid #1f1f30",
        },
    }
elif map_style == "Scatter Points":
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_pd,
        get_position=["longitude", "latitude"],
        get_fill_color=["r", "g", "b", 200],
        get_radius=50,
        radius_min_pixels=2,
        radius_max_pixels=10,
        pickable=True,
        opacity=0.85,
        stroked=False,
    )
    tooltip = {
        "html": (
            "<div style='font-family:DM Mono,monospace;font-size:0.75rem;"
            "background:#13131f;color:#e8e3f0;padding:0.6rem 0.9rem;"
            "border:1px solid #1f1f30;border-radius:6px;'>"
            "<b style='color:#ef4444;'>{accident_severity}</b><br/>"
            "{date}<br/>"
            "<span style='color:#6b6880;'>{factor}</span>"
            "</div>"
        ),
        "style": {"backgroundColor": "transparent", "border": "none"},
    }
else:
    # ScreenGridLayer: counts points per screen-space cell.
    # No weight column = no min/max range error.
    # No color_range override = avoids the RangeError we hit before.
    layer = pdk.Layer(
        "ScreenGridLayer",
        data=map_pd,
        get_position=["longitude", "latitude"],
        cell_size_pixels=14,
        opacity=0.8,
        pickable=False,
    )
    tooltip = None

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip=True,
    ),
    width="stretch",
    height=560,
)
if map_style == "Scatter Points":
    st.markdown(
        """
<div style="
  position: relative;
  display: inline-block;
  background: #13131f;
  border: 1px solid #1f1f30;
  border-radius: 6px;
  padding: 0.75rem 1rem;
  margin-top: -520px;
  margin-left: 1rem;
  z-index: 10;
  width: 180px;
">
  <div style="font-family:DM Mono,monospace;
              font-size:0.65rem;
              letter-spacing:0.12em;
              text-transform:uppercase;
              color:#6b6880;
              margin-bottom:0.6rem;">
    Severity
  </div>
  <div style="display:flex;flex-direction:column;gap:0.4rem;">
    <div><span style="background:#ef4444;width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:6px;"></span>Fatal</div>
    <div><span style="background:#f97316;width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:6px;"></span>Major Injury</div>
    <div><span style="background:#facc15;width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:6px;"></span>Minor Injury</div>
    <div><span style="background:#4ade80;width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:6px;"></span>No Injury</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
elif map_style == "Heatmap":
    st.markdown(
        """
    <div style="
      position: relative;
      display: inline-block;
      background: #13131f;
      border: 1px solid #1f1f30;
      border-radius: 6px;
      padding: 0.75rem 1rem;
      margin-top: -520px;
      margin-left: 1rem;
      z-index: 10;
      width: 200px;
    ">
      <div style="font-family:DM Mono,monospace;
                  font-size:0.65rem;
                  letter-spacing:0.12em;
                  text-transform:uppercase;
                  color:#6b6880;
                  margin-bottom:0.6rem;">
        Crash Density
      </div>
      <div style="
        height: 10px;
        background: linear-gradient(
          90deg,
          #13131f,
          #7c3aed,
          #ef4444
        );
        border-radius: 5px;
        margin-bottom: 0.4rem;
      "></div>
      <div style="display:flex;justify-content:space-between;
                  font-family:DM Mono,monospace;
                  font-size:0.6rem;color:#6b6880;">
        <span>Low</span>
        <span>High</span>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
# ── Top corridors — computed from data ────────────────────────────────────────
# Uses on_street_name from the crashes dataset (added to CRASH_COLS in clean.py).
# Falls back to lat/lon grid clusters if street names aren't in this parquet yet
# (i.e. before re-running clean.py with the new column list).

st.divider()

severe = df.filter(pl.col("accident_severity").is_in(["Fatal", "Major Injury"]))

if "on_street_name" in df.columns:
    corridors = (
        severe.filter(
            pl.col("on_street_name").is_not_null() & (pl.col("on_street_name") != "")
        )
        .with_columns(
            pl.col("on_street_name")
            .str.to_uppercase()
            .str.strip_chars()
            .alias("street")
        )
        .group_by("street")
        .agg(
            pl.len().alias("total_severe"),
            pl.col("accident_severity")
            .filter(pl.col("accident_severity") == "Fatal")
            .len()
            .alias("fatal_count"),
        )
        .sort("total_severe", descending=True)
        .head(5)
        .to_pandas()
    )

    total_severe = len(severe)
    top5_severe = int(corridors["total_severe"].sum())
    pct_top5 = top5_severe / total_severe * 100 if total_severe > 0 else 0
    corridors["share_pct"] = corridors["total_severe"] / total_severe * 100

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Top 5 Corridors by Severe Crashes")
        st.caption(
            f"Fatal + Major Injury only · % = each street\'s share of all severe crashes in Queens "
            f"(adds up to {pct_top5:.0f}%)"
        )
        for _, row in corridors.iterrows():
            share = row["share_pct"]
            bar_width = min(int(share / pct_top5 * 100), 100) if pct_top5 > 0 else 0
            st.markdown(
                f"""
            <div style="padding:0.75rem 1rem;margin-bottom:0.5rem;
                        background:#13131f;border:1px solid #1f1f30;border-radius:6px;">
              <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.4rem;">
                <div style="flex:1;font-family:'DM Sans',sans-serif;
                            font-size:0.95rem;color:#e8e3f0;">{row["street"].title()}</div>
                <div style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;
                            color:#ef4444;min-width:3rem;text-align:right;">
                  {int(row["total_severe"]):,}
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                            color:#ef4444;min-width:4rem;text-align:right;font-weight:600;">
                  {share:.1f}%
                </div>
              </div>
              <div style="background:#1f1f30;border-radius:3px;height:4px;width:100%;">
                <div style="background:#ef4444;border-radius:3px;height:4px;width:{bar_width}%;"></div>
              </div>
              <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                          color:#6b6880;margin-top:0.3rem;">
                {int(row["fatal_count"])} fatal crashes on this corridor
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        total_severe = len(severe)
        top5_severe = int(corridors["total_severe"].sum())
        pct_top5 = top5_severe / total_severe * 100 if total_severe > 0 else 0
        top_street = corridors.iloc[0]["street"].title()

        st.metric("Crashes on Top 5 Streets", f"{top5_severe:,}")
        st.metric("Share of All Severe Crashes", f"{pct_top5:.1f}%")
        st.metric("Highest Risk Corridor", top_street)

    # Policy callout — data-derived street names, sourced external stat
    top5 = ", ".join(corridors.head(5)["street"].str.title().tolist())
    st.markdown(
        f"""
    <div style="border-left:3px solid #ef4444;padding:1.2rem 1.5rem;
                background:#1a0f0f;border-radius:0 6px 6px 0;margin-top:1rem;">
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                  text-transform:uppercase;color:#ef4444;margin-bottom:0.5rem;">
        Data-Derived Insight
      </div>
      <div style="font-size:1rem;color:#e8e3f0;line-height:1.75;">
        Data shows the <strong>{top5}</strong> account for
        <strong>{pct_top5:.0f}%</strong> of all fatal and major injury crashes in Queens.
        NYC DOT's protected intersection program has demonstrated 20–40% reductions
        in serious injuries at treated locations across the city.
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

else:
    # Street name columns not yet in parquet — show fallback message
    st.info(
        "Street-level corridor analysis requires re-running `clean.py` — "
        "`on_street_name` has been added to `CRASH_COLS` and will appear after the next pipeline run.",
        icon="ℹ️",
    )
