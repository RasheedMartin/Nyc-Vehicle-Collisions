"""pages/2_hotspot_map.py — WebGL collision hotspot map."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import polars as pl
import pydeck as pdk
from main import load_data
from theme import SEVERITY_COLORS, SEVERITY_ORDER, CHART_BG, GRID_COLOR, TEXT, SUBTEXT, ACCENT, HEATMAP_COLOR_RANGE

df = load_data()

st.markdown("""
<div style="padding:2rem 0 1.5rem 0;">
  <div style="font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.2em;
              text-transform:uppercase;color:#6b6880;">02 · Hotspot Map</div>
  <h1 style="font-size:clamp(2rem,5vw,3.5rem);margin:0.3rem 0 0 0;color:#f0ecff;">
    Where Crashes Happen
  </h1>
</div>
""", unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
with col_f1:
    selected_severity = st.multiselect("Severity", SEVERITY_ORDER, default=["Fatal","Major Injury"])
with col_f2:
    map_style = st.selectbox("Map Style", ["Heatmap", "Density Grid", "Scatter Points"])
with col_f3:
    year_range = None

if "year" in df.columns:
    years = sorted(df["year"].drop_nulls().unique().to_list())

    if len(years) == 1:
        # Single-year dataset (current state)
        year_range = (years[0], years[0])
        st.markdown(
            f"<div style='font-family:DM Mono,monospace;"
            f"font-size:0.7rem;color:#6b6880;padding-top:0.6rem;'>"
            f"Year: {years[0]}</div>",
            unsafe_allow_html=True,
        )

    elif len(years) > 1:
        # Multi-year dataset (future-proof)
        year_range = st.select_slider(
            "Year",
            options=years,
            value=(years[0], years[-1]),
        )

map_df = df.filter(
    pl.col("accident_severity").is_in(selected_severity) &
    pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()
)
if year_range and "year" in df.columns:
    map_df = map_df.filter(pl.col("year").is_between(year_range[0], year_range[1]))
if len(map_df) > 150_000:
    map_df = map_df.sample(n=150_000, seed=42)

st.markdown(
    f"<div style='font-family:DM Mono,monospace;font-size:0.7rem;color:#6b6880;margin-bottom:1rem;'>"
    f"Showing {len(map_df):,} crashes</div>", unsafe_allow_html=True,
)

SEVERITY_RGB = {
    "No Injury":    [74,  222, 128],
    "Minor Injury": [250, 204, 21],
    "Major Injury": [249, 115, 22],
    "Fatal":        [239, 68,  68],
}

map_pd = map_df.select([
    "latitude", "longitude", "accident_severity",
    pl.col("crash_date").cast(pl.Utf8).alias("date"),
    pl.col("contributing_factor_vehicle_1").alias("factor"),
]).to_pandas()
map_pd["color"] = map_pd["accident_severity"].map(SEVERITY_RGB)

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
        color_range=HEATMAP_COLOR_RANGE
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
elif map_style == "Density Grid":
    layer = pdk.Layer(
        "ScreenGridLayer",
        data=map_pd,
        get_position=["longitude", "latitude"],
        cell_size_pixels=12,
        opacity=0.85,
        pickable=False,
    )
    tooltip = None
else:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_pd,
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius=40,
        radius_min_pixels=2,
        radius_max_pixels=8,
        pickable=True,
        opacity=0.85,
        stroked=False,
    )
    tooltip = {
        "html": """
        <div style='font-family:DM Mono,monospace;font-size:0.75rem;
                    background:#13131f;color:#e8e3f0;padding:0.7rem 1rem;
                    border:1px solid #1f1f30;border-radius:6px;'>
          <b style='color:#ef4444;'>{accident_severity}</b><br/>
          {date}<br/>
          <span style='color:#6b6880;'>{factor}</span>
        </div>""",
        "style": {"backgroundColor":"transparent","border":"none"},
    }

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    tooltip=True,
), width='stretch', height=560)

if map_style == "Scatter Points":
    st.markdown("""
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
""", unsafe_allow_html=True)
elif map_style == "Heatmap":
    st.markdown("""
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
    """, unsafe_allow_html=True)
    

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="border-left:3px solid #ef4444;padding:1.2rem 1.5rem;
            background:#1a0f0f;border-radius:0 6px 6px 0;">
  <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
              text-transform:uppercase;color:#ef4444;margin-bottom:0.5rem;">Policy Insight</div>
  <div style="font-size:1rem;color:#e8e3f0;line-height:1.75;">
    Crash hotspots in Queens cluster along major arterials — Jamaica Ave, Northern Blvd,
    and the Van Wyck corridor. Infrastructure interventions at these locations
    (protected bike lanes, pedestrian refuges, signal timing changes) have demonstrated
    30–50% reductions in pedestrian injuries in comparable NYC corridors.
  </div>
</div>
""", unsafe_allow_html=True)