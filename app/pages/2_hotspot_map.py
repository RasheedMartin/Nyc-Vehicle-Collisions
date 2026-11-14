"""
pages/2_hotspot_map.py

pydeck (deck.gl) map, fed with *pre-aggregated* data.

Why aggregate server-side: Queens has ~1.2M crashes. Shipping raw points to
the browser meant a ~20 MB JSON payload on every filter change (and random
sampling that distorted the heatmap). Binning into ~165 m grid cells collapses
that to ~8k rows / a few hundred KB with exact counts — heatmap is weighted by
the true count, so nothing is lost visually.

Scatter mode still shows individual crashes, but capped (all fatal crashes are
always kept) and with a slim column set.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import polars as pl
import pydeck as pdk
import streamlit as st
from theme import HEATMAP_COLOR_RANGE, SEVERITY_COLORS, SEVERITY_ORDER
from utils import load_data

# ~165 m cells at Queens' latitude (1° lat ≈ 111 km; 1° lon ≈ 84 km at 40.7°N)
LAT_STEP = 0.0015
LON_STEP = 0.002
CELL_SIZE_M = 165
SCATTER_CAP = 20_000

SEVERITY_RGB = {
    s: tuple(int(SEVERITY_COLORS[s][i : i + 2], 16) for i in (1, 3, 5))
    for s in SEVERITY_ORDER
}

TOOLTIP_STYLE = {
    "backgroundColor": "#13131f",
    "color": "#e8e3f0",
    "fontFamily": "DM Mono, monospace",
    "fontSize": "11px",
    "border": "1px solid #1f1f30",
    "borderRadius": "6px",
}

df = load_data()
# Cheap fingerprint so cached aggregates invalidate when the parquet refreshes
DATA_VERSION = (df.height, str(df["crash_date"].max()))


# ── Cached aggregations ──────────────────────────────────────────────────────


def _filtered(_df: pl.DataFrame, severities: tuple, years: tuple) -> pl.DataFrame:
    out = _df.filter(
        pl.col("accident_severity").is_in(list(severities))
        & pl.col("latitude").is_not_null()
        & pl.col("longitude").is_not_null()
    )
    if years:
        out = out.filter(pl.col("year").is_between(years[0], years[1]))
    return out


@st.cache_data(show_spinner=False, max_entries=64)
def grid_cells(
    _df: pl.DataFrame, version, severities: tuple, years: tuple
) -> pd.DataFrame:
    """Bin crashes into ~165 m cells with counts, severe/fatal counts, top street."""
    has_street = "on_street_name" in _df.columns
    aggs = [
        pl.len().alias("n"),
        (pl.col("accident_severity") == "Fatal").sum().alias("fatal"),
        pl.col("accident_severity")
        .is_in(["Fatal", "Major Injury"])
        .sum()
        .alias("severe"),
    ]
    if has_street:
        aggs.append(
            pl.col("on_street_name")
            .drop_nulls()
            .str.strip_chars()
            .str.to_titlecase()
            .mode()
            .first()
            .alias("street")
        )

    cells = (
        _filtered(_df, severities, years)
        .with_columns(
            (pl.col("latitude") / LAT_STEP).floor().cast(pl.Int32).alias("gy"),
            (pl.col("longitude") / LON_STEP).floor().cast(pl.Int32).alias("gx"),
        )
        .group_by("gy", "gx")
        .agg(aggs)
        .with_columns(
            # cell corner (GridCellLayer) and centre (HeatmapLayer)
            (pl.col("gy") * LAT_STEP).round(5).alias("lat0"),
            (pl.col("gx") * LON_STEP).round(5).alias("lon0"),
            ((pl.col("gy") + 0.5) * LAT_STEP).round(5).alias("lat"),
            ((pl.col("gx") + 0.5) * LON_STEP).round(5).alias("lon"),
        )
        .to_pandas()
    )
    if not has_street:
        cells["street"] = ""
    cells["street"] = cells["street"].fillna("—")

    # Log-scaled colour ramp over the theme's heatmap palette
    if len(cells):
        t = np.log1p(cells["n"]) / np.log1p(cells["n"].max())
        stops = np.array(HEATMAP_COLOR_RANGE, dtype=float)
        pos = t * (len(stops) - 1)
        i = np.clip(pos.astype(int), 0, len(stops) - 2)
        frac = (pos - i).to_numpy()[:, None]
        rgb = stops[i] * (1 - frac) + stops[i + 1] * frac
        cells[["r", "g", "b"]] = rgb.round().astype(np.uint8)
    return cells.drop(columns=["gy", "gx"])


@st.cache_data(show_spinner=False, max_entries=32)
def scatter_points(_df: pl.DataFrame, version, severities: tuple, years: tuple):
    """Individual crashes, capped; every fatal crash is always kept."""
    pts = _filtered(_df, severities, years)
    total = pts.height
    if total > SCATTER_CAP:
        fatal = pts.filter(pl.col("accident_severity") == "Fatal")
        rest = pts.filter(pl.col("accident_severity") != "Fatal")
        n_rest = max(SCATTER_CAP - fatal.height, 0)
        pts = pl.concat([fatal, rest.sample(n=min(n_rest, rest.height), seed=42)])

    out = pts.select(
        pl.col("longitude").round(5).alias("lon"),
        pl.col("latitude").round(5).alias("lat"),
        pl.col("accident_severity").alias("severity"),
        pl.col("crash_date").cast(pl.Utf8).alias("date"),
        pl.col("contributing_factor_vehicle_1")
        .fill_null("Unspecified")
        .alias("factor"),
    ).to_pandas()
    for k, ch in enumerate("rgb"):
        out[ch] = (
            out["severity"]
            .map(lambda s, k=k: SEVERITY_RGB.get(s, (150, 150, 150))[k])
            .astype(np.uint8)
        )
    return out, total


# ── Header ───────────────────────────────────────────────────────────────────

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

# ── Filters ──────────────────────────────────────────────────────────────────

col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
with col_f1:
    selected_severity = st.multiselect(
        "Severity", SEVERITY_ORDER, default=["Fatal", "Major Injury"]
    )
with col_f2:
    map_style = st.selectbox("Map Style", ["Heatmap", "Density Grid", "Scatter Points"])
with col_f3:
    year_range: tuple = ()
    if len(_year_list) == 1:
        st.caption(f"Year: {_year_list[0]}")
    elif len(_year_list) > 1:
        year_range = tuple(
            st.select_slider(
                "Year", options=_year_list, value=(_year_list[0], _year_list[-1])
            )
        )

if not selected_severity:
    st.info("Select at least one severity level to draw the map.", icon="ℹ️")
    st.stop()

sev_key = tuple(sorted(selected_severity, key=SEVERITY_ORDER.index))

# ── Layers ───────────────────────────────────────────────────────────────────

view_state = pdk.ViewState(latitude=40.7282, longitude=-73.7949, zoom=10.8, pitch=0)
tooltip = None

if map_style == "Scatter Points":
    data, total = scatter_points(df, DATA_VERSION, sev_key, year_range)
    shown = len(data)
    caption = f"Showing {shown:,} of {total:,} crashes" + (
        " · sampled for speed, all fatal crashes kept" if shown < total else ""
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=["lon", "lat"],
        get_fill_color=["r", "g", "b", 200],
        get_radius=40,
        radius_min_pixels=2,
        radius_max_pixels=8,
        pickable=True,
        stroked=False,
    )
    tooltip = {
        "html": "<b>{severity}</b><br/>{date}<br/><span style='color:#6b6880'>{factor}</span>",
        "style": TOOLTIP_STYLE,
    }
    legend_html = "".join(
        f"<span style='display:inline-flex;align-items:center;margin-right:1rem;'>"
        f"<span style='background:{SEVERITY_COLORS[s]};width:10px;height:10px;"
        f"border-radius:50%;display:inline-block;margin-right:6px;'></span>{s}</span>"
        for s in reversed(SEVERITY_ORDER)
    )
else:
    cells = grid_cells(df, DATA_VERSION, sev_key, year_range)
    total = int(cells["n"].sum()) if len(cells) else 0
    caption = f"{total:,} crashes aggregated into {len(cells):,} ~{CELL_SIZE_M} m cells"

    if map_style == "Heatmap":
        layer = pdk.Layer(
            "HeatmapLayer",
            data=cells[["lon", "lat", "n"]],
            get_position=["lon", "lat"],
            get_weight="n",
            radius_pixels=35,
            intensity=1.0,
            threshold=0.03,
            color_range=HEATMAP_COLOR_RANGE,
        )
    else:
        layer = pdk.Layer(
            "GridCellLayer",
            data=cells,
            get_position=["lon0", "lat0"],
            cell_size=CELL_SIZE_M,
            get_fill_color=["r", "g", "b", 210],
            extruded=False,
            pickable=True,
        )
        tooltip = {
            "html": (
                "<b>{street}</b><br/>{n} crashes<br/>"
                "<span style='color:#f97316'>{severe} severe</span> · "
                "<span style='color:#ef4444'>{fatal} fatal</span>"
            ),
            "style": TOOLTIP_STYLE,
        }
    legend_html = (
        "<span style='margin-right:0.6rem;'>Low</span>"
        "<span style='display:inline-block;width:140px;height:10px;border-radius:5px;"
        "background:linear-gradient(90deg,#13131f,#7c3aed,#ef4444);vertical-align:middle;'></span>"
        "<span style='margin-left:0.6rem;'>High</span>"
        + (
            ""
            if map_style == "Heatmap"
            else " <span style='color:#6b6880'>(log scale)</span>"
        )
    )

st.markdown(
    f"""<div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:0.5rem;
    font-family:'DM Mono',monospace;font-size:0.72rem;color:#e8e3f0;margin-bottom:0.4rem;">
    <div style="color:#6b6880;">{caption}</div><div>{legend_html}</div></div>""",
    unsafe_allow_html=True,
)

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip=tooltip or False,
    ),
    width="stretch",
    height=560,
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
            f"Fatal + Major Injury only · % = each street's share of all severe crashes in Queens "
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
