"""
clean.py
--------
Cleans and merges the raw NYC Motor Vehicle Collision datasets using Polars.
Faithful to the original notebook logic with production-grade improvements.

Note on the borough join:
  The original notebook used geopandas for a spatial join against the shapefile.
  Polars has no native geospatial support, so we keep geopandas for that step
  only and immediately convert back to Polars. Everything else is pure Polars.

Output:
  data/processed/collisions.parquet

Pipeline order:
  fetch.py  →  clean.py  →  features.py  →  train.py
"""

import logging
import polars as pl
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
BOROUGH_SHP = Path("Borough Boundaries/geo_borough.shp")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Column selections ─────────────────────────────────────────────────────────

CRASH_COLS = [
    "collision_id",
    "crash_date",
    "crash_time",
    "latitude",
    "longitude",
    "on_street_name",
    "off_street_name",
    "number_of_persons_injured",
    "number_of_persons_killed",
    "contributing_factor_vehicle_1",
    "contributing_factor_vehicle_2",
]

PERSON_COLS = [
    "collision_id",
    "person_age",
    "person_type",
    "person_sex",
]

# Typo/casing fixes — notebook 01 exact rename dict
FACTOR_RENAME: dict[str, str] = {
    "Cell Phone (hand-held)": "Cell Phone (Hand-Held)",
    "Cell Phone (hand-Held)": "Cell Phone (Hand-Held)",
    "Cell Phone (hands-free)": "Cell Phone (Hands-Free)",
    "Drugs (illegal)": "Drugs (Illegal)",
    "Illnes": "Illness",
}

# NYC bounding box — notebook 01 exact bounds
LAT_MIN, LAT_MAX = 40.5774, 45.01585
LON_MIN, LON_MAX = -74.2591, -73.7004

VALID_BOROUGHS = {"MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"}

# ── Loaders ───────────────────────────────────────────────────────────────────


def _available_cols(col_names: list[str], wanted: list[str]) -> list[str]:
    """Return only the columns from `wanted` that exist in `df`."""
    available = [c for c in wanted if c in col_names]
    missing = set(wanted) - set(available)
    if missing:
        log.warning(f"  Missing columns (skipped): {missing}")
    return available


def load_parquet(path: Path, required_cols: list[str]) -> pl.DataFrame:
    log.info(f"Loading {path}")
    # scan_parquet for lazy pushdown — only materialise the columns we need
    lf = pl.scan_parquet(path)
    lf = lf.rename(
        {c: c.lower().replace(" ", "_") for c in lf.collect_schema().names()}
    )
    available = _available_cols(lf.collect_schema().names(), required_cols)
    return lf.select(available).collect()


# Mark for deletion
def load_csv(path: Path, required_cols: list[str]) -> pl.DataFrame:
    """Fallback for users still on local CSV files."""
    log.info(f"Loading CSV {path}")
    lf = pl.scan_csv(path, infer_schema_length=10_000)
    # Normalise column names
    lf = lf.rename(
        {c: c.strip().lower().replace(" ", "_") for c in lf.collect_schema().names()}
    )
    available = _available_cols(lf.collect_schema().names(), required_cols)
    return lf.select(available).collect()


# ── Crash cleaning ────────────────────────────────────────────────────────────
# Good crashes cleaning
def clean_crashes(df: pl.DataFrame) -> pl.DataFrame:
    log.info("Cleaning crashes …")
    n_start = len(df)

    df = (
        df
        # Cast numeric columns — coerce bad strings to null
        .with_columns(
            [
                pl.col("collision_id").cast(pl.Int64, strict=False),
                pl.col("latitude").cast(pl.Float64, strict=False),
                pl.col("longitude").cast(pl.Float64, strict=False),
                pl.col("number_of_persons_injured")
                .cast(pl.Int32, strict=False)
                .fill_null(0),
                pl.col("number_of_persons_killed")
                .cast(pl.Int32, strict=False)
                .fill_null(0),
                # Parse crash_date to Date
                pl.col("crash_date")
                .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f", strict=False)
                .dt.date()
                .alias("crash_date"),
            ]
        )
        # Drop rows with no location (notebook 01: dropna on LOCATION)
        .filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
        # NYC bounding box — notebook 01 exact bounds
        .filter(
            pl.col("latitude").is_between(LAT_MIN, LAT_MAX)
            & pl.col("longitude").is_between(LON_MIN, LON_MAX)
        )
        # Drop null collision IDs and dates
        .filter(
            pl.col("collision_id").is_not_null() & pl.col("crash_date").is_not_null()
        )
        # Deduplicate
        .unique(subset=["collision_id"], keep="last")
    )

    # ── Contributing factors ──────────────────────────────────────────────────
    # Drop rows where BOTH factors are null (notebook 01 exact)
    df = df.filter(
        pl.col("contributing_factor_vehicle_1").is_not_null()
        & pl.col("contributing_factor_vehicle_2").is_not_null()
    )

    # Drop rows where both are "Unspecified" (notebook 01 exact OR logic)
    df = df.filter(
        ~(
            (pl.col("contributing_factor_vehicle_1") == "Unspecified")
            & (pl.col("contributing_factor_vehicle_2") == "Unspecified")
        )
    )

    # Drop junk codes (notebook 01 exact)
    df = df.filter(
        ~pl.col("contributing_factor_vehicle_1").is_in(["80", "1"])
        & ~pl.col("contributing_factor_vehicle_2").is_in(["80", "1"])
    )

    # Apply typo/casing renames
    df = df.with_columns(
        [
            pl.col("contributing_factor_vehicle_1").replace(FACTOR_RENAME),
            pl.col("contributing_factor_vehicle_2").replace(FACTOR_RENAME),
        ]
    )

    # Drop location column — borough will come from shapefile join
    if "location" in df.columns:
        df = df.drop("location")

    log.info(f"  Crashes: {n_start:,} -> {len(df):,} rows")
    return df


# ── Spatial borough join (geopandas)  ────────


def assign_borough(df: pl.DataFrame, shp_path: Path) -> pl.DataFrame:
    """
    Spatially joins each point to the NYC borough shapefile.
    Mirrors notebook 01's geopandas sjoin exactly.
    We use pandas/geopandas only for this step, then convert straight back.
    """
    if not shp_path.exists():
        log.warning(
            f"Shapefile not found at {shp_path} — borough column will be 'UNKNOWN'"
        )
        return df.with_columns(pl.lit("UNKNOWN").alias("borough"))

    log.info("Running spatial borough join (geopandas) …")

    # Convert to pandas for geopandas — only lat/lon + collision_id needed
    pdf = df.select(["collision_id", "latitude", "longitude"]).to_pandas()

    gdf_points = gpd.GeoDataFrame(
        pdf,
        geometry=pdf.apply(lambda r: Point(r["longitude"], r["latitude"]), axis=1),
        crs="EPSG:4326",
    )

    boroughs = gpd.read_file(shp_path)
    if boroughs.crs is None:
        boroughs = boroughs.set_crs(epsg=2263)
    boroughs = boroughs.to_crs(epsg=4326)

    joined = gpd.sjoin(gdf_points, boroughs, how="inner", predicate="within")
    joined = joined.drop(
        columns=["geometry", "index_right", "boro_code", "shape_area", "shape_leng"],
        errors="ignore",
    )
    joined = joined.rename(columns={"boro_name": "borough"})
    joined["borough"] = joined["borough"].str.upper()

    # Convert borough mapping back and join to Polars df
    borough_map = pl.from_pandas(joined[["collision_id", "borough"]])
    result = df.join(borough_map, on="collision_id", how="inner")

    log.info(f"  Borough join: {len(df):,} -> {len(result):,} rows")
    return result


# ── Person cleaning ───────────────────────────────────────────────────────────
# Good person cleaning
def clean_person(df: pl.DataFrame) -> pl.DataFrame:
    log.info("Cleaning person records …")
    n_start = len(df)

    df = df.with_columns(pl.col("collision_id").cast(pl.Int64, strict=False)).filter(
        pl.col("collision_id").is_not_null()
    )

    # Fill missing age with dataset mean (notebook 01 approach)
    df = df.with_columns(
        pl.col("person_age").cast(pl.Float64, strict=False).alias("person_age")
    )
    avg_age = df["person_age"].mean()
    df = df.with_columns(
        pl.col("person_age").fill_null(avg_age).round(0).cast(pl.Int32)
    )

    # Sex mapping (notebook 01 exactly: M→Male, F→Female, U→Unknown)
    df = df.with_columns(
        pl.col("person_sex")
        .replace({"M": "Male", "F": "Female", "U": "Unknown"})
        .fill_null("Unknown")
    )

    # Drop rows where bodily injury is "Unknown" (notebook 01 exactly)
    if "bodily_injury" in df.columns:
        df = df.filter(pl.col("bodily_injury") != "Unknown")

    # Fill remaining nulls and normalise "Does Not Apply"
    fill_map = {
        "person_type": "Unknown",
        "person_sex": "Unknown",
    }
    fill_exprs = [
        pl.col(c).fill_null(v) for c, v in fill_map.items() if c in df.columns
    ]
    if fill_exprs:
        df = df.with_columns(fill_exprs)

    # Replace "Does Not Apply" → "Unknown" across all string columns
    str_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8]
    df = df.with_columns(
        [pl.col(c).replace({"Does Not Apply": "Unknown"}) for c in str_cols]
    )

    log.info(f"  Person: {n_start:,} -> {len(df):,} rows")
    return df


# ── Merge (inner join — notebook 01 exact approach) ───────────────────────────


def merge_datasets(crashes: pl.DataFrame, person: pl.DataFrame) -> pl.DataFrame:
    """
    Inner join on collision_id — one row per person, same as notebook 01.
    The model step will sample 50k per borough from this merged frame.
    """
    log.info("Merging crashes + person (inner join on collision_id) …")
    merged = crashes.join(person, on="collision_id", how="inner")
    log.info(f"  Merged shape: {merged.shape}")
    return merged


# ── Feature engineering ───────────────────────────────────────────────────────


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    log.info("Engineering features …")

    # ── Accident severity (notebook 01 exact thresholds) ─────────────────────
    df = df.with_columns(
        pl.when(pl.col("number_of_persons_killed") > 0)
        .then(pl.lit("Fatal"))
        .when(pl.col("number_of_persons_injured") >= 3)
        .then(pl.lit("Major Injury"))
        .when(pl.col("number_of_persons_injured") > 0)
        .then(pl.lit("Minor Injury"))
        .otherwise(pl.lit("No Injury"))
        .alias("accident_severity")
    )

    # ── Date/time features (notebook 02 exact logic) ──────────────────────────
    df = df.with_columns(pl.col("crash_date").cast(pl.Date, strict=False))
    df = df.with_columns(
        [
            pl.col("crash_date").dt.year().alias("year"),
            pl.col("crash_date").dt.month().alias("month"),
            pl.col("crash_date")
            .dt.weekday()
            .alias("day_of_week"),  # Mon=1, Sun=7 in Polars
            pl.col("crash_date").dt.to_string("%A").alias("day_name"),
        ]
    )

    # is_weekend: Polars weekday() is 1=Mon … 7=Sun, so weekend = 6 or 7
    df = df.with_columns(
        pl.col("day_of_week").is_in([6, 7]).cast(pl.Int8).alias("is_weekend")
    )

    # Season (notebook 02 quarter logic, expressed as labels)
    df = df.with_columns(
        pl.when(pl.col("month") <= 3)
        .then(pl.lit("Winter"))
        .when(pl.col("month") <= 6)
        .then(pl.lit("Spring"))
        .when(pl.col("month") <= 9)
        .then(pl.lit("Summer"))
        .otherwise(pl.lit("Fall"))
        .alias("season")
    )

    # Parse hour from crash_time string ("HH:MM")
    if "crash_time" in df.columns:
        df = df.with_columns(
            pl.col("crash_time")
            .str.slice(0, 2)
            .cast(pl.Int8, strict=False)
            .alias("hour")
        )
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Int8).alias("hour"))

    # Time of day bucket
    df = df.with_columns(
        pl.when(pl.col("hour") < 6)
        .then(pl.lit("Late Night"))
        .when(pl.col("hour") < 10)
        .then(pl.lit("Morning Rush"))
        .when(pl.col("hour") < 16)
        .then(pl.lit("Midday"))
        .when(pl.col("hour") < 20)
        .then(pl.lit("Evening Rush"))
        .otherwise(pl.lit("Night"))
        .alias("time_of_day")
    )

    # Rush hour flag
    df = df.with_columns(
        (pl.col("hour").is_in(list(range(7, 10)) + list(range(16, 20))))
        .cast(pl.Int8)
        .alias("is_rush_hour")
    )

    log.info(f"  Final shape: {df.shape}")
    log.info(
        f"  Severity distribution:\n"
        f"{df.group_by('accident_severity').len().sort('len', descending=True)}"
    )
    return df


# ── Quality report ────────────────────────────────────────────────────────────


def log_quality_report(df: pl.DataFrame):
    log.info("── Quality report ───────────────────────────────────────")
    log.info(f"  Total rows   : {len(df):,}")
    if "crash_date" in df.columns:
        log.info(
            f"  Date range   : {df['crash_date'].min()} → {df['crash_date'].max()}"
        )
    if "borough" in df.columns:
        log.info(
            f"  Borough dist :\n"
            f"{df.group_by('borough').len().sort('len', descending=True)}"
        )
    null_pct = (
        pl.DataFrame(
            {
                "column": df.columns,
                "null_pct": [df[c].null_count() / len(df) * 100 for c in df.columns],
            }
        )
        .filter(pl.col("null_pct") > 20)
        .sort("null_pct", descending=True)
    )
    if len(null_pct) > 0:
        log.warning(f"  Columns >20% null:\n{null_pct}")
    log.info("─────────────────────────────────────────────────────────")


def save_colliision_by_borough(df: pl.DataFrame):

    for borough in VALID_BOROUGHS:
        output_path = PROCESSED_DIR / borough
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_path / f"collisions_{borough.lower().replace(' ', '_')}.parquet"
        )
        temp_df = df.filter(pl.col("borough") == borough)
        temp_df.write_parquet(output_path, compression="snappy")


# ── Main ──────────────────────────────────────────────────────────────────────


def run_clean(
    crashes_path: Path = RAW_DIR / "crashes.parquet",
    person_path: Path = RAW_DIR / "person.parquet",
    borough_shp: Path = BOROUGH_SHP,
) -> pl.DataFrame:

    # Crashes — parquet first, CSV fallback
    if crashes_path.exists():
        crashes = load_parquet(crashes_path, CRASH_COLS)
    else:
        csv_fallback = Path(
            "data_files/raw_data/Motor_Vehicle_Collisions_-_Crashes.csv"
        )
        if csv_fallback.exists():
            log.info("Parquet not found — falling back to local CSV")
            crashes = load_csv(csv_fallback, CRASH_COLS)
        else:
            raise FileNotFoundError(
                f"{crashes_path} not found. Run fetch.py first, or place "
                f"the raw CSV at {csv_fallback}"
            )
    crashes = clean_crashes(crashes)
    crashes = assign_borough(crashes, borough_shp)

    # Person — parquet first, CSV fallback
    if person_path.exists():
        person = load_parquet(person_path, PERSON_COLS)
        person = clean_person(person)
        df = merge_datasets(crashes, person)
    else:
        csv_fallback = Path("data_files/raw_data/Motor_Vehicle_Collisions_-_Person.csv")
        if csv_fallback.exists():
            person = load_csv(csv_fallback, PERSON_COLS)
            person = clean_person(person)
            df = merge_datasets(crashes, person)
        else:
            log.warning("No person data found — using crash-only dataset")
            df = crashes

    df = engineer_features(df)
    log_quality_report(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = PROCESSED_DIR / "collisions.parquet"
    df.write_parquet(OUTPUT_PATH, compression="snappy")

    # Save the Borough specific collisions
    save_colliision_by_borough(df)

    log.info(f"Saved → {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")

    return df


if __name__ == "__main__":
    run_clean()
