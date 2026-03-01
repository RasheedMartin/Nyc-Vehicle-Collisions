"""
Transforms the cleaned collisions dataset into a model-ready feature matrix.

Supports an optional --borough filter so models can be trained on a single
borough for focused analysis (e.g. Queens). All output artifacts are namespaced
by borough so multiple borough models can coexist.

Encoding strategy:
  - OneHotEncoder  → nominal categories (contributing factor, person_type, etc.)
  - OrdinalEncoder → ordered categories (time_of_day, season)
  - Pass-through   → numeric features (age, is_weekend, month, etc.)

NOTE: number_of_persons_injured and number_of_persons_killed are intentionally
excluded — severity is derived directly from them which causes data leakage.

Output (borough=QUEENS example):
  data/processed/QUEENS/features.parquet
  data/processed/QUEENS/preprocessor.joblib
  data/processed/QUEENS/feature_meta.json

Pipeline order:
  fetch.py  →  clean.py  →  features.py [--borough QUEENS]  →  train.py [--borough QUEENS]
"""

import json
import logging
import argparse
import joblib
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR      = Path("data/processed")
INPUT_PATH         = PROCESSED_DIR / "collisions.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Feature definitions ───────────────────────────────────────────────────────

# Borough is only included as a feature when training across ALL boroughs
BOROUGH_FEATURE = "borough"

# Top N street names to keep individually — the rest are collapsed to "Other"
# Keeps OHE cardinality manageable (~100 binary columns) while preserving the
# highest-volume corridors that drive the most statistical signal.
TOP_STREETS_N = 100

# Nominal categoricals — no order between values
# OneHotEncoder: borough=BROOKLYN gets its own binary column,
# completely independent of borough=MANHATTAN
NOMINAL_FEATURES = [
    "on_street_name",   # location feature: top-N streets + "Other"
]

# Ordered categoricals — a real sequence exists between values
# OrdinalEncoder: model can learn that Evening Rush > Midday numerically
ORDERED_FEATURES = {
    "time_of_day": ["Late Night", "Morning Rush", "Midday", "Evening Rush", "Night"],
    "season":      ["Winter", "Spring", "Summer", "Fall"],
}

# Numeric — passed through as-is, no encoding needed
NUMERIC_FEATURES = [
    "is_weekend",
    "is_rush_hour",
    "month",
    "day_of_week",
]

# Target
TARGET_COL     = "accident_severity"
SEVERITY_ORDER = ["No Injury", "Minor Injury", "Major Injury", "Fatal"]
SEVERITY_MAP   = {label: i for i, label in enumerate(SEVERITY_ORDER)}
VALID_BOROUGHS = {"MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def output_dir(borough: str | None) -> Path:
    """Borough-namespaced output directory."""
    slug = borough.upper().replace(" ", "_") if borough else "ALL"
    return PROCESSED_DIR / slug

def select_available(df: pl.DataFrame, cols: list[str], label: str) -> list[str]:
    available = [c for c in cols if c in df.columns]
    missing   = set(cols) - set(available)
    if missing:
        log.warning(f"  {label} — columns not found (skipped): {missing}")
    return available


def fill_nulls(df: pl.DataFrame, cat_cols: list[str], num_cols: list[str]) -> pl.DataFrame:
    cat_exprs = [pl.col(c).fill_null("Unknown") for c in cat_cols if c in df.columns]
    num_exprs = [pl.col(c).fill_null(0)         for c in num_cols if c in df.columns]
    exprs = cat_exprs + num_exprs
    return df.with_columns(exprs) if exprs else df


def cap_street_names(df: pl.DataFrame, n: int = TOP_STREETS_N) -> pl.DataFrame:
    """
    Normalise casing then replace streets outside the top-N by crash volume with 'Other'.
    Keeps OHE cardinality manageable while preserving the highest-signal corridors.
    Null / blank street names are left as-is (fill_nulls handles them downstream).
    """
    df = df.with_columns(
        pl.col("on_street_name").str.to_uppercase().str.strip_chars()
    )
    top = (
        df.filter(pl.col("on_street_name").is_not_null() & (pl.col("on_street_name") != ""))
          .group_by("on_street_name").len()
          .sort("len", descending=True).head(n)
          ["on_street_name"].to_list()
    )
    return df.with_columns(
        pl.when(pl.col("on_street_name").is_in(top))
          .then(pl.col("on_street_name"))
          .otherwise(pl.lit("Other"))
          .alias("on_street_name")
    )


def encode_target(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col(TARGET_COL)
          .replace(SEVERITY_MAP)
          .cast(pl.Int8)
          .alias("severity_encoded")
    )


def build_category_catalogue(df: pl.DataFrame, nominal_cols: list[str]) -> dict:
    """
    Record unique sorted values per nominal feature.
    Used by the Streamlit app to populate select boxes — values always
    match what the encoder was trained on.
    """
    catalogue = {}
    for col in nominal_cols:
        if col in df.columns:
            catalogue[col] = df[col].drop_nulls().unique().sort().to_list()
    # Also record the ordered categories
    for col, order in ORDERED_FEATURES.items():
        catalogue[col] = order
    return catalogue


# ── Build preprocessor ────────────────────────────────────────────────────────

def build_preprocessor(
    nominal_cols: list[str],
    ordered_cols: list[str],
    numeric_cols: list[str],
    ordered_categories: dict,
) -> ColumnTransformer:
    """
    ColumnTransformer that applies the correct encoding to each group:

      Nominal  → OneHotEncoder
                 handle_unknown='ignore' drops unseen categories at inference
                 (produces a zero vector — safe for XGBoost)
                 sparse_output=False keeps everything as dense numpy arrays

      Ordered  → OrdinalEncoder with explicit category order
                 handle_unknown='use_encoded_value' returns -1 for unseen values

      Numeric  → passthrough (no transformation)
    """
    transformers = []

    if nominal_cols:
        transformers.append((
            "nominal",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            ),
            nominal_cols,
        ))

    if ordered_cols:
        categories = [ordered_categories[c] for c in ordered_cols]
        transformers.append((
            "ordered",
            OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32,
            ),
            ordered_cols,
        ))

    if numeric_cols:
        transformers.append((
            "numeric",
            "passthrough",
            numeric_cols,
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract output column names from the fitted ColumnTransformer."""
    names = []
    for _, transformer, cols in preprocessor.transformers_:
        if transformer == "passthrough":
            names.extend(cols)
        elif hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(cols).tolist())
        else:
            names.extend(cols)
    return names


# ── Main ──────────────────────────────────────────────────────────────────────

def run_features(
        borough: str | None = None,
        input_path: Path = INPUT_PATH
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    
    log.info(f"Loading {input_path}")
    df = pl.read_parquet(input_path)
    log.info(f"  {len(df):,} rows, {len(df.columns)} columns")

# ── Borough filter ────────────────────────────────────────────────────────
    if borough:
        borough = borough.upper()
        if borough not in VALID_BOROUGHS:
            raise ValueError(f"Invalid borough '{borough}'. Choose from: {VALID_BOROUGHS}")
        df = df.filter(pl.col("borough") == borough)
        log.info(f"  Filtered to {borough}: {len(df):,} rows")
        # Borough is now constant — exclude it; street name carries the location signal
        nominal_cols = NOMINAL_FEATURES.copy()
    else:
        log.info("  No borough filter — training on all boroughs")
        # Include borough as a feature so the model can distinguish boroughs
        nominal_cols = NOMINAL_FEATURES + [BOROUGH_FEATURE]

    if len(df) == 0:
        raise ValueError(f"No rows remaining after filtering to borough='{borough}'")

    # Deduplicate to one row per crash.
    # The cleaned parquet has one row per person-in-crash; since we no longer use
    # person-level features (type, sex, age) we must collapse to crash-level to
    # avoid identical duplicate rows inflating the training set.
    before_dedup = len(df)
    df = df.unique(subset=["collision_id"])
    log.info(f"  Deduplicated to crash level: {before_dedup:,} → {len(df):,} rows")

    # Cap on_street_name cardinality before encoding
    if "on_street_name" in df.columns:
        df = cap_street_names(df)
        log.info(f"  Street names capped to top-{TOP_STREETS_N} + 'Other'")

    # Drop rows with no target
    before = len(df)
    df = df.filter(pl.col(TARGET_COL).is_not_null())
    if len(df) < before:
        log.warning(f"  Dropped {before - len(df):,} rows with null target")

    # Resolve which feature columns exist in this dataset
    nominal_cols = select_available(df, nominal_cols, "nominal")
    ordered_cols = select_available(df, list(ORDERED_FEATURES.keys()), "ordered")
    numeric_cols = select_available(df, NUMERIC_FEATURES, "numeric")
    all_input_cols = nominal_cols + ordered_cols + numeric_cols

    # Fill nulls before handing to sklearn
    df = fill_nulls(df, nominal_cols + ordered_cols, numeric_cols)
    df = encode_target(df)

    log.info(
        f"  Feature groups — "
        f"nominal: {len(nominal_cols)}, "
        f"ordered: {len(ordered_cols)}, "
        f"numeric: {len(numeric_cols)}"
    )
    log.info(
        f"  Severity distribution:\n"
        f"{df.group_by(TARGET_COL).len().sort('len', descending=True)}"
    )

    # Convert to pandas for sklearn
    pdf = df.select(all_input_cols + [TARGET_COL, "severity_encoded", "collision_id"]).to_pandas()

    X_raw = pdf[all_input_cols]
    y     = pdf["severity_encoded"].to_numpy().astype(np.int8)

    # Build, fit, apply preprocessor
    preprocessor = build_preprocessor(
        nominal_cols=nominal_cols,
        ordered_cols=ordered_cols,
        numeric_cols=numeric_cols,
        ordered_categories=ORDERED_FEATURES,
    )
    X = preprocessor.fit_transform(X_raw).astype(np.float32)
    feature_names = get_feature_names(preprocessor)

    log.info(f"  Encoded feature matrix: {X.shape[0]:,} rows × {X.shape[1]} columns")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = output_dir(borough)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path     = out_dir / "features.parquet"
    preprocessor_path = out_dir / "preprocessor.joblib"
    meta_path         = out_dir / "feature_meta.json"

    features_df = pl.DataFrame(X, schema=feature_names)
    features_df = pl.concat([
        features_df,
        pl.from_pandas(pdf[["collision_id", TARGET_COL, "severity_encoded"]]),
    ], how="horizontal")
    features_df.write_parquet(features_path, compression="snappy")
    log.info(f"Saved features -> {features_path}")

    joblib.dump(preprocessor, preprocessor_path)
    log.info(f"Saved preprocessor -> {preprocessor_path}")

    catalogue = build_category_catalogue(df, nominal_cols)
    meta = {
        "borough":               borough or "ALL",
        "nominal_features":      nominal_cols,
        "ordered_features":      ordered_cols,
        "ordered_categories":    ORDERED_FEATURES,
        "numeric_features":      numeric_cols,
        "encoded_feature_names": feature_names,
        "target_col":            TARGET_COL,
        "severity_order":        SEVERITY_ORDER,
        "severity_map":          SEVERITY_MAP,
        "severity_inverse":      {str(v): k for k, v in SEVERITY_MAP.items()},
        "category_catalogue":    catalogue,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Saved feature meta  -> {meta_path}")

    log.info(f"Feature engineering complete!  [{borough or 'ALL'}]")
    return X, y, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature matrix for model training")
    parser.add_argument(
        "--borough",
        type=str,
        default=None,
        help="Filter to a single borough (e.g. QUEENS). Omit to train on all boroughs.",
    )
    args = parser.parse_args()
    run_features(borough=args.borough)