"""
main.py — Queens Vehicle Collision Dashboard
Run with: streamlit run app/main.py
"""

import json
import joblib
import streamlit as st
import polars as pl
from pathlib import Path
import os

st.set_page_config(
    page_title="Queens Collision Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed" / "QUEENS"
MODELS_DIR    = ROOT / "models" / "QUEENS"

DATA_PATH         = PROCESSED_DIR / "collisions_queens.parquet"
FEATURE_META_PATH = PROCESSED_DIR / "feature_meta.json"
TRAIN_META_PATH   = MODELS_DIR    / "train_meta.json"
MODEL_PATH        = MODELS_DIR    / "severity_model.joblib"
PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"

# ── R2 artifact loader ────────────────────────────────────────────────────────

def _r2_client():
    """Build a boto3 S3 client pointed at Cloudflare R2."""
    import boto3
    from botocore.config import Config
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def _pull_from_r2(r2_key: str, local_path: Path) -> None:
    """Download a single artifact from R2 to local_path if not already present."""
    if local_path.exists():
        return
    bucket = os.environ.get("R2_BUCKET_NAME", "nyc-collisions")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    st.toast(f"Downloading {local_path.name} from R2…", icon="⬇️")
    _r2_client().download_file(bucket, r2_key, str(local_path))

def _ensure_artifacts() -> None:
    """Pull all required artifacts from R2 if running in cloud (no local files)."""
    # Skip if all files already exist locally (local dev)
    if all(p.exists() for p in [DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH,
                                  FEATURE_META_PATH, TRAIN_META_PATH]):
        return

    # Only attempt R2 pull if credentials are present
    if not os.environ.get("R2_ACCOUNT_ID"):
        st.error(
            "Artifact files not found locally and R2 credentials are not set. "
            "Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME "
            "as environment variables.",
            icon="🚨",
        )
        st.stop()

    artifacts = [
        ("processed/QUEENS/collisions_queens.parquet", DATA_PATH),
        ("processed/QUEENS/feature_meta.json",         FEATURE_META_PATH),
        ("processed/QUEENS/preprocessor.joblib",       PREPROCESSOR_PATH),
        ("models/QUEENS/severity_model.joblib",        MODEL_PATH),
        ("models/QUEENS/train_meta.json",              TRAIN_META_PATH),
    ]
    for r2_key, local_path in artifacts:
        _pull_from_r2(r2_key, local_path)

# Pull artifacts before anything else runs
_ensure_artifacts()


# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pl.DataFrame:
    if not DATA_PATH.exists():
        full_path = ROOT / "data" / "processed" / "collisions.parquet"
        return pl.read_parquet(full_path).filter(pl.col("borough") == "QUEENS")
    return pl.read_parquet(DATA_PATH)

@st.cache_data
def load_feature_meta() -> dict:
    with open(FEATURE_META_PATH) as f:
        return json.load(f)

@st.cache_data
def load_train_meta() -> dict:
    with open(TRAIN_META_PATH) as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


pages_dir = Path(__file__).parent / "pages"

pg = st.navigation(
    {
        "Analysis": [
            st.Page(str(pages_dir / "1_overview.py"),
                    title="Overview",
                    icon="📊",
                    default=True),
            st.Page(str(pages_dir / "2_hotspot_map.py"),
                    title="Hotspot Map",
                    icon="🗺️"),
            st.Page(str(pages_dir / "3_contributing_factors.py"),
                    title="Contributing Factors",
                    icon="🔍"),
        ],
        "Model": [
            st.Page(str(pages_dir / "4_severity_predictor.py"),
                    title="Severity Predictor",
                    icon="🤖"),
        ],
    }
)

pg.run()