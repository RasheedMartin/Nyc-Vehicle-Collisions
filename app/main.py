"""
main.py — Queens Vehicle Collision Dashboard
Run with: streamlit run app/main.py

Data strategy:
  - Parquet is cached on a Railway Volume (/data) and shared across all users.
  - On startup, we check R2's Last-Modified timestamp against the local file.
    If R2 is newer (i.e. Airflow has run a fresh pipeline), we re-download.
  - Model/preprocessor are NOT loaded here — all inference goes through the API.
  - train_meta and feature_meta are fetched from the API's /meta endpoint.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import polars as pl
import streamlit as st
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

st.set_page_config(
    page_title="Queens Collision Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────

# Railway Volume mount path (set RAILWAY_VOLUME_MOUNT_PATH=/data in Railway env)
VOLUME_DIR = Path(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/data"))
DATA_PATH = VOLUME_DIR / "collisions_queens.parquet"

# R2 key for the parquet
R2_PARQUET_KEY = "processed/QUEENS/collisions_queens.parquet"

# How old (in seconds) the local file can be before we check R2 for a newer version.
# Default: 6 days — slightly less than the weekly pipeline schedule.
CACHE_TTL_SECONDS = int(os.environ.get("PARQUET_CACHE_TTL", 6 * 24 * 3600))


# ── R2 client ─────────────────────────────────────────────────────────────────


def _r2_client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


# ── Volume cache with R2 TTL check ────────────────────────────────────────────


def _local_mtime() -> datetime | None:
    """Return the local file's mtime as a timezone-aware UTC datetime, or None."""
    if not DATA_PATH.exists():
        return None
    ts = DATA_PATH.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _r2_last_fetch_date(client) -> datetime | None:
    """
    Read last_fetch_date from raw/.last_fetch.json in R2.
    This is written by fetch.py after every successful Airflow run,
    so it reflects when new data was actually pulled from NYC Open Data.
    """
    bucket = os.environ.get("R2_BUCKET_NAME", "nyc-collisions")
    try:
        resp = client.get_object(Bucket=bucket, Key="raw/.last_fetch.json")
        meta = json.loads(resp["Body"].read().decode())
        return datetime.fromisoformat(meta["last_fetch_date"]).replace(
            tzinfo=timezone.utc
        )
    except Exception as e:
        log.warning(f"Could not read .last_fetch.json from R2: {e}")
        return None


def _needs_refresh(client) -> bool:
    """
    Return True if we should re-download the parquet from R2.

    Rules:
      1. File doesn't exist locally → always download.
      2. File age is within CACHE_TTL_SECONDS → use cache, skip all R2 checks.
      3. File is past TTL → read raw/.last_fetch.json from R2 (tiny request).
         If Airflow has fetched new data since the parquet was last downloaded
         → re-download. Otherwise keep serving the cached file.
    """
    local_mtime = _local_mtime()

    # Rule 1 — no local file
    if local_mtime is None:
        log.info("No local parquet found — downloading from R2")
        return True

    age_seconds = time.time() - local_mtime.timestamp()

    # Rule 2 — within TTL, don't even hit R2
    if age_seconds < CACHE_TTL_SECONDS:
        log.info(
            f"Local parquet is {age_seconds / 3600:.1f}h old — within TTL, using cache"
        )
        return False

    # Rule 3 — past TTL, check .last_fetch.json
    log.info(
        f"Local parquet is {age_seconds / 3600:.1f}h old — checking .last_fetch.json"
    )
    last_fetch = _r2_last_fetch_date(client)

    if last_fetch is None:
        # Can't read metadata — serve stale file rather than failing
        log.warning("Could not read last_fetch.json — serving existing cache")
        return False

    if last_fetch > local_mtime:
        log.info(
            f"Airflow fetched new data at {last_fetch.isoformat()} "
            f"(parquet last downloaded {local_mtime.isoformat()}) — refreshing"
        )
        return True

    log.info(
        f"No new data since last download "
        f"(last fetch: {last_fetch.isoformat()}) — using cache"
    )
    return False


def _download_parquet(client) -> None:
    """Download the parquet from R2 to the Volume path."""
    bucket = os.environ.get("R2_BUCKET_NAME", "nyc-collisions")
    VOLUME_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {R2_PARQUET_KEY} → {DATA_PATH}")
    client.download_file(bucket, R2_PARQUET_KEY, str(DATA_PATH))
    log.info(f"Download complete: {DATA_PATH.stat().st_size:,} bytes")


def _ensure_parquet() -> None:
    """Download or refresh the parquet on the Volume if needed."""
    if not os.environ.get("R2_ACCOUNT_ID"):
        st.error(
            "R2 credentials not set. Please set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY, and R2_BUCKET_NAME as environment variables.",
            icon="🚨",
        )
        st.stop()

    client = _r2_client()
    if _needs_refresh(client):
        _download_parquet(client)


# ── Loaders ───────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading collision data…")
def load_data() -> pl.DataFrame:
    """
    Load the Queens collision parquet from the Railway Volume.
    Cached for the lifetime of the Streamlit process — all users share this.
    """
    _ensure_parquet()
    log.info(f"Reading parquet from {DATA_PATH}")
    df = pl.read_parquet(DATA_PATH)
    log.info(f"Loaded {len(df):,} rows")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_train_meta() -> dict:
    """
    Fetch train metadata from the inference API's /meta endpoint.
    Cached for 1 hour — used by 6_pipeline.py.
    """
    import requests

    api_url = os.environ.get("INFERENCE_API_URL", "http://localhost:8000").rstrip("/")
    api_key = os.environ.get("API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}

    try:
        resp = requests.get(f"{api_url}/meta", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Return in the shape the pipeline page expects
        return {
            "test_metrics": data["test_metrics"],
            "cv_f1_mean": data["cv_f1_mean"],
            "cv_f1_std": data["cv_f1_std"],
            "n_train": data["n_train"],
            "n_test": data["n_test"],
            "trained_at": data["trained_at"],
            "feature_importances": data["feature_importances"],
            "n_features": len(data["feature_importances"]),
            # xgboost_params not in /meta — add to API if needed
            "xgboost_params": {},
            "borough": "QUEENS",
        }
    except Exception as e:
        log.warning(f"Could not fetch train meta from API: {e}")
        return {
            "test_metrics": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0},
            "cv_f1_mean": 0,
            "cv_f1_std": 0,
            "n_train": 0,
            "n_test": 0,
            "trained_at": "—",
            "feature_importances": {},
            "n_features": 0,
            "xgboost_params": {},
            "borough": "QUEENS",
        }


# ── Navigation ────────────────────────────────────────────────────────────────

pages_dir = Path(__file__).parent / "pages"

pg = st.navigation(
    {
        "Analysis": [
            st.Page(
                str(pages_dir / "1_overview.py"),
                title="Overview",
                icon="📊",
                default=True,
            ),
            st.Page(str(pages_dir / "2_hotspot_map.py"), title="Hotspot Map", icon="🗺️"),
            st.Page(
                str(pages_dir / "3_contributing_factors.py"),
                title="Contributing Factors",
                icon="🔍",
            ),
            st.Page(str(pages_dir / "4_trends.py"), title="Trends", icon="📈"),
        ],
        "Model": [
            st.Page(
                str(pages_dir / "5_severity_predictor.py"),
                title="Severity Predictor",
                icon="🤖",
            ),
            st.Page(str(pages_dir / "6_pipeline.py"), title="Data Pipeline", icon="⚙️"),
        ],
    }
)

pg.run()