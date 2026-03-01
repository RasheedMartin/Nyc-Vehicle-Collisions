"""
fetch.py
--------
Data Pipeline to Fetch NYC Motor Vehicle Collision data from the NYC Open Data Socrata API.

Datasets:
  - Crashes : https://data.cityofnewyork.us/resource/h9gi-nx95.json
  - Person  : https://data.cityofnewyork.us/resource/f55k-p6yu.json

Usage:
  # Full pull (first time)
  python fetch.py --mode full

  # Incremental pull (scheduled retraining — only new records since last run)
  python fetch.py --mode incremental

  # Pull a specific date range
  python fetch.py --mode range --start 2024-01-01 --end 2024-12-31

Environment variables (optional but recommended for higher rate limits):
  NYC_OPEN_DATA_APP_TOKEN  — register free at https://data.cityofnewyork.us/profile/app_tokens
"""

import os
import json
import logging
import argparse
import requests
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

from typing import Tuple

# Load all environment variables
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL   = "https://data.cityofnewyork.us/resource"
CRASHES_ID = "h9gi-nx95"
PERSON_ID  = "f55k-p6yu"
PAGE_SIZE  = 50_000
RAW_DIR    = Path("data/raw")
META_FILE  = RAW_DIR / ".last_fetch.json"

logging.basicConfig(
    filename="log/fetch.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Socrata client ────────────────────────────────────────────────────────────

class SocrataClient:
    """Thin wrapper around the Socrata JSON API with pagination + token support."""

    def __init__(self, app_token: str | None = None):
        self.app_token = app_token or os.getenv("NYC_OPEN_DATA_APP_TOKEN")
        self.session = requests.Session()
        if self.app_token:
            self.session.headers.update({"X-App-Token": self.app_token})
            log.info("Socrata app token loaded!")
        else:
            log.warning(
                "No app token — requests will be throttled. "
                "Set NYC_OPEN_DATA_APP_TOKEN to remove limits."
            )

    def fetch_dataset(
        self,
        dataset_id: str,
        date_col: str,
        since: str | None = None,
        until: str | None = None,
    ) -> pl.DataFrame:
        """
        Fetch a full dataset or date-filtered slice via paginated SoQL queries.

        Args:
            dataset_id : Socrata 4x4 dataset identifier
            date_col   : Name of the date column to filter on
            since      : ISO date string lower bound  e.g. '2024-01-01'
            until      : ISO date string upper bound  e.g. '2024-12-31'

        Returns:
            pl.DataFrame with all matching rows
        """
        url    = f"{BASE_URL}/{dataset_id}.json"
        frames = []
        offset = 0

        where_parts = []
        if since:
            where_parts.append(f"{date_col} >= '{since}T00:00:00'")
        if until:
            where_parts.append(f"{date_col} <= '{until}T23:59:59'")
        where = " AND ".join(where_parts) if where_parts else None

        log.info(f"Fetching dataset {dataset_id}  filter={where or 'none'}")

        while True:
            params: dict = {
                "$limit":  PAGE_SIZE,
                "$offset": offset,
                "$order":  f"{date_col} ASC",
            }
            if where:
                params["$where"] = where

            resp = self.session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            batch = resp.json()

            if not batch:
                break

            # Socrata returns a list of dicts — read directly into Polars
            frames.append(pl.DataFrame(batch))
            log.info(f"  offset={offset:>8,}  rows_fetched={len(batch):>6,}")
            offset += PAGE_SIZE

            if len(batch) < PAGE_SIZE:
                break  # last page

        if not frames:
            log.warning(f"No data returned for dataset {dataset_id}")
            return pl.DataFrame()

        normalized_frames = normalize_frames(frames)
        df = pl.concat(normalized_frames, how="vertical")  # now all columns match

        # df = pl.concat(frames, how="diagonal")  # diagonal handles mismatched schemas across pages
        log.info(f"Dataset {dataset_id}: {len(df):,} total rows fetched")
        return df


# ── Metadata helpers ──────────────────────────────────────────────────────────

def load_meta() -> dict:
    if META_FILE.exists():
        with open(META_FILE) as f:
            return json.load(f)
    return {}


def save_meta(meta: dict):
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


# ── Save / merge helpers ──────────────────────────────────────────────────────

def save_parquet(df: pl.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="snappy")
    log.info(f"Saved -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def merge_with_existing(new_df: pl.DataFrame, path: Path, id_col: str) -> pl.DataFrame:
    """
    Append new records to an existing parquet, deduplicating on id_col.
    Used in incremental mode to avoid double-counting rows.
    """
    if not path.exists():
        return new_df

    existing = pl.read_parquet(path)

    # Align schemas before concat — diagonal handles extra/missing columns
    combined = pl.concat([existing, new_df], how="diagonal")
    combined = combined.unique(subset=[id_col], keep="last")

    log.info(
        f"Merged: {len(existing):,} existing + {len(new_df):,} new "
        f"→ {len(combined):,} unique rows"
    )
    return combined

def normalize_frames(frames: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """Ensure all frames have the same columns (add missing columns as nulls)."""
    all_cols = set()
    for f in frames:
        all_cols.update(f.columns)
    all_cols = list(all_cols)

    normalized = []
    for f in frames:
        missing = [c for c in all_cols if c not in f.columns]
        for c in missing:
            f = f.with_columns(pl.lit(None).alias(c))
        # Reorder to match
        f = f.select(all_cols)
        normalized.append(f)
    return normalized



# ── Main pull logic ───────────────────────────────────────────────────────────

def run_fetch(mode: str, start: str | None = None, end: str | None = None) -> Tuple[pl.DataFrame, pl.DataFrame]:

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Initialize Client
    client = SocrataClient()
    # Load MetaData
    meta   = load_meta()

    today  = datetime.now().strftime("%Y-%m-%d")

    match mode:
        case "full":
            since, until = None, None
            log.info("Mode: FULL — fetching entire dataset (this may take a while)")
        case "incremental":
            last_run = meta.get("last_successful_fetch")
            if not last_run:
                log.warning("No previous fetch found — falling back to full pull")
                since, until = None, None
            else:
                # One day overlap to catch late-arriving records
                since = (datetime.fromisoformat(last_run) - timedelta(days=1)).strftime("%Y-%m-%d")
                until = today
                log.info(f"Mode: INCREMENTAL — fetching {since} -> {until}")
        case "range":
            since, until = start, end
            log.info(f"Mode: RANGE — fetching {since} -> {until}")
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    # ── Crashes ───────────────────────────────────────────────────────────────
    crashes_path = RAW_DIR / "crashes.parquet"
    crashes: pl.DataFrame = client.fetch_dataset(
        dataset_id=CRASHES_ID,
        date_col="crash_date",
        since=since,
        until=until,
    )
    if not crashes.is_empty():
        if mode == "incremental":
            crashes = merge_with_existing(crashes, crashes_path, id_col="collision_id")
        save_parquet(crashes, crashes_path)

    # ── Person ────────────────────────────────────────────────────────────────
    person_path = RAW_DIR / "person.parquet"
    person: pl.DataFrame = client.fetch_dataset(
        dataset_id=PERSON_ID,
        date_col="crash_date",
        since=since,
        until=until,
    )
    if not person.is_empty():
        if mode == "incremental":
            person = merge_with_existing(person, person_path, id_col="unique_id")
        save_parquet(person, person_path)

    # ── Update metadata ───────────────────────────────────────────────────────
    meta["last_successful_fetch"] = today
    meta["last_mode"]             = mode
    meta["crashes_rows"]          = len(crashes) if not crashes.is_empty() else meta.get("crashes_rows", 0)
    meta["person_rows"]           = len(person)  if not person.is_empty()  else meta.get("person_rows", 0)
    save_meta(meta)

    log.info("Fetch complete")
    return crashes, person


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NYC collision data from Socrata API")
    parser.add_argument(
        "--mode",
        choices=["full", "incremental", "range"],
        default="incremental",
        help="full=all data, incremental=since last run, range=custom dates",
    )
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD) for range mode")
    parser.add_argument("--end",   type=str, help="End date   (YYYY-MM-DD) for range mode")
    args = parser.parse_args()

    if args.mode == "range" and not (args.start and args.end):
        parser.error("--start and --end are required for range mode")

    run_fetch(mode=args.mode, start=args.start, end=args.end)