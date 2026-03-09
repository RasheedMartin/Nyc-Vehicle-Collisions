"""
fetch.py
--------
Data Pipeline to Fetch NYC Motor Vehicle Collision data from the NYC Open Data Socrata API.

Storage strategy:
  - Cloudflare R2 is the primary store for raw parquets + metadata.
    On incremental runs the existing parquet is downloaded from R2 into memory,
    merged with newly-fetched Socrata records, then pushed back — no large files
    ever accumulate on disk.
  - Local files under data/raw/ are used as a fallback when R2 is unreachable
    (e.g. offline dev), and are written after every successful R2 push so the
    local copy stays fresh for the next fallback opportunity.

Datasets:
  - Crashes : https://data.cityofnewyork.us/resource/h9gi-nx95.json
  - Person  : https://data.cityofnewyork.us/resource/f55k-p6yu.json

Usage:
  # Full pull (first time — seeds R2)
  python fetch.py --mode full

  # Incremental pull (scheduled retraining — only new records since last run)
  python fetch.py --mode incremental

  # Pull a specific date range
  python fetch.py --mode range --start 2024-01-01 --end 2024-12-31

Required environment variables (R2):
  R2_ACCOUNT_ID
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_BUCKET_NAME

Optional environment variables:
  NYC_OPEN_DATA_APP_TOKEN  — register free at https://data.cityofnewyork.us/profile/app_tokens
"""

import io
import os
import json
import logging
import argparse
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Tuple

# Load all environment variables
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL = "https://data.cityofnewyork.us/resource"
CRASHES_ID = "h9gi-nx95"
PERSON_ID = "f55k-p6yu"
PAGE_SIZE = 50_000
RAW_DIR = Path("data/raw")
META_FILE = RAW_DIR / ".last_fetch.json"

# R2 object keys
R2_CRASHES_KEY = "raw/crashes.parquet"
R2_PERSON_KEY = "raw/person.parquet"
R2_META_KEY = "raw/.last_fetch.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

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

# ── R2 client ─────────────────────────────────────────────────────────────────


class R2Client:
    """
    Thin wrapper around boto3 using Cloudflare R2's S3-compatible endpoint.

    All methods return None / empty values on failure rather than raising, so
    the pipeline can fall back gracefully to local files.
    """

    def __init__(self):
        account_id = os.getenv("R2_ACCOUNT_ID")
        access_key = os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket = os.getenv("R2_BUCKET_NAME")

        if not all([account_id, access_key, secret_key, self.bucket]):
            log.warning(
                "R2 credentials incomplete — R2 storage disabled. "
                "Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME."
            )
            self._enabled = False
            return

        self._enabled = True
        self._s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )
        log.info(f"R2 client initialised  bucket={self.bucket}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Download ──────────────────────────────────────────────────────────────

    def download_parquet(self, key: str) -> pl.DataFrame | None:
        """
        Download a parquet object from R2 directly into a Polars DataFrame.
        Returns None if the object does not exist or R2 is unavailable.
        """
        if not self._enabled:
            return None
        try:
            log.info(f"R2 ↓  {key}")
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            buf = io.BytesIO(resp["Body"].read())
            df = pl.read_parquet(buf)
            log.info(f"  Downloaded {len(df):,} rows from R2:{key}")
            return df
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                log.info(f"  R2:{key} does not exist yet — treating as empty")
            else:
                log.warning(f"  R2 download failed for {key}: {e}")
            return None
        except (BotoCoreError, Exception) as e:
            log.warning(f"  R2 download error for {key}: {e}")
            return None

    def download_json(self, key: str) -> dict:
        """Download a JSON object from R2. Returns {} on any failure."""
        if not self._enabled:
            return {}
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read().decode())
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("NoSuchKey", "404"):
                log.warning(f"  R2 JSON download failed for {key}: {e}")
            return {}
        except (BotoCoreError, Exception) as e:
            log.warning(f"  R2 JSON download error for {key}: {e}")
            return {}

    # ── Upload ────────────────────────────────────────────────────────────────

    def upload_parquet(self, df: pl.DataFrame, key: str) -> bool:
        """
        Serialise a Polars DataFrame to parquet in memory and upload to R2.
        Returns True on success, False on failure.
        """
        if not self._enabled:
            return False
        try:
            log.info(f"R2 ↑  {key}  ({len(df):,} rows)")
            buf = io.BytesIO()
            df.write_parquet(buf, compression="snappy")
            buf.seek(0)
            self._s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=buf,
                ContentType="application/octet-stream",
            )
            log.info(f"  Uploaded {buf.tell() / 1e6:.1f} MB to R2:{key}")
            return True
        except (BotoCoreError, ClientError, Exception) as e:
            log.error(f"  R2 upload failed for {key}: {e}")
            return False

    def upload_json(self, data: dict, key: str) -> bool:
        """Upload a dict as JSON to R2. Returns True on success."""
        if not self._enabled:
            return False
        try:
            body = json.dumps(data, indent=2).encode()
            self._s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
            return True
        except (BotoCoreError, ClientError, Exception) as e:
            log.warning(f"  R2 JSON upload failed for {key}: {e}")
            return False


# ── Socrata client ────────────────────────────────────────────────────────────


class SocrataClient:
    """Thin wrapper around the Socrata JSON API with pagination + token support."""

    def __init__(self, app_token: str | None = None):
        self.app_token = app_token or os.getenv("NYC_OPEN_DATA_APP_TOKEN")
        self.session = requests.Session()

        # Retry on transient errors: 429 (rate limit), 500/502/503/504 (server errors).
        # Exponential backoff: waits 2s, 4s, 8s between retries.
        retry = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

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
        url = f"{BASE_URL}/{dataset_id}.json"
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
                "$limit": PAGE_SIZE,
                "$offset": offset,
                "$order": f"{date_col} ASC",
            }
            if where:
                params["$where"] = where

            # Retry loop handles both ReadTimeout and truncated JSON responses.
            # Socrata occasionally drops the connection mid-transfer, producing
            # a partial payload that parses as JSONDecodeError.
            MAX_ATTEMPTS = 6
            BASE_TIMEOUT = 120
            batch = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                wait = 2**attempt  # 2, 4, 8, 16, 32, 64 seconds
                try:
                    resp = self.session.get(url, params=params, timeout=BASE_TIMEOUT)
                    resp.raise_for_status()

                    # Validate JSON before accepting — a truncated transfer
                    # returns 200 OK but the body is incomplete
                    batch = resp.json()
                    break  # clean parse — proceed
                except (
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                ) as e:
                    if attempt == MAX_ATTEMPTS:
                        log.error(
                            f"  Connection error after {MAX_ATTEMPTS} attempts at offset={offset:,}"
                        )
                        raise
                    log.warning(
                        f"  {type(e).__name__} at offset={offset:,} "
                        f"(attempt {attempt}/{MAX_ATTEMPTS}) — retrying in {wait}s"
                    )
                    time.sleep(wait)

                except Exception as e:
                    if attempt == MAX_ATTEMPTS:
                        raise
                    log.warning(
                        f"  {type(e).__name__} at offset={offset:,} "
                        f"(attempt {attempt}/{MAX_ATTEMPTS}) — retrying in {wait}s"
                    )
                    time.sleep(wait)

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

        normalized_frames = normalize_frames(frames, dataset_id)
        df = pl.concat(normalized_frames, how="vertical")  # now all columns match
        log.info(f"Dataset {dataset_id}: {len(df):,} total rows fetched")
        return df


# ── Metadata helpers ──────────────────────────────────────────────────────────


def load_meta(r2: R2Client) -> dict:
    """
    Load fetch metadata.
    Priority: R2 → local file → empty dict.
    R2 is authoritative because it reflects runs from any machine.
    """
    meta = r2.download_json(R2_META_KEY)
    if meta:
        log.info("Metadata loaded from R2")
        return meta

    if META_FILE.exists():
        log.info("Metadata loaded from local fallback")
        with open(META_FILE) as f:
            return json.load(f)

    log.info("No existing metadata found — starting fresh")
    return {}


def save_meta(meta: dict, r2: R2Client):
    """Persist metadata to R2 (primary) and local file (fallback copy)."""
    r2.upload_json(meta, R2_META_KEY)

    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


# ── Save / merge helpers ──────────────────────────────────────────────────────


def save_parquet(df: pl.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="snappy")
    log.info(f"Saved -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")


# ── Merge helper ──────────────────────────────────────────────────────────────


def merge_incremental(
    existing: pl.DataFrame,
    new_df: pl.DataFrame,
    id_col: str,
) -> pl.DataFrame:
    """
    Concat existing + new records and deduplicate on id_col.
    new_df rows win (keep='last') so late-updated records are corrected.
    """
    if existing.is_empty():
        return new_df

    combined = pl.concat([existing, new_df], how="diagonal")
    combined = combined.unique(subset=[id_col], keep="last")
    log.info(
        f"  Merged: {len(existing):,} existing + {len(new_df):,} new "
        f"→ {len(combined):,} unique rows"
    )
    return combined


# ── Local fallback helpers ────────────────────────────────────────────────────


def _local_read_parquet(path: Path) -> pl.DataFrame | None:
    """Read a local parquet file, returning None if it doesn't exist."""
    if path.exists():
        log.info(f"Local fallback ↓  {path}")
        return pl.read_parquet(path)
    return None


def _local_write_parquet(df: pl.DataFrame, path: Path):
    """Write a parquet to local disk (used to keep the fallback copy fresh)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="snappy")
    log.info(f"Local copy saved → {path}  ({path.stat().st_size / 1e6:.1f} MB)")


# ── Normalise helpers ─────────────────────────────────────────────────────────


def normalize_frames(
    frames: list[pl.DataFrame],
    database_id: str | None = None,
) -> list[pl.DataFrame]:
    """Ensure all pages share the same column set before concat."""
    normalized = []
    for f in frames:
        if database_id == PERSON_ID and "person_sex" not in f.columns:
            f = f.with_columns(pl.lit("U").alias("person_sex"))
        f = f.select(CRASH_COLS) if database_id == CRASHES_ID else f.select(PERSON_COLS)
        normalized.append(f)
    return normalized


# ── Dataset load: R2 → local fallback ────────────────────────────────────────


def load_existing(r2: R2Client, r2_key: str, local_path: Path) -> pl.DataFrame:
    """
    Load an existing raw parquet for incremental merging.

    Resolution order:
      1. R2 (primary — always up-to-date across machines)
      2. Local file (fallback — used when R2 is unavailable)
      3. Empty DataFrame (first-ever run, or both sources missing)
    """
    df = r2.download_parquet(r2_key)
    if df is not None:
        return df

    log.warning(f"R2 unavailable for {r2_key} — trying local fallback")
    df = _local_read_parquet(local_path)
    if df is not None:
        return df

    log.warning(f"No existing data found for {r2_key} — starting empty")
    return pl.DataFrame()


# ── Dataset save: R2 + local copy ────────────────────────────────────────────


def save_dataset(
    df: pl.DataFrame,
    r2: R2Client,
    r2_key: str,
    local_path: Path,
):
    """
    Upload to R2 (primary), then write a local copy for offline fallback.
    The local copy is only written if the R2 upload succeeds so the two
    stores never diverge silently.
    """
    uploaded = r2.upload_parquet(df, r2_key)
    if uploaded:
        _local_write_parquet(df, local_path)
    else:
        log.warning(
            f"R2 upload failed for {r2_key} — writing local only. "
            "R2 and local may be out of sync."
        )
        _local_write_parquet(df, local_path)


# ── Main pull logic ───────────────────────────────────────────────────────────


def run_fetch(
    mode: str, start: str | None = None, end: str | None = None
) -> Tuple[pl.DataFrame, pl.DataFrame]:

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize Client
    r2 = R2Client()
    client = SocrataClient()
    meta = load_meta(r2)
    today = datetime.now().strftime("%Y-%m-%d")
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
                since = (datetime.fromisoformat(last_run) - timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                )
                until = today
                log.info(f"Mode: INCREMENTAL — fetching {since} -> {until}")
        case "range":
            since, until = start, end
            log.info(f"Mode: RANGE — fetching {since} -> {until}")
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    crashes_local = RAW_DIR / "crashes.parquet"
    person_local = RAW_DIR / "person.parquet"

    # ── Crashes ───────────────────────────────────────────────────────────────
    new_crashes: pl.DataFrame = client.fetch_dataset(
        dataset_id=CRASHES_ID,
        date_col="crash_date",
        since=since,
        until=until,
    )

    if not new_crashes.is_empty():
        if mode == "incremental":
            existing_crashes = load_existing(r2, R2_CRASHES_KEY, crashes_local)
            crashes = merge_incremental(
                existing_crashes, new_crashes, id_col="collision_id"
            )
        else:
            crashes = new_crashes

        save_dataset(crashes, r2, R2_CRASHES_KEY, crashes_local)
    else:
        log.warning("No new crash records — loading existing data unchanged")
        crashes = load_existing(r2, R2_CRASHES_KEY, crashes_local)

    # ── Person ────────────────────────────────────────────────────────────────
    new_person: pl.DataFrame = client.fetch_dataset(
        dataset_id=PERSON_ID,
        date_col="crash_date",
        since=since,
        until=until,
    )

    if not new_person.is_empty():
        if mode == "incremental":
            existing_person = load_existing(r2, R2_PERSON_KEY, person_local)
            person = merge_incremental(
                existing_person, new_person, id_col="collision_id"
            )
        else:
            person = new_person

        save_dataset(person, r2, R2_PERSON_KEY, person_local)
    else:
        log.warning("No new person records — loading existing data unchanged")
        person = load_existing(r2, R2_PERSON_KEY, person_local)

    # ── Update metadata ───────────────────────────────────────────────────────
    meta["last_successful_fetch"] = today
    meta["last_mode"] = mode
    meta["crashes_rows"] = (
        len(crashes) if not crashes.is_empty() else meta.get("crashes_rows", 0)
    )
    meta["person_rows"] = (
        len(person) if not person.is_empty() else meta.get("person_rows", 0)
    )
    save_meta(meta, r2)

    log.info("Fetch complete")
    return crashes, person


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch NYC collision data from Socrata API"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental", "range"],
        default="incremental",
        help="full=all data, incremental=since last run, range=custom dates",
    )
    parser.add_argument(
        "--start", type=str, help="Start date (YYYY-MM-DD) for range mode"
    )
    parser.add_argument(
        "--end", type=str, help="End date   (YYYY-MM-DD) for range mode"
    )
    args = parser.parse_args()

    if args.mode == "range" and not (args.start and args.end):
        parser.error("--start and --end are required for range mode")

    run_fetch(mode=args.mode, start=args.start, end=args.end)
