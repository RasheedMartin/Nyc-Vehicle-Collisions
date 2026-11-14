"""
upload.py
---------
Push trained artifacts to Cloudflare R2 so the API and dashboard pick them up.

Fails loudly if any required artifact is missing — a partial upload (e.g. a new
model with a stale preprocessor) would break inference.

Pipeline order:
  fetch.py → clean.py → features.py → train.py → upload.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def artifacts_for(borough: str) -> list[tuple[Path, str]]:
    b = borough.upper()
    return [
        # (local path, r2 key)
        (
            Path(f"data/processed/{b}/collisions_{b.lower()}.parquet"),
            f"processed/{b}/collisions_{b.lower()}.parquet",
        ),
        (
            Path(f"data/processed/{b}/feature_meta.json"),
            f"processed/{b}/feature_meta.json",
        ),
        (
            Path(f"data/processed/{b}/preprocessor.joblib"),
            f"processed/{b}/preprocessor.joblib",
        ),
        (
            Path(f"models/{b}/severity_model.joblib"),
            f"models/{b}/severity_model.joblib",
        ),
        (Path(f"models/{b}/train_meta.json"), f"models/{b}/train_meta.json"),
    ]


def main(borough: str) -> None:
    artifacts = artifacts_for(borough)
    missing = [str(p) for p, _ in artifacts if not p.exists()]
    if missing:
        log.error("Missing artifacts — refusing partial upload: %s", missing)
        sys.exit(1)

    account_id = os.environ["R2_ACCOUNT_ID"]
    bucket = os.environ.get("R2_BUCKET_NAME", "nyc-collisions")
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    # Upload the model last so the API never sees a new model paired with an
    # old preprocessor if the run dies midway.
    for local_path, key in sorted(artifacts, key=lambda a: "severity_model" in a[1]):
        log.info(
            "R2 ↑  %s → %s (%.1f MB)", local_path, key, local_path.stat().st_size / 1e6
        )
        s3.upload_file(str(local_path), bucket, key)

    log.info("R2 upload complete — %d artifacts", len(artifacts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload artifacts to Cloudflare R2")
    parser.add_argument("--borough", default="QUEENS")
    main(parser.parse_args().borough)
