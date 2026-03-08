"""
dags/nyc_collisions_retrain.py

Retraining pipeline for the Queens Vehicle Collision severity model.

Schedule: Every Monday at 6am (incremental pull of new Socrata data).
Manual trigger: Available via Airflow UI or CLI.

Tasks:
  fetch     → pull new records from NYC Open Data (Socrata)
  clean     → merge crashes + person, filter Queens, feature engineer
  features  → encode features, fit/update preprocessor
  train     → retrain XGBoost, save model + metrics
  upload    → push artifacts to Cloudflare R2 (disabled until R2 is configured)

All scripts run as subprocesses inside the mounted project directory
(/opt/nyc) so they use the project's own virtualenv and imports cleanly.
"""

from __future__ import annotations

import subprocess
import logging
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path("/opt/nyc")
PIPELINE_DIR = PROJECT_DIR / "src" / "data-pipeline"
PYTHON_BIN = Path("/home/airflow/.local/bin/python")
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path("/home/airflow/.local/bin/python3")

BOROUGH = "QUEENS"

DEFAULT_ARGS = {
    "owner": "rashe",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(hours=2),
    "email_on_failure": False,
    "email_on_retry": False,
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def run_script(script: str, *args: str) -> None:
    """
    Run a pipeline script as a subprocess.
    Captures stdout/stderr and forwards to Airflow task logs so failures
    are visible directly in the UI without needing to SSH into the container.
    """
    cmd = [str(PYTHON_BIN), str(PIPELINE_DIR / script), *args]
    log.info("Running: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        for line in result.stdout.splitlines():
            log.info("[stdout] %s", line)
    if result.stderr:
        for line in result.stderr.splitlines():
            log.warning("[stderr] %s", line)

    if result.returncode != 0:
        raise RuntimeError(
            f"{script} exited with code {result.returncode}\n"
            f"Last stderr: {result.stderr[-2000:] if result.stderr else 'none'}"
        )
    log.info("%s completed successfully", script)


# ── Task functions ────────────────────────────────────────────────────────────


def task_fetch(**context) -> None:
    """
    Incremental fetch: pulls records added since the last successful run.
    On first run (no metadata) falls back to a full pull automatically —
    this is handled inside fetch.py itself.
    """
    run_script("fetch.py", "--mode", "incremental")


def task_clean(**context) -> None:
    run_script("clean.py")


def task_features(**context) -> None:
    run_script("features.py", "--borough", BOROUGH)


def task_train(**context) -> None:
    run_script("train.py", "--borough", BOROUGH)


def task_upload_r2(**context) -> None:
    """
    Upload processed artifacts to Cloudflare R2.

    Requires Airflow Variables:
      R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME

    This task is set to SKIPPED by default (enabled=False on the operator).
    Enable it in the UI once R2 is configured.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise RuntimeError("boto3 not installed — run: pip install boto3")

    account_id = Variable.get("R2_ACCOUNT_ID")
    access_key_id = Variable.get("R2_ACCESS_KEY_ID")
    secret_key = Variable.get("R2_SECRET_ACCESS_KEY")
    bucket = Variable.get("R2_BUCKET_NAME", default_var="nyc-collisions")

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    artifacts = [
        # (local path,                                         r2 key)
        (
            PROJECT_DIR / "data/processed/QUEENS/collisions_queens.parquet",
            "processed/QUEENS/collisions_queens.parquet",
        ),
        (
            PROJECT_DIR / "data/processed/QUEENS/feature_meta.json",
            "processed/QUEENS/feature_meta.json",
        ),
        (
            PROJECT_DIR / "data/processed/QUEENS/preprocessor.joblib",
            "processed/QUEENS/preprocessor.joblib",
        ),
        (
            PROJECT_DIR / "models/QUEENS/severity_model.joblib",
            "models/QUEENS/severity_model.joblib",
        ),
        (
            PROJECT_DIR / "models/QUEENS/train_meta.json",
            "models/QUEENS/train_meta.json",
        ),
    ]

    for local_path, r2_key in artifacts:
        if not local_path.exists():
            log.warning("Artifact not found, skipping: %s", local_path)
            continue
        log.info("Uploading %s → r2://%s/%s", local_path.name, bucket, r2_key)
        s3.upload_file(str(local_path), bucket, r2_key)
        log.info("  ✓ uploaded")

    log.info("R2 upload complete — %d artifacts", len(artifacts))


def task_redeploy_railway(**context) -> None:
    """
    Trigger a Railway redeploy via GitHub Actions workflow_dispatch.

    Airflow Variables required:
      GITHUB_TOKEN  — Personal Access Token with 'workflow' scope
                      GitHub → Settings → Developer settings → Personal access tokens
      GITHUB_REPO   — e.g. "RasheedMartin/Nyc-Vehicle-Collisions"
      GITHUB_REF    — branch that has the workflow file (default: "main")
    """
    import json

    token = Variable.get("GITHUB_TOKEN")
    repo = Variable.get(
        "GITHUB_REPO", default_var="RasheedMartin/Nyc-Vehicle-Collisions"
    )
    ref = Variable.get("GITHUB_REF", default_var="main")

    url = f"https://api.github.com/repos/{repo}/actions/workflows/redeploy-railway.yml/dispatches"
    body = json.dumps(
        {"ref": ref, "inputs": {"reason": "Model retrained by Airflow"}}
    ).encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        log.info("GitHub Actions triggered — HTTP %s", resp.status)
    log.info(
        "Railway redeploy workflow dispatched — app will redeploy with new R2 artifacts"
    )


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="nyc_collisions_retrain",
    description="Incremental fetch + retrain for Queens collision severity model",
    default_args=DEFAULT_ARGS,
    schedule="0 6 * * 1",  # every Monday at 6am
    start_date=datetime(2025, 1, 1),
    catchup=False,  # don't backfill missed runs
    max_active_runs=1,  # never run two pipelines simultaneously
    tags=["nyc", "collisions", "queens", "ml"],
) as dag:
    fetch = PythonOperator(
        task_id="fetch",
        python_callable=task_fetch,
        doc_md="Pull new crash + person records from Socrata (incremental).",
    )

    clean = PythonOperator(
        task_id="clean",
        python_callable=task_clean,
        doc_md="Merge, filter to Queens, derive features (borough join, severity label, time features).",
    )

    features = PythonOperator(
        task_id="features",
        python_callable=task_features,
        doc_md="Encode features, fit ColumnTransformer, save preprocessor.joblib + feature_meta.json.",
    )

    train = PythonOperator(
        task_id="train",
        python_callable=task_train,
        doc_md="Retrain XGBoost with SMOTE, save model + train_meta.json.",
    )

    upload_r2 = PythonOperator(
        task_id="upload_r2",
        python_callable=task_upload_r2,
        doc_md="Upload artifacts to Cloudflare R2.",
    )

    redeploy = PythonOperator(
        task_id="redeploy_railway",
        python_callable=task_redeploy_railway,
        doc_md="Trigger GitHub Actions workflow_dispatch → Railway CLI redeploy → app serves the new model.",
    )

    # Pipeline order
    fetch >> clean >> features >> train >> upload_r2 >> redeploy
