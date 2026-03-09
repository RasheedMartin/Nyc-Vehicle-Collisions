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

import json
import logging
import os
import subprocess
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from pendulum import timezone as pendulum_tz

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.email import EmailOperator
from airflow.providers.http.sensors.http import HttpSensor

log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_DIR  = Path("/opt/nyc")
PIPELINE_DIR = PROJECT_DIR / "src" / "data-pipeline"
PYTHON_BIN = Path("/home/airflow/.local/bin/python")
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path("/home/airflow/.local/bin/python3")

BOROUGH = "QUEENS"

DEFAULT_ARGS = {
    "owner":                     "rasheedmartin",
    "retries":                   2,
    "retry_delay":               timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "execution_timeout":         timedelta(hours=2),
    "email_on_failure":          False,   # handled by on_failure_callback below
    "email_on_retry":            False,
}

NOTIFY_EMAIL = Variable.get("AIRFLOW_EMAIL")
# ── Email callbacks ───────────────────────────────────────────────────────────

def _notify_success(context) -> None:
    """Send a success email when the full pipeline completes."""
    if not NOTIFY_EMAIL:
        return
    dag_run   = context["dag_run"]
    run_id    = dag_run.run_id
    exec_date = context["ds"]
    send_email(
        to=NOTIFY_EMAIL,
        subject=f"✅ nyc_collisions_retrain succeeded — {exec_date}",
        html_content=f"""
        <h3>Pipeline completed successfully</h3>
        <table style="font-family:monospace;font-size:13px;">
          <tr><td><b>DAG</b></td><td>nyc_collisions_retrain</td></tr>
          <tr><td><b>Run ID</b></td><td>{run_id}</td></tr>
          <tr><td><b>Date</b></td><td>{exec_date}</td></tr>
          <tr><td><b>Borough</b></td><td>{BOROUGH}</td></tr>
        </table>
        <p>Model artifacts uploaded to R2 and Railway redeploy triggered.</p>
        """,
    )


def _notify_failure(context) -> None:
    """Send a failure email on any task failure."""
    if not NOTIFY_EMAIL:
        return
    from airflow.utils.email import send_email
    task_instance = context["task_instance"]
    exec_date     = context["ds"]
    exception     = context.get("exception", "Unknown error")
    send_email(
        to=NOTIFY_EMAIL,
        subject=f"❌ nyc_collisions_retrain FAILED — {exec_date}",
        html_content=f"""
        <h3>Pipeline task failed</h3>
        <table style="font-family:monospace;font-size:13px;">
          <tr><td><b>DAG</b></td><td>nyc_collisions_retrain</td></tr>
          <tr><td><b>Task</b></td><td>{task_instance.task_id}</td></tr>
          <tr><td><b>Date</b></td><td>{exec_date}</td></tr>
          <tr><td><b>Error</b></td><td><pre>{str(exception)[:1000]}</pre></td></tr>
        </table>
        <p>Check the Airflow UI for full logs.</p>
        """,
    )


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

def task_has_new_data(**context) -> bool:
    """
    Short-circuit gate: returns True if fetch.py pulled new records, False to
    skip the rest of the pipeline.

    Reads raw/.last_fetch.json from R2 and compares last_fetch_date to the
    DAG execution date. If the dates match, new data was pulled this run.
    If not, the pipeline short-circuits and all downstream tasks are skipped —
    saving ~30–40 min of compute on weeks where NYPD hasn't pushed new data.
    """
    import boto3
    from botocore.config import Config

    account_id = Variable.get("R2_ACCOUNT_ID", "")
    if not account_id:
        log.warning("R2_ACCOUNT_ID not set — assuming new data and proceeding")
        return True

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=Variable.get("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=Variable.get("R2_SECRET_ACCESS_KEY"),
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        bucket = Variable.get("R2_BUCKET_NAME", "nyc-collisions")
        resp   = s3.get_object(Bucket=bucket, Key="raw/.last_fetch.json")
        meta   = json.loads(resp["Body"].read().decode())

        last_fetch = meta.get("last_fetch_date", "")[:10]   # YYYY-MM-DD
        exec_date  = context["ds"]                           # YYYY-MM-DD

        if last_fetch == exec_date:
            log.info("New data confirmed for %s — proceeding with pipeline", exec_date)
            return True

        log.info(
            "No new data for %s (last_fetch_date=%s) — short-circuiting",
            exec_date, last_fetch,
        )
        return False

    except Exception as e:
        log.warning("Could not read .last_fetch.json (%s) — proceeding anyway", e)
        return True

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

    account_id       = Variable.get("R2_ACCOUNT_ID")
    access_key_id    = Variable.get("R2_ACCESS_KEY_ID")
    secret_key       = Variable.get("R2_SECRET_ACCESS_KEY")
    bucket           = Variable.get("R2_BUCKET_NAME", default_var="nyc-collisions")

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
        (PROJECT_DIR / "data/processed/QUEENS/collisions_queens.parquet",
         "processed/QUEENS/collisions_queens.parquet"),
        (PROJECT_DIR / "data/processed/QUEENS/feature_meta.json",
         "processed/QUEENS/feature_meta.json"),
        (PROJECT_DIR / "data/processed/QUEENS/preprocessor.joblib",
         "processed/QUEENS/preprocessor.joblib"),
        (PROJECT_DIR / "models/QUEENS/severity_model.joblib",
         "models/QUEENS/severity_model.joblib"),
        (PROJECT_DIR / "models/QUEENS/train_meta.json",
         "models/QUEENS/train_meta.json"),
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
    repo  = Variable.get("GITHUB_REPO", default_var="RasheedMartin/Nyc-Vehicle-Collisions")
    ref   = Variable.get("GITHUB_REF",  default_var="main")

    url  = f"https://api.github.com/repos/{repo}/actions/workflows/redeploy-railway.yml/dispatches"
    body = json.dumps({"ref": ref, "inputs": {"reason": "Model retrained by Airflow"}}).encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Accept":               "application/vnd.github+json",
            "Authorization":        f"Bearer {token}",
            "Content-Type":         "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        log.info("GitHub Actions triggered — HTTP %s", resp.status)
    log.info("Railway redeploy workflow dispatched — app will redeploy with new R2 artifacts")

def task_reload_api():
    """
        Hot-swap the XGBoost model and preprocessor on the live FastAPI inference
    service by calling POST /reload.

    This runs immediately after upload_r2 completes, so predictions are served
    from the new model as soon as artifacts land in R2 — without waiting for
    Railway's full redeploy cycle (~2–3 min).

    Requires ``INFERENCE_API_URL`` and ``API_KEY`` to be set as environment
    variables on the Airflow worker (same block as ``R2_*`` vars in
    docker-compose.yaml).
    """
    api_url = Variable.get("INFERENCE_API_URL").rstrip("/")
    api_key = Variable.get("API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = requests.post(f"{api_url}/reload", headers=headers, timeout=30)
    resp.raise_for_status()
    log.info("API reload successful — model hot-swapped in memory")

# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="nyc_collisions_retrain",
    description="Incremental fetch + retrain for Queens collision severity model",
    default_args=DEFAULT_ARGS,
    schedule="0 6 * * 1",          # every Monday at 6am
    start_date=datetime(2025, 1, 1, tzinfo=pendulum_tz("America/New_York")),
    catchup=False,                  # don't backfill missed runs
    max_active_runs=1,              # never run two pipelines simultaneously
    tags=["nyc", "collisions", "queens", "ml"],
) as dag:

    fetch = PythonOperator(
        task_id="fetch",
        python_callable=task_fetch,
        doc_md="Pull new crash + person records from Socrata (incremental).",
    )

    check_new_data = ShortCircuitOperator(
        task_id="check_new_data",
        python_callable=task_has_new_data,
        doc_md="""
        Gate task — reads `raw/.last_fetch.json` from R2 to confirm fetch.py
        actually pulled new records for this execution date. Returns False
        (skipping all downstream tasks) if NYPD has not pushed new data since
        the last run. Prevents unnecessary compute on no-op weeks.
        """,
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

    wait_for_api = HttpSensor(
        task_id="wait_for_api",
        http_conn_id="inference_api",  
        endpoint="/health",
        method="GET",
        response_check=lambda resp: resp.status_code == 200,
        poke_interval=30,               # check every 30 seconds
        timeout=300,                    # give up after 5 minutes
        mode="poke",
        doc_md="""
        Sensor — waits for the FastAPI inference service to return HTTP 200
        on GET /health before attempting the model hot-swap. Prevents
        reload_api from hitting a cold-starting or redeploying instance.
        Requires an Airflow Connection named 'inference_api'.
        """,
    )

    reload_api = PythonOperator(
        task_id="reload_api",
        python_callable=task_reload_api,
        doc_md="""
        Hot-swap the XGBoost model and preprocessor on the live FastAPI
        inference service by calling POST /reload. Runs immediately after
        upload_r2 so predictions are served from the new model without
        waiting for Railway's full redeploy cycle (~2–3 min).
        """,
    )

    redeploy_railway = PythonOperator(
        task_id="redeploy_railway",
        python_callable=task_redeploy_railway,
        doc_md="Trigger GitHub Actions workflow_dispatch → Railway CLI redeploy → app serves new artifacts.",
    )

    notify_success = EmailOperator(
        task_id="notify_success",
        to=NOTIFY_EMAIL or "rasheedjmartin@gmail.com",
        subject="✅ nyc_collisions_retrain succeeded — {{ ds }}",
        html_content="""
        <h3>Pipeline completed successfully</h3>
        <table style="font-family:monospace;font-size:13px;">
          <tr><td><b>DAG</b></td><td>nyc_collisions_retrain</td></tr>
          <tr><td><b>Run ID</b></td><td>{{ run_id }}</td></tr>
          <tr><td><b>Date</b></td><td>{{ ds }}</td></tr>
          <tr><td><b>Borough</b></td><td>QUEENS</td></tr>
        </table>
        <p>Model artifacts uploaded to R2 and Railway redeploy triggered.</p>
        """,
        doc_md="Send success email after Railway redeploy is triggered.",
    )


    # Pipeline order
    fetch >> check_new_data >> clean >> features >> train
    train >> upload_r2 >> wait_for_api >> reload_api >> redeploy_railway >> notify_success