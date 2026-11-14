"""
dags/nyc_collisions_retrain.py

Weekly retraining pipeline for the Queens Vehicle Collision severity model.

Schedule: every Monday 06:00 America/New_York.

    fetch → check_new_data → clean → features → train → upload_r2
          → wait_for_api → reload_api → redeploy_railway → notify_success

Pipeline scripts run as subprocesses with a dedicated interpreter
(NYC_PIPELINE_PYTHON) so the ML stack (polars, geopandas, xgboost, …) never has
to share dependency pins with Airflow itself.

Configuration comes from environment variables first, then Airflow Variables,
so the same DAG works in the Railway image and in local docker compose:

  NYC_PROJECT_DIR       project root inside the container   (default /opt/nyc)
  NYC_PIPELINE_PYTHON   interpreter for pipeline scripts    (default /opt/nyc-venv/bin/python)
  NOTIFY_EMAIL          success/failure recipient           (optional)
  INFERENCE_API_URL     FastAPI base URL                    (required for reload)
  API_KEY               X-API-Key for /reload               (optional)
  GITHUB_TOKEN          PAT with `workflow` scope           (required for redeploy)
  GITHUB_REPO / GITHUB_REF
  AIRFLOW_CONN_INFERENCE_API  connection URI used by wait_for_api

R2_* credentials only need to be present in the environment; the pipeline
scripts read them directly.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pendulum
import requests

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.utils.email import send_email

log = logging.getLogger(__name__)

LOCAL_TZ = pendulum.timezone("America/New_York")


def cfg(name: str, default: str | None = None) -> str | None:
    """Env var → Airflow Variable → default. Only call inside tasks (not at parse time)."""
    value = os.environ.get(name)
    if value:
        return value
    return Variable.get(name, default_var=default)


# ── Static config (env only — safe at parse time, no DB hits) ────────────────

PROJECT_DIR = Path(os.environ.get("NYC_PROJECT_DIR", "/opt/nyc"))
PIPELINE_DIR = PROJECT_DIR / "src" / "data-pipeline"
PYTHON_BIN = os.environ.get("NYC_PIPELINE_PYTHON", "/opt/nyc-venv/bin/python")
META_FILE = PROJECT_DIR / "data" / "raw" / ".last_fetch.json"
BOROUGH = "QUEENS"
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", "")


# ── Callbacks ────────────────────────────────────────────────────────────────


def _notify_failure(context) -> None:
    if not NOTIFY_EMAIL:
        return
    ti = context["task_instance"]
    exception = context.get("exception", "Unknown error")
    try:
        send_email(
            to=NOTIFY_EMAIL,
            subject=f"❌ nyc_collisions_retrain FAILED — {ti.task_id} — {context['ds']}",
            html_content=f"""
            <h3>Pipeline task failed</h3>
            <table style="font-family:monospace;font-size:13px;">
              <tr><td><b>Task</b></td><td>{ti.task_id}</td></tr>
              <tr><td><b>Try</b></td><td>{ti.try_number}</td></tr>
              <tr><td><b>Run</b></td><td>{context["run_id"]}</td></tr>
              <tr><td><b>Log</b></td><td><a href="{ti.log_url}">open</a></td></tr>
              <tr><td><b>Error</b></td><td><pre>{str(exception)[:1500]}</pre></td></tr>
            </table>
            """,
        )
    except Exception:  # never let alerting mask the real failure
        log.exception("Failure email could not be sent")


DEFAULT_ARGS = {
    "owner": "rasheedmartin",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(hours=2),
    "email_on_failure": False,
    "email_on_retry": False,
    "on_failure_callback": _notify_failure,
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def run_script(script: str, *args: str) -> None:
    """Run a pipeline script, streaming its output into the Airflow task log."""
    cmd = [PYTHON_BIN, "-u", str(PIPELINE_DIR / script), *args]
    log.info("Running: %s (cwd=%s)", " ".join(cmd), PROJECT_DIR)

    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        log.info("[%s] %s", script, line)
        tail = (tail + [line])[-40:]
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{script} exited with code {rc}\n" + "\n".join(tail))


# ── Tasks ────────────────────────────────────────────────────────────────────


def task_has_new_data(**context) -> bool:
    """
    Short-circuit gate. fetch.py writes data/raw/.last_fetch.json locally (and
    to R2) with `crashes_rows_added`; skip the expensive downstream tasks when
    the merged dataset didn't grow.

    Trigger with conf {"force": true} to retrain regardless (e.g. after a
    training-code change).
    """
    conf = context["dag_run"].conf or {}
    if conf.get("force"):
        log.info("force=true in dag_run.conf — retraining regardless of new data")
        return True
    if not META_FILE.exists():
        log.warning("%s missing — proceeding to be safe", META_FILE)
        return True

    meta = json.loads(META_FILE.read_text())
    added = int(meta.get("crashes_rows_added", -1))
    log.info("fetch metadata: %s", meta)

    if added == 0:
        log.info("No new crash records since last run — skipping retrain")
        return False
    return True


def task_reload_api(**_) -> None:
    """Hot-swap model + preprocessor on the live API via POST /reload."""
    api_url = cfg("INFERENCE_API_URL")
    if not api_url:
        raise RuntimeError("INFERENCE_API_URL is not configured")
    api_key = cfg("API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = requests.post(f"{api_url.rstrip('/')}/reload", headers=headers, timeout=120)
    resp.raise_for_status()
    log.info("API reload OK: %s", resp.json())


def task_redeploy_railway(**_) -> None:
    """Dispatch the redeploy-railway.yml GitHub Actions workflow (restarts Streamlit)."""
    token = cfg("GITHUB_TOKEN")
    if not token:
        # Optional: Streamlit re-checks R2 hourly on its own (app/utils.py), so
        # a forced redeploy only makes new data appear sooner.
        log.warning(
            "GITHUB_TOKEN not set — skipping redeploy; Streamlit refreshes within ~1h"
        )
        return
    repo = cfg("GITHUB_REPO", "RasheedMartin/Nyc-Vehicle-Collisions")
    ref = cfg("GITHUB_REF", "main")

    resp = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/redeploy-railway.yml/dispatches",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"ref": ref, "inputs": {"reason": "Model retrained by Airflow"}},
        timeout=30,
    )
    resp.raise_for_status()
    log.info("GitHub Actions redeploy dispatched (HTTP %s)", resp.status_code)


def task_notify_success(**context) -> None:
    if not NOTIFY_EMAIL:
        log.info("NOTIFY_EMAIL not set — skipping success email")
        return
    meta = json.loads(META_FILE.read_text()) if META_FILE.exists() else {}
    send_email(
        to=NOTIFY_EMAIL,
        subject=f"✅ nyc_collisions_retrain succeeded — {context['ds']}",
        html_content=f"""
        <h3>Pipeline completed successfully</h3>
        <table style="font-family:monospace;font-size:13px;">
          <tr><td><b>Run</b></td><td>{context["run_id"]}</td></tr>
          <tr><td><b>Borough</b></td><td>{BOROUGH}</td></tr>
          <tr><td><b>New crash rows</b></td><td>{meta.get("crashes_rows_added", "?")}</td></tr>
          <tr><td><b>Total crash rows</b></td><td>{meta.get("crashes_rows", "?")}</td></tr>
        </table>
        <p>Artifacts uploaded to R2, API hot-swapped, Railway redeploy triggered.</p>
        """,
    )


# ── DAG ──────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="nyc_collisions_retrain",
    description="Incremental fetch + retrain for Queens collision severity model",
    default_args=DEFAULT_ARGS,
    schedule="0 6 * * 1",
    start_date=datetime(2025, 1, 1, tzinfo=LOCAL_TZ),
    catchup=False,
    max_active_runs=1,
    tags=["nyc", "collisions", "queens", "ml"],
) as dag:
    fetch = PythonOperator(
        task_id="fetch",
        python_callable=run_script,
        op_args=["fetch.py", "--mode", "incremental"],
        doc_md="Pull new crash + person records from Socrata (incremental) and sync to R2.",
    )

    check_new_data = ShortCircuitOperator(
        task_id="check_new_data",
        python_callable=task_has_new_data,
        doc_md='Skip retraining when fetch added no new crash rows. Trigger with {"force": true} to override.',
    )

    clean = PythonOperator(
        task_id="clean",
        python_callable=run_script,
        op_args=["clean.py"],
        doc_md="Merge crashes + persons, borough spatial join, severity label, time features.",
    )

    features = PythonOperator(
        task_id="features",
        python_callable=run_script,
        op_args=["features.py", "--borough", BOROUGH],
        doc_md="Fit ColumnTransformer; write preprocessor.joblib + feature_meta.json.",
    )

    train = PythonOperator(
        task_id="train",
        python_callable=run_script,
        op_args=["train.py", "--borough", BOROUGH],
        doc_md="Retrain XGBoost with SMOTE; write severity_model.joblib + train_meta.json.",
    )

    upload_r2 = PythonOperator(
        task_id="upload_r2",
        python_callable=run_script,
        op_args=["upload.py", "--borough", BOROUGH],
        doc_md="Upload all artifacts to Cloudflare R2 (model last).",
    )

    wait_for_api = HttpSensor(
        task_id="wait_for_api",
        http_conn_id="inference_api",
        endpoint="/health",
        response_check=lambda resp: resp.status_code == 200,
        poke_interval=30,
        timeout=300,
        mode="reschedule",
        doc_md="Wait for FastAPI /health before hot-swapping. Uses AIRFLOW_CONN_INFERENCE_API.",
    )

    reload_api = PythonOperator(
        task_id="reload_api",
        python_callable=task_reload_api,
        doc_md="POST /reload — API serves the new model immediately.",
    )

    redeploy_railway = PythonOperator(
        task_id="redeploy_railway",
        python_callable=task_redeploy_railway,
        doc_md="GitHub Actions → Railway redeploy so Streamlit picks up the new parquet.",
    )

    notify_success = PythonOperator(
        task_id="notify_success",
        python_callable=task_notify_success,
        retries=0,
        doc_md="Success email (no-op when NOTIFY_EMAIL is unset).",
    )

    (
        fetch
        >> check_new_data
        >> clean
        >> features
        >> train
        >> upload_r2
        >> wait_for_api
        >> reload_api
        >> redeploy_railway
        >> notify_success
    )
