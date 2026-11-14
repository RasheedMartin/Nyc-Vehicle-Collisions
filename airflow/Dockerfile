# Airflow for the NYC collisions retrain pipeline.
#
# Build context: repository root
#   docker build -f airflow/Dockerfile -t nyc-airflow .
#
# Single container, LocalExecutor: scheduler + webserver (see entrypoint.sh).
# The ML stack lives in an isolated venv so it never conflicts with Airflow's pins.
FROM apache/airflow:2.10.5-python3.12

USER root
# libgomp1: OpenMP runtime required by XGBoost on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Pipeline venv ────────────────────────────────────────────────────────────
COPY airflow/requirements-pipeline.txt /tmp/requirements-pipeline.txt
RUN python -m venv /opt/nyc-venv \
    && /opt/nyc-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/nyc-venv/bin/pip install --no-cache-dir -r /tmp/requirements-pipeline.txt \
    && chown -R airflow:root /opt/nyc-venv

# ── Project code ─────────────────────────────────────────────────────────────
WORKDIR /opt/nyc
COPY --chown=airflow:root src/data-pipeline ./src/data-pipeline
COPY --chown=airflow:root ["Borough Boundaries", "./Borough Boundaries"]
COPY --chown=airflow:root dags /opt/airflow/dags
COPY --chown=airflow:root --chmod=755 airflow/entrypoint.sh /entrypoint-nyc.sh
RUN mkdir -p data/raw data/processed models log \
    && chown -R airflow:root /opt/nyc

USER airflow
WORKDIR /opt/airflow

ENV NYC_PROJECT_DIR=/opt/nyc \
    NYC_PIPELINE_PYTHON=/opt/nyc-venv/bin/python \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    AIRFLOW__CORE__PARALLELISM=2 \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False \
    AIRFLOW__WEBSERVER__WORKERS=2 \
    AIRFLOW__WEBSERVER__EXPOSE_CONFIG=False \
    AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK=True \
    AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session

ENTRYPOINT ["/usr/bin/dumb-init", "--", "/entrypoint-nyc.sh"]
