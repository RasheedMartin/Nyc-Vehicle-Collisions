#!/usr/bin/env bash
# Start Airflow (LocalExecutor) in a single container: migrate DB, ensure the
# admin user exists, then run scheduler + webserver. If either process exits,
# the container exits so Railway's restart policy brings it back.
set -euo pipefail

# Railway Postgres exposes DATABASE_URL; map it if SQL_ALCHEMY_CONN isn't set.
if [[ -z "${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN:-}" && -n "${DATABASE_URL:-}" ]]; then
  export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="${DATABASE_URL/postgres:\/\//postgresql+psycopg2://}"
  export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN/postgresql:\/\//postgresql+psycopg2://}"
fi

for var in AIRFLOW__DATABASE__SQL_ALCHEMY_CONN AIRFLOW__CORE__FERNET_KEY AIRFLOW__WEBSERVER__SECRET_KEY; do
  if [[ -z "${!var:-}" ]]; then
    echo "FATAL: $var must be set" >&2
    exit 1
  fi
done

airflow db migrate

if [[ -n "${AIRFLOW_ADMIN_PASSWORD:-}" ]]; then
  airflow users create \
    --username "${AIRFLOW_ADMIN_USER:-admin}" \
    --password "${AIRFLOW_ADMIN_PASSWORD}" \
    --firstname Admin --lastname User --role Admin \
    --email "${AIRFLOW_ADMIN_EMAIL:-admin@example.com}" 2>/dev/null \
    || echo "Admin user already exists"
fi

airflow scheduler &
airflow webserver --port "${PORT:-8080}" &

# Exit as soon as either process dies.
wait -n
exit $?
