# Queens Vehicle Collision Intelligence

An end-to-end MLOps project analyzing motor vehicle collisions in Queens, NY — from raw NYC Open Data through a weekly-retrained XGBoost model to a live Streamlit dashboard backed by a FastAPI inference service.

**Live app →** [deployed on Railway](https://nyc-vehicle-collisions-production.up.railway.app/)

---

## What It Does

Given a street and time in Queens — say Jamaica Ave on a winter weekday at noon — how severe are crashes likely to be? The dashboard answers that question across six pages:

| Page | Description |
| --- | --- |
| **Overview** | KPIs, severity breakdown, hourly bar (12hr), annual trend, day × season heatmap |
| **Hotspot Map** | Street-level density heatmap with top 5 severe corridors and share-of-all-severe % |
| **Contributing Factors** | Top crash causes ranked by severity; pedestrian/cyclist fatal breakdown |
| **Trends** | Year-over-year and seasonal patterns with Vision Zero context |
| **Severity Predictor** | XGBoost model: select a street + time → severity probability distribution via FastAPI |
| **Data Pipeline** |  MLOps flow, last run metrics, infrastructure table, XGBoost hyperparameters |


---

## Business Case

Queens records hundreds of crashes every week. The NYPD, NYC DOT, and city planners must decide where to deploy resources — traffic enforcement, protected intersections, speed cameras, pedestrian safety improvements — but those decisions are typically reactive, based on where crashes already happened rather than where the next serious one is likely to occur.

This project shifts that framing: **given a specific street and time of day, what is the expected severity distribution of crashes there?** That moves the question from incident response to risk-based resource allocation.

The value concentrates in three areas:

**Prioritisation** — the hotspot map and corridors analysis surface the streets that account for a disproportionate share of severe crashes. A protected intersection program costs money; this identifies where it has the highest expected return.

**Enforcement timing** — the hourly severity analysis identifies the peak windows for fatal and major injury crashes. Speed cameras and increased patrol presence are more cost-effective when targeted at those windows rather than spread evenly across the day.

**Ongoing monitoring** — because the pipeline retrains weekly on fresh NYPD data, the model and dashboard stay current. If a newly opened road or changed traffic pattern shifts the severity distribution, it surfaces in the next Monday's run rather than waiting for an annual report.

> **Honest limitation:** location and time explain crash frequency well but severity less so — the same corridor at the same hour can produce a fender-bender or a fatality depending on vehicle type, speed, and angle of impact. The model outputs a probability distribution, not a point prediction, and is most useful as a prioritisation signal rather than a precise forecast.
---

## Architecture

```
NYC Open Data (Socrata)
        │
        ▼
   fetch.py  ──── incremental pull, weekly
        │
   clean.py  ──── merge crashes + persons, filter Queens,
        │          derive severity label + time features
        │
 features.py ──── encode on_street_name (top-100 OHE),
        │          time_of_day, season (ordinal), weekend/
        │          rush-hour/month/day_of_week (numeric)
        │          fit ColumnTransformer → preprocessor.joblib
        │
  train.py   ──── XGBoost multiclass, calibrated (no resampling),
        │          early stopping, 5-fold CV → severity_model.joblib
        │
Cloudflare R2 ── artifact store (parquet, joblib, json metadata)
        │                          │
        ▼                          ▼
  FastAPI (Railway) ←── loads model + preprocessor on startup
  POST /predict                    │
  GET  /meta                       │ raw/.last_fetch.json
  POST /reload (hot-swap)          │ (written by fetch.py)
        │                          ▼
  Streamlit (Railway) ──── Railway Volume /data
  calls API for inference          parquet cached, refreshed
  reads parquet locally            only when Airflow fetches new data
```

**Orchestration**: Apache Airflow (Docker, deployed on Railway) runs the full pipeline every Monday at 6am ET.  
**Stack**: Python · Polars · XGBoost · scikit-learn · Streamlit · FastAPI · Plotly · boto3

---

## Repository Structure

```
├── app/
│   ├── main.py                        # Streamlit entry point — navigation only
│   ├── utils.py                       # load_data() + load_train_meta() — shared loaders
│   ├── theme.py                       # Color palette and chart defaults
│   └── pages/
│       ├── 1_overview.py
│       ├── 2_hotspot_map.py
│       ├── 3_contributing_factors.py
│       ├── 4_trends.py
│       ├── 5_severity_predictor.py
│       └── 6_pipeline.py
│
├── src/
│   ├── data-pipeline/
│   │   ├── fetch.py                   # Socrata API pull (full or incremental)
│   │   ├── clean.py                   # Merge, filter, feature engineering
│   │   ├── features.py                # Encoding, preprocessor fitting
│   │   ├── train.py                   # XGBoost training + evaluation
│   │   └── upload.py                  # Push artifacts to R2 (model last)
│   └── api/
│       ├── main.py                    # FastAPI inference service
│       ├── Dockerfile                 # Multi-stage build (libgomp1 for XGBoost on Linux)
│       ├── requirements.txt
│       └── railway.toml
│
├── airflow/
│   ├── Dockerfile                     # Airflow 2.10 + isolated pipeline venv
│   ├── entrypoint.sh                  # migrate → scheduler + webserver
│   ├── docker-compose.yaml            # local mirror of the Railway service
│   ├── railway.toml
│   └── requirements-pipeline.txt      # pinned ML stack (matches API pins)
├── dags/
│   └── nyc_collisions_retrain.py
│
├── .github/workflows/
│   ├── ci.yml                         # ruff, Airflow image build, DAG integrity, pin check
│   └── redeploy-railway.yml           # Triggered by Airflow after R2 upload
│
└── requirements.txt
```

> `app/main.py` handles navigation only. Pages import `load_data()` and `load_train_meta()` from `app/utils.py` — never from `main.py` — to prevent Streamlit's double-execution bug on first load.

---

## Model

**Target**: 4-class accident severity — `No Injury`, `Minor Injury`, `Major Injury`, `Fatal`

**Features** (location + time only — no post-crash data):

- `on_street_name` — top 100 Queens streets by crash volume (OneHotEncoded), rest → "Other"
- `time_of_day` — Late Night / Morning Rush / Midday / Evening Rush / Night (ordinal)
- `season` — Winter / Spring / Summer / Fall (ordinal)
- `is_weekend`, `is_rush_hour`, `month`, `day_of_week` (numeric pass-through)

**Why no contributing factors or person details?** Those fields only exist after a crash occurs. The intended question is pre-crash: _given where and when, how bad do crashes tend to be here?_

**Training**: one row per crash (deduplicated from person-level raw data). The data is split 80/20 stratified *first*, so the test set keeps the real class mix. Training uses unweighted XGBoost (`multi:softprob`, `max_depth=6`, `learning_rate=0.05`) with early stopping on a validation split carved from train. There is no resampling, so the output probabilities are calibrated. 5-fold CV runs on the training split.

**Evaluation**: macro-F1 is the wrong yardstick for a calibrated model on 71%/26%/3%/0.1% classes. `train_meta.json` reports:

| Metric | Latest (Queens) | Meaning |
| --- | --- | --- |
| Severe-crash AUC | ~0.61 | ranks risky street/times above safe ones |
| Top-10% risk lift | ~1.9× | severe-crash rate in the riskiest decile vs overall |
| Calibration | predicted ≈ observed per class | "4% severe" really means ~4 in 100 |
| Log-loss skill | ~+1% vs base rate | location + time carry little severity signal |

The dashboard presents predictions as **relative risk vs the Queens average**, not as a single predicted label.

> **Why not SMOTE?** An earlier version oversampled with SMOTE. That interpolated one-hot street columns into impossible synthetic crashes, inflated predicted fatal and major-injury probabilities 10–70×, and ran CV on resampled data, which overstated performance. The calibrated model ranks risk better (AUC 0.61 vs 0.54) and trains 4× faster.

**Known limitations**: Fatal crashes are < 0.2% of Queens records. Location + time are strong signals for crash _frequency_ but weaker for crash _severity_ — the same corridor at the same hour can produce a fender-bender or a fatality depending on vehicle type, speed, and angle of impact. See the Severity Predictor page for full model caveats.

---

## Inference API

The FastAPI service runs as a separate Railway service on the project's private internal network. Streamlit never loads the model directly — all inference goes through the API.

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness + model load status |
| `GET` | `/meta` | Feature catalogue, train metrics, feature importances |
| `POST` | `/predict` | Run inference → per-class probabilities |
| `POST` | `/reload` | Hot-swap model from R2 without restarting |

Auth via `X-API-Key` header (`API_KEY` env var). Streamlit communicates with the API over Railway's private network at `http://<service>.railway.internal:<port>`.

---

## Running Locally

```bash
git clone https://github.com/RasheedMartin/Nyc-Vehicle-Collisions.git
cd Nyc-Vehicle-Collisions
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost requires OpenMP:
```bash
brew install libomp
```

Create a `.env` file:
```env
R2_ACCOUNT_ID=your-cloudflare-account-id
R2_ACCESS_KEY_ID=your-r2-access-key
R2_SECRET_ACCESS_KEY=your-r2-secret-key
R2_BUCKET_NAME=nyc-collisions

INFERENCE_API_URL=http://localhost:8000
API_KEY=your-api-key

RAILWAY_VOLUME_MOUNT_PATH=./data
```

**Run the pipeline:**
```bash
python src/data-pipeline/fetch.py --mode full
python src/data-pipeline/clean.py
python src/data-pipeline/features.py --borough QUEENS
python src/data-pipeline/train.py --borough QUEENS
```

**Start the inference API:**
```bash
cd src/api && uvicorn main:app --reload --port 8000
```

**Launch the dashboard:**
```bash
streamlit run app/main.py
```

---

## MLOps — Weekly Retraining

The Airflow DAG `nyc_collisions_retrain` runs every Monday at 06:00 America/New_York on Railway:

```
fetch → check_new_data → clean → features → train → upload_r2
      → wait_for_api → reload_api → redeploy_railway → notify_success
```

- **fetch** pulls new Socrata records incrementally, merges them into `raw/*.parquet` in R2 and records `crashes_rows_added` in `raw/.last_fetch.json`.
- **check_new_data** short-circuits the run when no new crash rows arrived.
- **upload_r2** (`src/data-pipeline/upload.py`) refuses partial uploads and writes the model last, so the API never pairs a new model with a stale preprocessor.
- **reload_api** calls `POST /reload` over Railway's private network to hot-swap the model in memory.
- **redeploy_railway** (optional, needs `GITHUB_TOKEN`) dispatches the GitHub Actions redeploy. Without it, Streamlit still picks up new data within `PARQUET_CACHE_TTL` (1h), because it compares the R2 parquet's `LastModified` with its Volume copy.
- Failures email `NOTIFY_EMAIL` via `on_failure_callback` when SMTP is configured.

Pipeline scripts run in an isolated venv (`/opt/nyc-venv`) inside the Airflow image. `scikit-learn`, `xgboost`, `joblib`, `numpy` and `pandas` are pinned identically in `airflow/requirements-pipeline.txt` and `src/api/requirements.txt` (enforced in CI), because the API unpickles what training pickles.

### Run Airflow locally

```bash
cp .env.example .env   # fill in R2_*, AIRFLOW__CORE__FERNET_KEY, AIRFLOW__WEBSERVER__SECRET_KEY, AIRFLOW_ADMIN_PASSWORD
cd airflow && docker compose up --build
# http://localhost:8080
```

---

## Deployment (Railway)

Everything runs in one Railway project:

| Service | Root dir | Config | Notes |
| --- | --- | --- | --- |
| `Frontend-Streamlit` | `/app` | `app/railway.toml` | Volume at `/data` caches the parquet |
| `Backend-API` | `/src/api` | `src/api/railway.toml` | Public + private (`*.railway.internal`) |
| `Airflow` | `/` | `/airflow/railway.toml` | Scheduler + webserver, LocalExecutor |
| `Postgres` | — | Railway template | Airflow metadata DB |

**Airflow service variables**

| Variable | Value |
| --- | --- |
| `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` | `${{Postgres.DATABASE_URL}}` |
| `AIRFLOW__CORE__FERNET_KEY` / `AIRFLOW__WEBSERVER__SECRET_KEY` | generated secrets |
| `AIRFLOW_ADMIN_USER` / `AIRFLOW_ADMIN_PASSWORD` | UI login |
| `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` | `${{Backend-API.R2_*}}` references |
| `API_KEY` | `${{Backend-API.API_KEY}}` |
| `INFERENCE_API_URL` | `http://${{Backend-API.RAILWAY_PRIVATE_DOMAIN}}:8080` |
| `AIRFLOW_CONN_INFERENCE_API` | same URL as above |
| `NYC_OPEN_DATA_APP_TOKEN` | Socrata app token (optional) |
| `GITHUB_TOKEN` | fine-grained PAT, Actions: write on this repo (optional) |
| `NOTIFY_EMAIL`, `AIRFLOW__SMTP__*` | alerting (optional) |

**GitHub Secrets** (for `redeploy-railway.yml`): `RAILWAY_TOKEN`, `RAILWAY_SERVICE_NAME`.

---

## Data Source

NYC Open Data — Motor Vehicle Collisions:

- [Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) — one row per crash event
- [Person](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Person/f55k-p6yu) — one row per person involved

Joined on `collision_id`. Data updated daily by NYPD.

---

## License

MIT