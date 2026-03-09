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
  train.py   ──── XGBoost multiclass, SMOTE on train split only,
        │          5-fold stratified CV → severity_model.joblib
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

**Orchestration**: Apache Airflow (Docker) runs the full pipeline every Monday at 6am.  
**Stack**: Python · Polars · XGBoost · scikit-learn · SMOTE · Streamlit · FastAPI · Plotly · boto3

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
│   │   └── train.py                   # XGBoost training + evaluation
│   └── api/
│       ├── main.py                    # FastAPI inference service
│       ├── Dockerfile                 # Multi-stage build (libgomp1 for XGBoost on Linux)
│       ├── requirements.txt
│       └── railway.toml
│
├── airflow-docker/
│   └── docker-compose.yaml            # Airflow + Postgres + Redis
├── dags/
│   └── nyc_collisions_retrain.py
│
├── .github/workflows/
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

**Training**: Stratified 70/30 split → SMOTE on train only → 5-fold CV → XGBoost (`n_estimators=500`, `max_depth=6`, `learning_rate=0.05`). One row per crash (deduplicated from person-level raw data).

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

The Airflow DAG `nyc_collisions_retrain` runs every Monday at 6am:

```
fetch (incremental) → clean → features → train → upload_r2 → reload_task → redeploy_railway
```

After `upload_r2`, `reload_task` calls `POST /reload` on the FastAPI service to hot-swap the model in memory immediately — so predictions are served from the new model as soon as artifacts land in R2, without waiting for Railway's redeploy cycle (~2–3 min). `redeploy_railway` then triggers a full Railway redeploy via GitHub Actions `workflow_dispatch` to refresh the Streamlit service.

The Streamlit app uses `raw/.last_fetch.json` (written by `fetch.py` to R2 after every successful pull) to decide whether to re-download the parquet from R2. Within the 6-day TTL it never hits R2 at all, keeping cold starts fast.

**Airflow Variables required** (set as OS env vars on the worker, not just Airflow Variables):

| Variable | Description |
| --- | --- |
| `R2_ACCOUNT_ID` | Cloudflare R2 account ID |
| `R2_ACCESS_KEY_ID` | R2 access key |
| `R2_SECRET_ACCESS_KEY` | R2 secret key |
| `R2_BUCKET_NAME` | R2 bucket (default: `nyc-collisions`) |
| `INFERENCE_API_URL` | FastAPI internal URL (e.g. `http://unique-harmony.railway.internal:8080`) |
| `API_KEY` | Shared secret for `X-API-Key` auth on `/reload` |
| `GITHUB_TOKEN` | PAT with `workflow` scope |
| `GITHUB_REPO` | `RasheedMartin/Nyc-Vehicle-Collisions` |
| `GITHUB_REF` | Branch with workflow file (default: `main`) |

**GitHub Secrets required:**

| Secret | Description |
| --- | --- |
| `RAILWAY_TOKEN` | Railway API token |
| `RAILWAY_SERVICE_NAME` | Service name in Railway |

---

## Deployment (Railway)

Two services in the same Railway project:

| Service | Root | Start command |
| --- | --- | --- |
| `Frontend-Streamlit` | `/` | `streamlit run app/main.py` |
| `Backend-API` | `/src/api` | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

Add a **Volume** to the Streamlit service mounted at `/data` for persistent parquet caching across redeploys.

Both services share env vars `R2_*` and `API_KEY`. Set `INFERENCE_API_URL` on the Streamlit service to the API's Railway internal URL so traffic stays on the private network.

---

## Data Source

NYC Open Data — Motor Vehicle Collisions:

- [Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) — one row per crash event
- [Person](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Person/f55k-p6yu) — one row per person involved

Joined on `collision_id`. Data updated daily by NYPD.

---

## License

MIT