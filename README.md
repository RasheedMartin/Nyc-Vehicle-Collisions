# Queens Vehicle Collision Intelligence

An end-to-end data science project analyzing motor vehicle collisions in Queens, NY — from raw NYC Open Data through a weekly-retrained XGBoost model to a live Streamlit dashboard.

**Live app →** [deployed on Railway](https://nyc-vehicle-collisions-production.up.railway.app/)

---

## What It Does

Given a street and time in Queens — say Jamaica Ave on a winter weekday at noon — how severe are crashes likely to be? The dashboard answers that question across five pages:

| Page                     | Description                                                                     |
| ------------------------ | ------------------------------------------------------------------------------- |
| **Overview**             | Crash counts, severity breakdown, KPIs by year                                  |
| **Hotspot Map**          | Street-level density heatmap of Queens collisions                               |
| **Contributing Factors** | Top crash causes ranked by severity class                                       |
| **Trends**               | Year-over-year and seasonal patterns (with Vision Zero context)                 |
| **Severity Predictor**   | XGBoost model: select a street + time → get a severity probability distribution |

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
Cloudflare R2 ── artifact store (parquet, joblib, json)
        │
GitHub Actions ── workflow_dispatch → railway redeploy
        │
   Railway    ──── Streamlit app pulls artifacts from R2 on cold start
```

**Orchestration**: Apache Airflow (Docker) runs the full pipeline every Monday at 6am.
**Stack**: Python · Polars · XGBoost · scikit-learn · SMOTE · Streamlit · Plotly · boto3

---

## Repository Structure

```
├── app/
│   ├── main.py                        # Streamlit entry point, R2 artifact loader
│   ├── theme.py                       # Color palette and chart defaults
│   └── pages/
│       ├── 1_overview.py
│       ├── 2_hotspot_map.py
│       ├── 3_contributing_factors.py
│       ├── 4_trends.py
│       └── 5_severity_predictor.py
│
├── src/data-pipeline/
│   ├── fetch.py                       # Socrata API pull (full or incremental)
│   ├── clean.py                       # Merge, filter, feature engineering
│   ├── features.py                    # Encoding, preprocessor fitting
│   └── train.py                       # XGBoost training + evaluation
│
├── data/                              # Git-ignored; populated by pipeline
│   ├── raw/                           # Socrata CSVs
│   └── processed/QUEENS/             # collisions_queens.parquet, preprocessor.joblib,
│                                      # feature_meta.json
├── models/QUEENS/                     # severity_model.joblib, train_meta.json
│
├── .github/workflows/
│   └── redeploy-railway.yml           # Triggered by Airflow after R2 upload
│
└── requirements.txt
```

> The Airflow DAG lives in a separate Docker environment at `dags/nyc_collisions_retrain.py`.

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

## Running Locally

```bash
git clone https://github.com/RasheedMartin/Nyc-Vehicle-Collisions.git
cd Nyc-Vehicle-Collisions
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Run the pipeline** (requires NYC Open Data access — no API key needed for public datasets):

```bash
python src/data-pipeline/fetch.py --mode full
python src/data-pipeline/clean.py
python src/data-pipeline/features.py --borough QUEENS
python src/data-pipeline/train.py --borough QUEENS
```

**Launch the app**:

```bash
streamlit run app/main.py
```

**Cloud deployment** (Railway): set `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` as environment variables. The app pulls all artifacts from Cloudflare R2 on startup if not found locally.

---

## MLOps — Weekly Retraining

The Airflow DAG `nyc_collisions_retrain` runs every Monday at 6am:

```
fetch (incremental) → clean → features → train → upload_r2 → redeploy_railway
```

`redeploy_railway` calls the GitHub Actions `workflow_dispatch` API, which triggers the Railway CLI to redeploy the live app with the new model artifacts.

**Airflow Variables required**:

| Variable               | Description                                 |
| ---------------------- | ------------------------------------------- |
| `R2_ACCOUNT_ID`        | Cloudflare R2 account ID                    |
| `R2_ACCESS_KEY_ID`     | R2 access key                               |
| `R2_SECRET_ACCESS_KEY` | R2 secret key                               |
| `R2_BUCKET_NAME`       | R2 bucket (default: `nyc-collisions`)       |
| `GITHUB_TOKEN`         | PAT with `workflow` scope                   |
| `GITHUB_REPO`          | `RasheedMartin/Nyc-Vehicle-Collisions`      |
| `GITHUB_REF`           | Branch with workflow file (default: `main`) |

**GitHub Secrets required** (for the Actions workflow):

| Secret                 | Description             |
| ---------------------- | ----------------------- |
| `RAILWAY_TOKEN`        | Railway API token       |
| `RAILWAY_SERVICE_NAME` | Service name in Railway |

---

## Data Source

NYC Open Data — Motor Vehicle Collisions:

- [Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) — one row per crash event
- [Person](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Person/f55k-p6yu) — one row per person involved

Joined on `collision_id`. Data updated daily by NYPD.

---

## License

MIT
