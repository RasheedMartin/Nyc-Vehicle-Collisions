"""
api/main.py
-----------
FastAPI inference service for the Queens Vehicle Collision severity model.

Loads the XGBoost model + preprocessor + feature metadata once at startup
from Cloudflare R2. Streamlit (or any client) calls POST /predict and gets
back probabilities — no model files ever touch the Streamlit server.

Endpoints:
  GET  /health        — liveness check + model status
  GET  /meta          — feature catalogue, severity labels, train metrics
  POST /predict       — run inference, return probabilities per class

Deploy as a separate Railway service alongside the Streamlit app.

Required environment variables:
  R2_ACCOUNT_ID
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_BUCKET_NAME

Optional:
  API_KEY   — if set, all requests must include  X-API-Key: <value>
  PORT      — defaults to 8000
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import boto3
import joblib
import numpy as np
import pandas as pd
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, model_validator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── R2 keys ───────────────────────────────────────────────────────────────────

R2_KEYS = {
    "model": "models/QUEENS/severity_model.joblib",
    "preprocessor": "processed/QUEENS/preprocessor.joblib",
    "feature_meta": "processed/QUEENS/feature_meta.json",
    "train_meta": "models/QUEENS/train_meta.json",
}

# ── Global model state (loaded once at startup) ───────────────────────────────


class ModelState:
    model = None
    preprocessor = None
    feature_meta: dict = {}
    train_meta: dict = {}
    loaded_at: float | None = None
    error: str | None = None


state = ModelState()


# ── R2 helpers ────────────────────────────────────────────────────────────────


def _r2_client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def _download_joblib(client, key: str) -> Any:
    bucket = os.environ.get("R2_BUCKET_NAME", "nyc-collisions")
    log.info(f"R2 ↓  {key}")
    resp = client.get_object(Bucket=bucket, Key=key)
    buf = io.BytesIO(resp["Body"].read())
    return joblib.load(buf)


def _download_json(client, key: str) -> dict:
    bucket = os.environ.get("R2_BUCKET_NAME", "nyc-collisions")
    log.info(f"R2 ↓  {key}")
    resp = client.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read().decode())


def load_artifacts() -> None:
    """Pull model + preprocessor + metadata from R2 into memory."""
    try:
        client = _r2_client()
        state.model = _download_joblib(client, R2_KEYS["model"])
        state.preprocessor = _download_joblib(client, R2_KEYS["preprocessor"])
        state.feature_meta = _download_json(client, R2_KEYS["feature_meta"])
        state.train_meta = _download_json(client, R2_KEYS["train_meta"])
        state.loaded_at = time.time()
        state.error = None
        log.info("All artifacts loaded successfully")
    except (BotoCoreError, ClientError) as e:
        state.error = f"R2 error: {e}"
        log.error(state.error)
        raise RuntimeError(state.error) from e
    except Exception as e:
        state.error = str(e)
        log.error(f"Artifact load failed: {e}")
        raise


# ── Lifespan (replaces @app.on_event) ────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model artifacts from R2…")
    load_artifacts()
    yield
    log.info("Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NYC Collision Severity API",
    description="XGBoost severity prediction for Queens vehicle collisions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Railway Streamlit URL in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── API key auth ─────────────────────────────────────────────────────

API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_key(key: str | None = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )


# ── Schemas ───────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    on_street_name: str = Field(
        "JAMAICA AVENUE", description="Street name (must match training vocabulary)"
    )
    time_of_day: str = Field(
        "Midday",
        description="Late Night | Morning Rush | Midday | Evening Rush | Night",
    )
    season: str = Field("Winter", description="Winter | Spring | Summer | Fall")
    is_weekend: int = Field(0, ge=0, le=1)
    is_rush_hour: int = Field(0, ge=0, le=1)
    month: int = Field(6, ge=1, le=12)
    day_of_week: int = Field(2, ge=0, le=6, description="0=Mon … 6=Sun")

    @model_validator(mode="after")
    def check_categoricals(self) -> "PredictRequest":
        valid_tod = ["Late Night", "Morning Rush", "Midday", "Evening Rush", "Night"]
        valid_seasons = ["Winter", "Spring", "Summer", "Fall"]
        if self.time_of_day not in valid_tod:
            raise ValueError(f"time_of_day must be one of {valid_tod}")
        if self.season not in valid_seasons:
            raise ValueError(f"season must be one of {valid_seasons}")
        return self


class SeverityProbability(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: list[SeverityProbability]
    borough: str = "QUEENS"


class MetaResponse(BaseModel):
    severity_order: list[str]
    category_catalogue: dict
    feature_importances: dict
    test_metrics: dict
    cv_f1_mean: float
    cv_f1_std: float
    n_train: int
    n_test: int
    trained_at: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    if state.error:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {state.error}")
    return {
        "status": "ok",
        "model": "loaded" if state.model else "not loaded",
        "loaded_at": state.loaded_at,
    }


@app.get("/meta", response_model=MetaResponse, dependencies=[Depends(verify_key)])
def meta() -> MetaResponse:
    """Return feature catalogue, severity labels, and training metrics."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MetaResponse(
        severity_order=state.feature_meta["severity_order"],
        category_catalogue=state.feature_meta["category_catalogue"],
        feature_importances=state.train_meta["feature_importances"],
        test_metrics=state.train_meta["test_metrics"],
        cv_f1_mean=state.train_meta["cv_f1_mean"],
        cv_f1_std=state.train_meta["cv_f1_std"],
        n_train=state.train_meta["n_train"],
        n_test=state.train_meta["n_test"],
        trained_at=state.train_meta["trained_at"],
    )


@app.post(
    "/predict", response_model=PredictResponse, dependencies=[Depends(verify_key)]
)
def predict(req: PredictRequest) -> PredictResponse:
    """Run severity inference and return per-class probabilities."""
    if state.model is None or state.preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    fm = state.feature_meta
    nom_feats = fm["nominal_features"]
    ord_feats = fm["ordered_features"]
    num_feats = fm["numeric_features"]
    all_cols = nom_feats + ord_feats + num_feats

    input_dict = req.model_dump()

    row = {
        col: [input_dict.get(col, 0 if col in num_feats else "Unknown")]
        for col in all_cols
    }

    try:
        X = state.preprocessor.transform(pd.DataFrame(row)).astype(np.float32)
        proba = state.model.predict_proba(X)[0]
    except Exception as e:
        log.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    severity_order = fm["severity_order"]
    predicted_idx = int(np.argmax(proba))
    predicted_label = severity_order[predicted_idx]

    return PredictResponse(
        predicted_label=predicted_label,
        confidence=round(float(proba[predicted_idx]), 4),
        probabilities=[
            SeverityProbability(label=label, probability=round(float(p), 4))
            for label, p in zip(severity_order, proba)
        ],
    )


@app.post("/reload", dependencies=[Depends(verify_key)])
def reload_model():
    """
    Hot-reload artifacts from R2 without restarting the server.
    Call this from your Airflow DAG after a successful retrain + R2 upload.
    """
    try:
        load_artifacts()
        return {"status": "reloaded", "loaded_at": state.loaded_at}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False
    )
