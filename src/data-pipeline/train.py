"""
train.py
--------
Trains an XGBoost classifier to predict NYC collision accident severity.

Supports --borough flag to train a focused borough-level model.
All artifacts are namespaced by borough so multiple models can coexist.

Improvements over the original notebook:
  - SMOTE applied AFTER the train/test split (never before — that's leakage)
  - Stratified split preserves class ratios in both train and test sets
  - 5-fold stratified cross-validation for honest pre-test estimates
  - XGBoost instead of RandomForest — better on tabular + imbalanced data

Output (borough=QUEENS example):
  models/QUEENS/severity_model.joblib
  models/QUEENS/train_meta.json

Pipeline order:
  fetch.py  →  clean.py  →  features.py [--borough QUEENS]  →  train.py [--borough QUEENS]
"""

import json
import logging
import argparse
import joblib
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR   = Path("data/processed")
MODELS_DIR      = Path("models")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

XGBOOST_PARAMS = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "eval_metric":       "mlogloss",
    "random_state":      42,
    "n_jobs":            2,
    "tree_method":       "hist",
}


VALID_BOROUGHS = {"MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"}


# ── Path helpers ──────────────────────────────────────────────────────────────

def borough_slug(borough: str | None) -> str:
    return borough.upper().replace(" ", "_") if borough else "ALL"


def features_path(borough: str | None) -> Path:
    return PROCESSED_DIR / borough_slug(borough) / "features.parquet"


def meta_path(borough: str | None) -> Path:
    return PROCESSED_DIR / borough_slug(borough) / "feature_meta.json"


def model_dir(borough: str | None) -> Path:
    return MODELS_DIR / borough_slug(borough)

# ── SMOTE ─────────────────────────────────────────────────────────────────────

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    severity_order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE only to the training set — never before the split.
    Applying SMOTE before splitting is a form of data leakage because
    synthetic minority samples would bleed into the test set.

    k_neighbors=2 instead of default 5 because Fatal has only ~125 samples —
    with 5 neighbours SMOTE would crash on classes smaller than k+1.
    """
    before = dict(zip(*np.unique(y_train, return_counts=True)))
    log.info(f"  Before SMOTE: { {severity_order[k]: v for k, v in before.items()} }")

    smote = SMOTE(random_state=42, k_neighbors=2)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    after = dict(zip(*np.unique(y_res, return_counts=True)))
    log.info(f"  After SMOTE : { {severity_order[k]: v for k, v in after.items()} }")
    return X_res, y_res


def stratified_sample(df: pl.DataFrame, max_per_class: int = 100_000) -> pl.DataFrame:
    """
    Cap each severity class at max_per_class rows.
    Keeps training fast while preserving all rare Fatal/Major Injury rows.
    """
    sampled = df.group_by("accident_severity").map_groups(
        lambda g: g.sample(n=min(max_per_class, len(g)), seed=42)
    )
    log.info(
        f"Stratified sample: {len(df):,} → {len(sampled):,} rows\n"
        f"{sampled.group_by('accident_severity').len().sort('len', descending=True)}"
    )
    return sampled


def log_metrics(metrics: dict, severity_order: list[str]):
    log.info("── Test set metrics ─────────────────────────────────────")
    log.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    log.info(f"  Precision : {metrics['precision']:.4f}  (macro)")
    log.info(f"  Recall    : {metrics['recall']:.4f}  (macro)")
    log.info(f"  F1        : {metrics['f1']:.4f}  (macro)")
    log.info("  Per-class breakdown:")
    for label in severity_order:
        pc = metrics["per_class"].get(label, {})
        log.info(
            f"    {label:<15}  "
            f"precision={pc.get('precision', 0):.3f}  "
            f"recall={pc.get('recall', 0):.3f}  "
            f"f1={pc.get('f1-score', 0):.3f}  "
            f"support={int(pc.get('support', 0)):,}"
        )
    log.info("─────────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_train(
    borough:   str | None = None,
    cv_folds:  int = 5,
) -> XGBClassifier:

    if borough:
        borough = borough.upper()
        if borough not in VALID_BOROUGHS:
            raise ValueError(f"Invalid borough '{borough}'. Choose from: {VALID_BOROUGHS}")

    slug = borough_slug(borough)
    out_dir = model_dir(borough)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path      = out_dir / "severity_model.joblib"
    train_meta_path = out_dir / "train_meta.json"

    # ── Load metadata + features ──────────────────────────────────────────────
    log.info(f"Training model  [borough={slug}]")

    with open(meta_path(borough)) as f:
        meta = json.load(f)

    severity_order = meta["severity_order"]
    feature_names  = meta["encoded_feature_names"]
    n_classes      = len(severity_order)

    fp = features_path(borough)
    log.info(f"Loading features from {fp}")
    df = pl.read_parquet(fp)
    log.info(f"  {len(df):,} rows")

    # ── Build X, y ────────────────────────────────────────────────────────────
    available = [c for c in feature_names if c in df.columns]
    missing   = set(feature_names) - set(available)
    if missing:
        log.warning(f"  Feature columns missing (skipped): {missing}")

    df = stratified_sample(df)

    X = df.select(available).to_numpy().astype(np.float32)
    y = df["severity_encoded"].to_numpy().astype(np.int8)
    del df

    # ── Stratified train / test split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    del X, y
    log.info(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    # ── SMOTE on training set only ────────────────────────────────────────────
    X_train_res, y_train_res = apply_smote(X_train, y_train, severity_order)

    # ── Cross-validation ──────────────────────────────────────────────────────
    log.info(f"Running {cv_folds}-fold stratified cross-validation …")
    cv_model = XGBClassifier(
        **XGBOOST_PARAMS,
        num_class=n_classes,
        objective="multi:softprob",
    )
    cv_scores = cross_val_score(
        cv_model, X_train_res, y_train_res,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=1,
    )
    log.info(
        f"  CV F1 (macro): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  "
        f"[{', '.join(f'{s:.3f}' for s in cv_scores)}]"
    )
    del cv_model

    # ── Final fit ─────────────────────────────────────────────────────────────
    log.info("Training final model …")
    model = XGBClassifier(
        **XGBOOST_PARAMS,
        num_class=n_classes,
        objective="multi:softprob",
    )
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── Evaluate on real (unbalanced) test set ────────────────────────────────
    y_pred  = model.predict(X_test)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "per_class": classification_report(
            y_test, y_pred,
            target_names=severity_order,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    log_metrics(metrics, severity_order)

    # ── Feature importances ───────────────────────────────────────────────────
    importances = dict(zip(available, model.feature_importances_.round(4).tolist()))
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    log.info("  Top 10 features by importance:")
    for feat, score in top[:10]:
        log.info(f"    {feat:<50} {score:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(model, model_path)
    log.info(f"Saved model -> {model_path}")

    train_meta = {
        "borough":             slug,
        "trained_at":          datetime.now().isoformat(),
        "n_train":             int(len(X_train)),
        "n_test":              int(len(X_test)),
        "features_used":       available,
        "n_features":          len(available),
        "xgboost_params":      XGBOOST_PARAMS,
        "cv_f1_mean":          round(float(cv_scores.mean()), 4),
        "cv_f1_std":           round(float(cv_scores.std()), 4),
        "test_metrics":        {k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix")},
        "confusion_matrix":    metrics["confusion_matrix"],
        "feature_importances": importances,
    }
    with open(train_meta_path, "w") as f:
        json.dump(train_meta, f, indent=2)
    log.info(f"Saved training meta → {train_meta_path}")

    log.info(f"Training complete ✓  [borough={slug}]")
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train severity model")
    parser.add_argument(
        "--borough",
        type=str,
        default=None,
        help="Train on a single borough (e.g. QUEENS). Omit for all boroughs.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    args = parser.parse_args()
    run_train(borough=args.borough, cv_folds=args.cv_folds)