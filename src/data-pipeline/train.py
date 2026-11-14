"""
train.py
--------
Trains an XGBoost classifier that outputs *calibrated* crash-severity
probabilities from location + time features.

Design (and why it changed from the SMOTE version):
  - Split first, on the full dataset. The test set keeps the real class mix
    (~71% No Injury, ~0.1% Fatal), so metrics describe production traffic.
  - No resampling. SMOTE interpolated one-hot street columns (synthetic
    "0.4 Jamaica Ave + 0.6 Queens Blvd" crashes) and inflated rare-class
    probabilities ~10–70×. Unweighted multi:softprob gives probabilities
    that match observed rates, which is what the dashboard displays.
  - Early stopping on a validation split carved from train (never the test set).
  - Metrics suited to a weak-signal, imbalanced, probabilistic problem:
    log loss vs a base-rate baseline, severe/fatal ROC-AUC, top-decile lift,
    and calibration (mean predicted vs observed). Macro-F1 is still reported
    for continuity but argmax labels are not the product.

Output (borough=QUEENS example):
  models/QUEENS/severity_model.joblib
  models/QUEENS/train_meta.json

Pipeline order:
  fetch.py → clean.py → features.py [--borough QUEENS] → train.py [--borough QUEENS]
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

XGBOOST_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": 2,
    "tree_method": "hist",
}
MAX_TREES = 2000
EARLY_STOPPING_ROUNDS = 50
TEST_SIZE = 0.2
VAL_SIZE = 0.15  # of the training split

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


# ── Model + metrics ───────────────────────────────────────────────────────────


def make_model(n_classes: int, n_estimators: int = MAX_TREES, early_stop: bool = True):
    return XGBClassifier(
        **XGBOOST_PARAMS,
        n_estimators=n_estimators,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS if early_stop else None,
        num_class=n_classes,
        objective="multi:softprob",
    )


def severe_score(proba: np.ndarray, severity_order: list[str]) -> np.ndarray:
    """P(Major Injury) + P(Fatal)."""
    idx = [severity_order.index(s) for s in ("Major Injury", "Fatal")]
    return proba[:, idx].sum(axis=1)


def evaluate(
    y: np.ndarray, proba: np.ndarray, prior: np.ndarray, severity_order: list[str]
) -> dict:
    """Probabilistic + ranking metrics, plus the classic ones for continuity."""
    fatal_idx = severity_order.index("Fatal")
    severe_idx = [severity_order.index(s) for s in ("Major Injury", "Fatal")]
    is_severe = np.isin(y, severe_idx).astype(int)
    is_fatal = (y == fatal_idx).astype(int)
    s = severe_score(proba, severity_order)
    top = s >= np.quantile(s, 0.9)
    pred = proba.argmax(axis=1)
    labels = list(range(len(severity_order)))

    ll = log_loss(y, proba, labels=labels)
    ll_base = log_loss(y, np.tile(prior, (len(y), 1)), labels=labels)
    return {
        # Probabilistic quality
        "log_loss": round(float(ll), 4),
        "baseline_log_loss": round(float(ll_base), 4),
        "log_loss_skill": round(float(1 - ll / ll_base), 4),
        # Ranking quality — can the model tell risky street/times from safe ones?
        "severe_auc": round(float(roc_auc_score(is_severe, s)), 4),
        "fatal_auc": round(float(roc_auc_score(is_fatal, proba[:, fatal_idx])), 4)
        if is_fatal.sum()
        else None,
        "top_decile_severe_rate": round(float(is_severe[top].mean()), 4),
        "overall_severe_rate": round(float(is_severe.mean()), 4),
        "top_decile_lift": round(float(is_severe[top].mean() / is_severe.mean()), 3),
        # Calibration — mean predicted probability vs observed frequency
        "calibration": {
            label: {
                "predicted": round(float(proba[:, k].mean()), 5),
                "observed": round(float((y == k).mean()), 5),
            }
            for k, label in enumerate(severity_order)
        },
        # Classic argmax metrics (kept for continuity; not the product)
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(
            precision_score(y, pred, average="macro", zero_division=0), 4
        ),
        "recall": round(recall_score(y, pred, average="macro", zero_division=0), 4),
        "f1": round(f1_score(y, pred, average="macro", zero_division=0), 4),
    }


def log_metrics(m: dict) -> None:
    log.info("── Test set metrics (real class distribution) ───────────")
    log.info(
        f"  Log loss  : {m['log_loss']:.4f}  (base-rate baseline {m['baseline_log_loss']:.4f}, "
        f"skill {m['log_loss_skill']:+.2%})"
    )
    log.info(f"  Severe AUC: {m['severe_auc']:.4f}   Fatal AUC: {m['fatal_auc']}")
    log.info(
        f"  Top-10% risk: {m['top_decile_severe_rate']:.2%} severe vs "
        f"{m['overall_severe_rate']:.2%} overall (lift {m['top_decile_lift']:.2f}×)"
    )
    for label, c in m["calibration"].items():
        log.info(
            f"  {label:<13} predicted {c['predicted']:.4f}  observed {c['observed']:.4f}"
        )
    log.info(f"  Macro-F1  : {m['f1']:.4f}  Accuracy: {m['accuracy']:.4f}")
    log.info("─────────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────


def run_train(borough: str | None = None, cv_folds: int = 5) -> XGBClassifier:
    if borough:
        borough = borough.upper()
        if borough not in VALID_BOROUGHS:
            raise ValueError(
                f"Invalid borough '{borough}'. Choose from: {VALID_BOROUGHS}"
            )

    slug = borough_slug(borough)
    out_dir = model_dir(borough)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "severity_model.joblib"
    train_meta_path = out_dir / "train_meta.json"

    log.info(f"Training model  [borough={slug}]")
    with open(meta_path(borough)) as f:
        meta = json.load(f)
    severity_order = meta["severity_order"]
    n_classes = len(severity_order)

    df = pl.read_parquet(features_path(borough))
    available = [c for c in meta["encoded_feature_names"] if c in df.columns]
    missing = set(meta["encoded_feature_names"]) - set(available)
    if missing:
        log.warning(f"  Feature columns missing (skipped): {missing}")

    X = df.select(available).to_numpy().astype(np.float32)
    y = df["severity_encoded"].to_numpy().astype(np.int8)
    del df
    log.info(f"  {len(y):,} crashes")

    # ── Splits: test (real distribution) → train / validation ─────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    del X, y
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=42
    )
    prior = np.bincount(y_train, minlength=n_classes) / len(y_train)
    base_rates = {
        label: round(float(prior[k]), 5) for k, label in enumerate(severity_order)
    }
    log.info(
        f"  Train {len(X_fit):,} · Val {len(X_val):,} · Test {len(X_test):,}  base rates {base_rates}"
    )

    # ── Fit with early stopping ───────────────────────────────────────────────
    model = make_model(n_classes)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=100)
    n_trees = int(model.best_iteration) + 1
    log.info(f"  Early stopping → {n_trees} trees")

    # ── Cross-validation on the training split (no resampling → no leakage) ──
    log.info(f"Running {cv_folds}-fold stratified CV with {n_trees} trees …")
    cv_auc, cv_ll = [], []
    for k, (tr, va) in enumerate(
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42).split(
            X_train, y_train
        )
    ):
        m = make_model(n_classes, n_estimators=n_trees, early_stop=False)
        m.fit(X_train[tr], y_train[tr], verbose=False)
        p = m.predict_proba(X_train[va])
        sev = np.isin(
            y_train[va],
            [severity_order.index("Major Injury"), severity_order.index("Fatal")],
        )
        cv_auc.append(roc_auc_score(sev, severe_score(p, severity_order)))
        cv_ll.append(log_loss(y_train[va], p, labels=list(range(n_classes))))
        log.info(
            f"  fold {k + 1}: severe AUC {cv_auc[-1]:.4f}  log loss {cv_ll[-1]:.4f}"
        )
    cv_auc, cv_ll = np.array(cv_auc), np.array(cv_ll)

    # ── Evaluate on the untouched, real-distribution test set ─────────────────
    proba = model.predict_proba(X_test)
    metrics = evaluate(y_test, proba, prior, severity_order)
    log_metrics(metrics)
    y_pred = proba.argmax(axis=1)

    importances = dict(zip(available, model.feature_importances_.round(4).tolist()))
    for feat, score in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        log.info(f"    {feat:<50} {score:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(model, model_path)
    log.info(f"Saved model -> {model_path}")

    train_meta = {
        "borough": slug,
        "trained_at": datetime.now(UTC).isoformat(),
        "model_version": 2,
        "approach": "calibrated (no resampling), early stopping",
        "n_train": int(len(X_fit)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_trees": n_trees,
        "class_counts_train": {
            label: int((y_train == k).sum()) for k, label in enumerate(severity_order)
        },
        "base_rates": base_rates,
        "features_used": available,
        "n_features": len(available),
        "xgboost_params": {**XGBOOST_PARAMS, "n_estimators": n_trees},
        "cv_severe_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_severe_auc_std": round(float(cv_auc.std()), 4),
        "cv_log_loss_mean": round(float(cv_ll.mean()), 4),
        "test_metrics": metrics,
        "per_class": classification_report(
            y_test,
            y_pred,
            labels=list(range(n_classes)),
            target_names=severity_order,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=list(range(n_classes))
        ).tolist(),
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
