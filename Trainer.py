"""
trainer.py
──────────
Trains your own ML models on data stored in YOUR database.
All training runs locally on your machine — no cloud, no API.

Models trained here:
  1. profit_loss_classifier  — predicts PROFIT / LOSS from financial ratios
  2. demand_model            — predicts business demand score (0.0 – 1.0)

Trained models are saved as .pkl files in data/models/
and registered in the ml_model_registry table.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.models import Report, MLModelRegistry


# ─── Feature columns used for training ────────────────────────────────────────
# These map directly to the columns in the reports table

PROFIT_LOSS_FEATURES = [
    "revenue",
    "net_profit",
    "gross_profit",
    "operating_income",
    "ebitda",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "cash_flow",
    "profit_margin",
    "revenue_growth",
    "debt_to_equity",
    "current_ratio",
    "return_on_equity",
    "return_on_assets",
]

DEMAND_FEATURES = [
    "revenue",
    "revenue_growth",
    "profit_margin",
    "return_on_equity",
    "cash_flow",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _next_version(model_name: str, db: Session) -> str:
    """Auto-increment version number for a model."""
    count = db.query(MLModelRegistry).filter(
        MLModelRegistry.model_name == model_name
    ).count()
    return f"v{count + 1}.0"


def _save_model(model, model_name: str, version: str) -> str:
    """Save a trained model to disk as a .pkl file. Returns the file path."""
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    path = os.path.join(settings.MODEL_DIR, f"{model_name}_{version}.pkl")
    joblib.dump(model, path)
    return path


def _register_model(
    db:               Session,
    model_name:       str,
    version:          str,
    file_path:        str,
    algorithm:        str,
    n_samples:        int,
    accuracy:         float,
    precision:        float,
    recall:           float,
    f1:               float,
    feature_names:    list,
    notes:            str = "",
):
    """Deactivate old versions and register the new one in the database."""
    # Mark all previous versions inactive
    db.query(MLModelRegistry).filter(
        MLModelRegistry.model_name == model_name
    ).update({"is_active": False})

    record = MLModelRegistry(
        model_name       = model_name,
        version          = version,
        file_path        = file_path,
        algorithm        = algorithm,
        is_active        = True,
        training_samples = n_samples,
        accuracy         = accuracy,
        precision_score  = precision,
        recall_score     = recall,
        f1_score         = f1,
        feature_names    = feature_names,
        training_notes   = notes,
    )
    db.add(record)
    db.commit()
    return record


# ─── 1. Profit / Loss Classifier ──────────────────────────────────────────────

def train_profit_loss_model(db: Session, notes: str = "") -> dict:
    """
    Trains an XGBoost classifier to predict PROFIT or LOSS.

    Training data:  all processed reports in your database that have
                    net_profit filled in.
    Label:          net_profit > 0  →  1 (PROFIT)
                    net_profit <= 0 →  0 (LOSS)

    The more reports you upload and process, the better this model gets.
    """
    # ── Load training data from your database ─────────────────────
    rows = db.query(Report).filter(
        Report.is_processed == True,
        Report.net_profit   != None,
    ).all()

    if len(rows) < settings.MIN_TRAINING_SAMPLES:
        return {
            "error": f"Not enough training data. Need at least {settings.MIN_TRAINING_SAMPLES} processed reports, have {len(rows)}."
        }

    # ── Build feature matrix ───────────────────────────────────────
    records = []
    for r in rows:
        record = {f: getattr(r, f, None) for f in PROFIT_LOSS_FEATURES}
        record["label"] = 1 if (r.net_profit or 0) > 0 else 0
        records.append(record)

    df = pd.DataFrame(records)
    df = df.fillna(df.median(numeric_only=True))  # impute missing values with median

    X = df[PROFIT_LOSS_FEATURES].values
    y = df["label"].values

    # ── Handle class imbalance with SMOTE ─────────────────────────
    if len(np.unique(y)) > 1:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

    # ── Train / test split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Model pipeline: scale + XGBoost ───────────────────────────
    model = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", XGBClassifier(
            n_estimators     = 200,
            max_depth        = 5,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            use_label_encoder= False,
            eval_metric      = "logloss",
            random_state     = 42,
        )),
    ])
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────
    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n── Profit/Loss Classifier ──────────────────────")
    print(classification_report(y_test, y_pred, target_names=["LOSS", "PROFIT"]))

    # ── Save and register ─────────────────────────────────────────
    version   = _next_version("profit_loss_classifier", db)
    file_path = _save_model(model, "profit_loss_classifier", version)
    _register_model(
        db=db, model_name="profit_loss_classifier",
        version=version, file_path=file_path,
        algorithm="XGBoost", n_samples=len(rows),
        accuracy=accuracy, precision=precision,
        recall=recall, f1=f1,
        feature_names=PROFIT_LOSS_FEATURES, notes=notes,
    )

    return {
        "model_name":       "profit_loss_classifier",
        "version":          version,
        "training_samples": len(rows),
        "accuracy":         round(accuracy,  4),
        "precision":        round(precision, 4),
        "recall":           round(recall,    4),
        "f1_score":         round(f1,        4),
    }


# ─── 2. Demand Score Regressor ────────────────────────────────────────────────

def train_demand_model(db: Session, notes: str = "") -> dict:
    """
    Trains a GradientBoosting regressor to predict a demand score (0.0 – 1.0).

    The demand score is derived from revenue_growth and profit_margin
    as a proxy for business demand strength.
    """
    rows = db.query(Report).filter(
        Report.is_processed  == True,
        Report.revenue_growth != None,
        Report.profit_margin  != None,
    ).all()

    if len(rows) < settings.MIN_TRAINING_SAMPLES:
        return {
            "error": f"Need at least {settings.MIN_TRAINING_SAMPLES} reports with revenue_growth and profit_margin."
        }

    records = []
    for r in rows:
        record = {f: getattr(r, f, None) for f in DEMAND_FEATURES}
        # Demand score: normalised composite of growth + margin
        rg = r.revenue_growth or 0
        pm = r.profit_margin  or 0
        score = min(max((rg * 0.6 + pm * 0.4 + 1) / 2, 0.0), 1.0)
        record["demand_score"] = score
        records.append(record)

    df = pd.DataFrame(records).fillna(df.median(numeric_only=True) if False else 0)

    X = df[DEMAND_FEATURES].values
    y = df["demand_score"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("scaler",    StandardScaler()),
        ("regressor", GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)),
    ])
    model.fit(X_train, y_train)

    from sklearn.metrics import mean_absolute_error, r2_score
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    print(f"\n── Demand Model ────────────────────────────────")
    print(f"   MAE: {mae:.4f}  |  R²: {r2:.4f}")

    version   = _next_version("demand_model", db)
    file_path = _save_model(model, "demand_model", version)
    _register_model(
        db=db, model_name="demand_model",
        version=version, file_path=file_path,
        algorithm="GradientBoosting", n_samples=len(rows),
        accuracy=r2, precision=0, recall=0, f1=0,
        feature_names=DEMAND_FEATURES, notes=notes,
    )

    return {
        "model_name":       "demand_model",
        "version":          version,
        "training_samples": len(rows),
        "r2_score":         round(r2,  4),
        "mae":              round(mae, 4),
    }