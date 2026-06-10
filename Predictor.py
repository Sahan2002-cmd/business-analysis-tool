"""
predictor.py
────────────
Loads your trained .pkl models from disk and runs predictions.
Uses Prophet for stock price time-series forecasting — fully local.

No internet. No API. Everything from your own trained models.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.models import MLModelRegistry, StockData, Report
from app.ml.trainer import PROFIT_LOSS_FEATURES, DEMAND_FEATURES


# ─── Load a model from disk ───────────────────────────────────────────────────

def _load_active_model(model_name: str, db: Session):
    """Find the active version in the registry and load it from disk."""
    record = db.query(MLModelRegistry).filter(
        MLModelRegistry.model_name == model_name,
        MLModelRegistry.is_active  == True,
    ).first()

    if not record:
        raise FileNotFoundError(
            f"No trained model found for '{model_name}'. "
            f"Run POST /api/v1/train first."
        )

    if not os.path.exists(record.file_path):
        raise FileNotFoundError(
            f"Model file missing at {record.file_path}. Retrain the model."
        )

    return joblib.load(record.file_path), record


# ─── 1. Profit / Loss Prediction ─────────────────────────────────────────────

def predict_profit_loss(report: Report, db: Session) -> dict:
    """
    Predicts PROFIT or LOSS for a given report using your trained XGBoost model.
    Returns the label, probability, and confidence.
    """
    model, registry = _load_active_model("profit_loss_classifier", db)

    # Build feature vector from the report's extracted financial data
    features = np.array([[
        getattr(report, f, 0) or 0 for f in PROFIT_LOSS_FEATURES
    ]])

    proba      = model.predict_proba(features)[0]   # [prob_loss, prob_profit]
    label_idx  = int(np.argmax(proba))
    label      = "PROFIT" if label_idx == 1 else "LOSS"
    probability= float(proba[label_idx])
    confidence = float(max(proba) - min(proba))      # spread = confidence proxy

    return {
        "label":       label,
        "probability": round(probability, 4),
        "confidence":  round(confidence,  4),
        "model_version": registry.version,
    }


# ─── 2. Demand Score Prediction ──────────────────────────────────────────────

def predict_demand(report: Report, db: Session) -> dict:
    """
    Predicts the business demand score (0.0 – 1.0) from financial metrics.
    1.0 = very high demand / strong growth
    0.0 = very low demand / declining
    """
    model, registry = _load_active_model("demand_model", db)

    features = np.array([[
        getattr(report, f, 0) or 0 for f in DEMAND_FEATURES
    ]])

    score = float(model.predict(features)[0])
    score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]

    if score >= 0.7:
        trend = "HIGH DEMAND"
    elif score >= 0.4:
        trend = "MODERATE DEMAND"
    else:
        trend = "LOW DEMAND"

    return {
        "score":         round(score, 4),
        "trend":         trend,
        "model_version": registry.version,
    }


# ─── 3. Stock Price Forecast (Prophet — fully local) ─────────────────────────

def predict_stock_forecast(company_id: int, db: Session) -> dict:
    """
    Trains a Prophet model on the historical stock data stored in your DB
    and forecasts the next 7, 30, and 90 days.

    Prophet runs entirely on your machine — no internet needed after
    stock data has been fetched once with yfinance.
    """
    try:
        from prophet import Prophet
    except ImportError:
        return {"error": "Prophet not installed. Run: pip install prophet"}

    # Load historical stock data from your database
    records = db.query(StockData).filter(
        StockData.company_id == company_id
    ).order_by(StockData.date).all()

    if len(records) < 30:
        return {
            "error": f"Need at least 30 days of stock data for company {company_id}. "
                     f"Currently have {len(records)} days. "
                     f"Run POST /api/v1/stocks/fetch/{company_id} first."
        }

    # Prophet expects a DataFrame with columns 'ds' (date) and 'y' (value)
    df = pd.DataFrame([{
        "ds": r.date,
        "y":  r.close_price,
    } for r in records if r.close_price is not None])

    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

    # Train Prophet model locally
    model = Prophet(
        daily_seasonality  = False,
        weekly_seasonality = True,
        yearly_seasonality = True,
        changepoint_prior_scale = 0.05,
    )
    model.fit(df)

    # Forecast 90 days ahead
    future   = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    # Extract key forecast points
    last_actual = float(df["y"].iloc[-1])
    tail        = forecast.tail(90)

    def _at_day(n):
        row = tail.iloc[n - 1]
        return round(float(row["yhat"]), 2)

    f7d  = _at_day(7)
    f30d = _at_day(30)
    f90d = _at_day(90)

    # Trend direction
    if f30d > last_actual * 1.03:
        trend = "BULLISH"
    elif f30d < last_actual * 0.97:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    return {
        "current_price":    round(last_actual, 2),
        "forecast_7d":      f7d,
        "forecast_30d":     f30d,
        "forecast_90d":     f90d,
        "trend":            trend,
        "training_records": len(df),
    }


# ─── 4. Business Value Estimate ──────────────────────────────────────────────

def estimate_business_value(report: Report) -> dict:
    """
    Estimates current and projected business value using financial multiples.
    A rules-based approach — no ML model needed, works immediately.

    Methods:
      - P/E multiple  (if net_profit available)
      - Revenue multiple (if revenue available)
      - Asset-based  (if total_equity available)
    """
    estimates = {}

    if report.net_profit and report.net_profit > 0:
        # Industry average P/E multiple ~15x for a baseline estimate
        estimates["pe_based"] = round(report.net_profit * 15, 2)

    if report.revenue and report.revenue > 0:
        # Revenue multiple ~1.5x for most industries
        estimates["revenue_based"] = round(report.revenue * 1.5, 2)

    if report.total_equity and report.total_equity > 0:
        estimates["asset_based"] = round(report.total_equity, 2)

    if not estimates:
        return {"error": "Insufficient financial data to estimate business value."}

    # Use average of available methods
    avg = sum(estimates.values()) / len(estimates)

    return {
        "estimated_value":   round(avg, 2),
        "methods_used":      list(estimates.keys()),
        "breakdown":         estimates,
        "6m_projection":     round(avg * 1.08, 2),  # +8% growth projection
        "12m_projection":    round(avg * 1.15, 2),  # +15% growth projection
    }