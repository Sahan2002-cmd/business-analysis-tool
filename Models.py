import enum
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text,
    DateTime, Boolean, ForeignKey, Enum
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.database import Base


# ─── Enums ────────────────────────────────────────────────────────

class PredictionStatus(str, enum.Enum):
    PENDING   = "pending"
    COMPLETED = "completed"
    FAILED    = "failed"

class ReportType(str, enum.Enum):
    ANNUAL    = "annual"
    QUARTERLY = "quarterly"
    MONTHLY   = "monthly"
    CUSTOM    = "custom"

class ProfitLossLabel(str, enum.Enum):
    PROFIT  = "PROFIT"
    LOSS    = "LOSS"
    NEUTRAL = "NEUTRAL"


# ─── Company ──────────────────────────────────────────────────────

class Company(Base):
    __tablename__ = "companies"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(200), nullable=False, index=True)
    ticker      = Column(String(20),  nullable=True)    # stock symbol e.g. APPL
    industry    = Column(String(100), nullable=True)
    country     = Column(String(100), nullable=True)
    description = Column(Text,        nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), onupdate=func.now())

    reports     = relationship("Report",     back_populates="company", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="company", cascade="all, delete-orphan")
    stock_data  = relationship("StockData",  back_populates="company", cascade="all, delete-orphan")


# ─── Report ───────────────────────────────────────────────────────

class Report(Base):
    """
    One uploaded business report (PDF / Excel / CSV).

    Core financial columns are extracted and stored as plain floats so
    they can be used directly as ML features without parsing JSON.

    raw_extracted_data (JSONB) stores everything else that was found in
    the document — flexible, no fixed schema required.
    """
    __tablename__ = "reports"

    id             = Column(Integer, primary_key=True, index=True)
    company_id     = Column(Integer, ForeignKey("companies.id"), nullable=False)
    report_type    = Column(Enum(ReportType), default=ReportType.ANNUAL)
    report_year    = Column(Integer, nullable=True)
    report_quarter = Column(Integer, nullable=True)

    # File storage
    original_filename = Column(String(300), nullable=True)
    file_path         = Column(String(500), nullable=True)
    file_type         = Column(String(20),  nullable=True)   # pdf / xlsx / csv

    # ── Extracted financial features (used directly by ML models) ──
    revenue            = Column(Float, nullable=True)
    net_profit         = Column(Float, nullable=True)
    gross_profit       = Column(Float, nullable=True)
    total_assets       = Column(Float, nullable=True)
    total_liabilities  = Column(Float, nullable=True)
    total_equity       = Column(Float, nullable=True)
    operating_income   = Column(Float, nullable=True)
    ebitda             = Column(Float, nullable=True)
    cash_flow          = Column(Float, nullable=True)
    revenue_growth     = Column(Float, nullable=True)   # % vs previous period
    profit_margin      = Column(Float, nullable=True)   # net_profit / revenue
    debt_to_equity     = Column(Float, nullable=True)
    current_ratio      = Column(Float, nullable=True)
    return_on_equity   = Column(Float, nullable=True)
    return_on_assets   = Column(Float, nullable=True)

    # ── Flexible storage for everything else found in the document ──
    raw_extracted_data = Column(JSONB, nullable=True)

    # Processing flags
    is_processed      = Column(Boolean, default=False)
    processing_notes  = Column(Text,    nullable=True)
    created_at        = Column(DateTime(timezone=True), server_default=func.now())

    company     = relationship("Company",    back_populates="reports")
    predictions = relationship("Prediction", back_populates="report", cascade="all, delete-orphan")


# ─── Prediction ───────────────────────────────────────────────────

class Prediction(Base):
    """
    Stores the output of every local ML model run.
    results_json holds the full model output — no schema change
    needed when you add new models.
    """
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True, index=True)
    company_id      = Column(Integer, ForeignKey("companies.id"), nullable=False)
    report_id       = Column(Integer, ForeignKey("reports.id"),   nullable=True)
    prediction_type = Column(String(50), nullable=False, index=True)
    status          = Column(Enum(PredictionStatus), default=PredictionStatus.PENDING)

    # Quick-access columns (copied from results_json for easy DB queries)
    profit_loss_label  = Column(Enum(ProfitLossLabel), nullable=True)
    profit_probability = Column(Float, nullable=True)
    confidence_score   = Column(Float, nullable=True)

    # Full output from all models
    results_json = Column(JSONB, nullable=True)
    # Stored structure example:
    # {
    #   "profit_loss":   {"label": "PROFIT", "probability": 0.84, "confidence": 0.91},
    #   "stock_forecast":{"7d": 145.2, "30d": 152.0, "trend": "bullish"},
    #   "demand":        {"score": 0.73, "trend": "rising"},
    #   "business_value":{"current": 1200000, "6m_forecast": 1350000},
    #   "model_versions":{"profit_loss": "v1.2", "stock": "v0.9"}
    # }

    error_message = Column(Text, nullable=True)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    company = relationship("Company", back_populates="predictions")
    report  = relationship("Report",  back_populates="predictions")


# ─── Stock Data ───────────────────────────────────────────────────

class StockData(Base):
    """
    Historical OHLCV data fetched via yfinance and stored locally.
    Once stored here, the app never needs to go online for stock data again.
    Prophet time-series models are trained from this table.
    """
    __tablename__ = "stock_data"

    id          = Column(Integer, primary_key=True, index=True)
    company_id  = Column(Integer, ForeignKey("companies.id"), nullable=False)
    date        = Column(DateTime(timezone=True), nullable=False, index=True)
    open_price  = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    high_price  = Column(Float, nullable=True)
    low_price   = Column(Float, nullable=True)
    volume      = Column(Float, nullable=True)

    company = relationship("Company", back_populates="stock_data")


# ─── ML Model Registry ────────────────────────────────────────────

class MLModelRegistry(Base):
    """
    Tracks every trained ML model saved to disk.
    When you retrain a model on new data, a new row is added here.
    The app always uses the latest ACTIVE version.
    """
    __tablename__ = "ml_model_registry"

    id           = Column(Integer, primary_key=True, index=True)
    model_name   = Column(String(100), nullable=False, index=True)
    # e.g. "profit_loss_classifier", "stock_forecaster", "demand_model"
    version      = Column(String(20),  nullable=False)
    file_path    = Column(String(500), nullable=False)   # path to .pkl file
    algorithm    = Column(String(100), nullable=True)    # e.g. "XGBoost", "Prophet"
    is_active    = Column(Boolean,     default=True)

    # Training metadata
    training_samples = Column(Integer, nullable=True)
    accuracy         = Column(Float,   nullable=True)
    precision_score  = Column(Float,   nullable=True)
    recall_score     = Column(Float,   nullable=True)
    f1_score         = Column(Float,   nullable=True)
    feature_names    = Column(JSONB,   nullable=True)   # list of features used
    training_notes   = Column(Text,    nullable=True)
    trained_at       = Column(DateTime(timezone=True), server_default=func.now())