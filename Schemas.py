from pydantic import BaseModel, Field
from typing import Optional, Any, List
from datetime import datetime


# ─── Company ──────────────────────────────────────────────────────

class CompanyCreate(BaseModel):
    name:        str
    ticker:      Optional[str] = None
    industry:    Optional[str] = None
    country:     Optional[str] = None
    description: Optional[str] = None

class CompanyResponse(CompanyCreate):
    id:         int
    created_at: datetime
    class Config:
        from_attributes = True


# ─── Report ───────────────────────────────────────────────────────

class ReportResponse(BaseModel):
    id:                int
    company_id:        int
    report_type:       str
    report_year:       Optional[int]
    original_filename: Optional[str]
    file_type:         Optional[str]
    # Core financials
    revenue:           Optional[float]
    net_profit:        Optional[float]
    gross_profit:      Optional[float]
    profit_margin:     Optional[float]
    revenue_growth:    Optional[float]
    debt_to_equity:    Optional[float]
    # Flexible extras
    raw_extracted_data: Optional[Any]
    is_processed:       bool
    created_at:         datetime
    class Config:
        from_attributes = True


# ─── Prediction ───────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    id:                 int
    company_id:         int
    report_id:          Optional[int]
    prediction_type:    str
    status:             str
    profit_loss_label:  Optional[str]
    profit_probability: Optional[float]
    confidence_score:   Optional[float]
    results_json:       Optional[Any]
    error_message:      Optional[str]
    created_at:         datetime
    class Config:
        from_attributes = True


# ─── ML Model Registry ────────────────────────────────────────────

class ModelRegistryResponse(BaseModel):
    id:               int
    model_name:       str
    version:          str
    algorithm:        Optional[str]
    is_active:        bool
    training_samples: Optional[int]
    accuracy:         Optional[float]
    f1_score:         Optional[float]
    trained_at:       datetime
    class Config:
        from_attributes = True


# ─── Training ─────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    model_type: str = Field(
        ...,
        description="Which model to train: 'profit_loss', 'demand', 'stock_forecast'"
    )
    notes: Optional[str] = None

class TrainResponse(BaseModel):
    message:          str
    model_name:       str
    version:          str
    training_samples: int
    accuracy:         Optional[float]
    f1_score:         Optional[float]