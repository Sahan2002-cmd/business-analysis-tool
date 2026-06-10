from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List

from app.db.database import get_db
from app.models.models import Prediction, Report, PredictionStatus, ProfitLossLabel
from app.schemas.schemas import PredictionResponse, TrainRequest, TrainResponse

router = APIRouter()


# ─── Trigger full analysis for a report ──────────────────────────────────────

@router.post("/report/{report_id}", response_model=PredictionResponse, status_code=202)
def run_prediction(
    report_id:        int,
    background_tasks: BackgroundTasks,
    db:               Session = Depends(get_db),
):
    """
    Trigger full ML analysis for an uploaded report.
    Runs in background — returns immediately with status=pending.
    Poll GET /api/v1/predict/{prediction_id} for the result.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if not report.is_processed:
        raise HTTPException(status_code=400, detail="Report parsing failed or not yet processed.")

    prediction = Prediction(
        company_id      = report.company_id,
        report_id       = report_id,
        prediction_type = "full_analysis",
        status          = PredictionStatus.PENDING,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    background_tasks.add_task(_run_all_models, prediction.id, report_id)
    return prediction


# ─── Train your own models ────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest, db: Session = Depends(get_db)):
    """
    Train (or retrain) one of your local ML models on all data in your database.

    model_type options:
      - "profit_loss"    → trains profit/loss XGBoost classifier
      - "demand"         → trains demand score regressor
      - "stock_forecast" → Prophet trains at prediction time (no pre-training needed)
    """
    from app.ml.trainer import train_profit_loss_model, train_demand_model

    if request.model_type == "profit_loss":
        result = train_profit_loss_model(db, notes=request.notes or "")
    elif request.model_type == "demand":
        result = train_demand_model(db, notes=request.notes or "")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_type '{request.model_type}'. Choose: profit_loss, demand, stock_forecast"
        )

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return TrainResponse(
        message          = f"Model '{request.model_type}' trained successfully.",
        model_name       = result.get("model_name", request.model_type),
        version          = result.get("version", "v1.0"),
        training_samples = result.get("training_samples", 0),
        accuracy         = result.get("accuracy") or result.get("r2_score"),
        f1_score         = result.get("f1_score"),
    )


# ─── Poll for result ──────────────────────────────────────────────────────────

@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction


@router.get("/company/{company_id}", response_model=List[PredictionResponse])
def get_company_predictions(company_id: int, db: Session = Depends(get_db)):
    return (
        db.query(Prediction)
        .filter(Prediction.company_id == company_id)
        .order_by(Prediction.created_at.desc())
        .all()
    )


# ─── Background: run all local ML models ─────────────────────────────────────

def _run_all_models(prediction_id: int, report_id: int):
    """
    Runs all trained local models in the background.
    Updates the prediction record when done.
    """
    from app.db.database import SessionLocal
    from app.ml.predictor import (
        predict_profit_loss,
        predict_demand,
        predict_stock_forecast,
        estimate_business_value,
    )

    db = SessionLocal()
    try:
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        report     = db.query(Report).filter(Report.id == report_id).first()

        results = {}

        # 1. Profit / Loss
        try:
            pl = predict_profit_loss(report, db)
            results["profit_loss"] = pl
            prediction.profit_loss_label  = ProfitLossLabel(pl["label"])
            prediction.profit_probability = pl["probability"]
            prediction.confidence_score   = pl["confidence"]
        except Exception as e:
            results["profit_loss"] = {"error": str(e)}

        # 2. Demand score
        try:
            results["demand"] = predict_demand(report, db)
        except Exception as e:
            results["demand"] = {"error": str(e)}

        # 3. Stock forecast (Prophet — trains on your stored historical data)
        try:
            results["stock_forecast"] = predict_stock_forecast(report.company_id, db)
        except Exception as e:
            results["stock_forecast"] = {"error": str(e)}

        # 4. Business value estimate (rules-based, always available)
        results["business_value"] = estimate_business_value(report)

        prediction.results_json = results
        prediction.status       = PredictionStatus.COMPLETED
        db.commit()

    except Exception as e:
        prediction.status        = PredictionStatus.FAILED
        prediction.error_message = str(e)
        db.commit()
    finally:
        db.close()