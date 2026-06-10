from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
import os

from app.db.database import get_db
from app.models.models import Report, Company
from app.schemas.schemas import ReportResponse
from app.services.report_parser import parse_report
from app.core.config import settings

router = APIRouter()


@router.post("/upload", response_model=ReportResponse, status_code=201)
async def upload_report(
    company_id:  int        = Form(...),
    report_year: int        = Form(...),
    report_type: str        = Form("annual"),
    file:        UploadFile = File(...),
    db:          Session    = Depends(get_db),
):
    """
    Upload a business report (PDF, Excel or CSV).
    The file is saved locally and parsed immediately using local NLP + regex.
    All extracted financial data is stored in the database.
    """
    # Verify company exists
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    # Validate file type
    allowed = {".pdf", ".xlsx", ".xls", ".csv"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {ext} not allowed. Use PDF, Excel or CSV.")

    # Check size
    contents = await file.read()
    if len(contents) / (1024 * 1024) > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File too large. Max {settings.MAX_UPLOAD_SIZE_MB}MB.")

    # Save file to local disk
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(settings.UPLOAD_DIR, f"{company_id}_{report_year}_{file.filename}")
    with open(save_path, "wb") as f:
        f.write(contents)

    # Parse report locally (no external API)
    extracted = parse_report(save_path, ext)
    parse_error = extracted.pop("error", None)

    # Store in database
    db_report = Report(
        company_id        = company_id,
        report_type       = report_type,
        report_year       = report_year,
        original_filename = file.filename,
        file_path         = save_path,
        file_type         = ext.lstrip("."),
        # Core financial columns
        revenue           = extracted.get("revenue"),
        net_profit        = extracted.get("net_profit"),
        gross_profit      = extracted.get("gross_profit"),
        operating_income  = extracted.get("operating_income"),
        ebitda            = extracted.get("ebitda"),
        total_assets      = extracted.get("total_assets"),
        total_liabilities = extracted.get("total_liabilities"),
        total_equity      = extracted.get("total_equity"),
        cash_flow         = extracted.get("cash_flow"),
        revenue_growth    = extracted.get("revenue_growth"),
        profit_margin     = extracted.get("profit_margin"),
        debt_to_equity    = extracted.get("debt_to_equity"),
        current_ratio     = extracted.get("current_ratio"),
        return_on_equity  = extracted.get("return_on_equity"),
        return_on_assets  = extracted.get("return_on_assets"),
        # Everything else
        raw_extracted_data = extracted,
        is_processed       = parse_error is None,
        processing_notes   = parse_error,
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report


@router.get("/company/{company_id}", response_model=List[ReportResponse])
def get_company_reports(company_id: int, db: Session = Depends(get_db)):
    return db.query(Report).filter(Report.company_id == company_id).all()


@router.get("/{report_id}", response_model=ReportResponse)
def get_report(report_id: int, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report