from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.db.database import engine, Base
from app.api.routes import companies, reports, predictions, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    print("✅ All database tables created / verified")
    print("📂 Upload folder:", settings.UPLOAD_DIR)
    print("🤖 Models folder:", settings.MODEL_DIR)
    yield


app = FastAPI(
    title       = settings.APP_NAME,
    version     = settings.APP_VERSION,
    description = (
        "Business Analysis & Decision Tool — "
        "100% local ML: profit/loss prediction, stock forecasting, "
        "demand analysis. Your data, your models, your machine."
    ),
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(health.router,      prefix="/api/v1",           tags=["Health"])
app.include_router(companies.router,   prefix="/api/v1/companies",  tags=["Companies"])
app.include_router(reports.router,     prefix="/api/v1/reports",    tags=["Reports"])
app.include_router(predictions.router, prefix="/api/v1/predict",    tags=["Predictions & Training"])


@app.get("/")
def root():
    return {
        "app":     settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs":    "/docs",
        "note":    "100% local ML — no external APIs",
    }