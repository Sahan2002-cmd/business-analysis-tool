from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ── Database ───────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://bizadmin:bizpassword123@localhost:5432/bizanalysis"

    # ── App ────────────────────────────────────────────────────────
    APP_NAME:    str = "BizAnalysis API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    SECRET_KEY:  str = "change-this-before-production"

    # ── File uploads ───────────────────────────────────────────────
    UPLOAD_DIR:         str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 50

    # ── Local ML ───────────────────────────────────────────────────
    # Trained models are saved here as .pkl files
    MODEL_DIR:              str = "data/models"
    MIN_TRAINING_SAMPLES:   int = 10

    class Config:
        env_file      = ".env"
        case_sensitive = True


settings = Settings()