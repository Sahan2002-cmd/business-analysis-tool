from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping  = True,
    pool_size      = 10,
    max_overflow   = 20,
    echo           = settings.ENVIRONMENT == "development",
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — one DB session per request, auto-closed after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()