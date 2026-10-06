from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from ..core.config import settings


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create the v1 tables for local development.

    A future production deployment should use Alembic migrations instead of
    create_all. The application intentionally tolerates a temporarily
    unavailable database so /health and /predict remain usable during ML/API
    development.
    """
    from ..models import core  # noqa: F401

    Base.metadata.create_all(bind=engine)


def database_ping() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
