from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"


class Settings(BaseSettings):
    app_name: str = "X-IDS"
    environment: str = "development"
    debug: bool = True
    database_url: str = "postgresql+psycopg://xids:change-me@localhost:5432/xids"
    ml_artifact_dir: str = str(DEFAULT_ARTIFACT_DIR)
    jwt_secret_key: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    replay_default_rate: int = 50
    frontend_origin: str = "http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_FILE) if DEFAULT_ENV_FILE.exists() else None,
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
