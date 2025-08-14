"""
config.py
---------

Centralized configuration loader for DB, API, and environment settings.

Features:
    - Loads secrets from `.env` or secret manager (AWS Secrets Manager placeholder).
    - Uses Pydantic for validation & type safety.
    - Supports multiple environments (dev, staging, prod).
    - No hardcoded secrets in code.
    - Central place for logging settings.

Author: Varun-engineer mode (20+ years experience)
"""

from __future__ import annotations
import os
import logging
from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ---- Environment ----
    ENV: str = Field("dev", description="Environment: dev, staging, prod")

    # ---- Database ----
    DB_HOST: str
    DB_PORT: int = 5432
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    # ---- LLM API ----
    LLM_API_KEY: str = Field(..., description="API key for LLM provider (e.g., OpenAI)")
    LLM_MODEL: str = Field("gpt-4", description="Default LLM model for SQL generation")

    # ---- Logging ----
    LOG_LEVEL: str = Field("INFO", description="Logging level")

    # ---- Optional: External API configs ----
    EXTERNAL_API_BASE_URL: Optional[str] = None
    EXTERNAL_API_TOKEN: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @validator("ENV")
    def validate_env(cls, v):
        if v not in {"dev", "staging", "prod"}:
            raise ValueError("ENV must be one of: dev, staging, prod")
        return v

    @property
    def sqlalchemy_url(self) -> str:
        """Build SQLAlchemy connection string for PostgreSQL."""
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def psycopg2_dsn(self) -> str:
        """Build psycopg2 DSN connection string."""
        return f"host={self.DB_HOST} port={self.DB_PORT} dbname={self.DB_NAME} user={self.DB_USER} password={self.DB_PASSWORD}"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings loader.

    In production, you could extend this to load from AWS Secrets Manager:
        import boto3, json
        client = boto3.client("secretsmanager")
        secret = json.loads(client.get_secret_value(SecretId="my-secret-id")["SecretString"])
    """
    settings = Settings()

    # Configure logging globally
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger(__name__).info(f"Loaded settings for environment: {settings.ENV}")

    return settings


# ---------- Unit Test Stub ----------
if __name__ == "__main__":
    cfg = get_settings()
    print("ENV:", cfg.ENV)
    print("DB URL:", cfg.sqlalchemy_url)
    print("LLM Model:", cfg.LLM_MODEL)
      
