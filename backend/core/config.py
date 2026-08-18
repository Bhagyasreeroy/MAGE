"""
core/config.py
──────────────
Centralised application settings loaded from environment variables.
All settings can be overridden via a .env file at the project root.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────
    app_name: str = "MAGE Backend"
    environment: str = "development"
    log_level: str = "INFO"

    # ── Server ─────────────────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # ── API Keys ───────────────────────────────────────────────────────────
    # ── Data retention (NFR-04) ───────────────────────────────────────────
    # Uploaded files live in Postgres as bytes, so "keep forever" is a real
    # cost as well as a privacy problem. Deliberately generous: a sweep that is
    # too eager is indistinguishable from data loss, and a dataset referenced by
    # a completed run is never collected regardless of age.
    dataset_retention_days: int = 30

    # ── Rate limiting (M8) ────────────────────────────────────────────────
    # Windows are per key, and the key is the authenticated user where there is
    # one (see core/rate_limit.py) — not the host, which would put a whole lab
    # behind one NAT into a single bucket. Kept configurable so the test suite,
    # which hammers these endpoints, can switch it off rather than become
    # timing-dependent.
    rate_limit_enabled: bool = True
    rate_limit_auth: str = "20/minute"        # login / register — credential stuffing
    rate_limit_analysis: str = "30/minute"    # /analysis/run — CPU and the full pipeline
    rate_limit_llm: str = "15/minute"         # explain + ask-in-English — spends Gemini quota

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # ── Vector Store ───────────────────────────────────────────────────────
    vector_store_backend: str = "chroma"  # "chroma" | "faiss"
    faiss_index_path: str = "./data/faiss_index"
    chroma_db_path: str = "./data/chroma_db"  # Chroma runs embedded (PersistentClient); no host/port needed

    # ── Data Pipeline ──────────────────────────────────────────────────────
    max_upload_size_mb: int = 100
    processing_engine: str = "pandas"  # "pandas" | "spark" | "dask"

    # ── Authentication / JWT ───────────────────────────────────────────────
    jwt_secret_key: str = "CHANGE-ME-IN-PRODUCTION-use-openssl-rand-hex-32"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    session_secret_key: str = "CHANGE-ME-SESSION-SECRET"
    google_client_id: str = ""
    google_client_secret: str = ""
    frontend_url: str = "http://localhost:3000"

    # ── Database ───────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://mage:mage@localhost:5432/mage"


# Singleton instance — import this everywhere instead of re-instantiating.
settings = Settings()
