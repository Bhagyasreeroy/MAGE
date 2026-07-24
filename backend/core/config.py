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
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Vector Store ───────────────────────────────────────────────────────
    vector_store_backend: str = "chroma"  # "chroma" | "faiss"
    faiss_index_path: str = "./data/faiss_index"
    chroma_db_path: str = "./data/chroma_db"
    chroma_host: str = "chromadb"
    chroma_port: int = 8001

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
