"""
core/database.py
────────────────
Async SQLAlchemy database engine + session factory.

Uses aiosqlite for local development.  Switch the DATABASE_URL
in settings to PostgreSQL (asyncpg) for production.
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.core.config import settings


engine = create_async_engine(
    settings.database_url,
    echo=settings.environment == "development",
    # SQLite needs this to allow async usage
    connect_args={"check_same_thread": False}
    if "sqlite" in settings.database_url
    else {},
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields an async DB session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables (call once at startup)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
