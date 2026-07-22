"""
core/database.py
────────────────
Async SQLAlchemy database engine + session factory.

Postgres (asyncpg) is the primary target — see DATABASE_URL in
core/config.py / docker-compose.yml. sqlite+aiosqlite is still accepted
(e.g. for a dependency-free test run) since it needs no special handling
beyond the connect_args below.
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from backend.core.config import settings


engine = create_async_engine(
    settings.database_url,
    echo=settings.environment == "development",
    # NullPool: a pooled asyncpg connection is bound to the event loop it
    # was created on, and Starlette's TestClient can run each request on a
    # fresh loop — reusing a pooled connection across loops raises
    # "another operation is in progress". A fresh connection per request
    # avoids that at a small cost, appropriate at this project's scale.
    poolclass=NullPool,
    # SQLite needs this to allow async usage; ignored by other dialects.
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
