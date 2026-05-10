"""Asynchronous SQLAlchemy connection management."""

from collections.abc import AsyncIterator
from typing import Final

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings, get_settings
from app.models.db_models import Base

ASYNC_PG_SCHEME: Final[str] = "postgresql+asyncpg://"
PGVECTOR_SCHEME: Final[str] = "postgresql+psycopg://"

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


class DatabaseConnectionError(RuntimeError):
    """Raised when database initialization or connection management fails."""


def _require_database_url(database_url: str) -> str:
    """Return a non-empty database URL or raise a safe configuration error."""
    normalized_url = database_url.strip()
    if not normalized_url:
        raise DatabaseConnectionError("DATABASE_URL is not configured")
    return normalized_url


def _replace_postgres_scheme(database_url: str, target_scheme: str) -> str:
    """Replace the PostgreSQL driver scheme without changing credentials."""
    normalized_url = _require_database_url(database_url)
    if normalized_url.startswith(target_scheme):
        return normalized_url
    if normalized_url.startswith("postgresql+asyncpg://"):
        return target_scheme + normalized_url.removeprefix("postgresql+asyncpg://")
    if normalized_url.startswith("postgresql+psycopg://"):
        return target_scheme + normalized_url.removeprefix("postgresql+psycopg://")
    if normalized_url.startswith("postgresql://"):
        return target_scheme + normalized_url.removeprefix("postgresql://")
    if normalized_url.startswith("postgres://"):
        return target_scheme + normalized_url.removeprefix("postgres://")
    return normalized_url


def build_sqlalchemy_database_url(database_url: str) -> str:
    """Build a SQLAlchemy asyncpg database URL from the configured URL."""
    return _replace_postgres_scheme(database_url, ASYNC_PG_SCHEME)


def build_pgvector_database_url(database_url: str) -> str:
    """Build a LangChain PGVector psycopg database URL from the configured URL."""
    return _replace_postgres_scheme(database_url, PGVECTOR_SCHEME)


def get_async_engine(settings: Settings | None = None) -> AsyncEngine:
    """Return the process-wide async SQLAlchemy engine."""
    global _engine
    if _engine is None:
        resolved_settings = settings or get_settings()
        database_url = build_sqlalchemy_database_url(str(resolved_settings.database_url))
        _engine = create_async_engine(database_url, pool_pre_ping=True)
    return _engine


def get_sessionmaker(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """Return the process-wide async session maker."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_async_engine(settings=settings),
            expire_on_commit=False,
        )
    return _sessionmaker


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield an async database session for FastAPI dependencies."""
    try:
        async with get_sessionmaker()() as session:
            yield session
    except SQLAlchemyError as exc:
        raise DatabaseConnectionError("database session failed") from exc


async def init_db(settings: Settings | None = None) -> None:
    """Initialize the pgvector extension and create database tables."""
    try:
        engine = get_async_engine(settings=settings)
        async with engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(Base.metadata.create_all)
    except SQLAlchemyError as exc:
        raise DatabaseConnectionError("database initialization failed") from exc
