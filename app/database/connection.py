"""Asynchronous PostgreSQL connection management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from app.config import Settings


class Database:
    """Manage an asyncpg connection pool."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create the connection pool if it does not already exist."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(str(self._settings.database_url))

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        """Acquire one database connection from the pool."""
        if self._pool is None:
            await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as connection:
            yield connection

