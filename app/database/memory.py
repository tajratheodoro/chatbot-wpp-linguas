"""Async PostgreSQL chat history for LangChain conversational memory."""

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError

from app.config import Settings
from app.database.connection import get_sessionmaker
from app.models.db_models import ConversationHistory


class ChatMemoryError(RuntimeError):
    """Raised when persisted chat memory cannot be read or written."""


class AsyncSessionContext(Protocol):
    """Protocol for async SQLAlchemy session context managers."""

    async def __aenter__(self) -> Any:
        """Enter the async session context."""

    async def __aexit__(self, *args: object) -> None:
        """Exit the async session context."""


class AsyncSessionFactory(Protocol):
    """Protocol for async SQLAlchemy session factories."""

    def __call__(self) -> AsyncSessionContext:
        """Create an async session context manager."""


class PostgresAsyncChatMessageHistory(BaseChatMessageHistory):
    """Persist LangChain chat messages in PostgreSQL by WhatsApp session id."""

    def __init__(
        self,
        session_id: str,
        session_factory: AsyncSessionFactory | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize persisted history for a single student session."""
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise ValueError("session_id must not be empty")
        self.session_id = normalized_session_id
        self._session_factory = session_factory or get_sessionmaker(settings=settings)

    @property
    def messages(self) -> list[BaseMessage]:
        """Return messages synchronously only when no event loop is running."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aget_messages())
        raise ChatMemoryError("use aget_messages inside async code")

    async def aget_messages(self) -> list[BaseMessage]:
        """Return this session history ordered by timestamp."""
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(ConversationHistory)
                    .where(ConversationHistory.session_id == self.session_id)
                    .order_by(ConversationHistory.timestamp.asc(), ConversationHistory.id.asc())
                )
                rows = list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise ChatMemoryError("failed to load conversation history") from exc

        return [_to_langchain_message(row) for row in rows]

    async def add_user_message(self, message: HumanMessage | str) -> None:
        """Persist a user message for this session."""
        content = message.content if isinstance(message, HumanMessage) else message
        await self._add_message(role="user", content=str(content))

    async def add_ai_message(self, message: AIMessage | str) -> None:
        """Persist an assistant message for this session."""
        content = message.content if isinstance(message, AIMessage) else message
        await self._add_message(role="assistant", content=str(content))

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Persist a batch of LangChain messages asynchronously."""
        try:
            async with self._session_factory() as session:
                for message in messages:
                    role = "assistant" if isinstance(message, AIMessage) else "user"
                    session.add(
                        ConversationHistory(
                            session_id=self.session_id,
                            role=role,
                            content=str(message.content),
                        )
                    )
                await session.commit()
        except SQLAlchemyError as exc:
            raise ChatMemoryError("failed to persist conversation history") from exc

    async def clear(self) -> None:
        """Delete all persisted messages for this session."""
        try:
            async with self._session_factory() as session:
                await session.execute(
                    delete(ConversationHistory).where(
                        ConversationHistory.session_id == self.session_id
                    )
                )
                await session.commit()
        except SQLAlchemyError as exc:
            raise ChatMemoryError("failed to clear conversation history") from exc

    async def aclear(self) -> None:
        """Delete all persisted messages for this session asynchronously."""
        await self.clear()

    async def _add_message(self, role: str, content: str) -> None:
        normalized_content = content.strip()
        if not normalized_content:
            return
        try:
            async with self._session_factory() as session:
                session.add(
                    ConversationHistory(
                        session_id=self.session_id,
                        role=role,
                        content=normalized_content,
                    )
                )
                await session.commit()
        except SQLAlchemyError as exc:
            raise ChatMemoryError("failed to persist conversation message") from exc


def _to_langchain_message(row: ConversationHistory) -> BaseMessage:
    """Convert a persisted row into a LangChain message."""
    if row.role == "assistant":
        return AIMessage(content=row.content)
    return HumanMessage(content=row.content)
