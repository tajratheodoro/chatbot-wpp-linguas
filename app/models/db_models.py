"""SQLAlchemy database models."""

from typing import Any

from pgvector.sqlalchemy import Vector
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""


class CurriculumLesson(Base):
    """Curriculum lesson content indexed for RAG retrieval."""

    __tablename__ = "curriculum_lessons"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    lesson_title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(384), nullable=False)


class ConversationHistory(Base):
    """Persisted chat history separated by WhatsApp session."""

    __tablename__ = "conversation_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(180), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
