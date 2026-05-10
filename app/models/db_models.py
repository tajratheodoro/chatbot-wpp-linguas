"""SQLAlchemy database models."""

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Text
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
