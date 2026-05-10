"""PostgreSQL pgvector integration for curriculum retrieval and ingestion."""

import asyncio
from typing import Any, Final, Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.config import Settings, get_settings
from app.database.connection import get_sessionmaker
from app.models.db_models import CurriculumLesson

COLLECTION_NAME: Final[str] = "curriculum_lessons"
HUGGINGFACE_MODEL_NAME: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: Final[int] = 384


class VectorStoreError(RuntimeError):
    """Raised when curriculum context retrieval or ingestion fails."""


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


class CurriculumVectorStore:
    """RAG store backed by the custom `curriculum_lessons` pgvector table."""

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
        session_factory: AsyncSessionFactory | None = None,
    ) -> None:
        """Initialize the store with open-source Hugging Face embeddings."""
        self._settings = settings or get_settings()
        self._embeddings = embeddings or _create_huggingface_embeddings()
        self._session_factory = session_factory or get_sessionmaker(settings=self._settings)

    async def add_lesson(self, title: str, content: str) -> int:
        """Embed and persist a curriculum lesson for future RAG retrieval.

        For best retrieval quality, `content` should include the complete
        exercise statement, expected answer, target grammar/vocabulary topic,
        common learner mistakes, and concise correction guidelines.
        """
        normalized_title = title.strip()
        normalized_content = content.strip()
        if not normalized_title:
            raise VectorStoreError("lesson title must not be empty")
        if not normalized_content:
            raise VectorStoreError("lesson content must not be empty")

        embedding = await asyncio.to_thread(self._embeddings.embed_query, normalized_content)
        try:
            async with self._session_factory() as session:
                lesson = CurriculumLesson(
                    lesson_title=normalized_title,
                    content=normalized_content,
                    embedding=embedding,
                )
                session.add(lesson)
                await session.commit()
                await session.refresh(lesson)
                return int(lesson.id)
        except SQLAlchemyError as exc:
            raise VectorStoreError("curriculum lesson ingestion failed") from exc

    async def search_context(self, query: str, limit: int = 3) -> str:
        """Return curriculum context relevant to a student message."""
        normalized_query = query.strip()
        if not normalized_query:
            return ""

        safe_limit = max(1, min(limit, 10))
        query_embedding = await asyncio.to_thread(self._embeddings.embed_query, normalized_query)
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(CurriculumLesson)
                    .order_by(CurriculumLesson.embedding.cosine_distance(query_embedding))
                    .limit(safe_limit)
                )
                lessons = list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise VectorStoreError("curriculum context search failed") from exc

        documents = [
            Document(
                page_content=lesson.content,
                metadata={"lesson_title": lesson.lesson_title, "id": lesson.id},
            )
            for lesson in lessons
        ]
        return self._format_documents(documents)

    @staticmethod
    def _format_documents(documents: list[Document]) -> str:
        """Format retrieved documents into a compact prompt context."""
        chunks: list[str] = []
        for index, document in enumerate(documents, start=1):
            title = document.metadata.get("lesson_title") or document.metadata.get("title")
            prefix = f"[{index}] {title}" if title else f"[{index}]"
            chunks.append(f"{prefix}\n{document.page_content}")
        return "\n\n".join(chunks)


def _create_huggingface_embeddings() -> Embeddings:
    """Create all-MiniLM-L6-v2 embeddings from langchain-huggingface."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise VectorStoreError("langchain-huggingface is not installed") from exc

    return HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL_NAME)
