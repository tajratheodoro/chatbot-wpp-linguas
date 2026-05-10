"""LangChain PGVector integration for curriculum retrieval."""

from typing import Final

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from app.config import Settings, get_settings
from app.database.connection import build_pgvector_database_url

COLLECTION_NAME: Final[str] = "curriculum_lessons"
EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
EMBEDDING_DIMENSION: Final[int] = 1536


class VectorStoreError(RuntimeError):
    """Raised when curriculum context retrieval fails."""


def _require_openai_api_key(settings: Settings) -> str:
    """Return the configured OpenAI API key or raise a safe error."""
    if not settings.openai_api_key:
        raise VectorStoreError("OPENAI_API_KEY is not configured")
    return settings.openai_api_key


class CurriculumVectorStore:
    """Asynchronous RAG context provider backed by PostgreSQL and pgvector."""

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: OpenAIEmbeddings | None = None,
        vector_store: PGVector | None = None,
    ) -> None:
        """Initialize the LangChain PGVector store from application settings."""
        self._settings = settings or get_settings()
        self._embeddings = embeddings or OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=_require_openai_api_key(self._settings),
        )
        self._vector_store = vector_store or PGVector(
            embeddings=self._embeddings,
            collection_name=COLLECTION_NAME,
            connection=build_pgvector_database_url(str(self._settings.database_url)),
            embedding_length=EMBEDDING_DIMENSION,
            use_jsonb=True,
            async_mode=True,
        )

    async def search_context(self, query: str, limit: int = 3) -> str:
        """Return curriculum context relevant to a student message."""
        normalized_query = query.strip()
        if not normalized_query:
            return ""

        safe_limit = max(1, min(limit, 10))
        try:
            documents = await self._vector_store.asimilarity_search(
                normalized_query,
                k=safe_limit,
            )
        except Exception as exc:
            raise VectorStoreError("curriculum context search failed") from exc

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
