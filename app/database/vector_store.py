"""Vector-store boundary for PostgreSQL with pgvector."""

from dataclasses import dataclass


@dataclass(slots=True)
class VectorSearchResult:
    """A single semantic search result from the vector store."""

    content: str
    score: float
    metadata: dict[str, str]


class PgVectorStore:
    """Placeholder async interface for future pgvector operations."""

    async def similarity_search(self, query: str, limit: int = 5) -> list[VectorSearchResult]:
        """Return semantically similar documents for a query."""
        _ = (query, limit)
        return []

