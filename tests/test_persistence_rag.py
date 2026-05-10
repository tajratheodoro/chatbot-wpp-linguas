from types import SimpleNamespace

import pytest

from app.database.connection import (
    build_pgvector_database_url,
    build_sqlalchemy_database_url,
)
from app.models.db_models import CurriculumLesson
from app.services.ai_orchestrator import AIOrchestrator


class FakeVectorStore:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    async def search_context(self, query: str, limit: int = 3) -> str:
        self.queries.append((query, limit))
        return "Lesson context"


class FakeChatModel:
    def __init__(self) -> None:
        self.messages: object | None = None

    async def ainvoke(self, messages: object) -> SimpleNamespace:
        self.messages = messages
        return SimpleNamespace(content="Corrected answer")


def test_curriculum_lesson_uses_embedding_dimension() -> None:
    embedding_column = CurriculumLesson.__table__.columns["embedding"]

    assert embedding_column.type.dim == 1536


def test_database_url_driver_conversion_keeps_credentials_external() -> None:
    raw_url = "postgresql://user:secret@localhost:5432/chatbot"

    assert build_sqlalchemy_database_url(raw_url).startswith("postgresql+asyncpg://")
    assert build_pgvector_database_url(raw_url).startswith("postgresql+psycopg://")


@pytest.mark.asyncio()
async def test_orchestrator_searches_context_before_model_call() -> None:
    vector_store = FakeVectorStore()
    model = FakeChatModel()
    orchestrator = AIOrchestrator(model=model, vector_store=vector_store)

    response = await orchestrator.generate_response("I has a apple")

    assert response == "Corrected answer"
    assert vector_store.queries == [("I has a apple", 3)]
    assert "Lesson context" in str(model.messages)
