from fastapi.testclient import TestClient
import pytest

from app.api.curriculum import get_curriculum_vector_store
from app.config import get_settings
from app.config.settings import Settings
from app.database.vector_store import CurriculumVectorStore, EMBEDDING_DIMENSION
from app.models.schemas import LessonCreate
from main import app


class FakeCurriculumVectorStore:
    def __init__(self) -> None:
        self.lessons: list[tuple[str, str]] = []

    async def add_lesson(self, title: str, content: str) -> int:
        self.lessons.append((title, content))
        return 42


def test_lesson_create_normalizes_and_documents_content() -> None:
    lesson = LessonCreate(title="  Present simple  ", content="  Exercise: I go...\nRubric: ... ")

    assert lesson.title == "Present simple"
    assert lesson.content.startswith("Exercise:")
    assert "enunciado" in LessonCreate.model_fields["content"].description.lower()


def test_curriculum_admin_route_requires_api_key() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(ADMIN_API_KEY="secret")
    app.dependency_overrides[get_curriculum_vector_store] = lambda: FakeCurriculumVectorStore()
    try:
        response = TestClient(app).post(
            "/api/curriculum/lesson",
            json={"title": "Present simple", "content": "Exercise: correct the sentence."},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401


def test_curriculum_admin_route_requires_configured_api_key() -> None:
    response = TestClient(app).post(
        "/api/curriculum/lesson",
        headers={"X-Admin-API-Key": "secret"},
        json={"title": "Present simple", "content": "Exercise: correct the sentence."},
    )

    assert response.status_code == 503


def test_curriculum_admin_route_adds_lesson_with_valid_api_key() -> None:
    fake_store = FakeCurriculumVectorStore()
    app.dependency_overrides[get_settings] = lambda: Settings(ADMIN_API_KEY="secret")
    app.dependency_overrides[get_curriculum_vector_store] = lambda: fake_store
    try:
        response = TestClient(app).post(
            "/api/curriculum/lesson",
            headers={"X-Admin-API-Key": "secret"},
            json={
                "title": "Present simple",
                "content": "Exercise: Correct: I has a book.\nRubric: explain subject-verb agreement.",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json() == {
        "id": 42,
        "title": "Present simple",
        "status": "created",
    }
    assert fake_store.lessons == [
        (
            "Present simple",
            "Exercise: Correct: I has a book.\nRubric: explain subject-verb agreement.",
        )
    ]


def test_huggingface_mini_lm_embedding_dimension_is_384() -> None:
    assert EMBEDDING_DIMENSION == 384


@pytest.mark.asyncio()
async def test_vector_store_add_lesson_persists_open_source_embedding() -> None:
    class FakeEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            assert text == "Exercise: Correct the sentence.\nRubric: explain the tense."
            return [0.25] * EMBEDDING_DIMENSION

    class FakeSession:
        def __init__(self) -> None:
            self.lesson: object | None = None
            self.committed = False

        async def __aenter__(self) -> "FakeSession":
            return self

        async def __aexit__(self, *_exc_info: object) -> None:
            return None

        def add(self, lesson: object) -> None:
            self.lesson = lesson

        async def commit(self) -> None:
            self.committed = True

        async def refresh(self, lesson: object) -> None:
            lesson.id = 77

    fake_session = FakeSession()
    store = CurriculumVectorStore(
        embeddings=FakeEmbeddings(),
        session_factory=lambda: fake_session,
    )

    lesson_id = await store.add_lesson(
        title="Present simple",
        content="Exercise: Correct the sentence.\nRubric: explain the tense.",
    )

    assert lesson_id == 77
    assert fake_session.committed is True
    assert fake_session.lesson.lesson_title == "Present simple"
    assert fake_session.lesson.embedding == [0.25] * EMBEDDING_DIMENSION
