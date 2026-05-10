from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.database.memory import PostgresAsyncChatMessageHistory
from app.models.db_models import ConversationHistory
from app.services.ai_orchestrator import AIOrchestrator


def test_conversation_history_model_separates_sessions() -> None:
    columns = ConversationHistory.__table__.columns

    assert "session_id" in columns
    assert "role" in columns
    assert "content" in columns
    assert "timestamp" in columns


@pytest.mark.asyncio()
async def test_postgres_history_returns_messages_ordered_by_timestamp() -> None:
    history = PostgresAsyncChatMessageHistory(
        session_id="5511999999999@s.whatsapp.net",
        session_factory=lambda: FakeSession(
            rows=[
                ConversationHistory(
                    session_id="5511999999999@s.whatsapp.net",
                    role="assistant",
                    content="Corrected answer",
                    timestamp=datetime(2026, 1, 1, 10, 1, tzinfo=UTC),
                ),
                ConversationHistory(
                    session_id="5511999999999@s.whatsapp.net",
                    role="user",
                    content="I has a apple",
                    timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
                ),
            ]
        ),
    )

    messages = await history.aget_messages()

    assert [type(message) for message in messages] == [HumanMessage, AIMessage]
    assert [message.content for message in messages] == ["I has a apple", "Corrected answer"]


@pytest.mark.asyncio()
async def test_postgres_history_persists_user_and_ai_messages() -> None:
    fake_session = FakeSession(rows=[])
    history = PostgresAsyncChatMessageHistory(
        session_id="5511999999999@s.whatsapp.net",
        session_factory=lambda: fake_session,
    )

    await history.add_user_message("I has a apple")
    await history.add_ai_message("Use 'have' with I.")

    assert [(row.session_id, row.role, row.content) for row in fake_session.added] == [
        ("5511999999999@s.whatsapp.net", "user", "I has a apple"),
        ("5511999999999@s.whatsapp.net", "assistant", "Use 'have' with I."),
    ]
    assert fake_session.commits == 2


@pytest.mark.asyncio()
async def test_orchestrator_passes_session_id_to_message_history() -> None:
    model = FakeChatModel()
    history_factory = FakeHistoryFactory()
    orchestrator = AIOrchestrator(
        model=model,
        vector_store=FakeVectorStore(),
        guardrails_engine=AllowingGuardrails(),
        history_factory=history_factory,
    )

    response = await orchestrator.generate_response(
        student_message="I has a apple",
        session_id="5511999999999@s.whatsapp.net",
    )

    assert response == "Corrected answer"
    assert history_factory.session_ids == ["5511999999999@s.whatsapp.net"]


class FakeScalarResult:
    def __init__(self, rows: list[ConversationHistory]) -> None:
        self._rows = rows

    def all(self) -> list[ConversationHistory]:
        return self._rows


class FakeResult:
    def __init__(self, rows: list[ConversationHistory]) -> None:
        self._rows = rows

    def scalars(self) -> FakeScalarResult:
        return FakeScalarResult(self._rows)


class FakeSession:
    def __init__(self, rows: list[ConversationHistory]) -> None:
        self.rows = rows
        self.added: list[ConversationHistory] = []
        self.commits = 0
        self.deleted = False

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        return None

    async def execute(self, statement: object) -> FakeResult:
        if "DELETE" in str(statement):
            self.deleted = True
        return FakeResult(sorted(self.rows, key=lambda row: row.timestamp))

    def add(self, row: ConversationHistory) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        self.commits += 1


class FakeVectorStore:
    async def search_context(self, query: str, limit: int = 3) -> str:
        return "Lesson context"


class AllowingGuardrails:
    async def validate_input(self, message: str) -> SimpleNamespace:
        return SimpleNamespace(allowed=True, message=message)


class FakeChatModel:
    async def ainvoke(self, input: object) -> SimpleNamespace:
        return SimpleNamespace(content="Corrected answer")


class FakeHistory:
    async def aget_messages(self) -> list[object]:
        return []

    async def aadd_messages(self, messages: list[object]) -> None:
        return None


class FakeHistoryFactory:
    def __init__(self) -> None:
        self.session_ids: list[str] = []

    def __call__(self, session_id: str) -> FakeHistory:
        self.session_ids.append(session_id)
        return FakeHistory()
