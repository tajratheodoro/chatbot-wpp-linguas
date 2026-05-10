"""AI orchestration service built on LangChain chat models."""

from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.config import Settings, get_settings
from app.core.guardrails import GuardrailsEngine
from app.database.connection import DatabaseConnectionError
from app.database.memory import ChatMemoryError, PostgresAsyncChatMessageHistory
from app.database.vector_store import CurriculumVectorStore

SYSTEM_PERSONA = "Voce e um professor de ingles rigoroso que corrige exercicios de alunos"
GROQ_CHAT_MODEL = "llama3-8b-8192"


class ContextStore(Protocol):
    """Protocol for asynchronous curriculum context retrieval."""

    async def search_context(self, query: str, limit: int = 3) -> str:
        """Return context related to the incoming student message."""


class ChatModel(Protocol):
    """Protocol for asynchronous LangChain-compatible chat models."""

    async def ainvoke(self, input: Any) -> Any:
        """Invoke the chat model asynchronously."""


class HistoryFactory(Protocol):
    """Protocol for per-session chat history creation."""

    def __call__(self, session_id: str) -> BaseChatMessageHistory:
        """Return persisted chat history for a session id."""


class AIOrchestrationError(RuntimeError):
    """Raised when the LLM response flow fails."""


def _require_groq_api_key(settings: Settings) -> str:
    """Return the configured Groq API key or raise a safe error."""
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not configured")
    return settings.groq_api_key


def _create_groq_chat_model(settings: Settings) -> BaseChatModel:
    """Create a ChatGroq model using only settings-managed credentials."""
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ValueError("langchain-groq is not installed") from exc

    return ChatGroq(
        model=GROQ_CHAT_MODEL,
        temperature=0.2,
        api_key=_require_groq_api_key(settings),
    )


class AIOrchestrator:
    """Coordinate guardrails, RAG retrieval and asynchronous AI responses."""

    def __init__(
        self,
        model: ChatModel | BaseChatModel | None = None,
        vector_store: ContextStore | None = None,
        guardrails_engine: GuardrailsEngine | None = None,
        history_factory: HistoryFactory | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the orchestrator with injectable model and safety boundaries."""
        self._settings = settings or get_settings()
        self._model = model or _create_groq_chat_model(self._settings)
        self._vector_store = vector_store
        self._guardrails_engine = guardrails_engine or GuardrailsEngine()
        self._history_factory = history_factory or self._create_session_history
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PERSONA),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    "Contexto curricular recuperado:\n{context}\n\n"
                    "Mensagem do aluno:\n{question}\n\n"
                    "Use o contexto quando ele for relevante. Corrija o exercicio "
                    "com rigor e explique de forma objetiva.",
                ),
            ]
        )
        model_runnable = self._model if isinstance(self._model, Runnable) else RunnableLambda(
            self._model.ainvoke
        )
        self._chain_with_history = RunnableWithMessageHistory(
            self._prompt | model_runnable,
            self._history_factory,
            input_messages_key="question",
            history_messages_key="history",
        )

    async def generate_response(self, student_message: str, session_id: str) -> str:
        """Generate a RAG-grounded answer for a student's English exercise."""
        normalized_message = student_message.strip()
        if not normalized_message:
            raise ValueError("message must not be empty")
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise ValueError("session_id must not be empty")

        guardrails_check = await self._guardrails_engine.validate_input(normalized_message)
        if not guardrails_check.allowed:
            return guardrails_check.response or "Vamos focar no exercicio de ingles."

        vector_store = self._vector_store or CurriculumVectorStore(settings=self._settings)
        context = await vector_store.search_context(guardrails_check.message)
        try:
            response = await self._chain_with_history.ainvoke(
                {
                    "context": context or "Nenhum contexto curricular encontrado.",
                    "question": guardrails_check.message,
                },
                config={"configurable": {"session_id": normalized_session_id}},
            )
        except (ChatMemoryError, DatabaseConnectionError):
            raise
        except Exception as exc:
            raise AIOrchestrationError("AI response generation failed") from exc

        content = response.content
        if isinstance(content, str):
            return content
        return str(content)

    def _create_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Create PostgreSQL-backed LangChain history for one WhatsApp student."""
        return PostgresAsyncChatMessageHistory(
            session_id=session_id,
            settings=self._settings,
        )
