"""AI orchestration service built on LangChain chat models."""

from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings, get_settings
from app.core.guardrails import GuardrailsEngine
from app.database.vector_store import CurriculumVectorStore

SYSTEM_PERSONA = "Você é um professor de inglês rigoroso que corrige exercícios de alunos"
GROQ_CHAT_MODEL = "llama3-8b-8192"


class ContextStore(Protocol):
    """Protocol for asynchronous curriculum context retrieval."""

    async def search_context(self, query: str, limit: int = 3) -> str:
        """Return context related to the incoming student message."""


class ChatModel(Protocol):
    """Protocol for asynchronous LangChain-compatible chat models."""

    async def ainvoke(self, input: Any) -> Any:
        """Invoke the chat model asynchronously."""


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
        settings: Settings | None = None,
    ) -> None:
        """Initialize the orchestrator with injectable model and safety boundaries."""
        resolved_settings = settings or get_settings()
        self._model = model or _create_groq_chat_model(resolved_settings)
        self._vector_store = vector_store or CurriculumVectorStore(settings=resolved_settings)
        self._guardrails_engine = guardrails_engine or GuardrailsEngine()
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PERSONA),
                (
                    "human",
                    "Contexto curricular recuperado:\n{context}\n\n"
                    "Mensagem do aluno:\n{message}\n\n"
                    "Corrija o exercício com rigor e explique de forma objetiva.",
                ),
            ]
        )

    async def generate_response(self, message: str) -> str:
        """Generate an AI response after validating input with guardrails."""
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("message must not be empty")

        guardrails_check = await self._guardrails_engine.validate_input(normalized_message)
        if not guardrails_check.allowed:
            return guardrails_check.response or "Vamos focar no exercício de inglês."

        context = await self._vector_store.search_context(guardrails_check.message)
        prompt_value = await self._prompt.ainvoke(
            {
                "context": context or "Nenhum contexto curricular encontrado.",
                "message": guardrails_check.message,
            }
        )
        response = await self._model.ainvoke(prompt_value)
        content = response.content
        if isinstance(content, str):
            return content
        return str(content)
