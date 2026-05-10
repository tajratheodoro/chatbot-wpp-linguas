"""AI orchestration service built on LangChain chat models."""

from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.database.vector_store import CurriculumVectorStore

SYSTEM_PERSONA = "Você é um professor de inglês rigoroso que corrige exercícios de alunos"


class ContextStore(Protocol):
    """Protocol for asynchronous curriculum context retrieval."""

    async def search_context(self, query: str, limit: int = 3) -> str:
        """Return context related to the incoming student message."""


class ChatModel(Protocol):
    """Protocol for asynchronous LangChain-compatible chat models."""

    async def ainvoke(self, input: Any) -> Any:
        """Invoke the chat model asynchronously."""


class AIOrchestrator:
    """Coordinate RAG retrieval and asynchronous AI responses."""

    def __init__(
        self,
        model: ChatModel | BaseChatModel | None = None,
        vector_store: ContextStore | None = None,
    ) -> None:
        """Initialize the orchestrator with injectable model and context store."""
        self._model = model or ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self._vector_store = vector_store or CurriculumVectorStore()
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
        """Generate an AI response using retrieved curriculum context."""
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("message must not be empty")

        context = await self._vector_store.search_context(normalized_message)
        prompt_value = await self._prompt.ainvoke(
            {
                "context": context or "Nenhum contexto curricular encontrado.",
                "message": normalized_message,
            }
        )
        response = await self._model.ainvoke(prompt_value)
        content = response.content
        if isinstance(content, str):
            return content
        return str(content)
