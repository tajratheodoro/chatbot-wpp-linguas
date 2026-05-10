"""AI orchestration service built on LangChain runnables."""

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda


class AIOrchestrator:
    """Coordinate prompt construction and asynchronous model execution."""

    def __init__(self, model: Runnable[Any, Any] | None = None) -> None:
        """Initialize the orchestrator with an optional LangChain-compatible model."""
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful English tutor on WhatsApp. "
                    "Correct exercises clearly and keep responses concise.",
                ),
                (
                    "human",
                    "Conversation id: {conversation_id}\nStudent message: {message}",
                ),
            ]
        )
        self._model = model or RunnableLambda(self._fallback_response)
        self._chain: Runnable[dict[str, str], str] = self._prompt | self._model | StrOutputParser()

    async def generate_response(self, message: str, conversation_id: str | None = None) -> str:
        """Generate an AI response asynchronously for a WhatsApp message."""
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("message must not be empty")

        return await self._chain.ainvoke(
            {
                "conversation_id": conversation_id or "anonymous",
                "message": normalized_message,
            }
        )

    async def _fallback_response(self, prompt_value: Any) -> str:
        """Return a deterministic response when no external LLM is injected."""
        _ = prompt_value
        return "Recebi sua mensagem. Em breve vou corrigir o exercício com mais detalhes."

