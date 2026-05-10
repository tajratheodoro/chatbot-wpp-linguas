"""AI orchestration service built on LangChain chat models."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

SYSTEM_PERSONA = "Você é um professor de inglês rigoroso que corrige exercícios de alunos"


class AIOrchestrator:
    """Coordinate asynchronous AI responses for incoming WhatsApp messages."""

    def __init__(self, model: BaseChatModel | None = None) -> None:
        """Initialize the orchestrator with an injectable LangChain chat model."""
        self._model = model or ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    async def generate_response(self, message: str) -> str:
        """Generate an AI response asynchronously for a WhatsApp message."""
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("message must not be empty")

        response = await self._model.ainvoke(
            [
                SystemMessage(content=SYSTEM_PERSONA),
                HumanMessage(content=normalized_message),
            ]
        )
        content = response.content
        if isinstance(content, str):
            return content
        return str(content)
