"""NeMo Guardrails integration for chatbot input safety."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings, get_settings

OFF_TOPIC_RESPONSE = "Vamos focar no exercício de inglês."
JAILBREAK_RESPONSE = "Não posso ignorar minhas instruções. Vamos focar no exercício de inglês."

OFF_TOPIC_KEYWORDS = (
    "receita",
    "política",
    "politica",
    "futebol",
    "esporte",
    "esportes",
    "investimento",
    "criptomoeda",
)
JAILBREAK_PATTERNS = (
    "ignore as instruções anteriores",
    "ignore as instrucoes anteriores",
    "ignore todas as instruções",
    "ignore todas as instrucoes",
    "forget previous instructions",
    "ignore previous instructions",
    "system prompt",
    "modo desenvolvedor",
    "developer mode",
)


class GuardrailsApp(Protocol):
    """Protocol for NeMo Guardrails applications."""

    async def generate_async(self, messages: list[dict[str, str]]) -> Any:
        """Generate a guarded response asynchronously."""


@dataclass(frozen=True, slots=True)
class GuardrailsCheck:
    """Result of a guardrails input validation."""

    allowed: bool
    message: str
    response: str | None = None


class GuardrailsInitializationError(RuntimeError):
    """Raised when NeMo Guardrails cannot be initialized."""


class GuardrailsEngine:
    """Validate user input before expensive RAG and LLM processing."""

    def __init__(self, rails_app: GuardrailsApp | None = None) -> None:
        """Create the engine with an optional prebuilt NeMo app."""
        self._rails_app = rails_app

    async def validate_input(self, message: str) -> GuardrailsCheck:
        """Validate and normalize user input before AI processing."""
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("message must not be empty")

        static_response = self._static_response_for(normalized_message)
        if static_response is not None:
            return GuardrailsCheck(
                allowed=False,
                message=normalized_message,
                response=static_response,
            )

        if self._rails_app is None:
            return GuardrailsCheck(allowed=True, message=normalized_message)

        rails_response = await self._rails_app.generate_async(
            messages=[{"role": "user", "content": normalized_message}]
        )
        response_text = _extract_response_text(rails_response)
        if response_text in {OFF_TOPIC_RESPONSE, JAILBREAK_RESPONSE}:
            return GuardrailsCheck(
                allowed=False,
                message=normalized_message,
                response=response_text,
            )

        return GuardrailsCheck(allowed=True, message=normalized_message)

    async def validate_output(
        self,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Validate generated output before it is sent to the user."""
        _ = metadata
        return response.strip()

    @staticmethod
    def _static_response_for(message: str) -> str | None:
        """Return a deterministic refusal for obvious off-topic or jailbreak input."""
        normalized = message.casefold()
        if any(pattern in normalized for pattern in JAILBREAK_PATTERNS):
            return JAILBREAK_RESPONSE
        if any(keyword in normalized for keyword in OFF_TOPIC_KEYWORDS):
            return OFF_TOPIC_RESPONSE
        return None


@lru_cache(maxsize=1)
def _get_guardrails_app_cached(config_path: str, openai_api_key: str) -> GuardrailsApp:
    """Initialize and cache the NeMo Guardrails app."""
    if not openai_api_key:
        raise GuardrailsInitializationError("OPENAI_API_KEY is not configured")

    try:
        from nemoguardrails import LLMRails, RailsConfig
    except ImportError as exc:
        raise GuardrailsInitializationError("nemoguardrails is not installed") from exc

    config = RailsConfig.from_path(str(Path(config_path)))
    return LLMRails(config=config, llm=None, verbose=False)


async def get_guardrails_app(settings: Settings | None = None) -> GuardrailsApp:
    """Return the cached NeMo Guardrails app for the configured rails directory."""
    resolved_settings = settings or get_settings()
    return _get_guardrails_app_cached(
        resolved_settings.nemo_guardrails_config_path,
        resolved_settings.openai_api_key,
    )


def _extract_response_text(response: Any) -> str:
    """Extract text from common NeMo response shapes."""
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, str):
            return content.strip()
        output = response.get("output")
        if isinstance(output, str):
            return output.strip()
    return str(response).strip()
