"""NeMo Guardrails integration boundary."""

from typing import Any


class GuardrailsEngine:
    """Thin async boundary for future NeMo Guardrails checks."""

    async def validate_input(self, message: str) -> str:
        """Validate and normalize user input before AI processing."""
        return message.strip()

    async def validate_output(self, response: str, metadata: dict[str, Any] | None = None) -> str:
        """Validate generated output before it is sent to the user."""
        return response.strip()

