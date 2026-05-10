"""Evolution API WhatsApp client boundary."""

from typing import Any

import httpx

from app.config import Settings, get_settings


class WhatsAppClient:
    """Async client for sending messages through Evolution API."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the client with Evolution API settings."""
        settings = settings or get_settings()
        self._base_url = settings.evolution_api_url.rstrip("/")
        self._api_key = settings.evolution_api_key

    async def send_text_message(self, phone: str, text: str) -> dict[str, Any]:
        """Send a text message to a WhatsApp contact."""
        normalized_phone = phone.strip()
        normalized_text = text.strip()
        if not normalized_phone:
            raise ValueError("phone must not be empty")
        if not normalized_text:
            raise ValueError("text must not be empty")

        headers = {"apikey": self._api_key} if self._api_key else {}
        payload = {"number": normalized_phone, "text": normalized_text}
        async with httpx.AsyncClient(base_url=self._base_url, timeout=15.0) as client:
            response = await client.post("/message/sendText", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def send_text(self, phone: str, message: str) -> dict[str, Any]:
        """Backward-compatible alias for sending a text message."""
        return await self.send_text_message(phone=phone, text=message)
