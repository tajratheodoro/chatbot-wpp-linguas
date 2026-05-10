"""Evolution API WhatsApp client boundary."""

from typing import Any

import httpx

from app.config import Settings


class WhatsAppClient:
    """Async client for sending messages through Evolution API."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.evolution_api_url.rstrip("/")
        self._api_key = settings.evolution_api_key

    async def send_text(self, phone: str, message: str) -> dict[str, Any]:
        """Send a text message to a WhatsApp contact."""
        headers = {"apikey": self._api_key} if self._api_key else {}
        payload = {"number": phone, "text": message}
        async with httpx.AsyncClient(base_url=self._base_url, timeout=15.0) as client:
            response = await client.post("/message/sendText", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

