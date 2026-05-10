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

    async def get_base64_from_media(self, message_id: str) -> str:
        """Fetch base64 media content for a WhatsApp message."""
        normalized_message_id = message_id.strip()
        if not normalized_message_id:
            raise ValueError("message_id must not be empty")

        headers = {"apikey": self._api_key} if self._api_key else {}
        payload = {"messageId": normalized_message_id}
        async with httpx.AsyncClient(base_url=self._base_url, timeout=30.0) as client:
            response = await client.post(
                "/chat/getBase64FromMediaMessage",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        media_base64 = _extract_media_base64(data)
        if not media_base64:
            raise ValueError("media base64 was not returned by Evolution API")
        return media_base64

    async def send_audio_message(self, phone: str, base64_audio: str) -> dict[str, Any]:
        """Send a base64 audio message to a WhatsApp contact."""
        normalized_phone = phone.strip()
        normalized_audio = base64_audio.strip()
        if not normalized_phone:
            raise ValueError("phone must not be empty")
        if not normalized_audio:
            raise ValueError("base64_audio must not be empty")

        headers = {"apikey": self._api_key} if self._api_key else {}
        payload = {"number": normalized_phone, "audio": normalized_audio}
        async with httpx.AsyncClient(base_url=self._base_url, timeout=30.0) as client:
            response = await client.post(
                "/message/sendWhatsAppAudio",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    async def send_text(self, phone: str, message: str) -> dict[str, Any]:
        """Backward-compatible alias for sending a text message."""
        return await self.send_text_message(phone=phone, text=message)


def _extract_media_base64(payload: dict[str, Any]) -> str | None:
    """Extract base64 media from common Evolution API response shapes."""
    candidates: list[Any] = [
        payload.get("base64"),
        payload.get("base64MediaMessage"),
        payload.get("media"),
    ]
    data = payload.get("data")
    if isinstance(data, dict):
        candidates.extend(
            [
                data.get("base64"),
                data.get("base64MediaMessage"),
                data.get("media"),
            ]
        )

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None
