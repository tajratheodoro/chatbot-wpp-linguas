import base64

import pytest
from fastapi.testclient import TestClient

from app.api.webhook import get_ai_orchestrator, get_audio_engine, get_whatsapp_client
from app.models.schemas import WhatsAppWebhookPayload
from main import app


class FakeAIOrchestrator:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def generate_response(self, student_message: str, session_id: str) -> str:
        self.messages.append(f"{session_id}:{student_message}")
        return f"corrigido: {student_message}"


class FakeAudioEngine:
    def __init__(self) -> None:
        self.generated: list[str] = []

    async def transcribe_audio_from_base64(self, base64_data: str) -> str:
        if base64_data == "invalid-base64":
            raise ValueError("invalid base64 audio")
        return "I has a apple"

    async def generate_audio_base64(self, text: str) -> str:
        self.generated.append(text)
        return base64.b64encode(b"reply-audio").decode("ascii")


class FakeWhatsAppClient:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.sent_audio: list[tuple[str, str]] = []
        self.media_requests: list[str] = []
        self.events: list[str] = []

    async def get_base64_from_media(self, message_id: str) -> str:
        self.media_requests.append(message_id)
        return base64.b64encode(b"audio").decode("ascii")

    async def send_text_message(self, phone: str, text: str) -> dict[str, object]:
        self.sent.append((phone, text))
        self.events.append("text")
        return {"sent": True}

    async def send_audio_message(self, phone: str, base64_audio: str) -> dict[str, object]:
        self.sent_audio.append((phone, base64_audio))
        self.events.append("audio")
        return {"sent": True}


@pytest.fixture()
def ai_orchestrator() -> FakeAIOrchestrator:
    return FakeAIOrchestrator()


@pytest.fixture()
def whatsapp_client() -> FakeWhatsAppClient:
    return FakeWhatsAppClient()


@pytest.fixture()
def audio_engine() -> FakeAudioEngine:
    return FakeAudioEngine()


@pytest.fixture(autouse=True)
def override_dependencies(
    ai_orchestrator: FakeAIOrchestrator,
    audio_engine: FakeAudioEngine,
    whatsapp_client: FakeWhatsAppClient,
) -> None:
    app.dependency_overrides[get_ai_orchestrator] = lambda: ai_orchestrator
    app.dependency_overrides[get_whatsapp_client] = lambda: whatsapp_client
    app.dependency_overrides[get_audio_engine] = lambda: audio_engine
    yield
    app.dependency_overrides.clear()


def test_audio_payload_exposes_message_id_and_base64() -> None:
    payload = WhatsAppWebhookPayload.model_validate(
        {
            "event": "messages.upsert",
            "data": {
                "key": {
                    "id": "MSG123",
                    "remoteJid": "5511999999999@s.whatsapp.net",
                },
                "message": {
                    "audioMessage": {
                        "base64": base64.b64encode(b"audio").decode("ascii"),
                        "mimetype": "audio/ogg",
                    }
                },
            },
        }
    )

    assert payload.message_id == "MSG123"
    assert payload.is_audio_message is True
    assert payload.audio_base64 is not None


def test_webhook_transcribes_audio_before_ai_response(
    ai_orchestrator: FakeAIOrchestrator,
    audio_engine: FakeAudioEngine,
    whatsapp_client: FakeWhatsAppClient,
) -> None:
    payload = {
        "event": "messages.upsert",
        "data": {
            "key": {
                "id": "MSG123",
                "remoteJid": "5511999999999@s.whatsapp.net",
            },
            "message": {"audioMessage": {"mimetype": "audio/ogg"}},
        },
    }

    response = TestClient(app).post("/api/webhook", json=payload)

    assert response.status_code == 202
    assert ai_orchestrator.messages == ["5511999999999@s.whatsapp.net:I has a apple"]
    assert whatsapp_client.media_requests == ["MSG123"]
    assert whatsapp_client.sent == [
        ("5511999999999@s.whatsapp.net", "corrigido: I has a apple")
    ]
    assert audio_engine.generated == ["corrigido: I has a apple"]
    assert whatsapp_client.sent_audio == [
        (
            "5511999999999@s.whatsapp.net",
            base64.b64encode(b"reply-audio").decode("ascii"),
        )
    ]
    assert whatsapp_client.events == ["text", "audio"]


def test_webhook_rejects_corrupted_audio() -> None:
    payload = {
        "event": "messages.upsert",
        "data": {
            "key": {
                "id": "MSG123",
                "remoteJid": "5511999999999@s.whatsapp.net",
            },
            "message": {"audioMessage": {"base64": "invalid-base64"}},
        },
    }

    response = TestClient(app).post("/api/webhook", json=payload)

    assert response.status_code == 422
