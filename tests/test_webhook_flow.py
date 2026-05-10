import pytest
from fastapi.testclient import TestClient

from app.api.webhook import get_ai_orchestrator, get_whatsapp_client
from main import app


class FakeAIOrchestrator:
    async def generate_response(self, student_message: str, session_id: str) -> str:
        return f"corrigido: {student_message} ({session_id})"


class FakeWhatsAppClient:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.sent_audio: list[tuple[str, str]] = []

    async def send_text_message(self, phone: str, text: str) -> dict[str, object]:
        self.sent.append((phone, text))
        return {"sent": True}

    async def send_audio_message(self, phone: str, base64_audio: str) -> dict[str, object]:
        self.sent_audio.append((phone, base64_audio))
        return {"sent": True}


class FakeAudioEngine:
    async def generate_audio_base64(self, text: str) -> str:
        _ = text
        return "audio-base64"


@pytest.fixture()
def whatsapp_client() -> FakeWhatsAppClient:
    return FakeWhatsAppClient()


@pytest.fixture(autouse=True)
def override_dependencies(whatsapp_client: FakeWhatsAppClient) -> None:
    app.dependency_overrides[get_ai_orchestrator] = lambda: FakeAIOrchestrator()
    app.dependency_overrides[get_whatsapp_client] = lambda: whatsapp_client
    from app.api.webhook import get_audio_engine

    app.dependency_overrides[get_audio_engine] = lambda: FakeAudioEngine()
    yield
    app.dependency_overrides.clear()


def test_webhook_extracts_message_and_replies(whatsapp_client: FakeWhatsAppClient) -> None:
    payload = {
        "event": "messages.upsert",
        "data": {
            "key": {"remoteJid": "5511999999999@s.whatsapp.net"},
            "message": {"textMessage": {"text": "I has a apple"}},
        },
    }

    response = TestClient(app).post("/api/webhook", json=payload)

    assert response.status_code == 202
    assert response.json() == {
        "accepted": True,
        "event": "messages.upsert",
        "phone": "5511999999999@s.whatsapp.net",
    }
    assert whatsapp_client.sent == [
        (
            "5511999999999@s.whatsapp.net",
            "corrigido: I has a apple (5511999999999@s.whatsapp.net)",
        )
    ]


def test_webhook_rejects_payload_without_text_message() -> None:
    payload = {
        "event": "messages.upsert",
        "data": {
            "key": {"remoteJid": "5511999999999@s.whatsapp.net"},
            "message": {},
        },
    }

    response = TestClient(app).post("/api/webhook", json=payload)

    assert response.status_code == 422
