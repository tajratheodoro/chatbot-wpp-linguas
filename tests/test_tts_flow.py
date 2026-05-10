import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.services.audio_engine import AudioEngine, GROQ_WHISPER_MODEL


class FakeCommunicate:
    calls: list[dict[str, str]] = []

    def __init__(self, text: str, voice: str) -> None:
        self.text = text
        self.voice = voice
        self.calls.append({"text": text, "voice": voice})

    async def save(self, path: str) -> None:
        Path(path).write_bytes(b"edge-audio")


class FakeTranscriptions:
    def __init__(self) -> None:
        self.temp_path: str | None = None
        self.model: str | None = None
        self.existed_during_call = False

    async def create(self, file: object, model: str) -> SimpleNamespace:
        self.temp_path = getattr(file, "name")
        self.model = model
        self.existed_during_call = Path(self.temp_path).exists()
        return SimpleNamespace(text="I have an apple")


class FakeGroqAudio:
    def __init__(self, transcriptions: FakeTranscriptions) -> None:
        self.transcriptions = transcriptions


class FakeGroqClient:
    def __init__(self, transcriptions: FakeTranscriptions) -> None:
        self.audio = FakeGroqAudio(transcriptions)


@pytest.mark.asyncio()
async def test_generate_audio_base64_uses_edge_tts_tempfile_cleanup() -> None:
    FakeCommunicate.calls = []
    settings = Settings(GROQ_API_KEY="test-key")
    engine = AudioEngine(settings=settings, tts_communicate_factory=FakeCommunicate)

    result = await engine.generate_audio_base64("Short answer")

    assert result == base64.b64encode(b"edge-audio").decode("ascii")
    assert FakeCommunicate.calls == [{"text": "Short answer", "voice": "en-US-AriaNeural"}]


@pytest.mark.asyncio()
async def test_transcribe_audio_uses_groq_tempfile_and_cleans_up() -> None:
    transcriptions = FakeTranscriptions()
    engine = AudioEngine(
        settings=Settings(GROQ_API_KEY="test-key"),
        groq_client=FakeGroqClient(transcriptions),
    )
    audio_base64 = base64.b64encode(b"ogg-bytes").decode("ascii")

    result = await engine.transcribe_audio_from_base64(audio_base64)

    assert result == "I have an apple"
    assert transcriptions.model == GROQ_WHISPER_MODEL
    assert transcriptions.temp_path is not None
    assert transcriptions.existed_during_call is True
    assert Path(transcriptions.temp_path).exists() is False
