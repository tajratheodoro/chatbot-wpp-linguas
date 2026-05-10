"""Audio generation and transcription boundary."""

import base64
import binascii
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings, get_settings

GROQ_WHISPER_MODEL = "whisper-large-v3"
EDGE_TTS_DEFAULT_VOICE = "en-US-AriaNeural"


class AudioProcessingError(RuntimeError):
    """Raised when incoming audio cannot be processed safely."""


class GroqTranscriptions(Protocol):
    """Protocol for Groq audio transcriptions."""

    async def create(self, file: Any, model: str) -> Any:
        """Create an audio transcription asynchronously."""


class GroqAudio(Protocol):
    """Protocol for Groq audio APIs."""

    transcriptions: GroqTranscriptions


class GroqClient(Protocol):
    """Protocol for the async Groq client."""

    audio: GroqAudio


class EdgeTTSCommunicate(Protocol):
    """Protocol for edge_tts Communicate objects."""

    async def save(self, path: str) -> None:
        """Save generated speech to a file path."""


class AudioEngine:
    """Async interface for Groq STT and Edge-TTS generation."""

    def __init__(
        self,
        settings: Settings | None = None,
        groq_client: GroqClient | None = None,
        tts_communicate_factory: Callable[[str, str], EdgeTTSCommunicate] | None = None,
    ) -> None:
        """Initialize the audio engine with settings-managed credentials."""
        resolved_settings = settings or get_settings()
        self._groq_api_key = resolved_settings.groq_api_key
        self._groq_client = groq_client
        self._tts_communicate_factory = tts_communicate_factory or _create_edge_tts_communicate

    async def transcribe_audio_from_base64(self, base64_data: str) -> str:
        """Decode base64 audio into a temporary .ogg file and transcribe it with Groq."""
        try:
            audio_bytes = base64.b64decode(base64_data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise AudioProcessingError("invalid base64 audio") from exc

        if not audio_bytes:
            raise AudioProcessingError("empty audio payload")

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as audio_file:
                temp_path = audio_file.name
                audio_file.write(audio_bytes)
                audio_file.flush()

            groq_client = self._groq_client or _create_groq_client(self._groq_api_key)
            with open(temp_path, "rb") as audio_file:
                transcription = await groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model=GROQ_WHISPER_MODEL,
                )
        except OSError as exc:
            raise AudioProcessingError("temporary audio processing failed") from exc
        except Exception as exc:
            raise AudioProcessingError("audio transcription failed") from exc
        finally:
            if temp_path is not None:
                _safe_unlink(temp_path)

        text = _extract_transcription_text(transcription)
        if not text:
            raise AudioProcessingError("empty audio transcription")
        return text

    async def generate_audio_base64(self, text: str, voice_id: str | None = None) -> str:
        """Generate Edge-TTS speech in a temporary .mp3 and return base64 audio."""
        normalized_text = text.strip()
        normalized_voice = (voice_id or EDGE_TTS_DEFAULT_VOICE).strip()
        if not normalized_text:
            raise AudioProcessingError("text must not be empty")
        if not normalized_voice:
            raise AudioProcessingError("voice_id must not be empty")

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
                temp_path = audio_file.name

            communicate = self._tts_communicate_factory(normalized_text, normalized_voice)
            await communicate.save(temp_path)
            audio_bytes = Path(temp_path).read_bytes()
        except OSError as exc:
            raise AudioProcessingError("temporary audio generation failed") from exc
        except Exception as exc:
            raise AudioProcessingError("audio generation failed") from exc
        finally:
            if temp_path is not None:
                _safe_unlink(temp_path)

        if not audio_bytes:
            raise AudioProcessingError("empty generated audio")
        return base64.b64encode(audio_bytes).decode("ascii")

    async def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech audio bytes."""
        audio_base64 = await self.generate_audio_base64(text, EDGE_TTS_DEFAULT_VOICE)
        return base64.b64decode(audio_base64)


async def transcribe_audio_from_base64(base64_data: str) -> str:
    """Transcribe base64-encoded audio using the default audio engine."""
    return await AudioEngine().transcribe_audio_from_base64(base64_data)


async def generate_audio_base64(text: str, voice_id: str | None = None) -> str:
    """Generate base64-encoded speech using the default audio engine."""
    return await AudioEngine().generate_audio_base64(text=text, voice_id=voice_id)


def _create_groq_client(api_key: str) -> GroqClient:
    """Create a Groq async client using only settings-managed credentials."""
    if not api_key:
        raise AudioProcessingError("GROQ_API_KEY is not configured")
    try:
        from groq import AsyncGroq
    except ImportError as exc:
        raise AudioProcessingError("groq is not installed") from exc
    return AsyncGroq(api_key=api_key)


def _create_edge_tts_communicate(text: str, voice: str) -> EdgeTTSCommunicate:
    """Create an Edge-TTS communicate object lazily."""
    try:
        import edge_tts
    except ImportError as exc:
        raise AudioProcessingError("edge-tts is not installed") from exc
    return edge_tts.Communicate(text, voice=voice)


def _extract_transcription_text(transcription: Any) -> str:
    """Extract text from Groq transcription response shapes."""
    text = getattr(transcription, "text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(transcription, dict):
        value = transcription.get("text")
        if isinstance(value, str):
            return value.strip()
    if isinstance(transcription, str):
        return transcription.strip()
    return ""


def _safe_unlink(path: str) -> None:
    """Delete a temporary file without leaking path details in callers."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
