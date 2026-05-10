"""Audio generation/transcription boundary for ElevenLabs."""


class AudioEngine:
    """Async interface for future ElevenLabs audio features."""

    async def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech audio bytes."""
        _ = text
        return b""

