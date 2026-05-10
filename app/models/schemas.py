"""Pydantic schemas used by the API layer."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    """Health-check response payload."""

    status: str


class EvolutionMessageKey(BaseModel):
    """Message key metadata sent by Evolution API."""

    remote_jid: str = Field(alias="remoteJid", min_length=1)
    message_id: str | None = Field(default=None, alias="id")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class EvolutionTextMessage(BaseModel):
    """Text message payload sent by Evolution API."""

    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        """Normalize text and reject empty messages."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("textMessage.text must not be empty")
        return normalized


class EvolutionAudioMessage(BaseModel):
    """Audio message payload sent by Evolution API."""

    base64_data: str | None = Field(default=None, alias="base64")
    mimetype: str | None = None
    url: str | None = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @field_validator("base64_data")
    @classmethod
    def strip_base64(cls, value: str | None) -> str | None:
        """Normalize optional base64 audio data."""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class EvolutionMessageContent(BaseModel):
    """Supported message content from Evolution API."""

    text_message: EvolutionTextMessage | None = Field(default=None, alias="textMessage")
    audio_message: EvolutionAudioMessage | None = Field(default=None, alias="audioMessage")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_validator(mode="after")
    def validate_supported_message(self) -> "EvolutionMessageContent":
        """Ensure the webhook carries a supported message type."""
        if self.text_message is None and self.audio_message is None:
            raise ValueError("message must include textMessage or audioMessage")
        return self


class EvolutionWebhookData(BaseModel):
    """Nested data payload for messages.upsert webhooks."""

    key: EvolutionMessageKey
    message: EvolutionMessageContent

    model_config = ConfigDict(extra="ignore")


class WhatsAppWebhookPayload(BaseModel):
    """Webhook payload for Evolution API messages.upsert events."""

    event: str = Field(default="unknown")
    instance: str | None = None
    data: EvolutionWebhookData

    model_config = ConfigDict(extra="ignore")

    @property
    def phone(self) -> str:
        """Return the sender remote JID."""
        return self.data.key.remote_jid

    @property
    def message_id(self) -> str | None:
        """Return the Evolution API message identifier."""
        return self.data.key.message_id

    @property
    def is_audio_message(self) -> bool:
        """Return whether the webhook contains an audio message."""
        return self.data.message.audio_message is not None

    @property
    def audio_base64(self) -> str | None:
        """Return inline audio base64 when present."""
        audio_message = self.data.message.audio_message
        return audio_message.base64_data if audio_message is not None else None

    @property
    def text_message(self) -> str:
        """Return the incoming text message."""
        if self.data.message.text_message is None:
            raise ValueError("textMessage is not available")
        return self.data.message.text_message.text


class WebhookResponse(BaseModel):
    """Acknowledgement returned after accepting a webhook."""

    accepted: bool
    event: str
    phone: str
