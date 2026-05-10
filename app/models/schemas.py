"""Pydantic schemas used by the API layer."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    """Health-check response payload."""

    status: str


class EvolutionMessageKey(BaseModel):
    """Message key metadata sent by Evolution API."""

    remote_jid: str = Field(alias="remoteJid", min_length=1)

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


class EvolutionMessageContent(BaseModel):
    """Supported message content from Evolution API."""

    text_message: EvolutionTextMessage = Field(alias="textMessage")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


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
    def text_message(self) -> str:
        """Return the incoming text message."""
        return self.data.message.text_message.text


class WebhookResponse(BaseModel):
    """Acknowledgement returned after accepting a webhook."""

    accepted: bool
    event: str
    phone: str
