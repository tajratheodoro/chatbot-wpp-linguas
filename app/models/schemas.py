"""Pydantic schemas used by the API layer."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health-check response payload."""

    status: str


class WhatsAppWebhookPayload(BaseModel):
    """Initial Evolution API webhook payload shape."""

    event: str = Field(default="unknown")
    instance: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class WebhookResponse(BaseModel):
    """Acknowledgement returned after accepting a webhook."""

    accepted: bool
    event: str

