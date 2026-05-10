"""Webhook endpoint for Evolution API events."""

from fastapi import APIRouter, status

from app.models.schemas import WebhookResponse, WhatsAppWebhookPayload

router = APIRouter(prefix="/webhook", tags=["webhook"])


@router.post("", response_model=WebhookResponse, status_code=status.HTTP_202_ACCEPTED)
async def receive_webhook(payload: WhatsAppWebhookPayload) -> WebhookResponse:
    """Accept an incoming WhatsApp webhook event for later processing."""
    return WebhookResponse(accepted=True, event=payload.event)

