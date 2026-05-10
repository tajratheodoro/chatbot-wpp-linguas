"""Webhook endpoint for Evolution API events."""

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.models.schemas import WebhookResponse, WhatsAppWebhookPayload
from app.services.ai_orchestrator import AIOrchestrator
from app.services.whatsapp_client import WhatsAppClient

router = APIRouter(prefix="/webhook", tags=["webhook"])


def get_ai_orchestrator() -> AIOrchestrator:
    """Create the AI orchestrator dependency."""
    return AIOrchestrator()


def get_whatsapp_client(settings: Settings = Depends(get_settings)) -> WhatsAppClient:
    """Create the WhatsApp client dependency."""
    return WhatsAppClient(settings=settings)


@router.post("", response_model=WebhookResponse, status_code=status.HTTP_202_ACCEPTED)
async def receive_webhook(
    payload: WhatsAppWebhookPayload,
    ai_orchestrator: AIOrchestrator = Depends(get_ai_orchestrator),
    whatsapp_client: WhatsAppClient = Depends(get_whatsapp_client),
) -> WebhookResponse:
    """Process an incoming WhatsApp text message and send an AI response."""
    if payload.event != "messages.upsert":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="unsupported webhook event",
        )

    try:
        ai_response = await ai_orchestrator.generate_response(payload.text_message)
        await whatsapp_client.send_text_message(phone=payload.phone, text=ai_response)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="failed to send WhatsApp message",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evolution API is unavailable",
        ) from exc

    return WebhookResponse(accepted=True, event=payload.event, phone=payload.phone)
