"""Webhook endpoint for Evolution API events."""

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.database.connection import DatabaseConnectionError
from app.database.vector_store import VectorStoreError
from app.models.schemas import WebhookResponse, WhatsAppWebhookPayload
from app.services.ai_orchestrator import AIOrchestrator
from app.services.audio_engine import AudioEngine, AudioProcessingError
from app.services.whatsapp_client import WhatsAppClient

router = APIRouter(prefix="/webhook", tags=["webhook"])


def get_ai_orchestrator() -> AIOrchestrator:
    """Create the AI orchestrator dependency."""
    try:
        return AIOrchestrator()
    except DatabaseConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database is not configured",
        ) from exc


def get_whatsapp_client(settings: Settings = Depends(get_settings)) -> WhatsAppClient:
    """Create the WhatsApp client dependency."""
    return WhatsAppClient(settings=settings)


def get_audio_engine(settings: Settings = Depends(get_settings)) -> AudioEngine:
    """Create the audio engine dependency."""
    try:
        return AudioEngine(settings=settings)
    except AudioProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="audio transcription is not configured",
        ) from exc


@router.post("", response_model=WebhookResponse, status_code=status.HTTP_202_ACCEPTED)
async def receive_webhook(
    payload: WhatsAppWebhookPayload,
    settings: Settings = Depends(get_settings),
    ai_orchestrator: AIOrchestrator = Depends(get_ai_orchestrator),
    whatsapp_client: WhatsAppClient = Depends(get_whatsapp_client),
    audio_engine: AudioEngine = Depends(get_audio_engine),
) -> WebhookResponse:
    """Process an incoming WhatsApp text message and send an AI response."""
    if payload.event != "messages.upsert":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="unsupported webhook event",
        )

    try:
        incoming_text = await _extract_incoming_text(
            payload=payload,
            whatsapp_client=whatsapp_client,
            audio_engine=audio_engine,
        )
        ai_response = await ai_orchestrator.generate_response(incoming_text)
        await whatsapp_client.send_text_message(phone=payload.phone, text=ai_response)
        audio_response = await audio_engine.generate_audio_base64(
            text=ai_response,
            voice_id=settings.edge_tts_voice,
        )
        await whatsapp_client.send_audio_message(
            phone=payload.phone,
            base64_audio=audio_response,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="curriculum context is unavailable",
        ) from exc
    except AudioProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid or unsupported audio message",
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


async def _extract_incoming_text(
    payload: WhatsAppWebhookPayload,
    whatsapp_client: WhatsAppClient,
    audio_engine: AudioEngine,
) -> str:
    """Extract text from text or audio WhatsApp messages."""
    if not payload.is_audio_message:
        return payload.text_message

    media_base64 = payload.audio_base64
    if media_base64 is None:
        if payload.message_id is None:
            raise ValueError("audio message id is required to fetch media")
        media_base64 = await whatsapp_client.get_base64_from_media(payload.message_id)

    return await audio_engine.transcribe_audio_from_base64(media_base64)
